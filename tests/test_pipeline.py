"""Orchestration and the cost report.

The funnel counts and the token ledger are the project's two claims, so they are
asserted together: a run that reports a good ratio because a stage silently did
nothing is the failure mode worth guarding against.
"""

import pytest
from sqlmodel import select

from app.core.config import settings
from app.models.screening import (
    LLMCallLog,
    ResumeChunk,
    ScreeningResult,
    ScreeningRun,
    RunStatus,
)
from app.services.screening.pipeline import (
    estimate_baseline,
    run_calls,
    run_screening,
)
from app.services.screening.report import breakdown_by_stage, build_report, format_report
from app.services.screening.store import store_chunks
from app.services.screening.chunker import chunk_resume
from tests.conftest import FakeEmbedder, FakeLLM, add_resume, make_job

GO_RESUME = """Liu Guanji
EDUCATION
Huaqiao University Bachelor of Computer Science
Sep 2019 – Jun 2023
PROJECTS
Built an AI operations assistant with Go and the Feishu API.
Implemented a ReAct-style agent loop with Function Calling.
SKILLS
Go, Python, Kubernetes, PostgreSQL
"""

ML_RESUME = """Sun Qi
EDUCATION
Tsinghua University Master of Computer Science
SKILLS
Python, PyTorch, CUDA, LoRA, RAG, vLLM
"""

MARKETING_RESUME = """Wang Fang
EDUCATION
Sun Yat-sen University Bachelor of Marketing
EXPERIENCE
Marketing Intern. Ran social media campaigns.
SKILLS
Social media, copywriting, Excel
"""


@pytest.fixture
def stocked(session):
    """Three resumes spanning the funnel's outcomes, plus a job."""
    job = make_job(session)
    documents = {
        "go": add_resume(session, "go.pdf", GO_RESUME),
        "ml": add_resume(session, "ml.pdf", ML_RESUME),
        "marketing": add_resume(session, "marketing.pdf", MARKETING_RESUME),
    }
    return job, documents


class TestEstimateBaseline:
    """The counterfactual."""

    def test_uses_the_measured_prefix(self, session):
        """The baseline must not be built on a guessed prefix.

        The first version used a configured 800 while the real assembled prompt
        measured about 860, which understated the baseline — and, worse, was not
        measured at all.
        """
        documents = [add_resume(session, "a.pdf", "x" * 1000)]

        with_default, _ = estimate_baseline(documents)
        with_measured, _ = estimate_baseline(documents, prefix_tokens=2000)

        assert with_measured > with_default

    def test_scales_with_resume_length(self, session):
        short = [add_resume(session, "a.pdf", "x" * 100)]
        long = [add_resume(session, "b.pdf", "x" * 10_000)]

        assert estimate_baseline(long)[0] > estimate_baseline(short)[0]

    def test_empty_batch_is_zero(self):
        assert estimate_baseline([]) == (0, 0)

    def test_completion_scales_with_document_count(self, session):
        # Distinct text per row: identical text hashes identically, and the unique
        # constraint on content_hash makes it one document, not three.
        documents = [add_resume(session, f"{i}.pdf", f"resume {i} " + "x" * 100) for i in range(3)]

        _, completion = estimate_baseline(documents)

        assert completion == 3 * settings.PIPELINE.baseline_output_tokens


class TestFullRun:
    """The funnel end to end, with fakes in place of providers."""

    def test_produces_a_ranked_shortlist(self, session, stocked):
        job, _ = stocked
        outcome = run_screening(
            session, job, llm=FakeLLM(default=70), embedder=FakeEmbedder()
        )

        assert outcome.status == RunStatus.SUCCEEDED.value
        assert outcome.verdicts
        assert [verdict.score for verdict in outcome.verdicts] == sorted(
            (verdict.score for verdict in outcome.verdicts), reverse=True
        )

        # Ordered in Python, not by the database. A rejected candidate has a NULL
        # final_rank, and SQLite sorts NULLs first while Postgres sorts them last —
        # so an ORDER BY here yields a different order on each backend. The API
        # layer guards against exactly this trap; the assertion has to as well.
        ranked_rows = sorted(
            (
                row
                for row in session.exec(
                    select(ScreeningResult).where(
                        ScreeningResult.run_id == outcome.run_id
                    )
                ).all()
                if row.final_rank is not None
            ),
            key=lambda row: row.final_rank,
        )

        assert [verdict.resume_id for verdict in outcome.verdicts] == [
            row.resume_id for row in ranked_rows
        ]

    def test_rule_layer_rejects_for_free(self, session, stocked):
        """The marketing resume must be dropped before anything is paid for."""
        job, documents = stocked
        outcome = run_screening(
            session, job, llm=FakeLLM(), embedder=FakeEmbedder()
        )

        assert outcome.counts["rule_rejected"] == 1

        marketing = session.exec(
            select(ScreeningResult).where(
                ScreeningResult.run_id == outcome.run_id,
                ScreeningResult.resume_id == documents["marketing"].id,
            )
        ).one()

        assert marketing.rule_passed is False
        assert marketing.judge_score is None
        assert marketing.final_rank is None
        # Rejection reasons are recorded in both directions and are auditable.
        assert any("无一命中" in reason for reason in marketing.rule_reasons)

    def test_no_judge_call_touches_a_rejected_candidate(self, session, stocked):
        """Every candidate the rules reject is one the paid stage never sees."""
        job, _ = stocked
        llm = FakeLLM()
        outcome = run_screening(session, job, llm=llm, embedder=FakeEmbedder())

        assert outcome.counts["judged"] == outcome.counts["shortlisted"]

    def test_counts_are_consistent(self, session, stocked):
        job, documents = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())

        counts = outcome.counts
        assert counts["documents"] == len(documents)
        assert counts["rule_rejected"] + counts["rule_passed"] == counts["documents"]
        assert counts["shortlisted"] <= counts["rule_passed"]
        assert counts["judged"] <= counts["shortlisted"]

    def test_results_carry_evidence_and_gaps(self, session, stocked):
        """HR reads evidence and gaps; a bare score is not actionable."""
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())

        rows = session.exec(
            select(ScreeningResult).where(ScreeningResult.run_id == outcome.run_id)
        ).all()
        scored = [row for row in rows if row.judge_score is not None]

        assert scored
        assert all(row.judge_evidence for row in scored)
        assert all(row.recommendation for row in scored)

    def test_ledger_matches_the_run_row(self, session, stocked):
        """The summary must be derived from the calls, not maintained alongside them."""
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())

        run = session.get(ScreeningRun, outcome.run_id)
        totals = outcome.ledger.totals()

        assert run.prompt_tokens == totals.prompt_tokens
        assert run.completion_tokens == totals.completion_tokens
        assert run.llm_calls == len(outcome.ledger.entries)

    def test_every_call_is_recorded(self, session, stocked):
        """An unrecorded call invalidates every number in the report."""
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())

        calls = run_calls(session, outcome.run_id)

        assert len(calls) == len(outcome.ledger.entries)
        assert all(call.run_id == outcome.run_id for call in calls)

    def test_ratio_is_reported_even_when_it_is_unflattering(self, session, stocked):
        """A tiny batch can cost *more* than the naive baseline.

        Fixed costs — the shared prompt, embedding the corpus — do not amortise
        over three short resumes, so the comparison can come out below 1. Asserting
        it is always a saving would assert something false; what has to hold is
        that the figure is produced and surfaced rather than hidden when it is
        unflattering.
        """
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())

        assert outcome.actual_tokens > 0
        assert outcome.baseline_tokens > 0
        assert outcome.token_ratio is not None

    def test_stage4_is_skipped_when_disabled(self, session, stocked):
        job, _ = stocked
        outcome = run_screening(
            session, job, llm=FakeLLM(), embedder=FakeEmbedder(), review_borderline=False
        )

        assert outcome.counts["reviewed"] == 0
        assert not any(
            call.purpose == "review" for call in outcome.ledger.entries
        )

    def test_stage4_runs_for_a_borderline_candidate(self, session, stocked):
        """A score inside the band must actually reach the review stage."""
        job, _ = stocked
        threshold = settings.PIPELINE.tier_qualified
        llm = FakeLLM(default=threshold)

        outcome = run_screening(session, job, llm=llm, embedder=FakeEmbedder())

        assert outcome.counts["reviewed"] > 0
        assert any(call.purpose == "review" for call in outcome.ledger.entries)

    def test_empty_pool_succeeds_without_spending(self, session):
        """A run with nothing to screen must not fail, and must not call out."""
        job = make_job(session)
        llm = FakeLLM()
        embedder = FakeEmbedder()

        outcome = run_screening(session, job, llm=llm, embedder=embedder)

        assert outcome.status == RunStatus.SUCCEEDED.value
        assert outcome.counts["documents"] == 0
        assert llm.calls == []
        assert embedder.batches == []

    def test_job_must_be_persisted(self, session):
        """An unsaved job has no id, so the run row could not reference it."""
        from app.models.job import JobDescription

        with pytest.raises(ValueError, match="job.id is None"):
            run_screening(
                session,
                JobDescription(title="unsaved"),
                llm=FakeLLM(),
                embedder=FakeEmbedder(),
            )


class TestFailureHandling:
    """What happens when a run breaks partway."""

    def test_ledger_is_kept_on_failure(self, session, stocked):
        """A failed run still cost money; hiding it hides the cost of the failure."""
        job, _ = stocked

        class Exploding(FakeLLM):
            def complete_json(self, messages, **kwargs):
                super().complete_json(messages, **kwargs)
                raise RuntimeError("provider melted")

        with pytest.raises(RuntimeError):
            run_screening(session, job, llm=Exploding(), embedder=FakeEmbedder())

        run = session.exec(select(ScreeningRun)).one()

        assert run.status == RunStatus.FAILED.value
        assert run.error and "provider melted" in run.error

    def test_rule_rows_survive_a_later_failure(self, session, stocked):
        """Stage-1 rows are committed before any paid stage, so a crash stays auditable."""
        job, _ = stocked

        class Exploding(FakeLLM):
            def complete_json(self, messages, **kwargs):
                raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            run_screening(session, job, llm=Exploding(), embedder=FakeEmbedder())

        run = session.exec(select(ScreeningRun)).one()
        rows = session.exec(
            select(ScreeningResult).where(ScreeningResult.run_id == run.id)
        ).all()

        assert rows
        assert all(row.rule_reasons for row in rows)


class TestCostReport:
    """The numbers, and how they are labelled."""

    @pytest.fixture
    def reported(self, session, stocked):
        """A completed run with its report."""
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(default=70), embedder=FakeEmbedder())
        run = session.get(ScreeningRun, outcome.run_id)
        return run, run_calls(session, outcome.run_id)

    def test_separates_measured_from_estimated(self, reported):
        """The distinction is the point; interleaving them invites over-claiming."""
        run, calls = reported
        report = build_report(run, calls)

        assert report.actual_tokens == run.prompt_tokens + run.completion_tokens
        assert report.baseline_tokens == (
            run.baseline_prompt_tokens + run.baseline_completion_tokens
        )
        assert report.actual_tokens != report.baseline_tokens

    def test_token_and_cost_ratios_are_both_reported(self, reported):
        """Token volume and money diverge, and neither tells the whole story.

        Embedding dominates tokens by volume while costing a fraction as much, so
        a token ratio understates the saving and a cost ratio hides the volume.
        """
        run, calls = reported
        report = build_report(run, calls)

        assert report.ratio is not None
        assert report.cost_ratio is not None
        assert report.cost_ratio != pytest.approx(report.ratio, rel=0.01)

    def test_stage_breakdown_is_ordered_and_complete(self, reported):
        """Stages appear in pipeline order so runs can be compared by eye."""
        _, calls = reported
        stages = breakdown_by_stage(calls)

        purposes = [stage.purpose for stage in stages]
        assert "embed" in purposes
        assert "judge" in purposes
        assert purposes.index("embed") < purposes.index("judge")

    def test_stage_totals_match_the_calls(self, reported):
        _, calls = reported
        stages = breakdown_by_stage(calls)

        assert sum(stage.prompt_tokens for stage in stages) == sum(
            call.prompt_tokens for call in calls
        )
        assert sum(stage.calls for stage in stages) == len(calls)

    def test_accepts_both_ledger_entries_and_rows(self, session, stocked):
        """The pipeline reports from memory; the API reports from the database."""
        job, _ = stocked
        outcome = run_screening(session, job, llm=FakeLLM(), embedder=FakeEmbedder())
        run = session.get(ScreeningRun, outcome.run_id)

        from_entries = build_report(run, outcome.ledger.entries)
        from_rows = build_report(run, run_calls(session, outcome.run_id))

        assert from_entries.actual_tokens == from_rows.actual_tokens
        assert from_entries.actual_est_cost == pytest.approx(from_rows.actual_est_cost)

    def test_rendered_report_labels_estimates(self, reported):
        """A reader must not be able to mistake the counterfactual for a measurement."""
        run, calls = reported
        text = format_report(build_report(run, calls))

        assert "估算" in text
        assert "实测" in text
        assert "倍数" in text

    def test_tokens_per_judged_is_normalised(self, reported):
        """The only figure comparable across runs of different sizes."""
        run, calls = reported
        report = build_report(run, calls)

        if report.funnel.get("judged"):
            assert report.tokens_per_judged == pytest.approx(
                report.actual_tokens / report.funnel["judged"]
            )

    def test_report_on_a_run_with_no_calls(self, session):
        """A run that spent nothing must not divide by zero."""
        run = ScreeningRun(job_id=1)
        session.add(run)
        session.commit()
        session.refresh(run)

        report = build_report(run, [])

        assert report.ratio is None
        assert report.cost_ratio is None
        assert report.stages == ()

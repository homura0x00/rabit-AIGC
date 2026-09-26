"""Stages 3 and 4 — batched judgement and borderline review.

The parsing tests matter more than they look. A judge reply is untrusted model
output arriving as JSON, and the ways it can be wrong — an omitted candidate, an
invented id, a score that is a string or a boolean — all fail *silently* by
default: the candidate simply disappears from the shortlist.
"""

import pytest

from app.services.screening.judge import (
    JudgeCandidate,
    JudgeVerdict,
    build_judge_system_prompt,
    judge,
    parse_verdicts,
    tier_for_score,
)
from app.services.screening.review import (
    ReviewTarget,
    parse_review,
    review,
    select_borderline,
)
from tests.conftest import FakeLLM, make_job


class TestParseVerdicts:
    """Turning untrusted model output into verdicts."""

    LOCAL = {"C1": 101, "C2": 102, "C3": 103}

    def test_valid_reply(self):
        verdicts, failed = parse_verdicts(
            {
                "results": [
                    {"id": "C1", "score": 82, "evidence": ["Go 微服务"], "gaps": []},
                    {"id": "C2", "score": 40, "evidence": [], "gaps": ["无 Go"]},
                    {"id": "C3", "score": 66, "evidence": ["Python"], "gaps": []},
                ]
            },
            self.LOCAL,
        )

        assert len(verdicts) == 3
        assert failed == []
        assert verdicts[101].tier == "strong"

    def test_omitted_candidate_is_reported_not_dropped(self):
        """A candidate missing from the reply must not vanish from the shortlist."""
        verdicts, failed = parse_verdicts(
            {"results": [{"id": "C1", "score": 82}, {"id": "C2", "score": 40}]},
            self.LOCAL,
        )

        assert len(verdicts) == 2
        assert failed == [103]

    def test_unknown_id_is_ignored(self):
        """Binding a score to a guessed candidate corrupts the ranking silently."""
        verdicts, failed = parse_verdicts(
            {"results": [{"id": "C1", "score": 82}, {"id": "C9", "score": 99}]},
            self.LOCAL,
        )

        assert set(verdicts) == {101}
        assert set(failed) == {102, 103}

    def test_duplicate_id_keeps_the_first(self):
        verdicts, _ = parse_verdicts(
            {"results": [{"id": "C1", "score": 82}, {"id": "C1", "score": 10}]},
            self.LOCAL,
        )

        assert verdicts[101].score == 82

    def test_ids_are_case_insensitive(self):
        verdicts, _ = parse_verdicts({"results": [{"id": "c1", "score": 70}]}, self.LOCAL)

        assert 101 in verdicts

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (82, 82.0),
            ("82", 82.0),
            ("82/100", 82.0),
            ("score: 82", 82.0),
            (82.5, 82.5),
            (150, 100.0),
            (-5, 0.0),
        ],
    )
    def test_score_forms_are_accepted(self, raw, expected):
        """Permissive about form, strict about range.

        Rejecting everything but a bare integer would discard usable verdicts over
        formatting, and each discard costs those candidates their score.
        """
        verdicts, failed = parse_verdicts({"results": [{"id": "C1", "score": raw}]}, {"C1": 1})

        assert verdicts[1].score == expected
        assert failed == []

    @pytest.mark.parametrize("raw", [True, False, None, "high", float("nan")])
    def test_unusable_scores_are_rejected(self, raw):
        """``True`` is an int subclass and would otherwise score as 1."""
        verdicts, failed = parse_verdicts({"results": [{"id": "C1", "score": raw}]}, {"C1": 1})

        assert verdicts == {}
        assert failed == [1]

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {"foo": 1},
            {"results": "nope"},
            {"results": None},
        ],
    )
    def test_malformed_payloads_fail_every_candidate(self, payload):
        """A reply with no usable array cannot be partially trusted."""
        verdicts, failed = parse_verdicts(payload, self.LOCAL)

        assert verdicts == {}
        assert set(failed) == set(self.LOCAL.values())

    def test_non_dict_entries_are_skipped(self):
        verdicts, _ = parse_verdicts(
            {"results": ["C1", {"id": "C2", "score": 50}]}, self.LOCAL
        )

        assert set(verdicts) == {102}

    def test_evidence_is_capped_and_cleaned(self):
        """Caps are requested in the prompt and enforced in code, because a prompt is a request."""
        verdicts, _ = parse_verdicts(
            {
                "results": [
                    {
                        "id": "C1",
                        "score": 70,
                        "evidence": ["a" * 500, "b", "c", "d"],
                        "gaps": ["g" * 200, "h", "i"],
                    }
                ]
            },
            {"C1": 1},
        )
        verdict = verdicts[1]

        assert len(verdict.evidence) == 2
        assert max(len(item) for item in verdict.evidence) <= 80
        assert len(verdict.gaps) == 2
        assert max(len(item) for item in verdict.gaps) <= 40

    def test_non_list_evidence_does_not_lose_the_score(self):
        """Missing evidence is a quality problem, not a reason to discard a verdict."""
        verdicts, _ = parse_verdicts(
            {"results": [{"id": "C1", "score": 70, "evidence": "Go"}]}, {"C1": 1}
        )

        assert verdicts[1].evidence == ()


class TestTierMapping:
    """Score to bucket."""

    @pytest.mark.parametrize(
        "score,expected",
        [(100, "strong"), (80, "strong"), (79, "qualified"), (65, "qualified"),
         (64, "borderline"), (50, "borderline"), (49, "weak"), (0, "weak")],
    )
    def test_boundaries_are_inclusive_at_the_bottom(self, score, expected):
        assert tier_for_score(score) == expected


class TestJudgeSystemPrompt:
    """The static prefix, which provider caching depends on."""

    def test_is_byte_identical_across_calls(self, session):
        """An unchanged prefix is the whole basis of the cache-hit saving.

        A prefix that shifts by one character silently turns every cache hit back
        into a miss, and the token totals still look plausible.
        """
        job = make_job(session)

        assert build_judge_system_prompt(job) == build_judge_system_prompt(job)

    def test_contains_the_rubric_and_the_job(self, session):
        job = make_job(session, title="专属岗位名称")
        prompt = build_judge_system_prompt(job)

        assert "评分维度" in prompt
        assert "专属岗位名称" in prompt
        assert "Kubernetes" in prompt

    def test_invariant_text_comes_before_job_text(self, session):
        """Order matters: caching matches the longest common prefix.

        Putting job-specific text first would make the rubric uncacheable across
        jobs that share it.
        """
        job = make_job(session)
        prompt = build_judge_system_prompt(job)

        assert prompt.index("评分维度") < prompt.index("岗位定义")


class TestJudge:
    """Batching, and accounting for what comes back."""

    def _candidates(self, count: int) -> list[JudgeCandidate]:
        return [JudgeCandidate(resume_id=100 + i) for i in range(count)]

    def test_scores_every_candidate(self, session):
        job = make_job(session)
        llm = FakeLLM(default=70)

        outcome = judge(llm, job, self._candidates(3), batch_size=3)

        assert set(outcome.verdicts) == {100, 101, 102}
        assert outcome.failed == ()

    def test_batches_by_configuration(self, session):
        """Five candidates per call is the largest single token lever in the pipeline."""
        job = make_job(session)
        llm = FakeLLM()

        outcome = judge(llm, job, self._candidates(7), batch_size=3)

        assert outcome.calls == 3
        assert [call["batch_size"] for call in llm.calls] == [3, 3, 1]

    def test_every_batch_receives_the_same_prefix(self, session):
        """Per-batch variation would defeat the cache the batching relies on."""
        job = make_job(session)
        llm = FakeLLM()

        judge(llm, job, self._candidates(6), batch_size=3)

        assert len({call["system"] for call in llm.calls}) == 1

    def test_local_ids_are_remapped_per_batch(self, session):
        """Ids are batch-scoped, so a score cannot bind to the wrong candidate.

        The fake re-reads the ids from each message, so a remapping bug would
        surface as a missing or duplicated resume id rather than passing.
        """
        job = make_job(session)
        llm = FakeLLM()

        outcome = judge(llm, job, self._candidates(5), batch_size=2)

        assert set(outcome.verdicts) == {100, 101, 102, 103, 104}
        assert all(call["ids"] == ["C1", "C2"] or call["ids"] == ["C1"] for call in llm.calls)

    def test_empty_shortlist_makes_no_calls(self, session):
        """A run that shortlists nobody must not spend anything."""
        job = make_job(session)
        llm = FakeLLM()

        outcome = judge(llm, job, [], batch_size=5)

        assert outcome.calls == 0
        assert llm.calls == []


class TestSelectBorderline:
    """Choosing who gets a second look."""

    def test_selects_only_the_band(self):
        """The band is in score points, matching the scale it bands.

        The first value was 0.15, carried over from 0-1 similarity thinking while
        scores run 0-100 — the band selected nothing and stage 4 was dead code with
        nothing failing.
        """
        verdicts = [
            JudgeVerdict(resume_id=i, score=score, tier="x")
            for i, score in [(1, 62), (2, 78), (3, 45), (4, 65), (5, 58)]
        ]

        assert select_borderline(verdicts, band=8.0, threshold=65.0) == [4, 1, 5]

    def test_orders_by_distance_from_the_line(self):
        """Nearest first, so a capped set reviews the most uncertain candidates."""
        verdicts = [
            JudgeVerdict(resume_id=i, score=score, tier="x")
            for i, score in [(1, 72), (2, 66), (3, 60)]
        ]

        assert select_borderline(verdicts, band=8.0, threshold=65.0) == [2, 3, 1]

    def test_empty_when_nothing_is_close(self):
        verdicts = [JudgeVerdict(resume_id=1, score=95, tier="strong")]

        assert select_borderline(verdicts, band=8.0, threshold=65.0) == []


class TestParseReview:
    """Reading a review reply."""

    def test_direction_is_derived_from_the_scores(self):
        """The label is narration; the score is the decision quantity.

        Models contradict themselves — reporting "lower" beside a higher number —
        and trusting the label would flip a candidate's outcome based on a field
        nothing downstream reads.
        """
        parsed = parse_review({"score": 90, "verdict": "lower"}, 70)
        assert parsed is not None, "a valid score must parse"
        score, direction, _ = parsed

        assert score == 90
        assert direction == "raise"

    @pytest.mark.parametrize(
        "score,prior,expected",
        [(70, 70, "confirm"), (71, 70, "raise"), (69, 70, "lower")],
    )
    def test_direction_values(self, score, prior, expected):
        parsed = parse_review({"score": score}, prior)
        assert parsed is not None
        assert parsed[1] == expected

    def test_reason_is_bounded(self):
        parsed = parse_review({"score": 70, "reason": "x" * 500}, 70)
        assert parsed is not None
        reason = parsed[2]

        assert len(reason) <= 160

    def test_missing_score_yields_nothing(self):
        assert parse_review({"verdict": "confirm"}, 70) is None
        assert parse_review(None, 70) is None


class TestReview:
    """The review stage."""

    def test_keeps_the_prior_gaps(self, session):
        """The review does not re-derive gaps, so replacing them loses real information.

        An earlier version wrote the review's rationale into ``gaps``, which both
        mislabelled it and silently discarded the shortfalls the judge had found.
        """
        job = make_job(session)
        prior = JudgeVerdict(
            resume_id=1, score=65, tier="qualified", gaps=("未提供 Go 证据",)
        )
        target = ReviewTarget(resume_id=1, full_text="some resume", prior=prior)
        llm = FakeLLM(default=65)

        outcome = review(llm, job, [target])
        revised = outcome.verdicts[1]

        assert revised.gaps == ("未提供 Go 证据",)
        assert revised.note  # the rationale goes in its own field
        assert revised.note != revised.gaps[0]

    def test_marks_confirmed_scores(self, session):
        job = make_job(session)
        prior = JudgeVerdict(resume_id=1, score=65, tier="qualified")
        llm = FakeLLM(default=65)

        outcome = review(llm, job, [ReviewTarget(1, "text", prior)])

        assert outcome.changes[1] == "confirm"

    def test_one_call_per_candidate(self, session):
        """Individual attention is the point; batching would re-truncate context."""
        job = make_job(session)
        targets = [
            ReviewTarget(i, f"resume {i}", JudgeVerdict(resume_id=i, score=65, tier="qualified"))
            for i in (1, 2, 3)
        ]
        llm = FakeLLM()

        outcome = review(llm, job, targets)

        assert outcome.calls == 3
        assert all(call["batch_size"] == 1 for call in llm.calls)

    def test_empty_targets_make_no_calls(self, session):
        job = make_job(session)
        llm = FakeLLM()

        assert review(llm, job, []).calls == 0

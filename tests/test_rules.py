"""Stage 1 — the free filter.

Most of these tests exist because the rule layer's failure mode is silent. Regex
that over-matches floods the paid stages; regex that under-matches discards
qualified people and nothing in the output says so. Neither raises, so the only
defence is asserting the behaviour directly.
"""

import pytest

from app.models.job import (
    JobDescription,
    JobSkillRequirement,
    SkillKind,
    SkillTerm,
    degree_label,
    degree_rank,
    meets_degree,
)
from app.services.screening import rules
from tests.conftest import make_job


def requirement(name: str, aliases=(), kind=SkillKind.REQUIRED.value, weight=1.0):
    """Build a detached requirement with its term attached.

    Rule functions take requirements rather than ids, so tests can exercise them
    without touching a database.
    """
    term = SkillTerm(id=1, name=name, aliases=list(aliases))
    req = JobSkillRequirement(job_id=1, term_id=1, kind=kind, weight=weight)
    req.term = term
    return req


def matched(text: str, name: str, aliases=()) -> bool:
    """Whether a single skill matches the given text."""
    return rules.match_skills(text, [requirement(name, aliases)])[0].matched


class TestSkillMatching:
    """Boundary-safe dictionary matching.

    Every case here is a false positive or false negative observed in practice,
    not a hypothetical.
    """

    @pytest.mark.parametrize(
        "text,should_match",
        [
            ("Google engineer. Django, category.", False),
            ("Built go to market plans and went live.", False),
            ("Backend Developer. Go, Kubernetes, PostgreSQL.", True),
            ("Experienced in Golang microservices.", True),
            ("Gophers unite", False),
        ],
    )
    def test_short_alphabetic_terms_are_case_sensitive(self, text, should_match):
        """``Go`` must not match the English verb, nor be missed in a skill list.

        Both directions are asserted together because the fix for one is the cause
        of the other: lowercasing the term to catch "Go," also makes it match "go
        to market".
        """
        assert matched(text, "Go", ("golang",)) is should_match

    @pytest.mark.parametrize(
        "term,text,should_match",
        [
            ("C++", "Wrote C++ and C# services.", True),
            ("C#", "Wrote C++ and C# services.", True),
            ("Java", "JavaScript and TypeScript only.", False),
            ("Python", "python scripts", True),
            ("Rust", "Wrote Rust daily.", True),
        ],
    )
    def test_symbol_and_length_handling(self, term, text, should_match):
        """A shorter term must not match inside a longer one."""
        assert matched(text, term) is should_match

    def test_digit_bearing_alias_is_case_insensitive(self):
        """``k8s`` must resolve against ``K8s``.

        Case sensitivity is decided by shape, not length alone: a term carrying a
        digit cannot collide with prose, so it is matched case-insensitively
        whatever its length.
        """
        assert matched("Deployed on K8s clusters.", "Kubernetes", ("k8s",)) is True
        assert matched("Stored data in s3 buckets.", "S3") is True

    @pytest.mark.parametrize(
        "text,should_match",
        [
            ("Worked in the R&D department.", False),
            ("Managed C&B budget.", False),
            ("Data analysis with R language.", True),
            ("Tidyverse, R, Python", True),
            ("Statistical modelling in R.", True),
        ],
    )
    def test_single_letter_terms_reject_compounds(self, text, should_match):
        """``R`` must not match ``R&D``.

        Research-and-development shorthand is on a large share of resumes, so a
        plain boundary match reads the leading letter as the R language.
        """
        assert matched(text, "R") is should_match

    def test_aliases_are_merged_not_replaced(self):
        """Surface forms are all tried, canonical name included."""
        assert rules.match_skills("Go and Golang", [requirement("Go", ("golang",))])[0].matched


class TestDegreeGuess:
    """Reading the highest degree from resume text."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Huaqiao University Bachelor of Computer Science", "bachelor"),
            ("硕士 计算机科学与技术", "master"),
            ("博士研究生 清华大学", "doctor"),
            ("大专 软件技术", "college"),
        ],
    )
    def test_recognises_both_scripts(self, text, expected):
        assert rules.guess_degree(text).level == expected

    def test_highest_degree_wins(self):
        """A resume listing a bachelor's and a master's has a master's."""
        guess = rules.guess_degree("Bachelor of Science, then Master of Science")

        assert guess.level == "master"

    def test_requirement_text_is_not_evidence(self):
        """A pasted job requirement must not be read as the candidate's degree.

        "硕士及以上学历优先" is a statement about the job. Counting it as the
        candidate's qualification would inflate exactly the candidates who pasted
        a job posting into their resume header.
        """
        assert rules.guess_degree("岗位要求：硕士及以上学历优先").level is None

    def test_absent_degree_has_zero_confidence(self):
        guess = rules.guess_degree("No degree information provided.")

        assert guess.level is None
        assert guess.confidence == 0.0

    def test_agreement_between_keywords_raises_confidence(self):
        """Confidence keys on *distinct keywords*, not on repetition.

        A resume that says "Bachelor" twice has told you no more than one that says
        it once; one that says both "Bachelor" and "BSc" has. Counting mentions
        would let a footer repeated on every page inflate the signal.
        """
        once = rules.guess_degree("Bachelor of Science")
        agreeing = rules.guess_degree("Bachelor of Science, BSc (Hons)")

        assert agreeing.confidence > once.confidence


class TestExperienceGuess:
    """Estimating years from dated ranges."""

    @pytest.mark.parametrize(
        "text,low,high",
        [
            ("Sep 2019 - Jun 2023", 3.5, 4.0),
            ("September 2019 – Present", 6.0, 7.5),
            ("2019.09 - 2023.06", 3.5, 4.0),
            ("2019年9月 - 2023年6月", 3.5, 4.0),
        ],
    )
    def test_parses_the_formats_resumes_actually_use(self, text, low, high):
        """Month-before-year is the common English form.

        A year-first-only pattern reads every such resume as having no dated
        experience, which suppresses the experience signal for exactly the
        candidates most likely to have it.
        """
        assert low <= rules.guess_experience_years(text).total_years <= high

    def test_overlapping_ranges_are_merged(self):
        """Concurrent roles must not be double counted."""
        guess = rules.guess_experience_years(
            "2019-2023 Company A\n2021-2023 Company B"
        )

        assert guess.total_years == pytest.approx(4.0, abs=0.6)

    def test_no_dates_yields_zero_confidence(self):
        guess = rules.guess_experience_years("No dates at all")

        assert guess.total_years == 0.0
        assert guess.confidence == 0.0

    def test_confidence_never_claims_precision(self):
        """Education ranges look identical to work ranges, so this stays an estimate."""
        guess = rules.guess_experience_years("2019-2023 A\n2020-2024 B\n2021-2025 C")

        assert guess.confidence <= 0.5


class TestDegreeComparison:
    """Vocabulary ordering."""

    def test_ranks_are_ordered(self):
        assert (
            degree_rank("any")
            < degree_rank("college")
            < degree_rank("bachelor")
            < degree_rank("master")
            < degree_rank("doctor")
        )

    def test_unknown_ranks_as_unconstrained(self):
        """A model returning something unexpected must not crash a run."""
        assert degree_rank("nonsense") == 0
        assert degree_rank(None) == 0

    @pytest.mark.parametrize(
        "actual,required,expected",
        [
            ("master", "bachelor", True),
            ("bachelor", "master", False),
            ("bachelor", "bachelor", True),
            (None, "bachelor", False),
            ("bachelor", "any", True),
        ],
    )
    def test_meets_degree(self, actual, required, expected):
        assert meets_degree(actual, required) is expected

    def test_labels_are_shared_across_callers(self):
        """One mapping, so the same value is never described two ways."""
        assert degree_label("master") == "硕士"
        assert degree_label(None) == "未知"


class TestEvaluate:
    """The stage-1 decision."""

    def test_rejects_when_no_required_skill_matches(self, session):
        """The only hard rejection the skill signal is allowed to make."""
        job = make_job(session)
        verdict = rules.evaluate("Marketing intern. Social media and copywriting.", job)

        assert verdict.passed is False
        assert "无一命中" in verdict.reasons[0]

    def test_keeps_a_partial_match(self, session):
        """Partial skill coverage must survive.

        This is the regression guard for the threshold that was originally 0.34:
        against a job listing several must-haves, candidates matching one were
        discarded before any stage that could have reconsidered them. Six of seven
        test candidates died that way.
        """
        job = make_job(session)
        verdict = rules.evaluate("Python and Spark data pipelines.", job)

        assert verdict.passed is True

    def test_unknown_signals_score_neutrally(self, session):
        """"Not stated" must not be scored as "does not have"."""
        job = make_job(session)
        verdict = rules.evaluate("Go developer", job)

        assert verdict.passed is True
        # Degree unstated, so it must not be scored as a failure.
        assert any("未识别" in reason for reason in verdict.reasons)

    def test_reasons_record_successes_too(self, session):
        """A reviewer has to tell a strong pass from a lucky one."""
        job = make_job(session)
        verdict = rules.evaluate(
            "Go and Python engineer. Bachelor of Science. 2019-2024 Backend Developer.", job
        )

        assert any("命中" in reason for reason in verdict.reasons)
        assert any("学历满足" in reason for reason in verdict.reasons)

    def test_foreign_degree_rejects_only_with_high_confidence(self, session):
        """The degree floor is enforced only on a confident reading."""
        job = make_job(session, min_degree="doctor")

        assert rules.evaluate("大专 学历", job).passed is False
        assert rules.evaluate("Go engineer, no degree stated", job).passed is True

    def test_score_is_bounded(self, session):
        """Weights summing above 1 must not produce a score above 1."""
        job = make_job(session)
        verdict = rules.evaluate(
            "Go Python LLM Kubernetes. Doctor of Science. 2015-2025 Senior Engineer.", job
        )

        assert 0.0 <= verdict.score <= 1.0

    def test_no_required_skills_never_hard_rejects(self, session):
        """A job with no must-haves must not reject everyone.

        The API rejects such a job at creation, but a row created another way
        would otherwise turn the free filter into a filter that removes everything.
        """
        job = make_job(
            session,
            skills=[("Kubernetes", ("k8s",), SkillKind.PREFERRED.value)],
        )
        verdict = rules.evaluate("Marketing intern, social media.", job)

        assert verdict.passed is True

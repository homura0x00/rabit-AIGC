"""Stage 1 — the free filter.

Everything here runs on raw resume text with regular expressions and dictionary
lookups, and costs zero tokens. That is the entire reason it exists: a resume can
be ruled out before any model sees it, and every resume ruled out here is one
that never enters an embedding or judgement call.

Why this layer is deliberately *timid*
--------------------------------------

Regex reads on a resume are noisy in a way that is asymmetric. A false positive
costs one extra paid evaluation on a candidate who was going to be rejected
anyway — a fraction of a cent. A false negative silently discards a qualified
person, and nothing in the output reveals that it happened; the run looks
completely healthy.

So the three signals are treated according to how much they can be trusted:

* **Skills** — near-binary and reliable. A required keyword is either present or
  it is not. This is the only signal allowed to hard-reject on its own.
* **Degree** — reliable when an explicit keyword appears (``硕士``, ``Master``),
  unreliable otherwise, since a resume may not state it at all. Penalises, and
  hard-rejects only on a high-confidence reading below the floor.
* **Years** — the least trustworthy. Date ranges in the education section look
  exactly like date ranges in the employment section, and the headline total
  printed on many resumes disagrees with the sum of the ranges. So the estimate
  is an *upper bound*, carries low confidence, and never hard-rejects.

An unknown always scores neutrally rather than badly. "The resume does not say"
is not evidence of "the candidate lacks it", and treating the two the same is how
a filter quietly becomes a bias.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from app.core.config import settings
from app.models.job import (
    DegreeLevel,
    JobDescription,
    JobSkillRequirement,
    SkillKind,
    degree_label,
    degree_rank,
)

# --- skill matching -------------------------------------------------------

# A term is bounded by non-alphanumerics on both sides. Plain ``\b`` is not
# enough: it treats ``+`` and ``#`` as boundaries, so ``C`` would match inside
# ``C++`` and ``C#``. The boundary is built per term in ``_compile_term``.

# Terms at or below this length that consist of letters *only* are matched
# case-sensitively. "go" as a lowercase English verb ("go to market", "go live")
# is far more common in a resume than the language spelled that way, so a
# case-insensitive match on a two-letter word produces constant false positives.
# Real skill lists write "Go".
_CASE_SENSITIVE_MAX_LEN = 3


# Single-character terms need one extra guard. Resume shorthand is full of
# compounds whose leading letter looks like a standalone token: "R&D" (research
# and development) is on a large share of Chinese and Western resumes, and a
# plain boundary match reads it as the R language. "C&B", "R/W" and "P&L" behave
# the same way. A one-letter term is therefore rejected when it heads such a
# compound, while "R language", "with R." and "R, Python" still match.
_COMPOUND_CHARS = "&+#/"


def _compile_term(term: str) -> re.Pattern[str]:
    """Build a boundary-safe pattern for one skill surface form.

    Case sensitivity is decided by the term's *shape*, not its length alone. The
    only genuinely dangerous case is a short, purely alphabetic term, because
    that is where a skill name collides with ordinary prose. Anything carrying a
    digit or a symbol has no such collision — nobody writes "k8s" as English — so
    it matches case-insensitively whatever its length, which is what lets the
    alias "k8s" resolve against "K8s" on a resume.

    Args:
        term: A surface form, e.g. ``"Go"``, ``"C++"``, ``"k8s"``.

    Returns:
        A compiled pattern.
    """
    case_sensitive = len(term) <= _CASE_SENSITIVE_MAX_LEN and term.isalpha()
    flags = 0 if case_sensitive else re.IGNORECASE

    follower = r"(?![A-Za-z0-9])"
    if len(term) == 1:
        follower = rf"(?![A-Za-z0-9{re.escape(_COMPOUND_CHARS)}])"

    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}{follower}"
    return re.compile(pattern, flags)


# --- degree detection -----------------------------------------------------

# Ordered most-specific first so "博士" is not shadowed by a "学士" elsewhere in
# the same line, and so "本科" wins over a bare "大学".
_DEGREE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        DegreeLevel.DOCTOR.value,
        ("博士", "phd", "ph.d", "doctorate", "doctor of philosophy", "doktor"),
    ),
    (
        DegreeLevel.MASTER.value,
        ("硕士", "研究生", "master", "msc", "m.sc", "mba", "meng", "m.eng"),
    ),
    (
        DegreeLevel.BACHELOR.value,
        ("本科", "学士", "bachelor", "bsc", "b.sc", "beng", "b.eng", "大学本科"),
    ),
    (
        DegreeLevel.COLLEGE.value,
        ("大专", "专科", "高职", "associate degree", "associate of"),
    ),
)

# Words that make a degree keyword a statement about a requirement rather than
# about the candidate. A JD pasted at the top of a resume, or a line reading
# "硕士及以上学历优先", is not evidence the candidate holds one.
_DEGREE_NEGATORS = ("要求", "优先", "及以上", "需", "须", "preferred", "required")


@dataclass(frozen=True)
class DegreeGuess:
    """Best-effort reading of the candidate's highest degree.

    Attributes:
        level: A :class:`~app.models.job.DegreeLevel` value, or ``None`` when
            nothing was recognised.
        confidence: 0.0-1.0. Low values must not be used to reject.
        evidence: The matched text, for the HR-facing reason string.
    """

    level: Optional[str]
    confidence: float
    evidence: Optional[str] = None


def guess_degree(text: str) -> DegreeGuess:
    """Read the highest degree mentioned in the resume.

    Args:
        text: Normalised resume text.

    Returns:
        The highest recognised level with a confidence score. Confidence is
        higher when the keyword appears near a negation-free context and when
        multiple distinct keywords agree.
    """
    lowered = text.lower()
    best: Optional[str] = None
    hits: dict[str, int] = {}
    evidence: Optional[str] = None

    for level, keywords in _DEGREE_KEYWORDS:
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), lowered):
                # Skip keywords that are part of a stated requirement.
                window = lowered[max(0, match.start() - 12) : match.end() + 12]
                if any(neg in window for neg in _DEGREE_NEGATORS):
                    continue
                hits[level] = hits.get(level, 0) + 1
                if evidence is None:
                    evidence = text[match.start() : match.end() + 12].strip()
                break

    if not hits:
        return DegreeGuess(level=None, confidence=0.0, evidence=None)

    # Highest recognised level wins; a resume citing both a bachelor's and a
    # master's has a master's.
    best = max(hits, key=lambda lvl: (degree_rank(lvl), hits[lvl]))

    # Two or more distinct mentions of the same level is a strong signal that it
    # is the candidate's own qualification rather than a passing reference.
    confidence = 0.9 if hits[best] >= 2 else 0.6

    return DegreeGuess(level=best, confidence=confidence, evidence=evidence)


# --- experience estimation ------------------------------------------------

_YEAR = r"(?:19|20)\d{2}"
_MONTH = r"(?:0?[1-9]|1[0-2])"
_MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_PRESENT = r"(?:至今|今|present|now|current|to\s*date)"

# A date atom is "Sep 2019", "2019.09", "2019年9月", or a bare "2019".
#
# Month-before-year is not a nicety. "Sep 2019 – Jun 2023" is how most English
# resumes are dated, and a year-first-only pattern reads every such resume as
# having no dated experience at all — which then silently suppresses the
# experience signal for exactly the candidates most likely to have it.
_START_ATOM = (
    rf"(?:(?P<smonname>{_MONTH_NAME})\s+)?"
    rf"(?P<sy>{_YEAR})"
    rf"(?:\s*[.\-/年]\s*(?P<sm>{_MONTH}))?月?"
)
_END_ATOM = (
    rf"(?:(?P<emonname>{_MONTH_NAME})\s+)?"
    rf"(?P<ey>{_YEAR})"
    rf"(?:\s*[.\-/年]\s*(?P<em>{_MONTH}))?月?"
)

_DATE_RANGE = re.compile(
    rf"{_START_ATOM}\s*(?:[-–—~～]|\bto\b|至)\s*(?:{_END_ATOM}|(?P<present>{_PRESENT}))",
    re.IGNORECASE,
)

_MONTH_NAMES: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Resumes are dated relative to "now" for ongoing roles.
_ASSUMED_CURRENT_YEAR = 2026


@dataclass(frozen=True)
class ExperienceGuess:
    """An upper-bound estimate of total professional experience.

    Attributes:
        total_years: Merged, de-overlapped span in years.
        confidence: 0.0-1.0. Always modest — see the module docstring.
        intervals: The merged ``(start, end)`` year pairs, kept so a human can
            see what the number was derived from.
    """

    total_years: float
    confidence: float
    intervals: tuple[tuple[float, float], ...] = ()


def _resolve_month(
    year: str,
    month_name: Optional[str],
    month_number: Optional[str],
) -> int:
    """Convert a date atom into a monotonic month index.

    A numeric month wins over a spelled-out one, since a token like
    "Sep 2019.09" is not meaningful and the numeric form is the precise one.

    A missing month resolves to January. That makes each span an *under*-estimate
    of up to a year, while merging ranges that include education over-counts —
    the two errors pull in opposite directions and neither is precise enough to
    justify rejecting anyone. This is exactly the ambiguity the paid extraction
    stage resolves later, and only for candidates that survive.
    """
    if month_number:
        month = int(month_number)
    elif month_name:
        month = _MONTH_NAMES.get(month_name[:3].lower(), 1)
    else:
        month = 1
    return int(year) * 12 + (month - 1)


def guess_experience_years(text: str) -> ExperienceGuess:
    """Estimate total experience by merging dated ranges in the resume.

    Overlapping ranges are merged so that concurrent roles are not double
    counted, but education ranges are *not* excluded — distinguishing a degree
    from a job would require understanding the document, which is exactly the
    paid work this stage exists to avoid. The result therefore over-counts, which
    is why it carries low confidence and never causes a rejection.

    Args:
        text: Normalised resume text.

    Returns:
        The merged span, its confidence, and the intervals behind it.
    """
    spans: list[tuple[int, int]] = []

    for match in _DATE_RANGE.finditer(text):
        start = _resolve_month(
            match.group("sy"), match.group("smonname"), match.group("sm")
        )

        if match.group("present"):
            end = _ASSUMED_CURRENT_YEAR * 12 + 11
        else:
            end = _resolve_month(
                match.group("ey"), match.group("emonname"), match.group("em")
            )

        if end >= start:
            spans.append((start, end))

    if not spans:
        return ExperienceGuess(total_years=0.0, confidence=0.0, intervals=())

    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total_months = sum(end - start for start, end in merged)

    # Confidence reflects how much material the estimate rests on rather than
    # how good the estimate is: more distinct ranges means more of the timeline
    # is actually dated, but never more than "moderate".
    confidence = 0.35 if len(merged) == 1 else 0.5

    return ExperienceGuess(
        total_years=round(total_months / 12, 1),
        confidence=confidence,
        intervals=tuple((start / 12, end / 12) for start, end in merged),
    )


# --- skill matching -------------------------------------------------------


@dataclass(frozen=True)
class SkillMatch:
    """One job requirement checked against the resume text.

    Attributes:
        term_id: ``SkillTerm`` primary key.
        name: Canonical skill name.
        kind: ``required`` or ``preferred``.
        weight: Relative importance within its kind.
        matched: Whether any surface form was found.
        matched_form: Which surface form matched.
    """

    term_id: int
    name: str
    kind: str
    weight: float
    matched: bool
    matched_form: Optional[str] = None


def match_skills(
    text: str,
    requirements: Iterable[JobSkillRequirement],
) -> list[SkillMatch]:
    """Check each job requirement against the resume text.

    Args:
        text: Normalised resume text.
        requirements: Job requirements, each with its ``term`` loaded.

    Returns:
        One :class:`SkillMatch` per requirement, in the order given.
    """
    results: list[SkillMatch] = []

    for requirement in requirements:
        term = requirement.term
        if term is None:
            continue

        found: Optional[str] = None
        for form in term.match_terms():
            if _compile_term(form).search(text):
                found = form
                break

        results.append(
            SkillMatch(
                # The requirement's own foreign key, not ``term.id``: it is typed
                # non-optional because a persisted requirement always has one, and
                # it is the value the rest of the schema keys on anyway.
                term_id=requirement.term_id,
                name=term.name,
                kind=requirement.kind,
                weight=requirement.weight,
                matched=found is not None,
                matched_form=found,
            )
        )

    return results


# --- verdict --------------------------------------------------------------

# Weights for the three free signals. Skills dominate because they are the only
# reliable one; degree and experience mostly break ties among skill-matched
# candidates.
_W_SKILLS = 0.6
_W_DEGREE = 0.2
_W_EXPERIENCE = 0.2

# Neutral score for an undetermined signal. Deliberately not 0.0: "not stated"
# must not be scored as "does not have".
_UNKNOWN = 0.5

# The rejection threshold lives in settings, not here, because getting it wrong is
# a measurement question rather than a code question. See
# PipelineSettings.rule_min_required_share for the numbers behind the default.


@dataclass(frozen=True)
class RuleVerdict:
    """The stage-1 decision for one resume.

    Attributes:
        passed: Whether the candidate continues to recall. ``False`` here means
            dropped without spending a token.
        score: 0.0-1.0 quality signal carried forward as a tie-breaker.
        reasons: Human-readable explanations, covering matches as well as
            failures. A reviewer has to be able to tell a strong pass from a
            lucky one, so successes are recorded too.
        skills: Per-requirement match detail.
        degree: The degree reading behind the decision.
        experience: The experience estimate behind the decision.
    """

    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    skills: list[SkillMatch] = field(default_factory=list)
    degree: Optional[DegreeGuess] = None
    experience: Optional[ExperienceGuess] = None

    @property
    def matched_required(self) -> list[str]:
        """Canonical names of matched required skills."""
        return [
            s.name
            for s in self.skills
            if s.kind == SkillKind.REQUIRED.value and s.matched
        ]

    @property
    def missing_required(self) -> list[str]:
        """Canonical names of required skills with no match."""
        return [
            s.name
            for s in self.skills
            if s.kind == SkillKind.REQUIRED.value and not s.matched
        ]


def _weighted_share(matches: Sequence[SkillMatch], kind: str) -> Optional[float]:
    """Weighted share of requirements of ``kind`` that matched.

    Returns:
        The share in 0.0-1.0, or ``None`` when there are no requirements of that
        kind — which is not the same as a share of zero.
    """
    relevant = [m for m in matches if m.kind == kind]
    if not relevant:
        return None

    total = sum(m.weight for m in relevant)
    if total <= 0:
        return None

    return sum(m.weight for m in relevant if m.matched) / total


def evaluate(text: str, job: JobDescription) -> RuleVerdict:
    """Score and filter one resume against one job, without any model call.

    Args:
        text: Normalised resume text.
        job: The job description, with ``requirements`` and their ``term``
            relationships loaded.

    Returns:
        The verdict, carrying the reasons behind it.
    """
    requirements = list(job.requirements or [])
    skills = match_skills(text, requirements)
    degree = guess_degree(text)
    experience = guess_experience_years(text)

    reasons: list[str] = []

    # --- skills ---
    required_share = _weighted_share(skills, SkillKind.REQUIRED.value)
    preferred_share = _weighted_share(skills, SkillKind.PREFERRED.value)

    matched_names = [s.name for s in skills if s.kind == SkillKind.REQUIRED.value and s.matched]
    missing_names = [s.name for s in skills if s.kind == SkillKind.REQUIRED.value and not s.matched]

    if matched_names:
        reasons.append(f"必备技能命中：{'、'.join(matched_names)}")
    if missing_names:
        reasons.append(f"必备技能未见：{'、'.join(missing_names)}")

    # --- degree ---
    degree_score = _UNKNOWN
    if degree.level is not None:
        floor = job.min_degree
        if degree_rank(degree.level) >= degree_rank(floor):
            degree_score = 1.0
            reasons.append(f"学历满足（{degree_label(degree.level)} ≥ {degree_label(floor)}）")
        else:
            degree_score = 0.0
            reasons.append(
                f"学历低于要求（{degree_label(degree.level)} < {degree_label(floor)}）"
            )
    else:
        reasons.append("学历未识别，按中性计分")

    # --- experience ---
    experience_score = _UNKNOWN
    if experience.confidence > 0:
        if experience.total_years >= job.min_years:
            experience_score = 1.0
            reasons.append(
                f"年限估计满足（约 {experience.total_years} 年 ≥ {job.min_years} 年）"
            )
        else:
            shortfall = job.min_years - experience.total_years
            # Partial credit, scaled by how far short: a candidate 0.5 years
            # under a 3-year bar is a different case from one 10 years under.
            experience_score = max(0.0, 1.0 - shortfall / max(job.min_years, 1.0))
            reasons.append(
                f"年限估计偏短（约 {experience.total_years} 年 < {job.min_years} 年，"
                "估计值为上界且非高置信）"
            )
    else:
        reasons.append("年限未识别，按中性计分")

    # --- score ---
    score = _W_SKILLS * (required_share if required_share is not None else _UNKNOWN)
    if preferred_share is not None:
        # Preferred skills nudge rather than dominate: fold into the tail of the
        # skill weight so they cannot push a requirement-missing candidate up.
        score = _W_SKILLS * (
            0.85 * (required_share if required_share is not None else _UNKNOWN)
            + 0.15 * preferred_share
        )
    score += _W_DEGREE * degree_score + _W_EXPERIENCE * experience_score
    score = round(min(max(score, 0.0), 1.0), 4)

    # --- decision ---
    # Hard rejections are limited to readings solid enough to defend. Everything
    # soft only lowers the score and lets recall ranking decide.
    threshold = settings.PIPELINE.rule_min_required_share
    if required_share is not None and required_share <= threshold:
        reasons.insert(0, "必备技能无一命中，规则层淘汰")
        return RuleVerdict(
            passed=False,
            score=score,
            reasons=reasons,
            skills=skills,
            degree=degree,
            experience=experience,
        )

    if (
        degree.level is not None
        and degree.confidence >= 0.9
        and degree_rank(degree.level) < degree_rank(job.min_degree)
    ):
        reasons.insert(0, "学历明确低于门槛，规则层淘汰")
        return RuleVerdict(
            passed=False,
            score=score,
            reasons=reasons,
            skills=skills,
            degree=degree,
            experience=experience,
        )

    reasons.insert(0, "通过规则层")
    return RuleVerdict(
        passed=True,
        score=score,
        reasons=reasons,
        skills=skills,
        degree=degree,
        experience=experience,
    )

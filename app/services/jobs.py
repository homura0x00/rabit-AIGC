"""Job description persistence.

Kept separate from the HTTP layer because the interesting part is not the
endpoint — it is the skill dictionary upsert, which decides whether a job
requirement reuses an existing :class:`~app.models.job.SkillTerm` or creates one.

Getting that wrong does not fail loudly. A duplicate term means the same skill
exists twice under slightly different names, so a resume matching one copy misses
the other, and the rule stage quietly rejects a qualified candidate. There is no
error to see; the shortlist is just worse.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from sqlmodel import Session, col, select

from app.core.log import get_logger
from app.models.base import require_id
from app.models.job import (
    DegreeLevel,
    JobDescription,
    JobSkillRequirement,
    SkillKind,
    SkillTerm,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """A skill requirement as supplied by a caller.

    Attributes:
        name: Canonical skill name.
        aliases: Alternative surface forms, e.g. ``("golang",)`` for ``Go``.
        kind: ``required`` or ``preferred``.
        weight: Relative importance within its kind.
    """

    name: str
    aliases: tuple[str, ...] = ()
    kind: str = SkillKind.REQUIRED.value
    weight: float = 1.0


def normalise_skill_name(name: str) -> str:
    """Canonical form used to match skill terms across jobs.

    Case-folded and whitespace-collapsed, so ``"Go"`` and ``" go "`` resolve to
    one dictionary entry instead of two.

    Args:
        name: The raw skill name.

    Returns:
        The normalised key.
    """
    return " ".join(name.split()).casefold()


def upsert_skill_term(
    session: Session,
    name: str,
    aliases: Sequence[str] = (),
) -> SkillTerm:
    """Find or create the dictionary entry for a skill.

    Aliases are merged rather than replaced: a second job that knows ``Go`` is
    also written ``Golang`` should enrich the shared entry, not silently drop the
    knowledge the first job contributed.

    Args:
        session: Database session.
        name: Canonical skill name.
        aliases: Additional surface forms.

    Returns:
        The persisted term.
    """
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("skill name must not be empty")

    key = normalise_skill_name(cleaned)
    term = session.exec(select(SkillTerm)).all()
    existing = next(
        (candidate for candidate in term if normalise_skill_name(candidate.name) == key),
        None,
    )

    incoming = [" ".join(alias.split()) for alias in aliases if alias.strip()]

    if existing is None:
        existing = SkillTerm(name=cleaned, aliases=incoming)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    merged = list(existing.aliases or [])
    known = {alias.casefold() for alias in merged}
    added = [alias for alias in incoming if alias.casefold() not in known]

    if added:
        # Reassign rather than mutate: SQLModel does not track in-place mutation
        # of a JSON column, so appending to the list would be silently discarded.
        existing.aliases = merged + added
        session.add(existing)
        session.commit()
        session.refresh(existing)

    return existing


def create_job(
    session: Session,
    *,
    title: str,
    department: Optional[str] = None,
    location: Optional[str] = None,
    min_degree: str = DegreeLevel.ANY.value,
    min_years: float = 0.0,
    accepts_internship: bool = True,
    weight_skills: float = 0.40,
    weight_experience: float = 0.35,
    weight_education: float = 0.15,
    weight_projects: float = 0.10,
    skills: Sequence[SkillSpec] = (),
) -> JobDescription:
    """Create a job description with its skill requirements.

    Args:
        session: Database session.
        title: Job title.
        department: Optional department.
        location: Optional location.
        min_degree: A :class:`~app.models.job.DegreeLevel` value.
        min_years: Minimum years of experience.
        accepts_internship: Whether internship experience counts.
        weight_skills: Rubric weight for skills.
        weight_experience: Rubric weight for experience.
        weight_education: Rubric weight for education.
        weight_projects: Rubric weight for projects.
        skills: Skill requirements to attach.

    Returns:
        The persisted job, with ``requirements`` and their ``term`` loaded.
    """
    job = JobDescription(
        title=" ".join(title.split()),
        department=department,
        location=location,
        min_degree=min_degree,
        min_years=min_years,
        accepts_internship=accepts_internship,
        weight_skills=weight_skills,
        weight_experience=weight_experience,
        weight_education=weight_education,
        weight_projects=weight_projects,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    for spec in skills:
        term = upsert_skill_term(session, spec.name, spec.aliases)
        session.add(
            JobSkillRequirement(
                job_id=require_id(job.id, "job_description"),
                term_id=require_id(term.id, "skill_term"),
                kind=spec.kind,
                weight=spec.weight,
            )
        )

    session.commit()
    session.refresh(job)

    logger.info("job %s created with %d skill requirements", job.id, len(skills))

    loaded = load_job(session, require_id(job.id, "job_description"))
    assert loaded is not None  # just created, so it must exist
    return loaded


def load_job(session: Session, job_id: int) -> Optional[JobDescription]:
    """Load a job with its requirements and their skill terms resolved.

    The relationships are touched during load on purpose. Every pipeline stage
    reads ``job.requirements[*].term``, so a job loaded without them raises a lazy
    load error deep inside scoring — or, worse, lazily loads after the session has
    closed. Resolving them here makes any problem surface at the call site, where
    the cause is obvious.

    Args:
        session: Database session.
        job_id: The job to load.

    Returns:
        The job, or ``None`` when it does not exist.
    """
    job = session.get(JobDescription, job_id)
    if job is None:
        return None

    # Touch the chain so it is loaded while the session is definitely open.
    for requirement in job.requirements:
        _ = requirement.term.name if requirement.term else None

    return job


def list_jobs(
    session: Session,
    *,
    active_only: bool = False,
) -> list[JobDescription]:
    """List job descriptions, newest first.

    Args:
        session: Database session.
        active_only: Restrict to jobs flagged active.

    Returns:
        Jobs without their requirements loaded; call :func:`load_job` for those.
    """
    statement = select(JobDescription).order_by(col(JobDescription.id).desc())
    if active_only:
        statement = statement.where(col(JobDescription.is_active).is_(True))
    return list(session.exec(statement).all())

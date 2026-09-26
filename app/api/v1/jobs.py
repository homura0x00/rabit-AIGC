"""Job description endpoints.

The JD is entered as structure rather than prose, and that is what makes the free
stage of the funnel possible: hard filters are plain comparisons and skill
matching is a dictionary lookup, with no model call anywhere in front of them.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.core.log import get_logger
from app.schemas.job import JobCreate, JobRead, JobSummary
from app.services.database import get_session
from app.services.jobs import SkillSpec, create_job, list_jobs, load_job

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/jobs",
    status_code=status.HTTP_201_CREATED,
    response_model=JobRead,
    summary="Create a job description",
)
def create(
    payload: JobCreate,
    session: Session = Depends(get_session),
) -> JobRead:
    """Create a job description with its skill requirements.

    Skill names are resolved against a shared dictionary, so aliases supplied here
    also improve matching for every other job that uses the same skill.

    Args:
        payload: The job to create.
        session: Injected database session.

    Returns:
        The created job, with its requirements resolved.
    """
    job = create_job(
        session,
        title=payload.title,
        department=payload.department,
        location=payload.location,
        min_degree=payload.min_degree.value,
        min_years=payload.min_years,
        accepts_internship=payload.accepts_internship,
        weight_skills=payload.weight_skills,
        weight_experience=payload.weight_experience,
        weight_education=payload.weight_education,
        weight_projects=payload.weight_projects,
        skills=[
            SkillSpec(
                name=skill.name,
                aliases=tuple(skill.aliases),
                kind=skill.kind.value,
                weight=skill.weight,
            )
            for skill in payload.skills
        ],
    )
    return JobRead.from_model(job)


@router.get("/jobs", response_model=list[JobSummary], summary="List job descriptions")
def index(
    active_only: bool = Query(default=False, description="Only jobs flagged active"),
    session: Session = Depends(get_session),
) -> list[JobSummary]:
    """List job descriptions, newest first.

    Args:
        active_only: Restrict to active jobs.
        session: Injected database session.

    Returns:
        Job summaries without their skill requirements.
    """
    return [JobSummary.from_model(job) for job in list_jobs(session, active_only=active_only)]


@router.get("/jobs/{job_id}", response_model=JobRead, summary="Get a job description")
def show(job_id: int, session: Session = Depends(get_session)) -> JobRead:
    """Load one job description with its requirements.

    Args:
        job_id: The job to load.
        session: Injected database session.

    Returns:
        The job.

    Raises:
        HTTPException: 404 when the job does not exist.
    """
    job = load_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"job {job_id} not found"
        )
    return JobRead.from_model(job)

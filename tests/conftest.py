"""Shared test fixtures.

Two hard rules, both enforced before any application module is imported:

* **No network.** Tests must pass on a laptop with the wifi off, and a suite that
  quietly spends money is a suite people stop running. Provider clients are
  replaced with deterministic fakes below, and ``RERANK_ENABLED`` is forced off so
  the reranker never reaches for its endpoint.
* **No real database.** ``DATABASE_URL`` is pinned to an in-memory SQLite instance
  before ``app.core.config`` is imported, because that module reads the
  environment once at import time and the project's ``.env`` points at a real
  Supabase instance. Getting this wrong would mean the test suite writing rows
  into production.

Environment variables are set with :func:`os.environ.__setitem__` rather than
``setdefault`` where the value matters, and ``load_dotenv`` does not override
existing values, so these win over ``.env``.
"""

import hashlib
import math
import os
import re

# Legacy scripts that predate the suite. They execute work at import time — one
# reads a PDF, another opens an interactive input loop — so collection must not
# touch them. This is a conftest variable, not an ini option; putting it in
# pyproject.toml silently did nothing and pytest warned about it.
collect_ignore = ["main.py", "pdf.py", "supabase.py"]

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["RERANK_ENABLED"] = "false"
os.environ["LLM_API_KEY"] = "test-key"
os.environ["EMBEDDING_API_KEY"] = "test-key"
os.environ["LLM_MODEL"] = "fake-chat"
os.environ["EMBEDDING_MODEL"] = "fake-embed"
os.environ["EMBEDDING_DIMENSIONS"] = "32"
os.environ["EMBEDDING_BATCH_SIZE"] = "4"

from typing import Any, Iterable, Optional, Sequence  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

import app.models  # noqa: E402,F401  (registers every table on the metadata)
from app.models.base import require_id  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.models.job import (  # noqa: E402
    DegreeLevel,
    JobDescription,
    JobSkillRequirement,
    SkillKind,
    SkillTerm,
)
from app.models.screening import ResumeDocument  # noqa: E402
from app.services.database import get_session  # noqa: E402
from app.models.screening import CallPurpose  # noqa: E402
from app.services.llm import (  # noqa: E402
    TokenLedger,
    Usage,
    estimate_tokens,
    extract_usage,
)

# Sample resume text, held as a constant rather than read from a PDF so tests do
# not depend on the sample file's exact bytes. Mirrors a real one-page resume,
# including the letter-spaced "PROJ ECTS" heading that PDF extraction produces.
SAMPLE_RESUME = """Liu Guanji
guanji_liu@icloud.com · +852 68240703 · github.com/homura0x00
Full-Stack Developer focused on Go and Java.

EDUCATION
Huaqiao University Bachelor of Computer Science
Sep 2019 – Jun 2023
Relevant Coursework: Data Structures, Machine Learning, NLP

PROJ ECTS
Operations Assistant (AI Agent)
Backend Developer
Mar 2023 – Present
Built an AI operations assistant with Go and the Feishu API.
Implemented a ReAct-style agent loop with Function Calling.
Deployed services on Kubernetes with PostgreSQL.

SKILLS
Languages: Go, Java, Python, JavaScript/TypeScript, SQL
Infrastructure: Docker, Kubernetes, PostgreSQL, Redis
"""


class FakeEmbedder:
    """Deterministic bag-of-words embedder.

    A hashing vectoriser rather than random vectors, because tests need similarity
    to *mean* something: two texts sharing vocabulary must score higher than two
    that share none, or a ranking assertion proves nothing. Deterministic across
    runs and processes so failures reproduce.
    """

    # Annotated, not bare: a bare `ledger = None` infers as `None`, and a mutable
    # protocol member is invariant, so the double would not satisfy `Embedder`
    # however compatible it is at runtime.
    ledger: Optional[TokenLedger] = None

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions
        self.batches: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, recording the batch and the usage.

        Usage is written through the ledger exactly as the real client does. A
        double that skipped this would leave the ledger empty, so every test of
        the cost report would pass vacuously against zeroes — verifying nothing
        about the one claim the project is built on.
        """
        self.batches.append(list(texts))

        if self.ledger is not None:
            tokens = sum(estimate_tokens(text) for text in texts)
            self.ledger.record(
                purpose=CallPurpose.EMBED.value,
                usage=Usage(
                    model=settings.EMBEDDING.model,
                    prompt_tokens=tokens,
                    cached_tokens=0,
                    completion_tokens=0,
                ),
                latency_ms=1,
                batch_size=len(texts),
            )

        return [self._vector(text) for text in texts]


class FakeLLM:
    """Judge/review stand-in that scores whatever it is shown.

    It parses the batch-local ids (``C1``, ``C2``, …) out of the user message
    instead of returning canned payloads. That matters: a fixture that returns a
    fixed reply would pass even if batching or id mapping were broken, which is
    exactly the code most worth testing.
    """

    ledger: Optional[TokenLedger] = None

    def __init__(self, scores: Optional[dict[str, float]] = None, default: float = 60.0):
        self.scores = scores or {}
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, messages, **kwargs):
        user = messages[-1]["content"]
        system = messages[0]["content"]
        purpose = kwargs.get("purpose")
        ids = re.findall(r"^\[(C\d+)\]", user, re.M)

        self.calls.append(
            {
                "batch_size": kwargs.get("batch_size", len(ids)),
                "ids": ids,
                "purpose": purpose,
                "system": system,
            }
        )

        self._record(messages, purpose, kwargs.get("batch_size", max(len(ids), 1)))

        # Stage 4 sends one candidate per call and expects a single-object reply,
        # so the double has to answer in the shape the caller asked for. Returning
        # the batch shape here made every review fail, which looked like a bug in
        # the review stage rather than in the fixture.
        if purpose == CallPurpose.REVIEW.value:
            return {
                "score": self.default,
                "verdict": "confirm",
                "reason": "对照完整材料后维持原判断。",
            }

        return {
            "results": [
                {
                    "id": local_id,
                    "score": self.scores.get(local_id, self.default),
                    "evidence": [f"evidence for {local_id}"],
                    "gaps": [],
                }
                for local_id in ids
            ]
        }

    def _record(self, messages, purpose, batch_size: int) -> None:
        """Write usage through the ledger, as a real client would."""
        if self.ledger is None:
            return

        prompt = sum(estimate_tokens(message.get("content") or "") for message in messages)
        self.ledger.record(
            purpose=purpose or CallPurpose.JUDGE.value,
            usage=Usage(
                model=settings.LLM.model,
                prompt_tokens=prompt,
                # A fixed, non-zero share so cache-hit assertions have something to
                # read without pretending to model the provider's block alignment.
                cached_tokens=prompt // 2,
                completion_tokens=32,
            ),
            latency_ms=1,
            batch_size=batch_size,
        )

    def complete(self, messages, **kwargs):  # pragma: no cover - unused path
        raise AssertionError("FakeLLM.complete should not be called directly")


@pytest.fixture
def engine():
    """An in-memory SQLite engine with every table created.

    ``StaticPool`` keeps the single in-memory database alive across connections,
    which the default pool does not: each new connection would otherwise get its
    own empty database and the schema would appear to vanish mid-test.
    """
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    SQLModel.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session(engine):
    """A session bound to the test engine."""
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(engine):
    """A TestClient whose database dependency points at the test engine.

    The override is installed on ``get_session`` rather than by rewriting the
    module-level engine, so the application's own wiring is left alone and the
    production path stays exactly as deployed.
    """

    def override() -> Iterable[Session]:
        with Session(engine) as session:
            yield session

    from app.main import app

    app.dependency_overrides[get_session] = override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_job(
    session: Session,
    *,
    title: str = "AI 应用工程师",
    min_degree: str = DegreeLevel.BACHELOR.value,
    min_years: float = 2.0,
    skills: Sequence[tuple[str, tuple[str, ...], str]] | None = None,
) -> JobDescription:
    """Create a persisted job description for tests.

    Args:
        session: Database session.
        title: Job title.
        min_degree: Degree floor.
        min_years: Experience floor.
        skills: ``(name, aliases, kind)`` triples. Defaults to a Go/Python/LLM
            must-have set, which is what most tests discriminate on.

    Returns:
        The job, with ``requirements`` and their ``term`` resolved.
    """
    specs = list(
        skills
        or [
            ("Go", ("golang",), SkillKind.REQUIRED.value),
            ("Python", (), SkillKind.REQUIRED.value),
            ("LLM", ("大模型",), SkillKind.REQUIRED.value),
            ("Kubernetes", ("k8s",), SkillKind.PREFERRED.value),
        ]
    )

    job = JobDescription(title=title, min_degree=min_degree, min_years=min_years)
    session.add(job)
    session.commit()
    session.refresh(job)

    for index, (name, aliases, kind) in enumerate(specs, start=1):
        term = session.exec(
            __import__("sqlmodel").select(SkillTerm).where(SkillTerm.name == name)
        ).first()
        if term is None:
            term = SkillTerm(name=name, aliases=list(aliases))
            session.add(term)
            session.commit()
            session.refresh(term)

        session.add(
            JobSkillRequirement(
                job_id=require_id(job.id, "job_description"),
                term_id=require_id(term.id, "skill_term"),
                kind=kind,
                weight=1.0,
            )
        )

    session.commit()
    session.refresh(job)

    for requirement in job.requirements:
        _ = requirement.term.name if requirement.term else None

    return job


def add_resume(
    session: Session,
    filename: str,
    text: str,
) -> ResumeDocument:
    """Persist a resume document.

    Args:
        session: Database session.
        filename: Display filename.
        text: Normalised resume text.

    Returns:
        The stored document.
    """
    document = ResumeDocument(
        filename=filename,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        raw_text=text,
        char_count=len(text),
        page_count=1,
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def doc_id(document: ResumeDocument) -> int:
    """The id of a resume the fixture just committed.

    ``ResumeDocument.id`` is ``Optional[int]`` — unset before the insert — so tests
    that feed it to functions taking an ``int`` need the same narrowing production
    code uses. Routing it through one helper keeps the reason in one place.

    Args:
        document: A committed resume document.

    Returns:
        Its identifier.
    """
    return require_id(document.id, "resume_document")


def usage_from(prompt: int, cached: int, completion: int, model: str = "fake-chat") -> Usage:
    """Build a :class:`Usage` for assertions."""
    return Usage(
        model=model,
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
    )


class _RawUsage:
    """Mimics the provider's usage object, including the cache-hit fields."""

    def __init__(self, prompt: int, cached: int, completion: int, *, detail_style: bool = False):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        if detail_style:
            self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()
        else:
            self.prompt_cache_hit_tokens = cached
            self.prompt_tokens_details = None


@pytest.fixture
def raw_usage():
    """Expose the fake provider usage object to tests."""
    return _RawUsage


@pytest.fixture
def extract():
    """Expose the usage normaliser to tests."""
    return extract_usage

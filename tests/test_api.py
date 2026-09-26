"""HTTP layer.

The API is what a reviewer actually touches, so these tests cover the contract
rather than the internals: status codes, validation that prevents a silently
broken configuration, and the asynchronous run lifecycle.
"""

import pymupdf
import pytest

from app.models.base import require_id


def make_pdf(text: str) -> bytes:
    """Render text into a one-page PDF in memory."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 780), text, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


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

MARKETING_RESUME = """Wang Fang
EDUCATION
Sun Yat-sen University Bachelor of Marketing
EXPERIENCE
Marketing Intern. Ran social media campaigns.
SKILLS
Social media, copywriting, Excel
"""

JOB_PAYLOAD = {
    "title": "AI 应用工程师",
    "department": "AI 平台部",
    "min_degree": "bachelor",
    "min_years": 2,
    "weight_skills": 40,
    "weight_experience": 35,
    "weight_education": 15,
    "weight_projects": 10,
    "skills": [
        {"name": "Go", "aliases": ["golang"], "kind": "required"},
        {"name": "Python", "kind": "required"},
        {"name": "LLM", "aliases": ["大模型"], "kind": "required"},
        {"name": "Kubernetes", "aliases": ["k8s"], "kind": "preferred"},
    ],
}


class TestHealth:
    """The one endpoint a load balancer touches."""

    def test_reports_healthy(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestResumeUpload:
    """Ingestion contract."""

    def test_accepts_a_pdf(self, client):
        response = client.post(
            "/api/v1/resumes",
            files={"file": ("cv.pdf", make_pdf(GO_RESUME), "application/pdf")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["duplicate"] is False
        assert body["char_count"] > 0

    def test_deduplicates_identical_text(self, client):
        """Submission twice must not become two candidates and two sets of paid calls."""
        pdf = make_pdf(GO_RESUME)
        first = client.post("/api/v1/resumes", files={"file": ("a.pdf", pdf, "application/pdf")})
        second = client.post("/api/v1/resumes", files={"file": ("b.pdf", pdf, "application/pdf")})

        assert first.json()["resume_id"] == second.json()["resume_id"]
        assert second.json()["duplicate"] is True

    def test_rejects_a_non_pdf(self, client):
        response = client.post(
            "/api/v1/resumes", files={"file": ("notes.txt", b"hello", "text/plain")}
        )

        assert response.status_code == 415

    def test_rejects_a_corrupt_pdf(self, client):
        response = client.post(
            "/api/v1/resumes",
            files={"file": ("broken.pdf", b"%PDF-1.4 not really", "application/pdf")},
        )

        assert response.status_code == 422

    def test_rejects_a_pdf_with_no_text_layer(self, client):
        """A scan needs OCR, and the message says so rather than failing generically."""
        document = pymupdf.open()
        document.new_page()
        blank = document.tobytes()
        document.close()

        response = client.post(
            "/api/v1/resumes", files={"file": ("scan.pdf", blank, "application/pdf")}
        )

        assert response.status_code == 422
        assert "OCR" in response.json()["detail"]


class TestJobs:
    """Job description contract."""

    def test_creates_a_job(self, client):
        response = client.post("/api/v1/jobs", json=JOB_PAYLOAD)

        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "AI 应用工程师"
        assert len(body["skills"]) == 4

    def test_normalises_percentage_weights(self, client):
        """40/35/15/10 and 0.40/0.35/0.15/0.10 express the same rubric."""
        body = client.post("/api/v1/jobs", json=JOB_PAYLOAD).json()

        assert sum(body["rubric"].values()) == pytest.approx(1.0)

    def test_rejects_a_job_without_must_have_skills(self, client):
        """A job with no required skills makes the free filter a silent no-op.

        It would cost real money on every run and report nothing wrong, so it is
        refused at creation rather than discovered later.
        """
        response = client.post(
            "/api/v1/jobs",
            json={"title": "x", "skills": [{"name": "Go", "kind": "preferred"}]},
        )

        assert response.status_code == 422
        assert "required" in response.text

    def test_rejects_all_zero_weights(self, client):
        response = client.post(
            "/api/v1/jobs",
            json={
                "title": "x",
                "weight_skills": 0,
                "weight_experience": 0,
                "weight_education": 0,
                "weight_projects": 0,
                "skills": [{"name": "Go", "kind": "required"}],
            },
        )

        assert response.status_code == 422

    def test_aliases_are_shared_across_jobs(self, client):
        """A second job naming the same skill must enrich the shared entry."""
        client.post("/api/v1/jobs", json=JOB_PAYLOAD)
        client.post(
            "/api/v1/jobs",
            json={
                "title": "另一个岗位",
                "skills": [{"name": "go", "aliases": ["golang", "go-lang"], "kind": "required"}],
            },
        )

        jobs = client.get("/api/v1/jobs").json()
        second = client.get(f"/api/v1/jobs/{jobs[0]['id']}").json()
        go = next(skill for skill in second["skills"] if skill["name"].lower() == "go")

        assert "go-lang" in go["aliases"]

    def test_lists_jobs(self, client):
        client.post("/api/v1/jobs", json=JOB_PAYLOAD)

        assert len(client.get("/api/v1/jobs").json()) == 1

    def test_missing_job_is_404(self, client):
        assert client.get("/api/v1/jobs/999").status_code == 404


class TestScreeningRuns:
    """The asynchronous run lifecycle."""

    @staticmethod
    def _prepare(client) -> tuple[int, list[int]]:
        job_id = client.post("/api/v1/jobs", json=JOB_PAYLOAD).json()["id"]
        ids = []
        for name, text in (("go.pdf", GO_RESUME), ("mkt.pdf", MARKETING_RESUME)):
            response = client.post(
                "/api/v1/resumes",
                files={"file": (name, make_pdf(text), "application/pdf")},
            )
            ids.append(response.json()["resume_id"])
        return job_id, ids

    def test_run_is_accepted_then_completes(self, client, monkeypatch):
        """A screening pass takes tens of seconds, so it cannot block a request.

        The provider is stubbed here; what is under test is the handshake — 202
        with an id, then a pollable status — not the pipeline, which has its own
        tests.
        """
        from app.services import screening as screening_pkg  # noqa: F401
        import app.api.v1.screening as screening_api
        from tests.conftest import FakeEmbedder, FakeLLM

        job_id, resume_ids = self._prepare(client)

        monkeypatch.setattr(
            screening_api,
            "execute_run",
            lambda *args, **kwargs: None,
        )

        response = client.post(
            "/api/v1/screening/runs",
            json={"job_id": job_id, "resume_ids": resume_ids},
        )

        assert response.status_code == 202
        run_id = response.json()["run_id"]

        polled = client.get(f"/api/v1/screening/runs/{run_id}")
        assert polled.status_code == 200
        assert polled.json()["job_id"] == job_id

    def test_unknown_job_is_404(self, client):
        response = client.post("/api/v1/screening/runs", json={"job_id": 999})

        assert response.status_code == 404

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/v1/screening/runs/999").status_code == 404
        assert client.get("/api/v1/screening/runs/999/results").status_code == 404
        assert client.get("/api/v1/screening/runs/999/cost").status_code == 404

    def test_results_are_ranked_and_rejections_are_optional(self, session, client):
        """The shortlist is what a reviewer acts on; rejections are the audit trail."""
        from app.services.screening.pipeline import run_screening
        from tests.conftest import FakeEmbedder, FakeLLM, make_job
        from app.services.jobs import load_job

        job = make_job(session)
        from tests.conftest import add_resume

        add_resume(session, "go.pdf", GO_RESUME)
        add_resume(session, "mkt.pdf", MARKETING_RESUME)

        loaded = load_job(session, require_id(job.id, "job_description"))
        assert loaded is not None

        outcome = run_screening(
            session, loaded, llm=FakeLLM(default=70), embedder=FakeEmbedder()
        )

        shortlist = client.get(f"/api/v1/screening/runs/{outcome.run_id}/results").json()
        everything = client.get(
            f"/api/v1/screening/runs/{outcome.run_id}/results",
            params={"include_rejected": True},
        ).json()

        assert all(item["final_rank"] is not None for item in shortlist)
        assert len(everything) > len(shortlist)
        assert [item["final_rank"] for item in shortlist] == sorted(
            item["final_rank"] for item in shortlist
        )
        # Rejections carry their reasons, which is what makes the filter auditable.
        rejected = [item for item in everything if item["final_rank"] is None]
        assert rejected and all(item["rule_reasons"] for item in rejected)

    def test_cost_report_separates_measured_and_estimated(self, session, client):
        """A client must not be able to render the counterfactual as a measurement."""
        from app.services.screening.pipeline import run_screening
        from tests.conftest import FakeEmbedder, FakeLLM, add_resume, make_job
        from app.services.jobs import load_job

        job = make_job(session)
        add_resume(session, "go.pdf", GO_RESUME)

        loaded = load_job(session, require_id(job.id, "job_description"))
        assert loaded is not None

        outcome = run_screening(
            session, loaded, llm=FakeLLM(), embedder=FakeEmbedder()
        )

        body = client.get(f"/api/v1/screening/runs/{outcome.run_id}/cost").json()

        assert set(body) >= {"measured", "estimated", "funnel", "stages", "report"}
        assert "tokens" in body["measured"]
        assert "tokens" in body["estimated"]
        assert body["report"]


class TestChat:
    """The follow-up endpoint's contract."""

    def test_unknown_run_is_404(self, client):
        response = client.post("/api/v1/chat", json={"run_id": 999, "question": "hi"})

        assert response.status_code == 404

    def test_empty_question_is_rejected(self, client):
        response = client.post("/api/v1/chat", json={"run_id": 1, "question": ""})

        assert response.status_code == 422

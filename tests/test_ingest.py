"""Stage 0 and stage 2 plumbing — parsing, chunking, storage, retrieval.

The parsing and chunking half is deterministic and cheap to test. The storage half
is where the expensive mistakes live: a vector bound to the wrong chunk, or a
stale vector silently reused after the embedding model changed, produces rankings
that look entirely plausible and are meaningless.
"""

import pymupdf
import pytest
from sqlmodel import select

from app.core.config import settings
from app.models.screening import ResumeChunk
from app.services.resume import ResumeParseError, normalise_text, parse_pdf_bytes
from app.services.screening import store
from app.services.screening.chunker import _classify_heading, chunk_resume, split_sections
from app.services.screening.store import search
from tests.conftest import FakeEmbedder, SAMPLE_RESUME, add_resume, doc_id


def make_pdf(text: str) -> bytes:
    """Render text into a one-page PDF in memory."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 780), text, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


class TestNormaliseText:
    """Making extraction output stable enough to hash."""

    def test_rejoins_hyphenated_line_breaks(self):
        """PDF wrapping turns one word into two; hashing must not notice."""
        assert normalise_text("machi-\nne learning") == "machine learning"

    def test_collapses_intra_line_whitespace(self):
        assert normalise_text("Go    and\t\tJava") == "Go and Java"

    def test_reduces_blank_runs(self):
        assert normalise_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_is_idempotent(self):
        """Normalising twice must change nothing, or hashes drift between runs."""
        once = normalise_text(SAMPLE_RESUME)

        assert normalise_text(once) == once

    def test_tolerates_windows_line_endings(self):
        assert normalise_text("a\r\nb") == "a\nb"


class TestParsePdf:
    """Turning bytes into text, and refusing to guess."""

    def test_round_trip(self):
        parsed = parse_pdf_bytes(make_pdf(SAMPLE_RESUME))

        assert parsed.page_count == 1
        assert "Huaqiao University" in parsed.text

    def test_hash_is_stable_across_extractions(self):
        """The same document parsed twice must produce one hash, or dedup fails."""
        first = parse_pdf_bytes(make_pdf(SAMPLE_RESUME))
        second = parse_pdf_bytes(make_pdf(SAMPLE_RESUME))

        assert first.content_hash == second.content_hash

    def test_rejects_a_non_pdf(self):
        with pytest.raises(ResumeParseError, match="not a readable PDF"):
            parse_pdf_bytes(b"this is not a pdf at all")

    def test_rejects_a_missing_text_layer(self):
        """A scan needs OCR, and saying so beats a generic parse failure."""
        document = pymupdf.open()
        document.new_page()
        blank = document.tobytes()
        document.close()

        with pytest.raises(ResumeParseError, match="no usable text layer"):
            parse_pdf_bytes(blank)


class TestChunking:
    """Splitting on the structure the document already has."""

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("EDUCATION", "EDUCATION"),
            ("PROJ ECTS", "PROJECTS"),
            ("工 作 经 历", "EXPERIENCE"),
            ("项 目 经 历", "PROJECTS"),
            ("Skills", "SKILLS"),
            ("教育经历", "EDUCATION"),
            ("AWARDS AND HONOURS", "AWARDS"),
            ("自我评价", "SUMMARY"),
        ],
    )
    def test_recognises_headings(self, line, expected):
        """Includes letter-spaced headings, which PDF extraction produces routinely.

        "PROJ ECTS" is how a designed template stores "PROJECTS"; a keyword lookup
        misses it entirely. The CJK case is subtler still — Chinese has no case, so
        an ``isupper``-based collapse test never fires for the headings that need
        it most.
        """
        assert _classify_heading(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "Huaqiao University Bachelor of Computer Science",
            "Built an AI Agent with ReAct and function calling in Go",
            "Sep 2019 – Jun 2023",
            "guanji_liu@icloud.com",
            "+852 68240703",
            "Operations Assistant (AI Agent)",
        ],
    )
    def test_body_text_is_not_a_heading(self, line):
        assert _classify_heading(line) is None

    def test_splits_the_sample_resume(self):
        sections = dict(split_sections(SAMPLE_RESUME))

        assert {"EDUCATION", "PROJECTS", "SKILLS"} <= set(sections)

    def test_each_chunk_carries_its_heading(self):
        """The heading prefix is what gives the embedder section context.

        Without it a chunk reading "Built a ReAct agent loop" retrieves far worse
        than the same text under "EXPERIENCE:".
        """
        for chunk in chunk_resume(SAMPLE_RESUME):
            assert chunk.text.startswith(f"{chunk.section}: ")

    def test_chunks_do_not_cross_sections(self):
        """A chunk containing two headings means a section boundary was ignored."""
        for chunk in chunk_resume(SAMPLE_RESUME):
            body = chunk.text.split(": ", 1)[1]
            assert not any(
                _classify_heading(line) for line in body.split("\n")
            ), f"chunk spans a section boundary: {chunk.text[:80]!r}"

    def test_respects_the_size_target(self):
        chunks = chunk_resume("EXPERIENCE\n" + "\n".join(f"Bullet {i}" for i in range(200)))

        # The overlap means the last chunk can exceed the target by its own width.
        assert all(len(chunk.text) < 900 for chunk in chunks)

    @pytest.mark.parametrize("text", ["", "   \n  \n", "\n\n\n"])
    def test_empty_input_yields_nothing(self, text):
        assert chunk_resume(text) == []

    def test_text_without_headings_still_produces_a_chunk(self):
        """A caller must never have to special-case a heading-less document."""
        chunks = chunk_resume("just some plain text about a person")

        assert len(chunks) == 1
        assert chunks[0].section == "HEADER"


class TestChunkStorage:
    """Persistence and vector reuse."""

    def test_stores_chunks_in_order(self, session):
        document = add_resume(session, "a.pdf", SAMPLE_RESUME)
        chunks = chunk_resume(SAMPLE_RESUME)

        rows = store.store_chunks(session, doc_id(document), chunks)

        assert [row.index for row in rows] == list(range(len(chunks)))
        assert [row.text for row in rows] == [chunk.text for chunk in chunks]

    def test_unchanged_chunks_keep_their_vectors(self, session):
        """Re-parsing a resume must not throw away embeddings it already paid for."""
        document = add_resume(session, "a.pdf", SAMPLE_RESUME)
        chunks = chunk_resume(SAMPLE_RESUME)

        rows = store.store_chunks(session, doc_id(document), chunks)
        for row in rows:
            row.embedding = [0.5] * settings.EMBEDDING.dimensions
            row.embedding_model = settings.EMBEDDING.model
            session.add(row)
        session.commit()

        again = store.store_chunks(session, doc_id(document), chunks)

        assert all(row.is_embedded for row in again)

    def test_changed_text_invalidates_its_vector(self, session):
        """A stale vector describes different text, which is worse than none."""
        document = add_resume(session, "a.pdf", "EXPERIENCE: old text")
        rows = store.store_chunks(session, doc_id(document), chunk_resume("EXPERIENCE: old text"))
        rows[0].embedding = [0.5] * settings.EMBEDDING.dimensions
        rows[0].embedding_model = settings.EMBEDDING.model
        session.add(rows[0])
        session.commit()

        updated = store.store_chunks(session, doc_id(document), chunk_resume("EXPERIENCE: new text"))

        assert updated[0].embedding is None

    def test_shrinking_a_document_removes_orphaned_chunks(self, session):
        """Leftovers from a longer previous parse would compete in the index."""
        document = add_resume(session, "a.pdf", SAMPLE_RESUME)
        store.store_chunks(session, doc_id(document), chunk_resume(SAMPLE_RESUME))

        store.store_chunks(session, doc_id(document), chunk_resume("SKILLS: Go, Java"))

        remaining = session.exec(
            select(ResumeChunk).where(ResumeChunk.resume_id == doc_id(document))
        ).all()
        assert len(remaining) == 1


class TestEmbedding:
    """Batching and staleness."""

    def test_batches_within_the_provider_limit(self, session):
        """The provider rejects oversized batches, so the split is enforced here."""
        document = add_resume(
            session, "a.pdf", "\n".join(f"SKILLS: skill number {i}" for i in range(20))
        )
        store.store_chunks(session, doc_id(document), chunk_resume(document.raw_text))

        embedder = FakeEmbedder()
        store.embed_pending_chunks(session, embedder)

        limit = settings.EMBEDDING.batch_size
        assert embedder.batches
        assert all(len(batch) <= limit for batch in embedder.batches)

    def test_skips_chunks_that_are_already_current(self, session):
        """Re-embedding unchanged text spends money for nothing."""
        document = add_resume(session, "a.pdf", SAMPLE_RESUME)
        store.store_chunks(session, doc_id(document), chunk_resume(SAMPLE_RESUME))

        first = store.embed_pending_chunks(session, FakeEmbedder())
        second = store.embed_pending_chunks(session, FakeEmbedder())

        assert first > 0
        assert second == 0

    def test_a_different_model_forces_re_embedding(self, session):
        """Vectors from different models are not comparable.

        Mixing them yields similarity scores that look plausible and mean nothing,
        with no error raised anywhere — so the model name is recorded per chunk and
        checked.
        """
        document = add_resume(session, "a.pdf", SAMPLE_RESUME)
        store.store_chunks(session, doc_id(document), chunk_resume(SAMPLE_RESUME))
        store.embed_pending_chunks(session, FakeEmbedder())

        rows = session.exec(select(ResumeChunk)).all()
        for row in rows:
            row.embedding_model = "some-other-model"
            session.add(row)
        session.commit()

        assert store.embed_pending_chunks(session, FakeEmbedder()) == len(rows)


class TestSearch:
    """Similarity retrieval."""

    def _embed(self, session, embedder):
        """Chunk and embed every resume attached to the session."""
        from app.models.screening import ResumeDocument

        for document in session.exec(select(ResumeDocument)).all():
            store.store_chunks(session, doc_id(document), chunk_resume(document.raw_text))
        store.embed_pending_chunks(session, embedder)

    def test_ranks_the_relevant_resume_first(self, session):
        """The similarity measure must actually discriminate, or nothing downstream will."""
        add_resume(session, "go.pdf", "EXPERIENCE: Go backend engineer, Kubernetes, PostgreSQL")
        add_resume(session, "marketing.pdf", "EXPERIENCE: social media campaigns, copywriting")

        embedder = FakeEmbedder()
        self._embed(session, embedder)

        query = embedder.embed(["Go backend engineer Kubernetes"])[0]
        hits = search(session, query, limit=2, resume_ids=[1, 2])

        assert hits
        assert hits[0].resume_id == 1

    def test_empty_index_returns_nothing(self, session):
        assert search(session, [0.1] * settings.EMBEDDING.dimensions) == []

    def test_zero_vector_does_not_poison_the_ranking(self, session):
        """One degenerate vector must not turn every score into NaN.

        A zero vector cannot be normalised; dividing by its norm yields NaN, and a
        single NaN propagates through the dot product to corrupt the whole batch,
        with the ranking silently wrong rather than raising.
        """
        document = add_resume(session, "a.pdf", "SKILLS: Go")
        rows = store.store_chunks(session, doc_id(document), chunk_resume("SKILLS: Go"))
        rows[0].embedding = [0.0] * settings.EMBEDDING.dimensions
        rows[0].embedding_model = settings.EMBEDDING.model
        session.add(rows[0])
        session.commit()

        hits = search(session, [1.0] * settings.EMBEDDING.dimensions)

        assert len(hits) == 1
        assert hits[0].score == 0.0

    def test_dimension_mismatch_says_what_to_do(self, session):
        """A width mismatch means stale vectors; the error names the remedy."""
        document = add_resume(session, "a.pdf", "SKILLS: Go")
        rows = store.store_chunks(session, doc_id(document), chunk_resume("SKILLS: Go"))
        rows[0].embedding = [0.1] * 8
        rows[0].embedding_model = settings.EMBEDDING.model
        session.add(rows[0])
        session.commit()

        with pytest.raises(ValueError, match="force=True"):
            search(session, [0.1] * settings.EMBEDDING.dimensions)

    def test_restricting_to_resume_ids_excludes_others(self, session):
        # Distinct text on purpose: content_hash is unique, so identical resumes
        # are one document by design and cannot be used to build a two-row fixture.
        add_resume(session, "a.pdf", "SKILLS: Go")
        add_resume(session, "b.pdf", "SKILLS: Go and Java")
        embedder = FakeEmbedder()
        self._embed(session, embedder)

        query = embedder.embed(["Go"])[0]

        assert {hit.resume_id for hit in search(session, query, resume_ids=[2])} == {2}

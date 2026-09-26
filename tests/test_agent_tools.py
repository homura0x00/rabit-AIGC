"""Agent tools and the follow-up loop.

Two things are worth testing here and not more. The tools are the agent's only
view of the world, so their *scope* matters: they must be unable to reach a
different run, and unable to reach the full pile of resumes. And the loop is the
one place in this codebase where a model can decide how many calls to make, so its
step cap has to hold.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from sqlmodel import select

from app.models.screening import ScreeningRun
from app.services.agent import _MAX_STEPS, AgentError, ask, build_agent, build_tools
from app.services.llm import TokenLedger
from app.services.screening.pipeline import run_screening
from tests.conftest import FakeEmbedder, FakeLLM, add_resume, make_job

GO_RESUME = """Liu Guanji
PROJECTS
Built an AI operations assistant with Go and the Feishu API.
Implemented a ReAct-style agent loop with Function Calling.
Deployed on Kubernetes with PostgreSQL.
SKILLS
Go, Python, Kubernetes
"""

MARKETING_RESUME = """Wang Fang
EXPERIENCE
Marketing Intern. Ran social media campaigns on WeChat.
SKILLS
Social media, copywriting, Excel
"""


@pytest.fixture
def completed_run(session):
    """A finished screening run with one kept and one rejected candidate."""
    job = make_job(session)
    add_resume(session, "go.pdf", GO_RESUME)
    add_resume(session, "mkt.pdf", MARKETING_RESUME)

    outcome = run_screening(
        session,
        job,
        llm=FakeLLM(default=70),
        embedder=FakeEmbedder(),
        review_borderline=False,
    )
    return outcome.run_id


class TestToolScoping:
    """What the tools can and cannot see."""

    def test_tools_are_bound_to_one_run(self, session, completed_run):
        """Another run's candidates must be unreachable.

        A tool that took a run id from the model could be pointed at any run —
        including one the asking HR user should not see — and the model would
        sometimes supply the wrong one.
        """
        other = ScreeningRun(job_id=1)
        session.add(other)
        session.commit()

        tools = build_tools(session, completed_run)
        listed = tools["list_shortlist"].handler({})

        assert listed["run_id"] == completed_run
        assert all(item["resume_id"] is not None for item in listed["candidates"])

    def test_shortlist_excludes_rejected_by_default(self, session, completed_run):
        """The shortlist is what a reviewer acts on; rejections are opt-in."""
        tools = build_tools(session, completed_run)

        default = tools["list_shortlist"].handler({})
        full = tools["list_shortlist"].handler({"include_rejected": True})

        assert all(item["rule_passed"] for item in default["candidates"])
        assert len(full["candidates"]) > len(default["candidates"])

    def test_shortlist_is_bounded(self, session, completed_run):
        """Tools exist to keep context small; an unbounded dump defeats that."""
        tools = build_tools(session, completed_run)

        listed = tools["list_shortlist"].handler({})

        assert len(listed["candidates"]) <= 30


class TestGetCandidate:
    """Single-candidate detail."""

    def test_looks_up_by_rank(self, session, completed_run):
        tools = build_tools(session, completed_run)
        first = tools["list_shortlist"].handler({})["candidates"][0]

        detail = tools["get_candidate"].handler({"rank": first["rank"]})

        assert detail["resume_id"] == first["resume_id"]
        assert "evidence" in detail
        assert "gaps" in detail
        assert "rule_reasons" in detail

    def test_looks_up_by_resume_id(self, session, completed_run):
        tools = build_tools(session, completed_run)
        first = tools["list_shortlist"].handler({})["candidates"][0]

        detail = tools["get_candidate"].handler({"resume_id": first["resume_id"]})

        assert detail["rank"] == first["rank"]

    def test_unknown_candidate_returns_an_error_not_an_exception(self, session, completed_run):
        """An error payload lets the model correct itself; an exception ends the question."""
        tools = build_tools(session, completed_run)

        result = tools["get_candidate"].handler({"rank": 999})

        assert "error" in result

    def test_notes_are_separate_from_gaps(self, session, completed_run):
        """Distinct concepts in the schema must stay distinct in the payload."""
        tools = build_tools(session, completed_run)

        detail = tools["get_candidate"].handler({"rank": 1})

        assert "gaps" in detail
        assert "review_note" in detail


class TestFindCandidates:
    """Keyword search over already-derived data."""

    def test_finds_a_skill_mentioned_in_evidence(self, session, completed_run):
        tools = build_tools(session, completed_run)

        found = tools["find_candidates"].handler({"keyword": "Go"})

        assert found["hits"]

    def test_finds_text_in_retrieved_chunks(self, session, completed_run):
        """Nuance lives in the retrieved text, not only in a verdict field."""
        tools = build_tools(session, completed_run)

        found = tools["find_candidates"].handler({"keyword": "Feishu"})

        assert found["hits"]
        assert any(
            match["field"].startswith("原文") for hit in found["hits"] for match in hit["matches"]
        )

    def test_requires_a_keyword(self, session, completed_run):
        tools = build_tools(session, completed_run)

        assert "error" in tools["find_candidates"].handler({})

    def test_no_match_is_an_empty_result(self, session, completed_run):
        """An empty hit list is information; pretending otherwise would invite invention."""
        tools = build_tools(session, completed_run)

        found = tools["find_candidates"].handler({"keyword": "quantum-tensor-flow"})

        assert found["hits"] == []

    def test_reports_how_many_candidates_were_searched(self, session, completed_run):
        """So the model can say what it looked at rather than implying completeness."""
        tools = build_tools(session, completed_run)

        found = tools["find_candidates"].handler({"keyword": "Go"})

        assert found["searched_candidates"] >= len(found["hits"])


class TestRunCostTool:
    """Cost lookups, with the measured/estimated split intact."""

    def test_separates_measured_from_estimated(self, session, completed_run):
        """The agent relays these to a human who will quote them."""
        tools = build_tools(session, completed_run)

        cost = tools["run_cost"].handler({})

        assert "measured_tokens" in cost
        assert "estimated_baseline_tokens" in cost
        assert cost["measured_tokens"] != cost["estimated_baseline_tokens"]


class TestToolSchemas:
    """Contract with the model."""

    def test_every_tool_has_a_description_and_schema(self, session, completed_run):
        """A vague tool description produces vague tool use."""
        for name, tool in build_tools(session, completed_run).items():
            schema = tool.schema()

            assert tool.description, name
            assert schema["type"] == "function"
            assert schema["function"]["name"] == name
            assert schema["function"]["parameters"]["type"] == "object"


@dataclass
class _Function:
    name: str
    arguments: str = "{}"


@dataclass
class _ToolCall:
    id: str
    function: _Function


@dataclass
class _Message:
    content: Optional[str] = None
    tool_calls: list[_ToolCall] = field(default_factory=list)


class ScriptedAgentLLM:
    """Agent stand-in that calls one tool and then answers.

    Returning a real tool call rather than text exercises the graph's routing,
    the argument parsing and the tool-result round trip, none of which a
    text-only fake would reach.
    """

    ledger: Optional[TokenLedger] = None

    def __init__(self, tool_name: str = "list_shortlist", arguments: str = "{}"):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0
        self.saw_tool_result = False

    def complete_with_tools(self, messages, **kwargs):
        self.calls += 1

        if any(message.get("role") == "tool" for message in messages):
            self.saw_tool_result = True
            return _Message(content="已根据工具结果作答。")

        return _Message(
            tool_calls=[
                _ToolCall(id="call_1", function=_Function(self.tool_name, self.arguments))
            ]
        )


class TestAgentLoop:
    """The graph, its routing and its bound."""

    def test_answers_after_using_a_tool(self, session, completed_run):
        llm = ScriptedAgentLLM()

        answer = ask(session, completed_run, "推荐哪几位？", llm=llm)

        assert answer.answer
        assert answer.tool_calls == ("list_shortlist",)
        assert llm.saw_tool_result is True

    def test_tool_arguments_are_parsed(self, session, completed_run):
        """A tool called with real arguments must reach the handler with them."""
        llm = ScriptedAgentLLM("find_candidates", '{"keyword": "Go"}')

        answer = ask(session, completed_run, "谁有 Go 经验？", llm=llm)

        assert answer.tool_calls == ("find_candidates",)

    def test_unknown_tool_is_reported_back_to_the_model(self, session, completed_run):
        """A hallucinated tool name must not end the question.

        Feeding the error back lets the model correct itself, which is cheaper than
        discarding a question the user already paid for.
        """
        llm = ScriptedAgentLLM("nonexistent_tool")

        answer = ask(session, completed_run, "?", llm=llm)

        assert answer.answer
        assert answer.tool_calls == ("nonexistent_tool",)

    def test_malformed_arguments_do_not_crash(self, session, completed_run):
        llm = ScriptedAgentLLM("find_candidates", "{not json")

        answer = ask(session, completed_run, "?", llm=llm)

        assert answer.answer

    def test_loop_is_bounded(self, session, completed_run):
        """An agent that can call tools without a bound can spend without a bound."""

        class NeverStops:
            ledger: Optional[TokenLedger] = None

            def complete_with_tools(self, messages, **kwargs):
                return _Message(
                    tool_calls=[_ToolCall(id="c", function=_Function("list_shortlist"))]
                )

        # Hitting the cap surfaces as an error rather than a blank answer: an
        # empty string reads as "there was nothing to say", which is a different
        # claim from "I ran out of steps".
        with pytest.raises(AgentError, match="within 3 steps"):
            ask(session, completed_run, "?", llm=NeverStops(), max_steps=3)

    def test_unknown_run_raises(self, session):
        """Failing before the graph is built keeps the error close to the cause."""
        with pytest.raises(AgentError, match="not found"):
            ask(session, 999, "?")

    def test_graph_compiles(self, session, completed_run):
        assert build_agent(session, completed_run, ScriptedAgentLLM()) is not None


class TestStepCap:
    """The cap itself."""

    def test_cap_is_small(self):
        """A follow-up question needs a lookup, not a research project."""
        assert 1 <= _MAX_STEPS <= 10

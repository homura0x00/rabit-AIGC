"""HR follow-up agent — the one place in this project where an agent is warranted.

Why here and nowhere else
-------------------------

Screening is a pipeline: a fixed sequence of transforms where each stage hands a
strictly smaller set to the next. Nothing in it needs deciding, so routing it
through a tool-calling loop would only resend context and burn tokens — which is
why the previous stages call the model directly and record exactly one row per
call.

Conversation is the opposite case. "Which of these have Go experience?" and "why
was candidate 7 rejected?" are different queries over the same small corpus, so
the model genuinely has to choose tools rather than follow a script.

The corpus is what keeps it affordable. Every tool reads data the pipeline already
derived — screening results, the extracted profile, the retrieved chunks — and
none of them can reach the full pile of uploaded resumes. The agent's context is
bounded by the shortlist, which is tens of rows, not by how many resumes were
screened.

LangGraph for control flow, own client for accounting
----------------------------------------------------

The graph is LangGraph's, because a two-node loop with a conditional exit is
exactly what it is for and the alternative is hand-rolling the same thing. The
provider client is this project's own, because every other stage records usage
through it and an assistant that silently spends tokens would be the one hole in
an otherwise complete ledger — and the hardest to notice, since an agent makes a
variable number of calls per question.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.log import get_logger
from app.core.prompts import load_prompt, load_system_prompt
from app.models.screening import (
    CallPurpose,
    ResumeChunk,
    ResumeDocument,
    ScreeningResult,
    ScreeningRun,
)
from app.services.llm import LLMClient, LLMError, TokenLedger, ToolCallingChatModel, Usage
from app.services.screening.pipeline import run_calls
from app.services.screening.report import build_report

logger = get_logger(__name__)

# Hard cap on loop iterations. An agent that can call tools without a bound is an
# agent that can spend without a bound, and a malformed question is enough to send
# one into a loop.
_MAX_STEPS = 6

# Bounds on what a tool will return. Tools exist to keep the agent's context
# small; a tool that dumps a whole resume defeats the point of having it.
_MAX_ROWS = 30
_MAX_SNIPPET_CHARS = 240


class AgentError(RuntimeError):
    """Raised when the agent cannot produce an answer."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool the agent may call.

    Attributes:
        name: Function name the model uses.
        description: What it does and when to reach for it. This is prompt text —
            a vague description produces vague tool use.
        parameters: JSON Schema for the arguments.
        handler: Callable taking parsed arguments and returning a JSON-serialisable
            value.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        """Render as an OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class AgentAnswer:
    """The agent's reply plus what it cost to produce.

    Attributes:
        answer: The final assistant text.
        steps: Model calls made.
        tool_calls: Names of tools invoked, in order.
        usage: Summed token usage across the loop.
    """

    answer: str
    steps: int = 0
    tool_calls: tuple[str, ...] = ()
    usage: Usage = field(default_factory=lambda: Usage(model=""))


def _candidate_label(result: ScreeningResult, document: Optional[ResumeDocument]) -> str:
    """Build a short human label for a candidate.

    Prefers the filename because no extraction stage runs in this pipeline
    version: the candidate's name exists inside the resume text, but nothing has
    parsed it into a column, and inventing one here would put two sources of truth
    side by side.

    Args:
        result: The persisted screening result.
        document: The owning resume, if loaded.

    Returns:
        A label like ``#2 real.pdf``.
    """
    rank = f"#{result.final_rank} " if result.final_rank else ""
    return f"{rank}{document.filename if document else f'resume {result.resume_id}'}"


def build_tools(session: Session, run_id: int) -> dict[str, ToolSpec]:
    """Build the tool set for one screening run.

    The session and run are bound by closure so the model never has to supply an
    identifier it could get wrong, and so it cannot reach a different run's data.

    Args:
        session: Database session.
        run_id: The run every tool is scoped to.

    Returns:
        Tools keyed by name.
    """

    def load_results() -> list[tuple[ScreeningResult, Optional[ResumeDocument]]]:
        rows = session.exec(
            select(ScreeningResult, ResumeDocument)
            .where(ScreeningResult.run_id == run_id)
            .join(ResumeDocument, col(ScreeningResult.resume_id) == col(ResumeDocument.id))
        ).all()
        ordered = sorted(
            rows,
            key=lambda pair: (
                pair[0].final_rank is None,
                pair[0].final_rank if pair[0].final_rank is not None else 0,
            ),
        )
        return [(result, document) for result, document in ordered]

    def list_shortlist(arguments: dict[str, Any]) -> Any:
        include_rejected = bool(arguments.get("include_rejected", False))
        rows = load_results()

        out = []
        for result, document in rows:
            if result.final_rank is None and not include_rejected:
                continue
            out.append(
                {
                    "rank": result.final_rank,
                    "resume_id": result.resume_id,
                    "label": _candidate_label(result, document),
                    "score": result.judge_score,
                    "tier": result.judge_tier,
                    "recommendation": result.recommendation,
                    "rule_passed": result.rule_passed,
                    "reviewed": result.reviewed,
                }
            )
            if len(out) >= _MAX_ROWS:
                break

        return {"run_id": run_id, "candidates": out, "total": len(rows)}

    def get_candidate(arguments: dict[str, Any]) -> Any:
        rank = arguments.get("rank")
        resume_id = arguments.get("resume_id")

        for result, document in load_results():
            if (rank is not None and result.final_rank == rank) or (
                resume_id is not None and result.resume_id == resume_id
            ):
                return {
                    "rank": result.final_rank,
                    "resume_id": result.resume_id,
                    "label": _candidate_label(result, document),
                    "score": result.judge_score,
                    "tier": result.judge_tier,
                    "recommendation": result.recommendation,
                    "evidence": list(result.judge_evidence or []),
                    "gaps": list(result.judge_gaps or []),
                    "review_note": result.judge_note,
                    "reviewed": result.reviewed,
                    "rule_passed": result.rule_passed,
                    "rule_score": result.rule_score,
                    "rule_reasons": list(result.rule_reasons or []),
                    "recall_score": result.recall_score,
                }

        return {"error": f"no candidate matching rank={rank} resume_id={resume_id}"}

    def find_candidates(arguments: dict[str, Any]) -> Any:
        keyword = " ".join(str(arguments.get("keyword", "")).split())
        if not keyword:
            return {"error": "keyword is required"}

        needle = keyword.casefold()
        rows = load_results()
        shortlisted = {result.resume_id for result, _ in rows}

        hits: list[dict[str, Any]] = []

        for result, document in rows:
            fields = [
                *[("证据", text) for text in (result.judge_evidence or [])],
                *[("缺口", text) for text in (result.judge_gaps or [])],
                *[("规则理由", text) for text in (result.rule_reasons or [])],
            ]
            matched = [
                {"field": name, "text": text[:_MAX_SNIPPET_CHARS]}
                for name, text in fields
                if needle in text.casefold()
            ]

            # Also search the chunks the pipeline actually retrieved, which is
            # where a nuance like "used Kafka at scale" lives when no verdict
            # field happened to mention it.
            if len(matched) < 2:
                chunks = session.exec(
                    select(ResumeChunk)
                    .where(ResumeChunk.resume_id == result.resume_id)
                    .order_by(col(ResumeChunk.index))
                ).all()
                for chunk in chunks:
                    if needle in chunk.text.casefold():
                        matched.append(
                            {
                                "field": f"原文[{chunk.section}]",
                                "text": chunk.text[:_MAX_SNIPPET_CHARS],
                            }
                        )
                    if len(matched) >= 4:
                        break

            if matched:
                hits.append(
                    {
                        "rank": result.final_rank,
                        "resume_id": result.resume_id,
                        "label": _candidate_label(result, document),
                        "matches": matched[:4],
                    }
                )

        return {
            "keyword": keyword,
            "searched_candidates": len(shortlisted),
            "hits": hits[:_MAX_ROWS],
        }

    def run_cost(arguments: dict[str, Any]) -> Any:
        run = session.get(ScreeningRun, run_id)
        if run is None:
            return {"error": f"run {run_id} not found"}

        report = build_report(run, run_calls(session, run_id))
        return {
            "funnel": report.funnel,
            # Flagged as estimates in the payload itself, not only in prose: the
            # agent relays these to a human who will quote them.
            "measured_tokens": report.actual_tokens,
            "measured_cost": round(report.actual_est_cost, 6),
            "estimated_baseline_tokens": report.baseline_tokens,
            "estimated_baseline_cost": round(report.baseline_est_cost, 6),
            "cost_ratio": round(report.cost_ratio, 2) if report.cost_ratio else None,
            "cache_hit_rate": report.cache_hit_rate,
            "currency": report.currency,
        }

    tools = [
        ToolSpec(
            name="list_shortlist",
            description=(
                "List the candidates for this screening run, ranked. Use this first "
                "for questions about who was recommended or how the run turned out. "
                "By default only candidates that reached scoring are returned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "include_rejected": {
                        "type": "boolean",
                        "description": (
                            "Also include candidates dropped by the rule stage, with "
                            "their rejection reasons. Use for questions about who was "
                            "filtered out and why."
                        ),
                    }
                },
                "required": [],
            },
            handler=list_shortlist,
        ),
        ToolSpec(
            name="get_candidate",
            description=(
                "Get one candidate's full detail: score, tier, the evidence quoted "
                "from their material, their gaps, the rule-stage reasons, and any "
                "review note. Use for 'why' questions about a specific candidate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "description": "Final rank, e.g. 1"},
                    "resume_id": {"type": "integer", "description": "Resume identifier"},
                },
                "required": [],
            },
            handler=get_candidate,
        ),
        ToolSpec(
            name="find_candidates",
            description=(
                "Find candidates whose stored evidence, gaps, rule reasons or "
                "retrieved resume text mention a keyword. Use for questions like "
                "'who has Go experience' or 'who worked on payments'. Only searches "
                "candidates in this run, never the full pile of uploaded resumes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Case-insensitive substring, e.g. 'Kubernetes'",
                    }
                },
                "required": ["keyword"],
            },
            handler=find_candidates,
        ),
        ToolSpec(
            name="run_cost",
            description=(
                "Get this run's token usage and cost. Measured figures and the "
                "estimated naive baseline are separate fields; when quoting a saving, "
                "say which is which."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=run_cost,
        ),
    ]

    return {tool.name: tool for tool in tools}


class AgentState(TypedDict):
    """Graph state: the conversation and a loop counter."""

    messages: list[dict[str, Any]]
    steps: int
    tool_names: list[str]


def build_agent(
    session: Session,
    run_id: int,
    llm: Optional[ToolCallingChatModel] = None,
    *,
    ledger: Optional[TokenLedger] = None,
    max_steps: int = _MAX_STEPS,
):
    """Compile the follow-up agent graph.

    Args:
        session: Database session the tools read from.
        run_id: The screening run to scope every tool to.
        llm: Chat client. One is created when omitted.
        ledger: Where the created client records usage.
        max_steps: Loop cap.

    Returns:
        A compiled LangGraph app.
    """
    tools = build_tools(session, run_id)
    client: ToolCallingChatModel = llm or LLMClient(ledger)
    schemas = [tool.schema() for tool in tools.values()]

    system_prompt = "\n\n".join([load_system_prompt(), load_prompt("agent")])

    def call_model(state: AgentState) -> AgentState:
        message = client.complete_with_tools(
            [{"role": "system", "content": system_prompt}, *state["messages"]],
            tools=schemas,
            purpose=CallPurpose.CHAT.value,
        )

        entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        return {
            "messages": [*state["messages"], entry],
            "steps": state["steps"] + 1,
            "tool_names": list(state["tool_names"]),
        }

    def call_tools(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        results: list[dict[str, Any]] = []
        invoked = list(state["tool_names"])

        for call in last.get("tool_calls", []):
            name = call["function"]["name"]
            invoked.append(name)

            tool = tools.get(name)
            if tool is None:
                # A hallucinated tool name. Feeding the error back lets the model
                # correct itself, which is cheaper than aborting the question.
                payload: Any = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    payload = {"error": f"arguments were not valid JSON: {exc}"}
                else:
                    try:
                        payload = tool.handler(arguments)
                    except Exception as exc:
                        logger.exception("tool %s failed", name)
                        payload = {"error": f"{type(exc).__name__}: {exc}"}

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                }
            )

        return {
            "messages": [*state["messages"], *results],
            "steps": state["steps"],
            "tool_names": invoked,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if state["steps"] >= max_steps:
            logger.warning("agent hit the %d-step cap", max_steps)
            return END
        if last.get("role") == "assistant" and last.get("tool_calls"):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    return graph.compile()


def ask(
    session: Session,
    run_id: int,
    question: str,
    *,
    llm: Optional[ToolCallingChatModel] = None,
    ledger: Optional[TokenLedger] = None,
    max_steps: int = _MAX_STEPS,
) -> AgentAnswer:
    """Answer one question about a screening run.

    Args:
        session: Database session.
        run_id: The run to scope the answer to.
        question: The HR question.
        llm: Chat client. One is created when omitted.
        ledger: Where a created client records usage.
        max_steps: Loop cap.

    Returns:
        The answer with its token cost.

    Raises:
        AgentError: If the run does not exist, or the model produces no answer.
    """
    if session.get(ScreeningRun, run_id) is None:
        raise AgentError(f"run {run_id} not found")

    ledger = ledger or TokenLedger(session=session, run_id=run_id)
    client: ToolCallingChatModel = llm or LLMClient(ledger)

    app = build_agent(session, run_id, client, max_steps=max_steps)

    initial: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "steps": 0,
        "tool_names": [],
    }

    try:
        final = app.invoke(initial, {"recursion_limit": max_steps * 2 + 2})
    except LLMError as exc:
        raise AgentError(f"agent call failed: {exc}") from exc

    answer = ""
    for message in reversed(final["messages"]):
        if message.get("role") == "assistant" and message.get("content"):
            answer = str(message["content"]).strip()
            break

    if not answer:
        # Reached when the loop hits its cap while the model was still calling
        # tools. Surfaced as an error rather than an empty string: a blank answer
        # reads as "there was nothing to say", which is a different claim.
        raise AgentError(
            f"agent produced no answer within {max_steps} steps; "
            "it may have been mid-tool-call when the loop hit its cap"
        )

    return AgentAnswer(
        answer=answer,
        steps=final["steps"],
        tool_calls=tuple(final["tool_names"]),
        usage=ledger.totals(),
    )

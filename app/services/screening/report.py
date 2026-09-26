"""Cost report — the part of this project that is meant to be read.

The whole claim of the pipeline is that it reaches a comparable shortlist for a
fraction of the tokens a naive per-resume evaluation would spend. A claim like
that is worth nothing unless it can be checked, so this module exists to make the
numbers visible and to be scrupulous about which of them are which.

**Measured.** Every token count here comes from the provider's own usage payload,
including the cache-hit split, recorded per call by
:class:`~app.services.llm.TokenLedger`. Funnel counts come from the database rows
the pipeline wrote.

**Estimated.** Nobody can measure what a different architecture would have spent,
because it was never run. The baseline is therefore a counterfactual computed from
the real resume lengths and a stated per-resume assumption, and every place it
appears says so. Presenting a constructed baseline with the same confidence as a
measured total would be the single most misleading thing this project could do —
and the estimate is conservative in direction anyway, since the naive approach
would also have to pay for the candidates the funnel eliminates for free.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.core.config import settings
from app.services.llm import LedgerEntry, Usage, estimate_cost
from app.models.screening import LLMCallLog, ScreeningRun

# Order stages appear in the report. Dict ordering is not something to rely on
# when the output is meant to be read by a person comparing runs.
_STAGE_ORDER = ("embed", "rerank", "judge", "review", "extract")

_STAGE_LABELS = {
    "embed": "L2 召回 embedding",
    "rerank": "L2.5 交叉编码重排",
    "judge": "L3 批量评分",
    "review": "L4 边界复核",
    "extract": "E  结构化抽取",
}


@dataclass(frozen=True)
class StageCost:
    """One stage's contribution to the bill.

    Attributes:
        purpose: Stage identifier.
        label: Human-readable stage name.
        calls: Provider calls issued.
        prompt_tokens: Input tokens billed.
        cached_tokens: Input tokens served from cache.
        completion_tokens: Output tokens generated.
        est_cost: Estimated cost at this model's rate.
        avg_batch_size: Mean candidates per call, which is the batching payoff
            made explicit.
    """

    purpose: str
    label: str
    calls: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    est_cost: float
    avg_batch_size: float

    @property
    def total_tokens(self) -> int:
        """All tokens billed by this stage."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Share of this stage's input tokens served from cache."""
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens


@dataclass(frozen=True)
class CostReport:
    """A complete accounting of one screening run.

    Attributes:
        run_id: The run this describes.
        funnel: Stage-by-stage candidate counts.
        actual_tokens: Tokens actually billed. Measured.
        actual_est_cost: Estimated cost of the measured tokens.
        baseline_tokens: Tokens the naive approach would likely have billed.
            Estimated.
        baseline_est_cost: Estimated cost of the estimated baseline.
        cache_hit_rate: Overall share of input tokens served from cache.
        stages: Per-stage breakdown.
        currency: Currency of every cost figure here.
    """

    run_id: Optional[int]
    funnel: dict[str, int]
    actual_tokens: int
    actual_est_cost: float
    baseline_tokens: int
    baseline_est_cost: float
    cache_hit_rate: Optional[float]
    stages: tuple[StageCost, ...] = ()
    currency: str = "CNY"

    @property
    def ratio(self) -> Optional[float]:
        """How many times cheaper than the baseline, or ``None`` if nothing spent."""
        if self.actual_tokens <= 0:
            return None
        return self.baseline_tokens / self.actual_tokens

    @property
    def cost_ratio(self) -> Optional[float]:
        """How many times cheaper than the baseline in money, or ``None``.

        Reported alongside the token ratio because the two diverge, sometimes
        sharply, and the divergence is itself the finding. Chunk embedding becomes
        the largest token consumer by volume as the pile grows, while costing
        roughly a thousandth as much per token as judgement — so a token ratio
        understates the actual saving. Quoting only the cost ratio would hide that
        the volume is real; quoting only the token ratio would undersell the
        result. Both are shown, and neither is allowed to stand alone.
        """
        if self.actual_est_cost <= 0:
            return None
        return self.baseline_est_cost / self.actual_est_cost

    @property
    def tokens_per_judged(self) -> Optional[float]:
        """Tokens per candidate that reached paid judgement.

        The only figure here that is comparable across runs of different sizes,
        which makes it the one worth tracking over time.
        """
        judged = self.funnel.get("judged", 0)
        if judged <= 0:
            return None
        return self.actual_tokens / judged


def _normalise(entry: Any) -> tuple[str, str, int, int, int, int]:
    """Extract comparable fields from a ledger entry or a persisted call row.

    Accepts both shapes on purpose: the pipeline reports from the in-memory ledger
    it just filled, while the API reports from rows loaded out of the database, and
    forcing one to be converted into the other would mean two code paths for the
    same arithmetic.

    Args:
        entry: A :class:`~app.services.llm.LedgerEntry` or an :class:`LLMCallLog`.

    Returns:
        ``(purpose, model, prompt_tokens, cached_tokens, completion_tokens,
        batch_size)``.
    """
    if isinstance(entry, LedgerEntry):
        usage = entry.usage
        return (
            entry.purpose,
            usage.model,
            usage.prompt_tokens,
            usage.cached_tokens,
            usage.completion_tokens,
            entry.batch_size,
        )

    return (
        str(getattr(entry, "purpose", "")),
        str(getattr(entry, "model", "")),
        int(getattr(entry, "prompt_tokens", 0) or 0),
        int(getattr(entry, "cached_tokens", 0) or 0),
        int(getattr(entry, "completion_tokens", 0) or 0),
        int(getattr(entry, "batch_size", 1) or 1),
    )


def breakdown_by_stage(entries: Sequence[Any]) -> tuple[StageCost, ...]:
    """Group calls by stage and total each.

    Args:
        entries: Ledger entries or persisted call rows.

    Returns:
        Per-stage costs, in a fixed reading order, with unknown purposes last.
    """
    grouped: dict[str, list[tuple[str, int, int, int, int]]] = {}

    for entry in entries:
        purpose, model, prompt, cached, completion, batch_size = _normalise(entry)
        grouped.setdefault(purpose, []).append((model, prompt, cached, completion, batch_size))

    stages: list[StageCost] = []

    for purpose, rows in grouped.items():
        usages = [
            Usage(model=model, prompt_tokens=prompt, cached_tokens=cached, completion_tokens=completion)
            for model, prompt, cached, completion, _ in rows
        ]
        stages.append(
            StageCost(
                purpose=purpose,
                label=_STAGE_LABELS.get(purpose, purpose or "未分类"),
                calls=len(rows),
                prompt_tokens=sum(row[1] for row in rows),
                cached_tokens=sum(row[2] for row in rows),
                completion_tokens=sum(row[3] for row in rows),
                est_cost=estimate_cost(usages),
                avg_batch_size=sum(row[4] for row in rows) / len(rows),
            )
        )

    rank = {purpose: position for position, purpose in enumerate(_STAGE_ORDER)}
    stages.sort(key=lambda stage: rank.get(stage.purpose, len(_STAGE_ORDER)))

    return tuple(stages)


def build_report(run: ScreeningRun, entries: Sequence[Any]) -> CostReport:
    """Build a cost report for a run.

    Args:
        run: The persisted run row.
        entries: Ledger entries or persisted call rows for this run.

    Returns:
        The report. Funnel counts and measured totals come from the run row;
        stage costs are recomputed from ``entries`` so the two cannot drift.
    """
    usages = [
        Usage(
            model=model,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=completion,
        )
        for _, model, prompt, cached, completion, _ in (_normalise(e) for e in entries)
    ]

    actual_tokens = run.prompt_tokens + run.completion_tokens

    return CostReport(
        run_id=run.id,
        funnel={
            "documents": run.total_documents,
            "rule_rejected": run.rule_rejected,
            "shortlisted": run.shortlisted,
            "judged": run.judged,
            "reviewed": run.reviewed,
            "failed": run.failed,
        },
        actual_tokens=actual_tokens,
        actual_est_cost=estimate_cost(usages),
        baseline_tokens=run.baseline_prompt_tokens + run.baseline_completion_tokens,
        baseline_est_cost=run.baseline_est_cost,
        cache_hit_rate=run.cache_hit_rate,
        stages=breakdown_by_stage(entries),
        currency=settings.LLM.price.currency,
    )


def format_report(report: CostReport) -> str:
    """Render a report as plain text for a terminal or a log line.

    Args:
        report: The report to render.

    Returns:
        A multi-line string. Every estimated figure is marked as such in place,
        not in a footnote.
    """
    funnel = report.funnel
    lines: list[str] = []

    lines.append(f"筛选运行 #{report.run_id}")
    lines.append("")
    lines.append("漏斗")
    lines.append(f"  入库简历      {funnel.get('documents', 0):>6}")
    lines.append(f"  规则层淘汰    {funnel.get('rule_rejected', 0):>6}   (0 token)")
    lines.append(f"  进入召回      {funnel.get('shortlisted', 0):>6}")
    lines.append(f"  完成评分      {funnel.get('judged', 0):>6}")
    lines.append(f"  边界复核      {funnel.get('reviewed', 0):>6}")
    if funnel.get("failed"):
        lines.append(f"  评分失败      {funnel.get('failed', 0):>6}")

    lines.append("")
    lines.append("按阶段消耗")
    lines.append(
        f"  {'阶段':<18}{'调用':>5}{'输入':>9}{'命中':>8}{'输出':>8}{'批次':>7}  {'成本':>10}"
    )
    for stage in report.stages:
        hit = f"{stage.cache_hit_rate:.0%}" if stage.cache_hit_rate is not None else "-"
        lines.append(
            f"  {stage.label:<18}{stage.calls:>5}{stage.prompt_tokens:>9}"
            f"{hit:>8}{stage.completion_tokens:>8}{stage.avg_batch_size:>7.1f}"
            f"  {stage.est_cost:>9.6f}"
        )

    lines.append("")
    lines.append("总量对比")
    lines.append(
        f"  实测消耗      {report.actual_tokens:>9} tokens   "
        f"≈ {report.actual_est_cost:.6f} {report.currency}"
    )
    lines.append(
        f"  朴素基线      {report.baseline_tokens:>9} tokens   "
        f"≈ {report.baseline_est_cost:.6f} {report.currency}   (估算)"
    )

    if report.ratio is not None:
        lines.append(f"  token 倍数    {report.ratio:>9.1f}x")
    if report.cost_ratio is not None:
        lines.append(f"  成本倍数      {report.cost_ratio:>9.1f}x")

    if report.cache_hit_rate is not None:
        lines.append(f"  缓存命中率    {report.cache_hit_rate:>9.1%}")

    if report.tokens_per_judged is not None:
        lines.append(f"  每份评分消耗  {report.tokens_per_judged:>9.0f} tokens")

    lines.append("")
    lines.append("说明：实测值来自 provider 返回的 usage，含 embedding 消耗；")
    lines.append("      基线为「每份简历单独全量评估」的估算值，前缀长度取自实际组装的 prompt，")
    lines.append("      但调用形态是假设，非实测。token 倍数与成本倍数背离的原因是")
    lines.append("      chunk embedding 在量上占比大、单价却低约三个数量级。")

    return "\n".join(lines)

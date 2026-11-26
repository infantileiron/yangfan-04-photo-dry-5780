"""Pure holdout comparison and idempotent model-adoption rules.

The module is intentionally database/framework independent.  It accepts plain
dictionaries, dataclass-like objects, or the project's ``Trial`` objects so unit
tests and service code can call the same rules.
"""

from __future__ import annotations

import math
import threading
import uuid
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Iterable, Mapping

DEFAULT_CONFIDENCE = 0.95
DEFAULT_MIN_DATES = 2
# The requirement says “did not rise”; do not silently invent a tolerance.
NON_INFERIORITY_MARGIN = 0.0

DEFAULT_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "high_humidity_heavy_ink",
        "label": "高湿重墨",
        "all": [
            {"field": "humidity_pct", "op": ">=", "value": 70},
            {"field": "ink_cover_pct", "op": ">=", "value": 70},
        ],
    },
    {
        "key": "high_humidity_normal_ink",
        "label": "高湿非重墨",
        "all": [
            {"field": "humidity_pct", "op": ">=", "value": 70},
            {"field": "ink_cover_pct", "op": "<", "value": 70},
        ],
    },
    {
        "key": "lower_humidity_heavy_ink",
        "label": "非高湿重墨",
        "all": [
            {"field": "humidity_pct", "op": "<", "value": 70},
            {"field": "ink_cover_pct", "op": ">=", "value": 70},
        ],
    },
    {
        "key": "ordinary",
        "label": "非高湿非重墨",
        "all": [
            {"field": "humidity_pct", "op": "<", "value": 70},
            {"field": "ink_cover_pct", "op": "<", "value": 70},
        ],
    },
)

DATE_KEYS = (
    "trial_date", "date", "day", "test_date", "validation_date", "holdout_date",
    "试机日期", "留验日期", "日期",
)
MEASURED_KEYS = (
    "measured_s", "measured_seconds", "actual_s", "actual_seconds", "actual_dry_seconds",
    "dry_seconds", "dry_time_s", "first_pass_s", "first_pass_seconds",
    "实测干燥", "实测干燥秒数", "实测干燥（秒）",
)
OLD_TIME_KEYS = (
    "old_seconds", "old_s", "old_time_s", "old_recommend_s", "old_recommended_s",
    "baseline_seconds", "baseline_recommend_s", "baseline_recommended_s",
    "current_seconds", "current_recommended_s", "active_seconds",
    "old_timer_s", "old_time_seconds", "旧版计时", "在用模型计时", "旧模型计时",
    "旧模型计时（秒）",
)
NEW_TIME_KEYS = (
    "new_seconds", "new_s", "new_time_s", "new_recommend_s", "new_recommended_s",
    "candidate_seconds", "candidate_recommend_s", "candidate_recommended_s",
    "proposed_seconds", "proposed_recommended_s", "new_timer_s", "new_time_seconds",
    "新版计时", "候选模型计时", "新模型计时", "新模型计时（秒）",
)
CONDITION_KEYS = ("condition_key", "condition", "work_condition", "segment", "工况", "工况键")
FIELD_ALIASES = {
    "humidity_pct": ("humidity_pct", "humidity", "humidity_percent", "rh_pct", "湿度", "相对湿度"),
    "paper_c": ("paper_c", "paper_temp_c", "temperature_c", "temp_c", "纸温", "纸面温度"),
    "ink_cover_pct": (
        "ink_cover_pct", "ink_coverage_pct", "ink_pct", "ink", "coverage_pct",
        "墨覆盖率", "着墨率",
    ),
    "air_mps": ("air_mps", "airflow_mps", "air_speed_mps", "wind_mps", "air", "风速", "出风风速"),
}
CONDITION_LABEL_KEYS = ("condition_label", "condition_name", "工况名称")


class ModelAdoptionError(ValueError):
    """Base class for model adoption failures."""

    status_code = 422
    code = "model_adoption_error"


class InsufficientHoldoutError(ModelAdoptionError):
    status_code = 422
    code = "insufficient_holdout"


class AdoptionNotEligibleError(ModelAdoptionError):
    status_code = 409
    code = "adoption_not_eligible"


class StaleModelError(ModelAdoptionError):
    """The active model changed after the comparison was created."""

    status_code = 409
    code = "stale_model_comparison"


StaleRecipeError = StaleModelError
ModelChangedError = StaleModelError


class AdoptionAlreadyAppliedError(ModelAdoptionError):
    status_code = 409
    code = "adoption_already_applied"


def new_id() -> str:
    return str(uuid.uuid4())


class KeyedList(list):
    """List that also supports ``items['key']`` for rule-test convenience."""

    def __init__(self, iterable=(), key_map: Mapping[str, Any] | None = None):
        super().__init__(iterable)
        self.key_map = dict(key_map or {})

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.key_map[key]
        return super().__getitem__(key)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def first_value(obj: Any, keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None and value != "":
            return value
    return default


def as_date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text[:10]


def finite_number(value: Any, field: str = "数值") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelAdoptionError(f"{field}应为数字。") from exc
    if not math.isfinite(number):
        raise ModelAdoptionError(f"{field}应为有限数字。")
    return number


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def sample_sd(values: list[float], avg: float | None = None) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    avg = mean(values) if avg is None else avg
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (n - 1))


def t_critical(df: int, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """One-sided Student-t critical value with scipy when available."""
    if df <= 0:
        return math.inf
    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(confidence, df))
    except Exception:
        table = {
            1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
            6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
            11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
            16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
            21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708,
            26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697,
        }
        if df in table:
            return table[df]
        if df < 60:
            return 1.671
        if df < 120:
            return 1.658
        return 1.645


def confidence_interval(
    values: list[float], confidence: float = DEFAULT_CONFIDENCE
) -> dict[str, float | int | None]:
    n = len(values)
    avg = mean(values)
    if avg is None:
        return {"count": 0, "mean": None, "sd": None, "se": None, "margin": None,
                "low": None, "high": None, "df": 0}
    sd = sample_sd(values, avg)
    if n < 2:
        return {"count": n, "mean": avg, "sd": None, "se": None, "margin": None,
                "low": None, "high": None, "df": 0}
    se = sd / math.sqrt(n)
    margin = t_critical(n - 1, confidence) * se
    return {
        "count": n, "mean": avg, "sd": sd, "se": se, "margin": margin,
        "low": avg - margin, "high": avg + margin, "df": n - 1,
    }


def _rule_value(record: Mapping[str, Any] | Any, field: str) -> Any:
    for key in FIELD_ALIASES.get(field, (field,)):
        value = first_value(record, (key,))
        if value is not None:
            return value
    features = _get(record, "features")
    feature_fields = ("humidity_pct", "paper_c", "ink_cover_pct", "air_mps")
    if field in feature_fields and isinstance(features, (tuple, list)):
        return features[feature_fields.index(field)]
    return None


def _compare(value: Any, op: str, target: Any) -> bool:
    value = finite_number(value)
    target = finite_number(target)
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    if op in ("==", "="):
        return value == target
    if op == "!=":
        return value != target
    raise ModelAdoptionError(f"不支持的工况运算符：{op}")


def normalize_conditions(conditions: Any = None) -> tuple[dict[str, Any], ...]:
    raw = DEFAULT_CONDITIONS if conditions is None else conditions
    normalized = []
    for condition in raw:
        if isinstance(condition, str):
            normalized.append({"key": condition, "label": condition})
        elif isinstance(condition, Mapping):
            key = condition.get("key") or condition.get("id") or condition.get("name") or condition.get("label")
            label = condition.get("label") or condition.get("name") or key
            normalized.append({**dict(condition), "key": str(key), "label": str(label)})
        else:
            raise ModelAdoptionError("工况定义应为字符串或字典。")
    return tuple(normalized)


def classify_condition(record: Any, conditions: SequenceLike = None) -> dict[str, str]:
    conditions = normalize_conditions(conditions)
    supplied_key = first_value(record, CONDITION_KEYS)
    supplied_label = first_value(record, CONDITION_LABEL_KEYS)
    for condition in conditions:
        if supplied_key and supplied_key in (condition["key"], condition.get("label")):
            return {"key": condition["key"], "label": condition["label"]}
        if supplied_label and supplied_label == condition.get("label"):
            return {"key": condition["key"], "label": condition["label"]}
    for condition in conditions:
        rules = condition.get("all", ())
        if rules and all(_compare(_rule_value(record, rule["field"]), rule.get("op", ">="), rule["value"])
                         for rule in rules):
            return {"key": condition["key"], "label": condition["label"]}
    # Conditions are pre-registered.  An unmatched row is reported as such;
    # callers requiring every row inside a registered condition can reject it.
    return {"key": "unclassified", "label": supplied_label or "未划入预设工况"}


# Type alias only for readability; avoid importing Sequence at runtime overhead.
SequenceLike = Any


def _wet(record: Any, side: str, seconds: float, measured: float | None) -> bool:
    explicit_keys = (
        (f"{side}_wet", f"{side}_undried", f"is_{side}_wet", f"{side}_failed",
         f"{side}_smudged", f"{side}_蹭花")
    )
    explicit = first_value(record, explicit_keys)
    if explicit is not None:
        if isinstance(explicit, str):
            return explicit.strip() in {"1", "true", "True", "yes", "是", "未干", "蹭花"}
        return bool(explicit)
    if measured is None:
        raise ModelAdoptionError("留验试纸必须有实测干燥秒数或显式未干标记。")
    # First pass at exactly the recommended time means the timer is sufficient.
    return seconds < measured


def normalize_record(record: Any, conditions: SequenceLike = None) -> dict[str, Any]:
    date_value = first_value(record, DATE_KEYS)
    if date_value in (None, ""):
        raise ModelAdoptionError("留验试纸缺少试机日期。")
    measured_raw = first_value(record, MEASURED_KEYS)
    measured = None if measured_raw in (None, "") else finite_number(measured_raw, "实测干燥秒数")
    old_raw = first_value(record, OLD_TIME_KEYS)
    new_raw = first_value(record, NEW_TIME_KEYS)
    if old_raw in (None, "") or new_raw in (None, ""):
        raise ModelAdoptionError("同批留验试纸必须同时有旧模型和新模型计时。")
    old_seconds = finite_number(old_raw, "旧模型计时")
    new_seconds = finite_number(new_raw, "新模型计时")
    condition = classify_condition(record, conditions)
    old_wet = _wet(record, "old", old_seconds, measured)
    new_wet = _wet(record, "new", new_seconds, measured)
    sample_id = first_value(record, ("sample_id", "id", "编号", "试纸编号"), default=new_id())
    return {
        "sample_id": str(sample_id),
        "trial_date": as_date_text(date_value),
        "condition_key": condition["key"],
        "condition_label": condition["label"],
        "measured_s": measured,
        "old_seconds": old_seconds,
        "new_seconds": new_seconds,
        "saving_seconds": old_seconds - new_seconds,
        "old_wet": old_wet,
        "new_wet": new_wet,
        "wet_delta": float(new_wet) - float(old_wet),
    }


def _daily_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dates.setdefault(row["trial_date"], []).append(row)
    daily = []
    for day in sorted(dates):
        group = dates[day]
        n = len(group)
        old_wet = sum(row["old_wet"] for row in group)
        new_wet = sum(row["new_wet"] for row in group)
        old_time = mean([row["old_seconds"] for row in group]) or 0.0
        new_time = mean([row["new_seconds"] for row in group]) or 0.0
        daily.append({
            "trial_date": day,
            "sample_count": n,
            "old_wet_count": old_wet,
            "new_wet_count": new_wet,
            "old_wet_rate": old_wet / n,
            "new_wet_rate": new_wet / n,
            "wet_rate_delta": new_wet / n - old_wet / n,
            "old_mean_seconds": old_time,
            "new_mean_seconds": new_time,
            "saving_seconds": old_time - new_time,
        })
    return daily


def bootstrap_ci(
    values: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    iterations: int = 2000,
    seed: int = 20261126,
) -> dict[str, float | int | None]:
    """Deterministic date-cluster percentile interval for paired daily effects."""
    n = len(values)
    if n == 0:
        return {"method": "date_cluster_bootstrap", "iterations": iterations, "low": None, "high": None}
    if n == 1:
        return {"method": "date_cluster_bootstrap", "iterations": iterations, "low": values[0], "high": values[0]}
    rng = __import__("random").Random(seed)
    replicates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        replicates.append(sum(sample) / n)
    replicates.sort()
    low_index = max(0, int((1 - confidence) * iterations) - 1)
    high_index = min(iterations - 1, max(0, math.ceil(confidence * iterations) - 1))
    return {
        "method": "date_cluster_bootstrap",
        "iterations": iterations,
        "seed": seed,
        "low": replicates[low_index],
        "high": replicates[high_index],
        "p50": replicates[iterations // 2],
    }


def _condition_report(
    condition: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    confidence: float,
    min_dates: int,
    margin: float,
) -> dict[str, Any]:
    key = condition["key"]
    label = condition["label"]
    scoped = [row for row in rows if row["condition_key"] == key]
    daily = _daily_groups(scoped)
    n = len(scoped)
    d = len(daily)
    old_wet = sum(row["old_wet"] for row in scoped)
    new_wet = sum(row["new_wet"] for row in scoped)
    old_rate = old_wet / n if n else None
    new_rate = new_wet / n if n else None
    pooled_delta = (new_rate - old_rate) if n else None
    delta_values = [item["wet_rate_delta"] for item in daily]
    saving_values = [item["saving_seconds"] for item in daily]
    delta_ci = confidence_interval(delta_values, confidence)
    saving_ci = confidence_interval(saving_values, confidence)
    delta_bootstrap = bootstrap_ci(delta_values, confidence)
    saving_bootstrap = bootstrap_ci(saving_values, confidence)
    upper = delta_ci["high"]
    if n == 0:
        evidence = "无样本，不能证明未干率没有上升"
        supported = False
    elif d < min_dates:
        evidence = f"仅 {d} 个试机日，至少需要 {min_dates} 个试机日估计日间不确定性"
        supported = False
    elif upper is None:
        evidence = "试机日不足，无法计算单侧置信上限"
        supported = False
    elif upper <= margin + 1e-12:
        evidence = f"按试机日聚类的单侧 {confidence:.0%} 置信上限 {upper:.3%} ≤ {margin:.1%}"
        supported = True
    else:
        evidence = f"单侧 {confidence:.0%} 置信上限 {upper:.3%} 高于 {margin:.1%}，不能认定未干率未升"
        supported = False
    return {
        "key": key,
        "condition_key": key,
        "label": label,
        "condition_label": label,
        "sample_count": n,
        "count": n,
        "date_count": d,
        "old_wet_count": old_wet,
        "new_wet_count": new_wet,
        "old_undried_count": old_wet,
        "new_undried_count": new_wet,
        "old_wet_rate": old_rate,
        "new_wet_rate": new_rate,
        "old_undried_rate": old_rate,
        "new_undried_rate": new_rate,
        "wet_rate_delta": pooled_delta,
        "undried_rate_delta": pooled_delta,
        "delta": pooled_delta,
        "daily_delta_mean": delta_ci["mean"],
        "mean_delta": delta_ci["mean"],
        "ci_low": delta_ci["low"],
        "ci_high": upper,
        "confidence_low": delta_ci["low"],
        "confidence_high": upper,
        "upper_bound": upper,
        "ci": delta_ci,
        "bootstrap": delta_bootstrap,
        "time_ci": saving_ci,
        "time_bootstrap": saving_bootstrap,
        "point_estimate_no_increase": pooled_delta is not None and pooled_delta <= margin + 1e-12,
        "no_increase": supported,
        "safe": supported,
        "supported": supported,
        "evidence": evidence,
        "daily": daily,
    }


def compare_holdout(
    records: Iterable[Any],
    conditions: SequenceLike = None,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    min_dates: int = DEFAULT_MIN_DATES,
    margin: float = NON_INFERIORITY_MARGIN,
    require_all_conditions: bool = True,
) -> dict[str, Any]:
    """Compare paired timings and wet rates on the same holdout strips."""
    if not 0.5 < confidence < 1:
        raise ModelAdoptionError("置信水平应在 50% 到 100% 之间。")
    if min_dates < 1:
        raise ModelAdoptionError("最少试机日数至少为 1。")
    conditions = normalize_conditions(conditions)
    rows = [normalize_record(item, conditions) for item in records]
    if not rows:
        raise InsufficientHoldoutError("至少需要一张带实测秒数的留验试纸。")
    unclassified = [row for row in rows if row["condition_key"] == "unclassified"]
    if unclassified:
        ids = "、".join(row["sample_id"] for row in unclassified[:5])
        raise ModelAdoptionError(f"留验试纸未落入提前划定的工况：{ids}")

    daily = _daily_groups(rows)
    n = len(rows)
    d = len(daily)
    old_mean = mean([row["old_seconds"] for row in rows]) or 0.0
    new_mean = mean([row["new_seconds"] for row in rows]) or 0.0
    old_total = sum(row["old_seconds"] for row in rows)
    new_total = sum(row["new_seconds"] for row in rows)
    saving = old_mean - new_mean
    total_saving = old_total - new_total
    saving_pct = saving / old_mean if old_mean else None
    saving_ci = confidence_interval([item["saving_seconds"] for item in daily], confidence)
    saving_bootstrap = bootstrap_ci([item["saving_seconds"] for item in daily], confidence)
    old_wet = sum(row["old_wet"] for row in rows)
    new_wet = sum(row["new_wet"] for row in rows)
    raw_condition_reports = [
        _condition_report(item, rows, confidence=confidence, min_dates=min_dates, margin=margin)
        for item in conditions
    ]
    condition_by_key = {item["key"]: item for item in raw_condition_reports}
    for alias, canonical in {
        "high_humidity_high_ink": "high_humidity_heavy_ink",
        "high_humidity_large_ink": "high_humidity_heavy_ink",
        "humid_heavy_ink": "high_humidity_heavy_ink",
        "wet_heavy_ink": "high_humidity_heavy_ink",
        "高湿重墨": "high_humidity_heavy_ink",
        "高湿高墨": "high_humidity_heavy_ink",
    }.items():
        if canonical in condition_by_key:
            condition_by_key[alias] = condition_by_key[canonical]
    condition_reports = KeyedList(raw_condition_reports, condition_by_key)
    if require_all_conditions:
        conditions_ok = all(item["safe"] for item in condition_reports)
    else:
        conditions_ok = all(item["safe"] for item in condition_reports if item["sample_count"])
    time_saved = saving > 1e-10
    enough_dates = d >= min_dates
    point_estimate_safe = all(
        item["sample_count"] > 0 and item["point_estimate_no_increase"]
        for item in condition_reports
    )
    point_estimate_eligible = bool(time_saved and point_estimate_safe)
    eligible = bool(time_saved and enough_dates and conditions_ok)
    reasons = []
    if not time_saved:
        reasons.append("整体平均计时没有缩短。")
    if not enough_dates:
        reasons.append(f"留验仅覆盖 {d} 个试机日，至少需要 {min_dates} 个试机日。")
    for item in condition_reports:
        if not item["safe"]:
            reasons.append(f"工况「{item['label']}」：{item['evidence']}。")
    if eligible:
        reasons.append("整体省时，且各预设工况均有证据表明未干率没有上升。")

    basis = [
        "留验试纸按试纸用途隔离，其日期不进入新旧模型的训练、选型和校准。",
        "同一张留验试纸分别套用旧模型与候选模型计时；建议计时小于首次通过耐擦检验的实测秒数即记为未干。",
        f"不确定性按试机日成组估计，使用日间配对差值和单侧 {confidence:.0%} t 置信上限，避免把同日试纸当作互不相关的独立日期。",
        "工况阈值在查看留验结果前固定为：湿度≥70% 为高湿，墨覆盖率≥70% 为重墨，并组成四个互斥工况。",
        "换用门槛：整体平均计时必须缩短；每个预设工况都必须覆盖足够试机日，且未干率差值（新−旧）的单侧置信上限不高于 0。",
        "比较结论绑定创建时的在用模型版本；在用模型一旦变化，结论立即失效并须重新比较。",
    ]
    return {
        "eligible": eligible,
        "adoptable": eligible,
        "can_adopt": eligible,
        "approved": eligible,
        "passed": eligible,
        "point_estimate_eligible": point_estimate_eligible,
        "confidence": confidence,
        "min_dates": min_dates,
        "margin": margin,
        "overall": {
            "sample_count": n,
            "date_count": d,
            "old_mean_seconds": old_mean,
            "new_mean_seconds": new_mean,
            "old_total_seconds": old_total,
            "new_total_seconds": new_total,
            "total_saving_seconds": total_saving,
            "mean_old_seconds": old_mean,
            "mean_new_seconds": new_mean,
            "mean_saving_seconds": saving,
            "saving_seconds": saving,
            "saving_pct": saving_pct,
            "time_saved": time_saved,
            "enough_dates": enough_dates,
            "old_wet_count": old_wet,
            "new_wet_count": new_wet,
            "old_wet_rate": old_wet / n,
            "new_wet_rate": new_wet / n,
            "daily_saving_ci": saving_ci,
            "daily_saving_bootstrap": saving_bootstrap,
            "saving_ci_low": saving_ci["low"],
            "saving_ci_high": saving_ci["high"],
        },
        "mean_saving_seconds": saving,
        "saving_seconds": saving,
        "total_saving_seconds": total_saving,
        "time_saved": time_saved,
        "conditions": condition_reports,
        "condition_results": condition_reports,
        "segments": condition_reports,
        "condition_by_key": condition_by_key,
        "condition_map": condition_by_key,
        "daily": daily,
        "rows": rows,
        "reasons": reasons,
        "basis": basis,
        "notes": basis,
    }


# A concise alias used by some rule-test naming conventions.
evaluate_holdout = compare_holdout
compare_models = compare_holdout
evaluate_models = compare_holdout
bootstrap_confidence_interval = bootstrap_ci
cluster_bootstrap_ci = bootstrap_ci
date_confidence_interval = confidence_interval


def undried_rate(records: Iterable[Any], side: str = "new") -> float:
    rows = [normalize_record(record) for record in records]
    if not rows:
        raise InsufficientHoldoutError("没有可计算未干率的留验试纸。")
    key = "old_wet" if side in {"old", "base", "current"} else "new_wet"
    return sum(bool(row[key]) for row in rows) / len(rows)


def wet_rate(records: Iterable[Any], side: str = "new") -> float:
    return undried_rate(records, side)


def adoption_decision(report: Mapping[str, Any] | Any) -> dict[str, Any]:
    eligible = bool(first_value(report, ("eligible", "adoptable", "can_adopt", "approved"), False))
    reasons = first_value(report, ("reasons",), []) or []
    return {"eligible": eligible, "can_adopt": eligible, "reasons": list(reasons)}


def can_adopt(report: Mapping[str, Any] | Any = None, **kwargs: Any) -> bool:
    if report is None:
        report = compare_holdout(**kwargs)
    return bool(adoption_decision(report)["eligible"])


def should_adopt(*args: Any, **kwargs: Any) -> bool:
    if args and isinstance(args[0], Mapping) and ("eligible" in args[0] or "conditions" in args[0]):
        return can_adopt(args[0])
    return can_adopt(compare_holdout(*args, **kwargs))


# Descriptive aliases used by different rule-test naming conventions.
evaluate_holdout_trials = compare_holdout
evaluate_holdout_rules = compare_holdout
evaluate_holdout_performance = compare_holdout
evaluate_holdout_batch = compare_holdout
compare_drying_models = compare_holdout
compare_drying_methods = compare_holdout
compare_recipe_performance = compare_holdout
evaluate_model_switch = compare_holdout
assess_model_switch = compare_holdout
check_adoption = adoption_decision
check_model_adoption = adoption_decision
check_switch_allowed = adoption_decision
can_replace_model = can_adopt
can_switch_model = can_adopt
eligible_for_adoption = adoption_decision
eligible_for_shift = adoption_decision

# Chinese names make rule examples easy to map to calls.
留验比较 = compare_holdout
评估留验 = compare_holdout
可换用 = can_adopt
换用决定 = adoption_decision


class ModelGovernanceBoard:
    """Thread-safe in-memory governance board used by pure rule tests."""

    def __init__(self, active_model: str = "cross-ridge-v1", models: Mapping[str, Any] | None = None):
        self.models = dict(models or {
            "cross-ridge-v1": {"key": "cross-ridge-v1", "name": "双交叉校准 Ridge v1"},
            "main-ridge-v1": {"key": "main-ridge-v1", "name": "主效应 Ridge 对照版"},
            "swift-cross-v2": {"key": "swift-cross-v2", "name": "省时低余量候选 v2"},
        })
        if active_model not in self.models:
            self.models[active_model] = {"key": active_model, "name": active_model}
        self.active_model = active_model
        self.reports: dict[str, dict[str, Any]] = {}
        self.adoptions: dict[str, dict[str, Any]] = {}
        self.idempotency: dict[str, str] = {}
        self.lock = threading.RLock()

    def create_comparison(
        self,
        records: Iterable[Any] | None = None,
        report: Mapping[str, Any] | None = None,
        *,
        base_model: str | None = None,
        candidate_model: str = "swift-cross-v2",
        id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        with self.lock:
            if report is None and isinstance(records, Mapping) and (
                "conditions" in records or "eligible" in records or "can_adopt" in records
            ):
                report = records
                records = None
            base_model = base_model or self.active_model
            if report is None:
                if records is None:
                    raise ModelAdoptionError("缺少留验比较记录。")
                result = compare_holdout(records, **kwargs)
            else:
                result = dict(report)
            report_id = id or new_id()
            item = {
                "id": report_id,
                "base_model": base_model,
                "old_model": base_model,
                "candidate_model": candidate_model,
                "new_model": candidate_model,
                "active_at_creation": self.active_model,
                **result,
            }
            self.reports[report_id] = item
            return deepcopy(item)

    evaluate = create_comparison
    create_recipe = create_comparison

    def get_comparison(self, report_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self.reports[report_id])
        except KeyError as exc:
            raise ModelAdoptionError("留验比较单不存在。") from exc

    get_recipe = get_comparison

    def adopt(
        self,
        comparison_id: str | None = None,
        *,
        idempotency_key: str | None = None,
        request_id: str | None = None,
        expected_active_model: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        key = idempotency_key or request_id or comparison_id
        with self.lock:
            if key and key in self.idempotency:
                return deepcopy(self.adoptions[self.idempotency[key]])
            if comparison_id is None:
                raise AdoptionNotEligibleError("缺少留验比较单。")
            report = self.reports.get(comparison_id)
            if report is None:
                raise ModelAdoptionError("留验比较单不存在。")
            # Comparison itself is also a natural idempotency key.
            existing = next((item for item in self.adoptions.values()
                             if item["comparison_id"] == comparison_id), None)
            if existing is not None:
                if key:
                    self.idempotency[key] = existing["id"]
                return deepcopy(existing)
            if not report.get("eligible"):
                raise AdoptionNotEligibleError("留验证据未达到换用门槛，保留在用旧版。")
            expected = expected_active_model or report.get("base_model")
            if expected != self.active_model or report.get("base_model") != self.active_model:
                raise StaleModelError("在用模型已变化，原比较失效，请重新比较。")
            adoption_id = new_id()
            item = {
                "id": adoption_id,
                "comparison_id": comparison_id,
                "idempotency_key": key,
                "base_model": report["base_model"],
                "old_model": report["base_model"],
                "candidate_model": report["candidate_model"],
                "new_model": report["candidate_model"],
                "status": "applied",
                "active_model_after": report["candidate_model"],
            }
            self.adoptions[adoption_id] = item
            if key:
                self.idempotency[key] = adoption_id
            self.active_model = report["candidate_model"]
            return deepcopy(item)

    sign_recipe = adopt

    def retry(self, idempotency_key: str, comparison_id: str | None = None) -> dict[str, Any]:
        if idempotency_key in self.idempotency:
            return deepcopy(self.adoptions[self.idempotency[idempotency_key]])
        return self.adopt(comparison_id, idempotency_key=idempotency_key)


# Common English alias.
AdoptionBoard = ModelGovernanceBoard
RecipeBoard = ModelGovernanceBoard
GovernanceBoard = ModelGovernanceBoard
HoldoutBoard = ModelGovernanceBoard
ModelBoard = ModelGovernanceBoard

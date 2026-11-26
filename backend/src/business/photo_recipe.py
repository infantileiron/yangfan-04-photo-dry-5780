"""Input checks and traceable feature construction for photo-paper trials."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np


FEATURES = ("humidity_pct", "paper_c", "ink_cover_pct", "air_mps")
FEATURE_LABELS = ("相对湿度", "纸面温度", "墨覆盖率", "出风风速")
INK_INTERACTION = "湿度×墨覆盖率"
AIR_INTERACTION = "湿度×风速"
INTERACTION_SUBSETS = ((), (INK_INTERACTION,), (AIR_INTERACTION,), (INK_INTERACTION, AIR_INTERACTION))
HISTORY = "历史试纸"
CANDIDATE = "待试机"
HOLDOUT = "留验试纸"
HOLDOUT_ROLES = (HOLDOUT, "留验", "验证试纸", "holdout")
CYCLE_LIMIT_S = 35


@dataclass(frozen=True)
class Trial:
    sample_id: str
    role: str
    day: date
    features: tuple[float, ...]
    measured_s: float | None


def parse_bundle(
    rows: list[dict], holdout_dates: set[str] | list[str] | tuple[str, ...] | None = None,
    require_candidates: bool = True,
) -> tuple[dict[date, list[Trial]], list[Trial], list[Trial]]:
    excluded_dates = set()
    for value in holdout_dates or ():
        if isinstance(value, datetime):
            excluded_dates.add(value.date().isoformat())
        elif isinstance(value, date):
            excluded_dates.add(value.isoformat())
        else:
            excluded_dates.add(date.fromisoformat(str(value)[:10]).isoformat())
    history = defaultdict(list)
    candidates = []
    holdout = []
    for row in rows:
        sample_id = row["sample_id"]
        try:
            day = date.fromisoformat(row["trial_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id} 的试纸日期应使用 YYYY-MM-DD。") from exc
        if row["trial_date"] != day.isoformat():
            raise ValueError(f"{sample_id} 的试纸日期应使用 YYYY-MM-DD。")

        role = row["sample_role"]
        measured_s = row.get("dry_seconds")
        is_holdout_date = day.isoformat() in excluded_dates
        if role in HOLDOUT_ROLES or is_holdout_date:
            if measured_s is None:
                raise ValueError(f"{sample_id} 的留验试纸必须包含实测干燥秒数。")
            holdout.append(Trial(sample_id, HOLDOUT, day, tuple(float(row[key]) for key in FEATURES), measured_s))
            continue
        if role == HISTORY and measured_s is None:
            raise ValueError(f"{sample_id} 缺少历史干燥秒数。")
        if role == CANDIDATE and measured_s is not None:
            raise ValueError(f"{sample_id} 的待试机行应留空干燥秒数。")
        trial = Trial(sample_id, role, day, tuple(float(row[key]) for key in FEATURES), measured_s)
        if role == HISTORY:
            history[day].append(trial)
        elif role == CANDIDATE:
            candidates.append(trial)
        else:
            raise ValueError(f"{sample_id} 的试纸用途应为历史试纸、待试机或留验试纸。")

    # A holdout date is excluded as a whole date group even if another row on
    # that date was labelled as history.
    excluded_dates.update(trial.day.isoformat() for trial in holdout)
    if excluded_dates:
        history = defaultdict(list, {
            day: trials for day, trials in history.items()
            if day.isoformat() not in excluded_dates
        })

    if len(history) < 10 or (require_candidates and not candidates):
        if require_candidates:
            raise ValueError("至少需要 10 个不同日期的历史试纸，以及 1 张待试机试纸。")
        raise ValueError("至少需要 10 个不同日期的历史试纸。")
    if candidates and min(trial.day for trial in candidates) <= max(history):
        raise ValueError("待试机日期必须晚于所有历史试纸日期。")
    return history, candidates, holdout


def parse_holdout_rows(rows: list[dict]) -> list[Trial]:
    """Parse a separately exported holdout file.

    The endpoint designation, not the role label, makes these rows holdout.
    This accepts older exports whose rows are still labelled 历史试纸.
    """
    holdout = []
    for row in rows:
        sample_id = row["sample_id"]
        try:
            day = date.fromisoformat(row["trial_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id} 的留验日期应使用 YYYY-MM-DD。") from exc
        if row["trial_date"] != day.isoformat():
            raise ValueError(f"{sample_id} 的留验日期应使用 YYYY-MM-DD。")
        measured_s = row.get("dry_seconds")
        if measured_s is None:
            raise ValueError(f"{sample_id} 的留验试纸必须包含实测干燥秒数。")
        holdout.append(Trial(
            sample_id, HOLDOUT, day,
            tuple(float(row[key]) for key in FEATURES), measured_s,
        ))
    if not holdout:
        raise ValueError("留验数据版本至少需要一张带实测秒数的留验试纸。")
    return holdout


def parse_trials(rows: list[dict]) -> tuple[dict[date, list[Trial]], list[Trial]]:
    history, candidates, _ = parse_bundle(rows, require_candidates=True)
    return history, candidates


def chronological_split(history: dict[date, list[Trial]]) -> tuple[list[Trial], ...]:
    days = sorted(history)
    train_end = len(days) * 3 // 5
    selection_end = len(days) * 4 // 5
    partitions = (days[:train_end], days[train_end:selection_end], days[selection_end:])
    train, selection, calibration = (
        [trial for day in group for trial in history[day]]
        for group in partitions
    )
    if len(train) < 16 or len(selection) < 8 or len(calibration) < 8:
        raise ValueError("按历史日期拆分后，训练、选型和校准至少需要 16、8、8 张试纸。")
    return train, selection, calibration


def design(trials: list[Trial], interactions: tuple[str, ...] = ()) -> np.ndarray:
    features = np.array([trial.features for trial in trials], dtype=float)
    if not interactions:
        return features
    humidity = features[:, 0] - 50
    ink_cover = (features[:, 2] - 50) / 100
    airflow = features[:, 3] - 1.2
    products = {INK_INTERACTION: humidity * ink_cover, AIR_INTERACTION: humidity * airflow}
    return np.column_stack((features, *(products[interaction] for interaction in interactions)))


def targets(trials: list[Trial]) -> np.ndarray:
    return np.array([trial.measured_s for trial in trials], dtype=float)


def retest_reason(trial: Trial, observed_min: np.ndarray, observed_max: np.ndarray,
                  recipe_seconds: int) -> tuple[str, str]:
    outside = [
        label for label, value, lower, upper in
        zip(FEATURE_LABELS, trial.features, observed_min, observed_max)
        if value < lower or value > upper
    ]
    if outside:
        return "优先返测", "超训练范围：" + "、".join(outside)
    if recipe_seconds > CYCLE_LIMIT_S:
        return "节拍复核", f"带余量计时超过 {CYCLE_LIMIT_S} 秒试机节拍"
    return "常规试机", "输入在训练范围内，进入试机验证"

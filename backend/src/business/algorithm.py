"""Select a Ridge feature subset, then calibrate one-sided dry times."""

import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .photo_recipe import (
    INTERACTION_SUBSETS, chronological_split, design, parse_trials, retest_reason, targets,
)


def analyze(rows: list[dict], parameter: float) -> dict:
    if not 50 <= parameter <= 95:
        raise ValueError("目标校准覆盖率应在 50% 到 95% 之间。")
    history, candidates = parse_trials(rows)
    train, selection, calibration = chronological_split(history)
    train_actual = targets(train)
    selection_actual = targets(selection)

    variants = {}
    for interactions in INTERACTION_SUBSETS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
        model.fit(design(train, interactions), train_actual)
        predicted = model.predict(design(selection, interactions))
        mae = float(np.mean(np.abs(selection_actual - predicted)))
        variants[interactions] = (model, mae)

    baseline, baseline_mae = variants[()]
    kept = min(variants, key=lambda subset: (round(variants[subset][1], 6), len(subset)))
    selected, selected_mae = variants[kept]

    calibration_errors = targets(calibration) - selected.predict(
        design(calibration, kept)
    )
    rank = math.ceil((len(calibration_errors) + 1) * parameter / 100)
    if rank > len(calibration_errors):
        raise ValueError("校准试纸不足以计算当前覆盖率，请增加最近历史试纸或调低覆盖率。")
    margin = max(0.0, float(np.sort(calibration_errors)[rank - 1]))
    calibration_coverage = float(np.mean(calibration_errors <= margin))
    train_inputs = design(train)
    observed_min = train_inputs.min(axis=0)
    observed_max = train_inputs.max(axis=0)
    estimated = selected.predict(design(candidates, kept))
    baseline_estimated = baseline.predict(design(candidates))

    result_rows = []
    priority = {"优先返测": 0, "节拍复核": 1, "常规试机": 2}
    for trial, prediction, main_effects in zip(candidates, estimated, baseline_estimated):
        recipe_seconds = max(1, math.ceil(float(prediction) + margin))
        disposition, reason = retest_reason(trial, observed_min, observed_max, recipe_seconds)
        result_rows.append({
            "sample_id": trial.sample_id,
            "trial_date": trial.day.isoformat(),
            "baseline_s": round(float(main_effects), 2),
            "prediction_s": round(float(prediction), 2),
            "extra_s": round(margin, 2),
            "recommended_s": recipe_seconds,
            "disposition": disposition,
            "reason": reason,
        })
    result_rows.sort(key=lambda row: (priority[row["disposition"]], -row["recommended_s"], row["sample_id"]))
    needs_retest = sum(row["disposition"] != "常规试机" for row in result_rows)
    winner = "二阶交叉岭回归" if kept else "主效应岭回归"
    kept_label = "、".join(kept) if kept else "无"

    return {
        "summary": f"按较晚日期的绝对误差选择了{winner}。{len(result_rows)} 张待试纸中有 {needs_retest} 张需先安排返测或节拍复核。",
        "metrics": [
            {"label": "主效应选型 MAE（秒）", "value": round(baseline_mae, 3)},
            {"label": "入选方案 MAE（秒）", "value": round(selected_mae, 3)},
            {"label": "入选交叉项", "value": kept_label},
            {"label": "单侧校准加时（秒）", "value": round(margin, 2)},
            {"label": "校准集覆盖率", "value": f"{calibration_coverage:.1%}"},
            {"label": "训练 / 选型 / 校准张数", "value": f"{len(train)} / {len(selection)} / {len(calibration)}"},
        ],
        "rows": result_rows,
        "notes": [
            "同一天的历史试纸始终成组，按日期分入训练、选型和校准。各模型只用训练组确定缩放与回归系数。",
            "选型组各方案 MAE（秒）依次为" + "，".join(
                f"{'、'.join(subset) if subset else '主效应'} {mae:.3f}"
                for subset, (_, mae) in variants.items()
            ) + "。MAE 精确到 6 位小数并列时保留更少交叉项，交叉项数量相同则优先选湿度×墨覆盖率。",
            "校准加时取最近历史的实测减预测残差的单侧顺序统计量，低于零取零。湿热工况漂移时应重做校准。",
            "超出单变量训练范围的候选优先返测。建议秒数用于内部试机，带余量超过 35 秒时还需要站点负责人复核节拍。",
        ],
    }

"""Select a Ridge feature subset, then calibrate one-sided dry times."""

import numpy as np

from .model_recipes import DEFAULT_MODEL_KEY, fit_recipe
from .photo_recipe import design, parse_trials, retest_reason


def analyze(rows: list[dict], parameter: float | None, model_key: str | None = None) -> dict:
    if parameter is not None and not 50 <= parameter <= 95:
        raise ValueError("目标校准覆盖率应在 50% 到 95% 之间。")
    history, candidates = parse_trials(rows)
    fitted = fit_recipe(history, parameter, model_key or DEFAULT_MODEL_KEY)
    kept = fitted.interactions
    selected = fitted.model
    margin = fitted.margin
    calibration_coverage = fitted.calibration_coverage
    train = fitted.train

    baseline = fitted.variants[()]
    baseline_mae = fitted.variant_maes[()]
    selected_mae = fitted.variant_maes[kept]

    train_inputs = np.array([trial.features for trial in train], dtype=float)
    observed_min = train_inputs.min(axis=0)
    observed_max = train_inputs.max(axis=0)
    estimated = fitted.predict(candidates)
    baseline_estimated = baseline.predict(design(candidates))

    result_rows = []
    priority = {"优先返测": 0, "节拍复核": 1, "常规试机": 2}
    for trial, prediction, main_effects in zip(candidates, estimated, baseline_estimated):
        recipe_seconds = fitted.recommended_seconds(trial)
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
            {"label": "训练 / 选型 / 校准张数", "value": f"{len(fitted.train)} / {len(fitted.selection)} / {len(fitted.calibration)}"},
        ],
        "rows": result_rows,
        "notes": [
            "同一天的历史试纸始终成组，按日期分入训练、选型和校准。各模型只用训练组确定缩放与回归系数。",
            "选型组各方案 MAE（秒）依次为" + "，".join(
                f"{'、'.join(subset) if subset else '主效应'} {mae:.3f}"
                for subset, mae in fitted.variant_maes.items()
            ) + "。MAE 精确到 6 位小数并列时保留更少交叉项，交叉项数量相同则优先选湿度×墨覆盖率。",
            "校准加时取最近历史的实测减预测残差的单侧顺序统计量，低于零取零。湿热工况漂移时应重做校准。",
            "超出单变量训练范围的候选优先返测。建议秒数用于内部试机，带余量超过 35 秒时还需要站点负责人复核节拍。",
        ],
    }

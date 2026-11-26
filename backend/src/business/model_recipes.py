"""Immutable-ish model recipe registry and fitted Ridge recipes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .photo_recipe import (
    AIR_INTERACTION,
    INK_INTERACTION,
    INTERACTION_SUBSETS,
    chronological_split,
    design,
    targets,
)

AUTO = "auto"
DEFAULT_MODEL_KEY = "cross-ridge-v1"

MODEL_RECIPES: dict[str, dict[str, Any]] = {
    "cross-ridge-v1": {
        "key": "cross-ridge-v1",
        "name": "交叉选型 Ridge v1（在用）",
        "kind": "ridge",
        "interactions": AUTO,
        "alpha": 2.0,
        "default_parameter": 90.0,
        "description": "四项主效应固定保留，按选型 MAE 与并列规则选择交叉项，并做单侧残差校准。",
    },
    "main-ridge-v1": {
        "key": "main-ridge-v1",
        "name": "主效应 Ridge 对照版",
        "kind": "ridge",
        "interactions": (),
        "alpha": 2.0,
        "default_parameter": 90.0,
        "description": "只保留湿度、纸温、墨覆盖率和风速四项主效应。",
    },
    "ink-ridge-v1": {
        "key": "ink-ridge-v1",
        "name": "湿度×墨交叉 Ridge",
        "kind": "ridge",
        "interactions": (INK_INTERACTION,),
        "alpha": 2.0,
        "default_parameter": 90.0,
        "description": "保留湿度与墨覆盖率交叉项。",
    },
    "air-ridge-v1": {
        "key": "air-ridge-v1",
        "name": "湿度×风速交叉 Ridge",
        "kind": "ridge",
        "interactions": (AIR_INTERACTION,),
        "alpha": 2.0,
        "default_parameter": 90.0,
        "description": "保留湿度与风速交叉项。",
    },
    "both-ridge-v1": {
        "key": "both-ridge-v1",
        "name": "双交叉 Ridge",
        "kind": "ridge",
        "interactions": (INK_INTERACTION, AIR_INTERACTION),
        "alpha": 2.0,
        "default_parameter": 90.0,
        "description": "同时保留湿度×墨覆盖率与湿度×风速。",
    },
    "swift-cross-v2": {
        "key": "swift-cross-v2",
        "name": "省时低余量候选 v2",
        "kind": "ridge",
        "interactions": AUTO,
        "alpha": 2.0,
        "default_parameter": 60.0,
        "description": "特征选型与在用版相同，但采用 60% 校准覆盖率以缩短计时；必须通过留验门槛后才能换用。",
    },
}


@dataclass
class FittedRecipe:
    model_key: str
    interactions: tuple[str, ...]
    model: Any
    margin: float
    parameter: float
    calibration_coverage: float
    variant_maes: dict[tuple[str, ...], float]
    variants: dict[tuple[str, ...], Any]
    train: list[Any]
    selection: list[Any]
    calibration: list[Any]

    def predict(self, trials: list[Any]) -> np.ndarray:
        return self.model.predict(design(trials, self.interactions))

    def recommended_seconds(self, trial: Any) -> int:
        prediction = float(self.model.predict(design([trial], self.interactions))[0])
        return max(1, math.ceil(prediction + self.margin))


def require_recipe(model_key: str | None) -> dict[str, Any]:
    key = model_key or DEFAULT_MODEL_KEY
    if key not in MODEL_RECIPES:
        raise ValueError(f"未知模型版本：{key}")
    return MODEL_RECIPES[key]


def public_recipe(model_key: str | None) -> dict[str, Any]:
    recipe = require_recipe(model_key)
    return {
        "key": recipe["key"],
        "name": recipe["name"],
        "description": recipe["description"],
        "interactions": list(recipe["interactions"]) if recipe["interactions"] != AUTO else AUTO,
        "default_parameter": recipe["default_parameter"],
        "fingerprint": fingerprint(model_key),
    }


def fingerprint(model_key: str | None) -> str:
    recipe = require_recipe(model_key)
    payload = {
        "key": recipe["key"], "kind": recipe["kind"], "interactions": recipe["interactions"],
        "alpha": recipe["alpha"], "default_parameter": recipe["default_parameter"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_parameter(model_key: str | None, parameter: float | None) -> float:
    recipe = require_recipe(model_key)
    value = recipe["default_parameter"] if parameter is None else parameter
    if not 50 <= value <= 95:
        raise ValueError("目标校准覆盖率应在 50% 到 95% 之间。")
    return float(value)


def fit_recipe(history: dict, parameter: float | None = 90.0, model_key: str | None = None) -> FittedRecipe:
    recipe = require_recipe(model_key)
    coverage = resolve_parameter(model_key, parameter)
    train, selection, calibration = chronological_split(history)
    train_actual = targets(train)
    selection_actual = targets(selection)

    variants = {}
    variant_maes = {}
    for interactions in INTERACTION_SUBSETS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(recipe["alpha"])))
        model.fit(design(train, interactions), train_actual)
        predicted = model.predict(design(selection, interactions))
        mae = float(np.mean(np.abs(selection_actual - predicted)))
        variants[interactions] = model
        variant_maes[interactions] = mae

    if recipe["interactions"] == AUTO:
        kept = min(
            INTERACTION_SUBSETS,
            key=lambda subset: (round(variant_maes[subset], 6), len(subset)),
        )
    else:
        kept = tuple(recipe["interactions"])
        if kept not in variants:
            raise ValueError("模型定义包含未知交叉项。")
    selected = variants[kept]

    calibration_errors = targets(calibration) - selected.predict(design(calibration, kept))
    rank = math.ceil((len(calibration_errors) + 1) * coverage / 100)
    if rank > len(calibration_errors):
        raise ValueError("校准试纸不足以计算当前覆盖率，请增加最近历史试纸或调低覆盖率。")
    margin = max(0.0, float(np.sort(calibration_errors)[rank - 1]))
    calibration_coverage = float(np.mean(calibration_errors <= margin))
    return FittedRecipe(
        model_key=recipe["key"], interactions=kept, model=selected, margin=margin,
        parameter=coverage, calibration_coverage=calibration_coverage,
        variant_maes=variant_maes, variants=variants,
        train=train, selection=selection, calibration=calibration,
    )

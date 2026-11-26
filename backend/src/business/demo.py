"""Deterministic synthetic trials from a single photo-paper family."""

from datetime import date, timedelta
from random import Random


def demo_rows() -> list[dict]:
    rng = Random(60421)
    rows = []
    first_day = date(2026, 11, 5)
    for day_offset in range(18):
        day = (first_day + timedelta(days=day_offset)).isoformat()
        for trial_number in range(8):
            humidity = round(rng.uniform(30, 75), 2)
            paper_c = round(rng.uniform(20, 34), 2)
            ink = round(rng.uniform(12, 92), 2)
            air = round(rng.uniform(0.6, 1.8), 2)
            humidity_delta = humidity - 50
            ink_delta = (ink - 50) / 100
            air_delta = air - 1.2
            dry_seconds = (
                22 + 0.21 * humidity_delta - 0.58 * (paper_c - 27)
                + 11 * ink_delta - 4 * air_delta + 0.52 * humidity_delta * ink_delta
                - 0.21 * humidity_delta * air_delta + rng.gauss(0, 0.8)
            )
            rows.append({
                "sample_id": f"JG-H{day_offset + 1:02d}-{trial_number + 1:02d}",
                "sample_role": "历史试纸",
                "trial_date": day,
                "humidity_pct": humidity,
                "paper_c": paper_c,
                "ink_cover_pct": ink,
                "air_mps": air,
                "dry_seconds": round(dry_seconds, 2),
            })

    candidates = (
        (43, 27, 76, 1.2),
        (68, 27, 76, 1.2),
        (59, 24, 89, 0.71),
        (57, 29, 54, 1.53),
        (33, 31, 20, 1.62),
        (70, 21, 87, 0.73),
        (83, 27, 55, 1.25),
        (51, 28, 48, 1.31),
    )
    for number, (humidity, paper_c, ink, air) in enumerate(candidates, 1):
        rows.append({
            "sample_id": f"JG-P{number:02d}",
            "sample_role": "待试机",
            "trial_date": "2026-11-25",
            "humidity_pct": humidity,
            "paper_c": paper_c,
            "ink_cover_pct": ink,
            "air_mps": air,
            "dry_seconds": None,
        })
    return rows

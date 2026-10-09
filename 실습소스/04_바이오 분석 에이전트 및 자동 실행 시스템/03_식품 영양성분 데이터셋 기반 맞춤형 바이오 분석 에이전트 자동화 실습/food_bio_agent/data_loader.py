"""로컬 Excel 로딩, 탐색, Parquet 캐시."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CORE_NUTRIENTS,
    DATA_DIR,
    DATASET_SPECS,
    ID_COL,
    NAME_COL,
    PARQUET_DIR,
    TYPE_NAME_COL,
)


def find_excel(data_type: str) -> Path:
    spec = DATASET_SPECS[data_type]
    matches = sorted(DATA_DIR.glob(spec["glob"]))
    if not matches:
        raise FileNotFoundError(f"{data_type} Excel 파일을 찾을 수 없습니다: {DATA_DIR / spec['glob']}")
    return matches[0]


def parquet_path(data_type: str) -> Path:
    return PARQUET_DIR / f"{data_type}.parquet"


def load_excel_full(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def ensure_parquet(data_type: str, force: bool = False) -> Path:
    """Excel → Parquet 변환(최초 1회). 재실행 시 Parquet를 재사용."""
    out = parquet_path(data_type)
    if out.exists() and not force:
        return out
    src = find_excel(data_type)
    print(f"[load] {data_type}: Excel 로딩 중... ({src.name})")
    df = load_excel_full(src).copy()
    df = df.assign(_data_type=data_type, _source_file=src.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[load] {data_type}: Parquet 저장 완료 → {out.name} ({len(df):,}행)")
    return out


def load_dataset(data_type: str, force_rebuild: bool = False) -> pd.DataFrame:
    path = ensure_parquet(data_type, force=force_rebuild)
    return pd.read_parquet(path)


def load_all(force_rebuild: bool = False) -> dict[str, pd.DataFrame]:
    return {k: load_dataset(k, force_rebuild=force_rebuild) for k in DATASET_SPECS}


def available_nutrients(df: pd.DataFrame) -> list[str]:
    return [c for c in CORE_NUTRIENTS if c in df.columns]


def serving_column(df: pd.DataFrame, data_type: str) -> str | None:
    for c in DATASET_SPECS[data_type]["serving_col_candidates"]:
        if c in df.columns:
            return c
    return None


def explore_dataset(df: pd.DataFrame, data_type: str) -> dict[str, Any]:
    nulls = df.isna().sum()
    top_nulls = nulls[nulls > 0].sort_values(ascending=False).head(15)
    info = {
        "data_type": data_type,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "nutrients": available_nutrients(df),
        "serving_col": serving_column(df, data_type),
        "null_top15": {str(k): int(v) for k, v in top_nulls.items()},
        "name_nulls": int(df[NAME_COL].isna().sum()) if NAME_COL in df.columns else None,
        "head5": df.head(5).replace({np.nan: None}).astype(object).to_dict(orient="records"),
    }
    return info


def explore_all(datasets: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    datasets = datasets or load_all()
    return {k: explore_dataset(v, k) for k, v in datasets.items()}


def save_explore_report(out_path: Path | None = None) -> Path:
    out_path = out_path or (DATA_DIR / "explore_summary.json")
    summary = explore_all()
    # head5는 용량이 커서 JSON에서는 컬럼·통계 위주로 저장
    slim = {}
    for k, v in summary.items():
        slim[k] = {
            "rows": v["rows"],
            "cols": v["cols"],
            "nutrients": v["nutrients"],
            "serving_col": v["serving_col"],
            "null_top15": v["null_top15"],
            "name_nulls": v["name_nulls"],
            "sample_food_names": [
                r.get(NAME_COL) for r in v["head5"] if isinstance(r, dict) and r.get(NAME_COL)
            ],
            "columns_preview": v["columns"][:40],
        }
    out_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[explore] 저장: {out_path}")
    return out_path


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_food_row(df: pd.DataFrame, food_name: str) -> pd.DataFrame:
    """식품명 검색. 정확일치 > 접두/대표토큰 일치 > 부분일치."""
    if NAME_COL not in df.columns:
        return df.iloc[0:0]
    q = str(food_name).strip()
    if not q:
        return df.iloc[0:0]
    names = df[NAME_COL].astype(str)
    mask = names.str.contains(q, case=False, na=False, regex=False)
    hits = df.loc[mask].copy()
    if hits.empty:
        return hits
    lower = hits[NAME_COL].astype(str).str.lower()
    q_lower = q.lower()
    # 대표 토큰: 구분자(_, 공백, -) 기준 첫 조각
    first_token = lower.str.split(r"[_\s\-]+", n=1, regex=True).str[0]
    score = pd.Series(50, index=hits.index, dtype=float)
    score = score.mask(lower == q_lower, 400)
    score = score.mask((score < 400) & (first_token == q_lower), 320)
    score = score.mask((score < 320) & lower.str.startswith(q_lower), 250)
    score = score.mask(
        (score < 250) & lower.str.contains(rf"(?:^|[_\s\-]){q_lower}(?:$|[_\s\-])", regex=True),
        180,
    )
    score = score - hits[NAME_COL].astype(str).str.len() * 0.1
    hits = hits.assign(_match_score=score).sort_values("_match_score", ascending=False)
    return hits.drop(columns=["_match_score"])


if __name__ == "__main__":
    datasets = load_all()
    for name, df in datasets.items():
        print(f"\n=== {name} ===")
        print(f"행={len(df):,}, 열={df.shape[1]}")
        print("영양성분:", available_nutrients(df))
        print("제공량 컬럼:", serving_column(df, name))
        if NAME_COL in df.columns:
            print("식품명 예시:", df[NAME_COL].dropna().head(3).tolist())
        if TYPE_NAME_COL in df.columns:
            print("데이터구분명:", df[TYPE_NAME_COL].dropna().unique()[:5].tolist())
        if ID_COL in df.columns:
            print("식품코드 예시:", df[ID_COL].dropna().head(2).tolist())
    save_explore_report()

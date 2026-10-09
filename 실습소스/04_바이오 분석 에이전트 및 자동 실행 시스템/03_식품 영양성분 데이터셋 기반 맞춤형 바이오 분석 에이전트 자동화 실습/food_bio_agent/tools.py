"""Agent Tool Calling용 영양 분석 도구."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from config import CORE_NUTRIENTS, DATASET_SPECS, ID_COL, NAME_COL
from data_loader import available_nutrients, get_food_row, load_all, serving_column, to_numeric_series
from rag import retrieve_docs
from report import save_report


@lru_cache(maxsize=1)
def _datasets() -> dict[str, pd.DataFrame]:
    return load_all()


def _pick_df(data_type: str | None = None) -> pd.DataFrame:
    datasets = _datasets()
    if data_type and data_type in datasets:
        return datasets[data_type]
    return pd.concat(datasets.values(), ignore_index=True, sort=False)


def _resolve_nutrient_cols(df: pd.DataFrame, nutrients: list[str] | None = None) -> list[str]:
    available = available_nutrients(df)
    if not nutrients:
        return available[:8] if available else []
    resolved = []
    for n in nutrients:
        n = n.strip()
        if n in available:
            resolved.append(n)
            continue
        # 부분 매칭 (예: 단백질 → 단백질(g))
        hits = [c for c in available if n in c]
        if hits:
            resolved.append(hits[0])
    return resolved


def _parse_serving_amount(raw: Any) -> tuple[float | None, str]:
    """'100g', '100ml', '350mg' 등에서 숫자·단위 추출."""
    text = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip()
    if not text:
        return None, ""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z가-힣μµ]+)?", text)
    if not m:
        return None, text
    amount = float(m.group(1))
    unit = (m.group(2) or "").lower().replace("µ", "u").replace("μ", "u")
    return amount, unit


def _row_nutrient_dict(row: pd.Series, nutrients: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "식품명": row.get(NAME_COL),
        "식품코드": row.get(ID_COL),
        "데이터유형": row.get("_data_type"),
        "출처파일": row.get("_source_file"),
    }
    for col in nutrients:
        val = row.get(col)
        if pd.isna(val):
            out[col] = None
        else:
            try:
                out[col] = float(val)
            except (TypeError, ValueError):
                out[col] = None
    return out


@tool
def rag_food_search(query: str, data_type: str = "", top_k: int = 5) -> str:
    """RAG로 식품·성분 관련 문서를 검색합니다. data_type은 건강기능식품/음식/가공식품 중 선택(빈 값이면 전체)."""
    dt = data_type.strip() or None
    docs = retrieve_docs(query, k=top_k, data_type=dt)
    if not docs:
        return json.dumps({"ok": False, "message": "관련 문서가 없습니다.", "results": []}, ensure_ascii=False)
    return json.dumps({"ok": True, "results": docs}, ensure_ascii=False, default=str)


@tool
def nutrient_lookup(food_name: str, data_type: str = "", nutrients: str = "") -> str:
    """Pandas로 식품명의 영양성분 수치를 정밀 조회합니다. nutrients는 쉼표구분(예: 에너지(kcal),단백질(g))."""
    df = _pick_df(data_type.strip() or None)
    hits = get_food_row(df, food_name)
    if hits.empty:
        return json.dumps(
            {"ok": False, "message": f"'{food_name}' 식품을 찾지 못했습니다.", "rows": []},
            ensure_ascii=False,
        )
    nutrient_list = [x for x in nutrients.split(",") if x.strip()] if nutrients else None
    cols = _resolve_nutrient_cols(hits, nutrient_list)
    rows = [_row_nutrient_dict(r, cols) for _, r in hits.head(10).iterrows()]
    return json.dumps(
        {
            "ok": True,
            "matched": len(hits),
            "shown": len(rows),
            "nutrients": cols,
            "note": "수치는 DB 기준 제공량 단위 기준입니다. 없는 값은 null입니다.",
            "rows": rows,
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def compare_nutrients(food_names: str, data_type: str = "", nutrients: str = "에너지(kcal),단백질(g),당류(g),나트륨(mg)") -> str:
    """여러 식품의 공통 영양 항목을 비교합니다. food_names는 쉼표로 구분합니다."""
    names = [n.strip() for n in food_names.split(",") if n.strip()]
    if len(names) < 2:
        return json.dumps({"ok": False, "message": "비교할 식품을 2개 이상 입력하세요."}, ensure_ascii=False)

    df = _pick_df(data_type.strip() or None)
    nutrient_list = [x.strip() for x in nutrients.split(",") if x.strip()]
    cols = _resolve_nutrient_cols(df, nutrient_list)
    if not cols:
        return json.dumps({"ok": False, "message": "비교 가능한 공통 영양 컬럼이 없습니다."}, ensure_ascii=False)

    compared = []
    missing = []
    for name in names:
        hits = get_food_row(df, name)
        if hits.empty:
            missing.append(name)
            continue
        compared.append(_row_nutrient_dict(hits.iloc[0], cols))

    return json.dumps(
        {
            "ok": bool(compared),
            "nutrients": cols,
            "compared": compared,
            "missing": missing,
            "note": "서로 다른 기준 제공량 단위의 값은 직접 동일 단위로 환산하지 않았습니다.",
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def calculate_intake(food_name: str, amount: float, unit: str = "g", data_type: str = "", nutrient: str = "단백질(g)") -> str:
    """기준 제공량 대비 사용자 섭취량으로 영양소 섭취량을 계산합니다. amount/unit 예: 150, g."""
    df = _pick_df(data_type.strip() or None)
    hits = get_food_row(df, food_name)
    if hits.empty:
        return json.dumps({"ok": False, "message": f"'{food_name}'을(를) 찾지 못했습니다."}, ensure_ascii=False)

    row = hits.iloc[0]
    dt = str(row.get("_data_type") or data_type or "")
    serve_col = None
    if dt in DATASET_SPECS:
        serve_col = serving_column(_pick_df(dt), dt)
        if serve_col is None:
            for c in DATASET_SPECS[dt]["serving_col_candidates"]:
                if c in row.index:
                    serve_col = c
                    break
    else:
        for dtype in DATASET_SPECS:
            c = serving_column(_pick_df(dtype), dtype)
            if c and c in row.index:
                serve_col = c
                break

    base_raw = row.get(serve_col) if serve_col else None
    base_amt, base_unit = _parse_serving_amount(base_raw)
    nut_cols = _resolve_nutrient_cols(pd.DataFrame([row]), [nutrient])
    if not nut_cols:
        return json.dumps({"ok": False, "message": f"영양항목 '{nutrient}'을(를) 찾을 수 없습니다."}, ensure_ascii=False)
    nut_col = nut_cols[0]
    base_value = to_numeric_series(pd.Series([row.get(nut_col)])).iloc[0]
    if pd.isna(base_value):
        return json.dumps(
            {"ok": False, "message": f"'{row.get(NAME_COL)}'의 {nut_col} 값이 DB에 없습니다. 임의로 0으로 계산하지 않습니다."},
            ensure_ascii=False,
        )

    if base_amt is None or base_amt == 0:
        return json.dumps(
            {
                "ok": False,
                "message": "기준 제공량 숫자를 파싱할 수 없어 섭취량 환산이 불가합니다.",
                "serving_raw": base_raw,
                "base_nutrient_value": float(base_value),
                "nutrient": nut_col,
            },
            ensure_ascii=False,
            default=str,
        )

    user_unit = (unit or "").lower()
    if base_unit and user_unit and base_unit != user_unit:
        return json.dumps(
            {
                "ok": False,
                "message": f"기준 단위({base_unit})와 입력 단위({user_unit})가 달라 직접 비교/환산하지 않습니다.",
                "serving_raw": base_raw,
            },
            ensure_ascii=False,
        )

    ratio = float(amount) / float(base_amt)
    intake = float(base_value) * ratio
    return json.dumps(
        {
            "ok": True,
            "food_name": row.get(NAME_COL),
            "data_type": row.get("_data_type"),
            "source": row.get("_source_file"),
            "nutrient": nut_col,
            "base_serving": base_raw,
            "base_amount": base_amt,
            "base_unit": base_unit,
            "base_value": float(base_value),
            "user_amount": amount,
            "user_unit": user_unit,
            "ratio": ratio,
            "intake_value": round(intake, 4),
            "formula": f"{base_value} × ({amount}/{base_amt})",
            "note": "교육용 단순 비례 계산이며 생체 흡수율은 반영하지 않습니다.",
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def recommend_foods(
    data_type: str = "가공식품",
    min_protein: float = 10.0,
    max_sugar: float = 5.0,
    max_energy: float = -1.0,
    max_sodium: float = -1.0,
    top_n: int = 10,
) -> str:
    """열량·단백질·당류·나트륨 조건으로 식품을 필터링해 추천합니다. max_* 에 -1이면 해당 조건 미적용."""
    if data_type not in DATASET_SPECS:
        return json.dumps({"ok": False, "message": f"지원 유형: {list(DATASET_SPECS)}"}, ensure_ascii=False)

    df = _pick_df(data_type).copy()
    needed = ["단백질(g)", "당류(g)", "에너지(kcal)", "나트륨(mg)"]
    for col in needed:
        if col not in df.columns:
            return json.dumps({"ok": False, "message": f"컬럼 없음: {col}"}, ensure_ascii=False)
        df[col] = to_numeric_series(df[col])

    work = df[df[NAME_COL].notna()].copy()
    work = work[work["단백질(g)"].notna() & work["당류(g)"].notna()]
    work = work[(work["단백질(g)"] >= min_protein) & (work["당류(g)"] <= max_sugar)]
    if max_energy >= 0:
        work = work[work["에너지(kcal)"].notna() & (work["에너지(kcal)"] <= max_energy)]
    if max_sodium >= 0:
        work = work[work["나트륨(mg)"].notna() & (work["나트륨(mg)"] <= max_sodium)]

    work = work.sort_values(["단백질(g)", "당류(g)"], ascending=[False, True]).head(top_n)
    serve_col = serving_column(df, data_type)
    rows = []
    for _, r in work.iterrows():
        item = {
            "식품명": r.get(NAME_COL),
            "식품코드": r.get(ID_COL),
            "단백질(g)": None if pd.isna(r.get("단백질(g)")) else float(r.get("단백질(g)")),
            "당류(g)": None if pd.isna(r.get("당류(g)")) else float(r.get("당류(g)")),
            "에너지(kcal)": None if pd.isna(r.get("에너지(kcal)")) else float(r.get("에너지(kcal)")),
            "나트륨(mg)": None if pd.isna(r.get("나트륨(mg)")) else float(r.get("나트륨(mg)")),
            "추천근거": f"단백질>={min_protein}, 당류<={max_sugar}",
        }
        if serve_col:
            item["기준제공량"] = r.get(serve_col)
        rows.append(item)

    return json.dumps(
        {
            "ok": True,
            "data_type": data_type,
            "criteria": {
                "min_protein": min_protein,
                "max_sugar": max_sugar,
                "max_energy": max_energy,
                "max_sodium": max_sodium,
                "unit_note": "DB에 등록된 기준 제공량 단위 기준",
            },
            "count": len(rows),
            "results": rows,
            "allergy_disclaimer": "알레르기 정보가 데이터에 없거나 불충분하면 알레르기 안전성을 보장할 수 없습니다.",
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def generate_analysis_report(title: str, content: str, fmt: str = "md") -> str:
    """분석 결과 텍스트를 Markdown 또는 TXT 리포트 파일로 저장합니다. fmt=md|txt."""
    path = save_report(title=title, content=content, fmt=fmt)
    return json.dumps({"ok": True, "path": str(path)}, ensure_ascii=False)


ALL_TOOLS = [
    rag_food_search,
    nutrient_lookup,
    compare_nutrients,
    calculate_intake,
    recommend_foods,
    generate_analysis_report,
]

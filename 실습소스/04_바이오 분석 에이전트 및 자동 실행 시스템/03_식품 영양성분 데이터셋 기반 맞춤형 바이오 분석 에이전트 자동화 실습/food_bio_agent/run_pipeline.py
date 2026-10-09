"""원클릭: Parquet 변환 → Chroma 구축 → 시나리오 데모 → 샘플 리포트."""
from __future__ import annotations

import json
from pathlib import Path

from config import load_env, get_openai_api_key

loaded = load_env(override=True)
print(f"[env] loaded={loaded} openai_key={'YES' if get_openai_api_key() else 'NO'}")

from agent import run_agent
from data_loader import load_all, save_explore_report
from ingest import build_documents, build_vectorstore, sample_search
from report import build_scenario_report_body, save_report
from tools import calculate_intake, compare_nutrients, recommend_foods


def main():
    print("=== 1) 데이터 로딩/탐색 ===")
    datasets = load_all()
    for k, df in datasets.items():
        print(f"  - {k}: {len(df):,} rows")
    save_explore_report()

    print("\n=== 2) 문서화 + ChromaDB ===")
    docs = build_documents(datasets)
    build_vectorstore(docs=docs, rebuild=True)
    print("샘플 검색:", sample_search("단백질 높은 식품", k=3))

    print("\n=== 3) 도구 시나리오 ===")
    cmp = compare_nutrients.invoke(
        {
            "food_names": "우유,닭가슴살",
            "nutrients": "에너지(kcal),단백질(g),당류(g),나트륨(mg)",
        }
    )
    print("비교:", cmp[:500])

    calc = calculate_intake.invoke(
        {"food_name": "우유", "amount": 150, "unit": "g", "nutrient": "단백질(g)"}
    )
    print("계산:", calc[:500])

    reco = recommend_foods.invoke(
        {"data_type": "가공식품", "min_protein": 15, "max_sugar": 5, "top_n": 5}
    )
    print("추천:", reco[:500])

    print("\n=== 4) Agent 데모(일부) ===")
    agent_out = run_agent("당류가 낮고 단백질이 높은 가공식품을 추천해줘.")
    Path("reports/agent_demo_log.json").write_text(
        json.dumps([{"question": "당류가 낮고 단백질이 높은 가공식품을 추천해줘.", **agent_out}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    body = build_scenario_report_body(
        question="당류가 낮고 단백질이 높은 가공식품 추천 + 우유/닭가슴살 비교 + 우유 150g 단백질 계산",
        foods=["우유", "닭가슴살"],
        evidence=cmp,
        calculations=calc,
        recommendations=reco,
    )
    path = save_report("자동생성_영양분석리포트", body, fmt="md")
    print(f"\n=== 완료: 리포트 {path} ===")


if __name__ == "__main__":
    main()

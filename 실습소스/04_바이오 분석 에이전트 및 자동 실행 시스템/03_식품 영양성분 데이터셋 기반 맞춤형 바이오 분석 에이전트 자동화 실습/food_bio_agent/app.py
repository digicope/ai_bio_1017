"""Streamlit 맞춤형 식품 영양 분석 UI."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agent import run_agent
from config import DATASET_SPECS, ENV_FILE, REPORT_DIR, get_openai_api_key, load_env
from data_loader import get_food_row, load_all, to_numeric_series
from rag import answer_with_rag
from report import build_scenario_report_body, save_report
from tools import compare_nutrients, recommend_foods

load_env(override=False)

st.set_page_config(page_title="식품영양 RAG Agent", layout="wide")
st.title("식품 영양성분 맞춤형 바이오 분석 에이전트")
st.caption("식약처 식품영양성분 DB(로컬) · RAG + Tool Calling · 교육용")
st.sidebar.write(f"환경파일: `{ENV_FILE}`")
st.sidebar.write(f"OpenAI 키: {'설정됨' if get_openai_api_key() else '없음'}")


@st.cache_resource(show_spinner="데이터셋 로딩 중...")
def get_datasets():
    return load_all()


datasets = get_datasets()
tab_agent, tab_rag, tab_compare, tab_reco, tab_report = st.tabs(
    ["Agent 질의", "RAG 검색", "영양 비교", "맞춤 추천", "리포트"]
)

with tab_agent:
    q = st.text_area(
        "자연어 질문",
        value="당류가 낮고 단백질이 높은 가공식품을 추천해줘.",
        height=100,
    )
    if st.button("Agent 실행", type="primary"):
        with st.spinner("도구 호출 중..."):
            result = run_agent(q)
        st.write(f"실행 모드: `{result.get('mode')}`")
        st.markdown(result.get("answer", ""))
        if result.get("tool_logs"):
            st.code("\n".join(result["tool_logs"]))

with tab_rag:
    rq = st.text_input("검색 질문", value="이 제품에 어떤 영양성분 정보가 등록되어 있어? 우유")
    dtype = st.selectbox("데이터 유형 필터", ["(전체)"] + list(DATASET_SPECS.keys()))
    if st.button("RAG 검색"):
        with st.spinner("Retriever 검색 중..."):
            out = answer_with_rag(rq, data_type=None if dtype == "(전체)" else dtype)
        st.markdown(out["answer"])
        st.subheader("출처")
        st.dataframe(pd.DataFrame(out["sources"]), use_container_width=True)

with tab_compare:
    c1, c2 = st.columns(2)
    with c1:
        food_a = st.text_input("식품 A", value="닭가슴살")
    with c2:
        food_b = st.text_input("식품 B", value="연어")
    dtype2 = st.selectbox("비교 범위", ["(전체)"] + list(DATASET_SPECS.keys()), key="cmp_type")
    if st.button("비교 실행"):
        raw = compare_nutrients.invoke(
            {
                "food_names": f"{food_a},{food_b}",
                "data_type": "" if dtype2 == "(전체)" else dtype2,
                "nutrients": "에너지(kcal),단백질(g),당류(g),나트륨(mg)",
            }
        )
        data = json.loads(raw)
        st.json(data)
        if data.get("compared"):
            df = pd.DataFrame(data["compared"]).set_index("식품명")
            nutrient_cols = [c for c in data.get("nutrients", []) if c in df.columns]
            st.dataframe(df[nutrient_cols], use_container_width=True)
            chart_df = df[nutrient_cols].apply(pd.to_numeric, errors="coerce")
            st.bar_chart(chart_df.T)

with tab_reco:
    rtype = st.selectbox("추천 대상", list(DATASET_SPECS.keys()), index=2)
    min_p = st.number_input("최소 단백질(g)", value=10.0)
    max_s = st.number_input("최대 당류(g)", value=5.0)
    max_e = st.number_input("최대 열량(kcal, -1=미적용)", value=-1.0)
    max_na = st.number_input("최대 나트륨(mg, -1=미적용)", value=-1.0)
    if st.button("추천 실행"):
        raw = recommend_foods.invoke(
            {
                "data_type": rtype,
                "min_protein": float(min_p),
                "max_sugar": float(max_s),
                "max_energy": float(max_e),
                "max_sodium": float(max_na),
                "top_n": 15,
            }
        )
        data = json.loads(raw)
        st.info(data.get("allergy_disclaimer", ""))
        st.write("기준:", data.get("criteria"))
        if data.get("results"):
            st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
        else:
            st.warning("조건에 맞는 식품이 없습니다.")

with tab_report:
    st.write("현재 세션 요약 리포트를 저장합니다.")
    title = st.text_input("리포트 제목", value="맞춤형_식품영양_분석")
    foods = st.text_input("대상 식품(쉼표)", value="우유,닭가슴살")
    evidence = st.text_area("근거 데이터", value="식약처 로컬 DB 조회 결과")
    calc = st.text_area("영양 계산 결과", value="")
    reco = st.text_area("추천 결과", value="")
    if st.button("리포트 저장"):
        body = build_scenario_report_body(
            question="Streamlit 수동 리포트",
            foods=[x.strip() for x in foods.split(",") if x.strip()],
            evidence=evidence,
            calculations=calc,
            recommendations=reco,
        )
        path = save_report(title, body, fmt="md")
        st.success(f"저장 완료: {path}")

st.sidebar.header("데이터셋 현황")
for name, df in datasets.items():
    st.sidebar.write(f"**{name}**: {len(df):,}행 / {df.shape[1]}열")
st.sidebar.write(f"리포트 폴더: `{REPORT_DIR}`")
st.sidebar.warning(
    "본 서비스는 교육용이며 의료 진단·치료 목적이 아닙니다. "
    "알레르기 안전성은 DB에 없으면 보장할 수 없습니다."
)

# 빠른 식품명 미리보기
with st.sidebar.expander("식품명 검색 미리보기"):
    preview_name = st.text_input("식품명 포함검색", value="우유")
    preview_type = st.selectbox("유형", list(DATASET_SPECS.keys()), key="side_type")
    if preview_name:
        hits = get_food_row(datasets[preview_type], preview_name).head(5)
        if hits.empty:
            st.write("결과 없음")
        else:
            cols = [c for c in ["식품명", "에너지(kcal)", "단백질(g)", "당류(g)", "나트륨(mg)"] if c in hits.columns]
            view = hits[cols].copy()
            for c in cols[1:]:
                view[c] = to_numeric_series(view[c])
            st.dataframe(view, use_container_width=True)

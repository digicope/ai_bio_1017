"""LangChain Agent Tool Calling + API 키 없을 때 규칙 기반 라우터."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from config import ENV_FILE, LLM_MODEL, get_openai_api_key
from tools import (
    ALL_TOOLS,
    calculate_intake,
    compare_nutrients,
    generate_analysis_report,
    nutrient_lookup,
    rag_food_search,
    recommend_foods,
)


def _tool_map() -> dict[str, Any]:
    return {t.name: t for t in ALL_TOOLS}


def run_langchain_agent(question: str) -> dict:
    """OpenAI 기반 LangChain Agent (버전에 따라 create_agent / create_react_agent)."""
    from langchain_openai import ChatOpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(f"{ENV_FILE} 에서 OPENAI_API_KEY를 읽지 못했습니다.")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    system = (
        "당신은 식품영양성분 DB 기반 맞춤형 바이오 분석 에이전트입니다. "
        "질문에 필요한 도구만 선택해 호출하고, 데이터에 없는 효능·수치를 추측하지 마세요. "
        "최종 답변에 데이터 근거, 계산 조건, 추천 이유를 포함하세요. "
        "의료 진단/치료를 단정하지 마세요."
    )

    # LangChain 신규 create_agent 우선, 실패 시 classic react agent
    try:
        from langchain.agents import create_agent

        agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=system)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        final = messages[-1].content if messages else str(result)
        return {"mode": "langchain_create_agent", "answer": final, "raw": str(result)[:4000]}
    except Exception:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(llm, ALL_TOOLS, prompt=system)
        result = agent.invoke({"messages": [("user", question)]})
        messages = result.get("messages", [])
        final = messages[-1].content if messages else str(result)
        return {"mode": "langgraph_react_agent", "answer": final, "raw": str(result)[:4000]}


def _rule_based_route(question: str) -> dict:
    """API 키 없이 시나리오별 도구 호출을 시연하는 로컬 라우터."""
    q = question.strip()
    logs: list[str] = []
    outputs: list[Any] = []

    def call(tool_fn, **kwargs):
        logs.append(f"TOOL {tool_fn.name} args={kwargs}")
        out = tool_fn.invoke(kwargs)
        outputs.append(out)
        return out

    # 추천
    if "추천" in q:
        dtype = "가공식품"
        for t in ("건강기능식품", "음식", "가공식품"):
            if t in q:
                dtype = t
                break
        min_protein = 10.0
        max_sugar = 5.0
        m = re.search(r"단백질[^\d]*(\d+(?:\.\d+)?)", q)
        if m:
            min_protein = float(m.group(1))
        m = re.search(r"당류[^\d]*(\d+(?:\.\d+)?)", q)
        if m:
            max_sugar = float(m.group(1))
        # '높고/낮은' 기본 시나리오
        if "단백질" in q and "당류" in q:
            call(
                recommend_foods,
                data_type=dtype,
                min_protein=min_protein,
                max_sugar=max_sugar,
                top_n=8,
            )
            # 부가 설명용 RAG
            call(rag_food_search, query=q, data_type=dtype, top_k=3)

    # 비교
    elif "비교" in q:
        names = re.findall(r"[\"'“”](.+?)[\"'“”]", q)
        if len(names) < 2:
            # A와 B 패턴
            m = re.search(r"(.+?)와\s*(.+?)의", q)
            if m:
                names = [m.group(1).strip(), m.group(2).strip()]
        if len(names) >= 2:
            call(
                compare_nutrients,
                food_names=",".join(names[:4]),
                nutrients="에너지(kcal),단백질(g),당류(g),나트륨(mg)",
            )
        else:
            call(rag_food_search, query=q, top_k=5)

    # 섭취량 계산
    elif "계산" in q or re.search(r"\d+\s*g", q, re.I):
        m_amt = re.search(r"(\d+(?:\.\d+)?)\s*(g|ml|mg)", q, re.I)
        amount = float(m_amt.group(1)) if m_amt else 100.0
        unit = m_amt.group(2).lower() if m_amt else "g"
        # 식품명 추정: '의 영양' 앞 또는 따옴표
        names = re.findall(r"[\"'“”](.+?)[\"'“”]", q)
        food = names[0] if names else None
        if not food:
            m = re.search(r"([가-힣A-Za-z0-9 ·_-]{2,40})의\s*영양", q)
            food = m.group(1).strip() if m else "우유"
        nutrient = "단백질(g)" if "단백" in q else "에너지(kcal)"
        call(rag_food_search, query=food, top_k=3)
        call(calculate_intake, food_name=food, amount=amount, unit=unit, nutrient=nutrient)

    # 리포트
    elif "보고" in q or "리포트" in q or "저장" in q:
        body = "자동 리포트 요청\n\n" + "\n\n".join(str(o) for o in outputs) if outputs else q
        # 간단 비교+추천 샘플을 먼저 수행
        rec = call(recommend_foods, data_type="가공식품", min_protein=15, max_sugar=5, top_n=5)
        call(
            generate_analysis_report,
            title="식품영양_분석리포트",
            content=f"질문: {q}\n\n추천결과:\n{rec}",
            fmt="md",
        )

    # 기본: RAG + 정밀조회
    else:
        call(rag_food_search, query=q, top_k=5)
        names = re.findall(r"[\"'“”](.+?)[\"'“”]", q)
        if names:
            call(nutrient_lookup, food_name=names[0])

    answer_parts = [
        "### 로컬 규칙 기반 Agent 실행 결과",
        f"질문: {q}",
        "",
        "#### 도구 호출 로그",
        *logs,
        "",
        "#### 도구 출력",
    ]
    for i, o in enumerate(outputs, 1):
        answer_parts.append(f"\n[{i}]\n{o}")
    answer_parts.append(
        f"\n※ {ENV_FILE} 에 OPENAI_API_KEY가 없어 규칙 기반 라우터로 도구를 호출했습니다. "
        "키를 설정하면 LangChain Agent Tool Calling을 사용합니다."
    )
    return {"mode": "rule_based_router", "answer": "\n".join(answer_parts), "tool_logs": logs}


def run_agent(question: str) -> dict:
    if get_openai_api_key():
        try:
            return run_langchain_agent(question)
        except Exception as e:
            fallback = _rule_based_route(question)
            fallback["answer"] = (
                f"LangChain Agent 호출 실패({e}). 규칙 기반 라우터로 대체합니다.\n\n"
                + fallback["answer"]
            )
            fallback["mode"] = "fallback_rule_based"
            return fallback
    return _rule_based_route(question)


def demo_scenarios() -> list[dict]:
    scenarios = [
        "우유에 어떤 영양성분 정보가 등록되어 있어?",
        "닭가슴살과 연어의 열량, 단백질, 당류, 나트륨을 비교해줘.",
        "우유의 영양정보를 설명하고 150g을 먹으면 단백질을 얼마나 섭취하는지 계산해줘.",
        "당류가 낮고 단백질이 높은 가공식품을 추천해줘.",
        "힘가네 혈당케어 앤 유산균 건강기능식품의 DB 등록 정보를 알려줘.",
        "오늘 비교한 식품의 영양 분석 결과를 보고서로 저장해줘.",
    ]
    results = []
    for q in scenarios:
        print(f"\n=== SCENARIO: {q}")
        r = run_agent(q)
        results.append({"question": q, **r})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default="")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        out = demo_scenarios()
        path = "reports/agent_demo_log.json"
        os.makedirs("reports", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("saved", path)
    else:
        q = args.q or "당류가 낮고 단백질이 높은 가공식품을 추천해줘."
        print(run_agent(q)["answer"][:3000])

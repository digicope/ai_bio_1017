"""Retriever 및 RAG 질의응답."""
from __future__ import annotations

from typing import Any

from config import ENV_FILE, LLM_MODEL, TOP_K, get_openai_api_key
from ingest import build_vectorstore


def get_retriever(k: int | None = None, data_type: str | None = None):
    vs, _ = build_vectorstore(rebuild=False)
    search_kwargs: dict[str, Any] = {"k": k or TOP_K}
    if data_type:
        search_kwargs["filter"] = {"data_type": data_type}
    return vs.as_retriever(search_kwargs=search_kwargs)


def retrieve_docs(query: str, k: int | None = None, data_type: str | None = None) -> list[dict]:
    retriever = get_retriever(k=k, data_type=data_type)
    docs = retriever.invoke(query)
    results = []
    for d in docs:
        results.append(
            {
                "food_name": d.metadata.get("food_name"),
                "data_type": d.metadata.get("data_type"),
                "source": d.metadata.get("source"),
                "food_id": d.metadata.get("food_id"),
                "content": d.page_content,
            }
        )
    return results


RAG_SYSTEM_PROMPT = """당신은 식품의약품안전처 식품영양성분 DB 기반 교육용 영양 분석 도우미입니다.
규칙:
1. 제공된 검색 문서에 있는 정보만 사용하세요.
2. 문서에 없는 영양 수치, 건강 효능, 질병 예방/치료 효과를 추측하거나 단정하지 마세요.
3. 답변에 식품명과 데이터 출처(source/data_type)를 명시하세요.
4. 검색 결과가 없거나 관련성이 낮으면 '정보가 부족합니다'라고 말하세요.
5. 이 시스템은 의료 진단 목적이 아닙니다.
"""


def answer_with_rag(question: str, k: int | None = None, data_type: str | None = None) -> dict:
    docs = retrieve_docs(question, k=k, data_type=data_type)
    if not docs:
        return {
            "question": question,
            "answer": "관련 문서를 찾지 못했습니다. 식품명을 더 구체적으로 입력해 주세요.",
            "sources": [],
            "mode": "empty",
        }

    context = "\n\n---\n\n".join(
        f"[{i+1}] 식품명={d['food_name']} | 유형={d['data_type']} | 출처={d['source']}\n{d['content']}"
        for i, d in enumerate(docs)
    )

    api_key = get_openai_api_key()
    if api_key:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=f"질문: {question}\n\n검색 문서:\n{context}\n\n위 문서만 근거로 한국어로 답하세요."
            ),
        ]
        answer = llm.invoke(messages).content
        mode = "llm"
    else:
        # API 키 없을 때 문서 요약형 응답
        lines = [
            f"질문: {question}",
            "",
            "검색된 DB 등록 정보(문서 근거):",
        ]
        for i, d in enumerate(docs, 1):
            lines.append(f"\n[{i}] {d['food_name']} ({d['data_type']} / {d['source']})")
            lines.append(d["content"])
        lines.append(
            f"\n※ {ENV_FILE} 에 OPENAI_API_KEY가 없어 문서 원문 기반 요약입니다. "
            "문서에 없는 효능·수치는 포함하지 않았습니다."
        )
        answer = "\n".join(lines)
        mode = "extractive"

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "food_name": d["food_name"],
                "data_type": d["data_type"],
                "source": d["source"],
                "food_id": d["food_id"],
            }
            for d in docs
        ],
        "mode": mode,
    }


if __name__ == "__main__":
    q = "우유에 어떤 영양성분 정보가 등록되어 있어?"
    result = answer_with_rag(q)
    print(result["answer"][:1500])
    print("\nSOURCES:", result["sources"])

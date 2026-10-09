"""문서 전처리 → 임베딩 → ChromaDB 저장."""
from __future__ import annotations

import hashlib
import json
import shutil
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CORE_NUTRIENTS,
    DATASET_SPECS,
    EMBEDDING_MODEL_NAME,
    ENV_FILE,
    ID_COL,
    NAME_COL,
    OPENAI_EMBEDDING_MODEL,
    RAG_SAMPLE_LIMITS,
    TYPE_NAME_COL,
    get_openai_api_key,
)
from data_loader import load_all

META_PATH = CHROMA_DIR / "embedding_meta.json"


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "해당없음"}:
        return ""
    return text


def row_to_document(row: pd.Series, data_type: str) -> Document | None:
    name = _clean_text(row.get(NAME_COL))
    if not name:
        return None

    food_id = _clean_text(row.get(ID_COL)) or hashlib.md5(
        f"{data_type}:{name}:{row.name}".encode("utf-8")
    ).hexdigest()[:16]

    parts = [
        f"식품명: {name}",
        f"데이터 유형: {data_type}",
    ]
    type_name = _clean_text(row.get(TYPE_NAME_COL))
    if type_name:
        parts.append(f"데이터구분명: {type_name}")

    for col in DATASET_SPECS[data_type]["extra_text_cols"]:
        val = _clean_text(row.get(col))
        if val:
            parts.append(f"{col}: {val}")

    serve_col = None
    for c in DATASET_SPECS[data_type]["serving_col_candidates"]:
        if c in row.index and _clean_text(row.get(c)):
            serve_col = c
            break
    if serve_col:
        serve_val = _clean_text(row.get(serve_col))
        if serve_val:
            parts.append(f"기준 제공량({serve_col}): {serve_val}")

    nutrient_bits = []
    for col in CORE_NUTRIENTS:
        if col not in row.index:
            continue
        val = row.get(col)
        if pd.isna(val):
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        nutrient_bits.append(f"{col}={num}")
    if nutrient_bits:
        parts.append("등록된 영양성분: " + ", ".join(nutrient_bits))
    elif len(parts) <= 2:
        return None

    page_content = "\n".join(parts)
    if len(page_content) > 1800:
        page_content = page_content[:1800] + "\n...(중략)"

    metadata = {
        "source": _clean_text(row.get("_source_file")) or data_type,
        "data_type": data_type,
        "food_id": food_id,
        "food_name": name,
    }
    return Document(page_content=page_content, metadata=metadata, id=f"{data_type}:{food_id}")


def build_documents(
    datasets: dict[str, pd.DataFrame] | None = None,
    sample_limits: dict[str, int] | None = None,
) -> list[Document]:
    datasets = datasets or load_all()
    sample_limits = sample_limits or RAG_SAMPLE_LIMITS
    docs: list[Document] = []
    seen: set[str] = set()

    for data_type, df in datasets.items():
        limit = sample_limits.get(data_type, len(df))
        work = df.copy()
        if NAME_COL in work.columns:
            work = work[work[NAME_COL].notna() & (work[NAME_COL].astype(str).str.strip() != "")]
        if len(work) > limit:
            work = work.sample(n=limit, random_state=42)
        print(f"[ingest] {data_type}: 문서화 {len(work):,}행")

        for _, row in work.iterrows():
            doc = row_to_document(row, data_type)
            if doc is None:
                continue
            key = doc.id or f"{doc.metadata['data_type']}:{doc.metadata['food_id']}"
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
    print(f"[ingest] 총 문서 수: {len(docs):,}")
    return docs


def get_embeddings():
    """OpenAI 키가 있으면 OpenAI, 없으면 로컬 sentence-transformers."""
    api_key = get_openai_api_key()
    if api_key:
        from langchain_openai import OpenAIEmbeddings

        print(f"[embed] OpenAI Embeddings: {OPENAI_EMBEDDING_MODEL} (env={ENV_FILE})")
        return (
            OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key),
            "openai",
            OPENAI_EMBEDDING_MODEL,
            1536,
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"[embed] Local HuggingFace: {EMBEDDING_MODEL_NAME}")
    return (
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        ),
        "local",
        EMBEDDING_MODEL_NAME,
        384,
    )


def _read_embed_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_embed_meta(backend: str, model: str, dim: int, doc_count: int) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps(
            {
                "backend": backend,
                "model": model,
                "dimension": dim,
                "doc_count": doc_count,
                "collection": COLLECTION_NAME,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _clear_chroma() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _needs_rebuild(backend: str, model: str, dim: int) -> bool:
    """임베딩 백엔드/모델/차원이 바뀌면 재구축 필요."""
    meta = _read_embed_meta()
    if not meta:
        # 메타 없이 예전 local(384) DB만 있는 경우 → OpenAI(1536)와 충돌
        return True
    return (
        meta.get("backend") != backend
        or meta.get("model") != model
        or int(meta.get("dimension") or 0) != dim
    )


def build_vectorstore(docs: list[Document] | None = None, rebuild: bool = False):
    """ChromaDB 영속 저장. 임베딩 차원/백엔드가 바뀌면 자동 재구축."""
    from langchain_chroma import Chroma

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, backend, model, dim = get_embeddings()
    persist = str(CHROMA_DIR)
    marker = CHROMA_DIR / "chroma.sqlite3"

    if rebuild or (marker.exists() and _needs_rebuild(backend, model, dim)):
        reason = "강제 재구축" if rebuild else f"임베딩 변경 감지({backend}/{model}/{dim}d)"
        print(f"[ingest] {reason} → ChromaDB 초기화")
        _clear_chroma()
        marker = CHROMA_DIR / "chroma.sqlite3"

    if marker.exists():
        print(f"[ingest] 기존 ChromaDB 재사용: {persist} ({backend}/{model})")
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist,
        )
        try:
            count = vs._collection.count()
            # 차원 불일치 조기 탐지
            _ = vs.similarity_search("영양", k=1)
            print(f"[ingest] 문서 수={count}")
            return vs, backend
        except Exception as e:
            print(f"[ingest] 기존 DB 사용 불가({e}) → 재구축")
            _clear_chroma()

    docs = docs or build_documents()
    if not docs:
        raise RuntimeError("임베딩할 문서가 없습니다.")

    print(
        f"[ingest] ChromaDB 구축 시작 "
        f"({len(docs):,} docs, backend={backend}, model={model}, dim={dim}, path={persist})"
    )
    batch_size = 100 if backend == "openai" else 200
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist,
    )
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d.id for d in batch]
        vs.add_documents(documents=batch, ids=ids)
        print(f"[ingest] 저장 진행 {min(i + batch_size, len(docs))}/{len(docs)}")
    _write_embed_meta(backend, model, dim, len(docs))
    print("[ingest] ChromaDB 저장 완료")
    return vs, backend


def sample_search(query: str = "단백질이 높은 식품", k: int = 3) -> list[dict]:
    vs, _ = build_vectorstore(rebuild=False)
    results = vs.similarity_search(query, k=k)
    return [
        {
            "food_name": r.metadata.get("food_name"),
            "data_type": r.metadata.get("data_type"),
            "source": r.metadata.get("source"),
            "preview": r.page_content[:300],
        }
        for r in results
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="문서 전처리 및 ChromaDB 구축")
    parser.add_argument("--rebuild", action="store_true", help="기존 ChromaDB 삭제 후 재구축")
    parser.add_argument("--explore-only", action="store_true", help="데이터 탐색만 수행")
    args = parser.parse_args()

    from data_loader import save_explore_report

    save_explore_report()
    if args.explore_only:
        raise SystemExit(0)

    docs = build_documents()
    build_vectorstore(docs=docs, rebuild=args.rebuild)
    print("\n[sample search]")
    for item in sample_search():
        print("-", item["food_name"], f"({item['data_type']})", item["source"])

"""프로젝트 경로·컬럼·영양성분 설정."""
from __future__ import annotations

import os
from pathlib import Path

# OpenAI 등 API 키는 C:/env/.env 에서만 읽는다 (소스/제출물에 키를 넣지 않음)
ENV_FILE = Path(os.getenv("FOOD_BIO_ENV_FILE", r"C:\env\.env"))


def load_env(override: bool = False) -> Path | None:
    """C:/env/.env 를 환경변수로 로드. 파일이 없으면 None."""
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise ImportError("python-dotenv 가 필요합니다. pip install python-dotenv") from e

    if not ENV_FILE.exists():
        return None
    load_dotenv(dotenv_path=ENV_FILE, override=override)
    return ENV_FILE


def get_openai_api_key() -> str:
    """OPENAI_API_KEY를 C:/env/.env 기준으로 반환."""
    load_env(override=False)
    return os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")


# 모듈 import 시점에 환경변수 선로드
_LOADED_ENV = load_env(override=False)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
REPORT_DIR = BASE_DIR / "reports"
# Windows + 한글 경로에서 Chroma HNSW 로딩 오류를 피하기 위해 ASCII 경로 사용
# (프로젝트 내 chroma_db는 심볼릭/안내용으로 유지 가능)
CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", r"C:\MyCursorLab\chroma_food_bio_agent")
)

COLLECTION_NAME = "food_nutrition_docs"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# RAG 문서화 시 유형별 최대 샘플 수 (전체 행은 Pandas 도구에서 사용)
RAG_SAMPLE_LIMITS = {
    "건강기능식품": int(os.getenv("RAG_SAMPLE_HEALTH", "1500")),
    "음식": int(os.getenv("RAG_SAMPLE_FOOD", "2000")),
    "가공식품": int(os.getenv("RAG_SAMPLE_PROCESSED", "3000")),
}

# 실제 파일명 패턴 (다운로드 버전에 따라 숫자·날짜가 달라질 수 있음)
DATASET_SPECS = {
    "건강기능식품": {
        "glob": "*건강기능식품DB*.xlsx",
        "type_code": "F",
        "serving_col_candidates": ["영양성분제공단위량", "1회분량중량/부피", "식품중량/부피"],
        "extra_text_cols": ["제조사명", "대표식품명", "식품대분류명", "식품중분류명", "유형명"],
    },
    "음식": {
        "glob": "*음식DB*.xlsx",
        "type_code": "D",
        "serving_col_candidates": ["영양성분함량기준량", "1인(회)분량", "식품중량"],
        "extra_text_cols": ["출처명", "대표식품명", "식품대분류명", "식품중분류명"],
    },
    "가공식품": {
        "glob": "*가공식품DB*.xlsx",
        "type_code": "P",
        "serving_col_candidates": ["영양성분함량기준량", "1회 제공량", "식품중량"],
        "extra_text_cols": ["제조사명", "대표식품명", "식품대분류명", "식품중분류명", "출처명"],
    },
}

# 공통 핵심 영양성분 (실제 컬럼명이 존재하는 경우만 사용)
CORE_NUTRIENTS = [
    "에너지(kcal)",
    "단백질(g)",
    "지방(g)",
    "탄수화물(g)",
    "당류(g)",
    "나트륨(mg)",
    "포화지방산(g)",
    "트랜스지방산(g)",
    "콜레스테롤(mg)",
    "식이섬유(g)",
    "칼슘(mg)",
    "철(mg)",
    "칼륨(mg)",
    "비타민 C(mg)",
    "비타민A(μg RAE)",
    "비타민 A(μg RAE)",
]

ID_COL = "식품코드"
NAME_COL = "식품명"
TYPE_NAME_COL = "데이터구분명"

for d in (DATA_DIR, PARQUET_DIR, CHROMA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

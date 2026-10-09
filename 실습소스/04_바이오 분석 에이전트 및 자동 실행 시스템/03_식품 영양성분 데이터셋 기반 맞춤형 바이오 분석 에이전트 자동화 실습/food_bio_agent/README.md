# 식품 영양성분 RAG + Tool Calling 맞춤형 바이오 분석 에이전트

식품의약품안전처 식품영양성분 DB(가공식품·음식·건강기능식품) 로컬 파일을 사용해
RAG 검색과 Agent Tool Calling을 결합한 교육용 영양 분석 서비스입니다.

> 의료 진단·치료·개인 생체반응 예측 목적이 아닙니다.

## 폴더 구조

```text
food_bio_agent/
  app.py              # Streamlit UI
  ingest.py           # 문서 전처리 + 임베딩 + ChromaDB
  rag.py              # Retriever / RAG 질의응답
  tools.py            # Agent 도구 6종
  agent.py            # LangChain Agent / 로컬 라우터
  report.py           # Markdown·TXT 리포트
  data_loader.py      # Excel→Parquet 로딩·탐색
  config.py           # 경로·컬럼 매핑
  run_pipeline.py     # 원클릭 파이프라인
  data/               # 원본 xlsx 3종
  data/parquet/       # 변환 캐시
  reports/            # 자동 리포트
  requirements.txt
```

ChromaDB 기본 저장 경로: `C:\MyCursorLab\chroma_food_bio_agent`  
(Windows에서 한글 경로의 HNSW 로딩 오류를 피하기 위함. `CHROMA_DIR`로 변경 가능)

## 데이터 배치

`data/` 폴더에 다음 유형의 파일을 둡니다(파일명은 다운로드 버전에 따라 다를 수 있음).

| 유형 | 예시 파일명 |
|------|-------------|
| 건강기능식품 | `20260623_건강기능식품DB_5556건.xlsx` |
| 음식 | `20260828_음식DB_19617건.xlsx` |
| 가공식품 | `20260929_가공식품DB_323197건.xlsx` |

원본 DB는 용량이 크므로 제출물에서 제외할 수 있습니다.  
공식 다운로드: 식품의약품안전처 식품영양성분 데이터베이스

## 설치

```bash
cd food_bio_agent
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

OpenAI API 키는 **`C:\env\.env`** 에서 읽습니다 (`config.load_env` / `get_openai_api_key`).

```env
# C:\env\.env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

경로 변경이 필요하면 환경변수 `FOOD_BIO_ENV_FILE`로 지정할 수 있습니다.  
키가 있으면 OpenAI 임베딩/LLM Agent를 사용하고, 없으면 로컬 임베딩 + 규칙 기반 라우터로 동작합니다.  
API 키는 소스 코드·제출물에 포함하지 않습니다.

## 실행 순서

### 1) 데이터 탐색 + Parquet 변환

```bash
python data_loader.py
```

가공식품 DB(약 32만 행)는 최초 Excel 로딩에 시간이 걸립니다.  
이후에는 `data/parquet/*.parquet`를 재사용합니다.

### 2) 문서 전처리 및 ChromaDB 구축

```bash
python ingest.py --rebuild
```

- 유형별로 샘플링해 RAG 문서를 만듭니다(전체 수치는 Pandas 도구에서 사용).
- 환경변수 `RAG_SAMPLE_*`로 샘플 수를 조절할 수 있습니다.

### 3) RAG / Agent 데모

```bash
python rag.py
python agent.py --demo
```

### 4) Streamlit UI

```bash
streamlit run app.py
```

## 구현된 도구

1. `rag_food_search` — Chroma Retriever 기반 식품 정보 검색  
2. `nutrient_lookup` — Pandas 영양성분 정밀 조회  
3. `compare_nutrients` — 식품 간 공통 영양항목 비교  
4. `calculate_intake` — 기준 제공량 대비 섭취량 비례 계산  
5. `recommend_foods` — 단백질/당류/열량/나트륨 조건 추천  
6. `generate_analysis_report` — Markdown/TXT 리포트 저장  

## 유의사항

- 데이터에 없는 값은 0으로 채우거나 임의 생성하지 않습니다.
- 기준 단위가 다른 영양값은 직접 환산·비교하지 않습니다.
- RAG(설명·관련 문서)와 정량 계산(원본 수치)을 분리합니다.
- 알레르기 정보가 없으면 안전성을 보장할 수 없다고 안내합니다.
- API 키는 환경변수로만 관리합니다.

# GSE2034 유전자 발현 기반 분류·회귀 모델

[NCBI GEO GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034) (Wang et al., *Lancet* 2005) 유방암 발현 데이터를 이용해 **원격전이(재발) 분류**와 **무재발 기간(개월) 회귀** 모델을 학습하는 노트북입니다.

## 데이터 요약

| 항목 | 내용 |
|------|------|
| 플랫폼 | Affymetrix HG-U133A (GPL96) |
| 샘플 | 림프절 음성 원발 유방암 286명 |
| 라벨 | 재발 없음 ~180명 / 원격전이 발생 ~106명 |
| 발현 | Series Matrix (~22,283 probes) |
| 임상 | SOFT family의 Patient clinical parameters |

## 환경 설정

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
jupyter notebook GSE2034_classification_regression.ipynb
```

노트북을 위에서 아래로 실행하면 `data/`에 GEO 파일을 캐시하고, `outputs/`에 지표·그래프·모델을 저장합니다.

## 파이프라인 개요

1. Series Matrix / SOFT 다운로드 및 파싱
2. GSM ID 기준 발현·임상 병합, log2 변환
3. 분산 필터 + SelectKBest로 프로브 축소
4. **분류**: Logistic / RF / Gradient Boosting → 원격전이(재발) 예측
5. **회귀**: Ridge / RF / Gradient Boosting → 재발 환자의 전이까지 개월 수 예측
6. **보조**: ER+/ER− 분류
7. 혼동행렬·ROC·잔차·상위 프로브 중요도 시각화

## 검증 실행 결과 (요약)

| 과제 | 최고 모델 | 지표 |
|------|-----------|------|
| 재발 분류 | Logistic Regression | Acc ≈ 0.69, ROC-AUC ≈ 0.71 |
| 전이까지 개월 회귀 (재발군) | Random Forest | MAE ≈ 16개월, R² ≈ 0.19 |
| ER 분류 | Logistic Regression | Acc ≈ 0.89, ROC-AUC ≈ 0.94 |

## 참고

- Wang Y et al. Gene-expression profiles to predict distant metastasis of lymph-node-negative primary breast cancer. *Lancet*. 2005.
- 교육·연구용 데모이며 임상 진단용으로 사용할 수 없습니다.

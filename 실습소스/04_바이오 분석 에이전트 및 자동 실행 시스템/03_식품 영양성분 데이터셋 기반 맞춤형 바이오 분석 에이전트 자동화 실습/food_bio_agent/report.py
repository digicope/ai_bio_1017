"""Markdown/TXT 분석 리포트 생성."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import REPORT_DIR


def save_report(title: str, content: str, fmt: str = "md") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:40] or "report"
    ext = "md" if fmt.lower() in {"md", "markdown"} else "txt"
    path = REPORT_DIR / f"{stamp}_{safe}.{ext}"

    header = [
        f"# {title}" if ext == "md" else title,
        "",
        f"생성일시: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "본 리포트는 식품의약품안전처 식품영양성분 DB(로컬) 기반 교육용 분석 결과입니다.",
        "의료 진단·치료·개인 생체반응 예측 목적이 아닙니다.",
        "",
        "---",
        "",
        content.strip(),
        "",
        "---",
        "",
        "## 데이터 한계",
        "- 데이터에 없는 값은 임의 생성하지 않았습니다.",
        "- 기준 제공량 단위가 다른 항목은 직접 비교하지 않았습니다.",
        "- 알레르기 안전성은 DB에 명시되지 않은 경우 보장할 수 없습니다.",
        "",
    ]
    path.write_text("\n".join(header), encoding="utf-8")
    return path


def build_scenario_report_body(
    question: str,
    foods: list[str],
    evidence: str,
    calculations: str,
    recommendations: str,
) -> str:
    return "\n".join(
        [
            "## 질문",
            question,
            "",
            "## 검색/비교 식품",
            ", ".join(foods) if foods else "(없음)",
            "",
            "## 근거 데이터",
            evidence or "(없음)",
            "",
            "## 영양 계산 결과",
            calculations or "(없음)",
            "",
            "## 추천 기준 및 결과",
            recommendations or "(없음)",
        ]
    )

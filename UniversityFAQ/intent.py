"""Intent classification for the University FAQ assistant.

Two classifiers over one label set, so the project can compare a hand-written
matcher with a trained model on the same held-out paraphrase queries:

  RuleIntent     keyword matcher, no training, transparent triggers
  TrainedIntent  TF-IDF (word 1-2 grams with the retrieval tokeniser, plus
                 character 2-4 grams) with a logistic-regression head

Both are evaluated against labels derived from the FAQ corpus, never from the
queries they are scored on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

UNKNOWN = "UNKNOWN"

INTENTS: dict[str, str] = {
    "ASSESSMENT_EXTENSION": "Assignment extension",
    "ASSESSMENT_GRADE": "Grades and appeals",
    "EXAM_TIMETABLE": "Exam timetable",
    "EXAM_ABSENCE": "Missed exam",
    "ENROLMENT_CHANGE": "Add or drop a course",
    "FEES_PAYMENT": "Tuition payment",
    "FEES_REFUND": "Refund",
    "SCHOLARSHIP": "Scholarship",
    "IT_ACCOUNT": "IT account",
    "LIBRARY": "Library",
    "ACADEMIC_INTEGRITY": "Academic integrity",
    "STUDENT_SUPPORT": "Student support",
}

INTENTS_ZH: dict[str, str] = {
    "ASSESSMENT_EXTENSION": "作業延期",
    "ASSESSMENT_GRADE": "成績與申訴",
    "EXAM_TIMETABLE": "考試時間表",
    "EXAM_ABSENCE": "缺席與補考",
    "ENROLMENT_CHANGE": "加退選",
    "FEES_PAYMENT": "學費繳交",
    "FEES_REFUND": "退款",
    "SCHOLARSHIP": "獎學金",
    "IT_ACCOUNT": "資訊系統帳號",
    "LIBRARY": "圖書館",
    "ACADEMIC_INTEGRITY": "學術誠信",
    "STUDENT_SUPPORT": "學生支援",
}

# Keyword triggers, English and Chinese. Deliberately hand-written and shallow:
# that is the point of the comparison.
RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ASSESSMENT_EXTENSION": (
        ("extension", "extend", "extra time", "more time", "late submission",
         "deadline", "mitigating", "postpone"),
        ("延期", "延後", "延后", "遲交", "迟交", "截止", "寬限", "宽限", "補交", "补交"),
    ),
    "ASSESSMENT_GRADE": (
        ("appeal", "re-mark", "remark", "grade", "mark", "result", "scaling",
         "review"),
        ("申訴", "申诉", "成績", "成绩", "分數", "分数", "重評", "重评", "複核",
         "复核", "上調", "上调"),
    ),
    "EXAM_TIMETABLE": (
        ("timetable", "exam schedule", "exam room", "venue", "seat", "clash",
         "start time", "exam date"),
        ("時間表", "时间表", "考場", "考场", "課室", "课室", "撞期", "幾點", "几点",
         "考試日期", "考试日期"),
    ),
    "EXAM_ABSENCE": (
        ("missed exam", "absent", "sick", "deferred", "resit", "consideration",
         "late arrival", "hospital"),
        ("缺席", "生病", "住院", "補考", "补考", "特殊考慮", "特殊考虑", "遲到",
         "迟到", "考不了"),
    ),
    "ENROLMENT_CHANGE": (
        ("add a course", "drop", "enrol", "enroll", "tutorial group", "waiting list",
         "switch", "section"),
        ("加課", "加课", "退課", "退课", "退選", "退选", "選課", "选课", "導修",
         "导修", "候補", "候补", "換組", "换组"),
    ),
    "FEES_PAYMENT": (
        ("pay", "payment", "tuition", "instalment", "installment", "receipt", "fee"),
        ("繳", "缴", "交費", "交费", "學費", "学费", "分期", "收據", "收据", "付款"),
    ),
    "FEES_REFUND": (
        ("refund", "repay", "duplicate", "overpayment", "charged twice", "money back"),
        ("退錢", "退钱", "退款", "重複", "重复", "多付", "退回"),
    ),
    "SCHOLARSHIP": (
        ("scholarship", "bursary", "grant", "award", "stipend"),
        ("獎學金", "奖学金", "助學金", "助学金", "資助", "资助", "補助", "补助"),
    ),
    "IT_ACCOUNT": (
        ("password", "log in", "login", "locked", "account", "two-factor",
         "verification", "email"),
        ("密碼", "密码", "登入", "帳戶", "帐户", "鎖定", "锁定", "驗證", "验证",
         "郵箱", "邮箱"),
    ),
    "LIBRARY": (
        ("library", "borrow", "loan", "renew", "database", "journal", "fine"),
        ("圖書", "图书", "借書", "借书", "續借", "续借", "數據庫", "数据库", "罰款",
         "罚款", "文獻", "文献"),
    ),
    "ACADEMIC_INTEGRITY": (
        ("plagiarism", "similarity", "citation", "cite", "reference", "integrity",
         "copying", "reuse"),
        ("抄襲", "抄袭", "相似度", "引用", "參考", "参考", "誠信", "诚信", "重複使用",
         "重复使用"),
    ),
    "STUDENT_SUPPORT": (
        ("counsel", "stress", "wellbeing", "disability", "hardship", "support",
         "adjustment"),
        ("輔導", "辅导", "壓力", "压力", "情緒", "情绪", "殘障", "残障", "困難",
         "困难", "支援", "援助", "心理"),
    ),
}


def _compile_rule(keywords: Sequence[str]) -> list[re.Pattern[str]]:
    patterns = []
    for word in keywords:
        if word.isascii():
            patterns.append(re.compile(rf"\b{re.escape(word)}", re.IGNORECASE))
        else:
            patterns.append(re.compile(re.escape(word)))
    return patterns


_COMPILED = {
    intent: (_compile_rule(en), _compile_rule(zh))
    for intent, (en, zh) in RULES.items()
}


@dataclass
class RuleIntent:
    """Transparent keyword matcher: score = number of distinct triggers hit."""

    def predict(self, text: str) -> tuple[str, float]:
        best, best_hits = UNKNOWN, 0
        for intent, (en_patterns, zh_patterns) in _COMPILED.items():
            hits = sum(bool(p.search(text)) for p in en_patterns)
            hits += sum(bool(p.search(text)) for p in zh_patterns)
            if hits > best_hits:
                best, best_hits = intent, hits
        if best_hits == 0:
            return UNKNOWN, 0.0
        confidence = min(1.0, best_hits / 3)
        return best, confidence

    def triggers(self, intent: str) -> list[str]:
        en, zh = RULES[intent]
        return list(en) + list(zh)


@dataclass
class TrainedIntent:
    """TF-IDF features plus a logistic-regression head, fitted on FAQ questions."""

    # Default threshold is the uniform class prior: "better than chance".
    min_prob: float = 1.0 / 12.0
    labels: list[str] = field(default_factory=list)
    head: object | None = None

    def fit(self, entries: Sequence[dict]) -> "TrainedIntent":
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        from faq_engine import tokens

        usable = [e for e in entries if e.get("intent") in INTENTS]
        if len(usable) < 2:
            raise ValueError("Not enough intent-labelled entries to train on.")
        texts = [f"{e['q_en']} {e['q_zh']}" for e in usable]
        labels = [e["intent"] for e in usable]
        self.labels = sorted(INTENTS)

        # Same tokeniser as retrieval, so Chinese character overlap counts.
        self._word = TfidfVectorizer(
            tokenizer=tokens, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True, min_df=1,
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1
        )
        features = hstack([
            self._word.fit_transform(texts),
            self._char.fit_transform(texts),
        ]).tocsr()
        self.head = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")
        self.head.fit(features, labels)
        return self

    def _features(self, text: str):
        from scipy.sparse import hstack

        return hstack([
            self._word.transform([text]),
            self._char.transform([text]),
        ]).tocsr()

    def predict(self, text: str) -> tuple[str, float]:
        if self.head is None:
            raise RuntimeError("TrainedIntent has not been fitted.")
        probabilities = self.head.predict_proba(self._features(text))[0]
        index = int(probabilities.argmax())
        intent = self.head.classes_[index]
        confidence = float(probabilities[index])
        if confidence < self.min_prob:
            return UNKNOWN, confidence
        return str(intent), confidence


def label_text(intent: str, lang: str = "en") -> str:
    """Display name for an intent label, in English or Chinese."""
    if intent == UNKNOWN:
        return "超出範圍" if lang == "zh" else "Out of scope"
    if lang == "zh":
        return INTENTS_ZH.get(intent, intent)
    return INTENTS.get(intent, intent)


def evaluate_intent(
    entries: Sequence[dict],
    tests: Sequence[dict],
    min_prob: float = 1.0 / 12.0,
) -> dict:
    """Score both classifiers on paraphrase queries never used for training."""
    from sklearn.metrics import precision_recall_fscore_support

    intent_of = {e["id"]: e.get("intent") for e in entries}
    in_scope = [
        t for t in tests
        if t["relevant"] and intent_of.get(t["relevant"][0]) in INTENTS
    ]
    out_scope = [t for t in tests if not t["relevant"]]
    if not in_scope:
        raise ValueError("No intent-labelled test queries.")

    rule = RuleIntent()
    trained = TrainedIntent(min_prob=min_prob).fit(entries)

    rows = []
    for item in in_scope:
        truth = intent_of[item["relevant"][0]]
        rule_intent, rule_conf = rule.predict(item["query"])
        trained_intent, trained_conf = trained.predict(item["query"])
        rows.append({
            "query_id": item["id"],
            "query": item["query"],
            "truth": truth,
            "rule": rule_intent,
            "rule_confidence": round(rule_conf, 2),
            "trained": trained_intent,
            "trained_confidence": round(trained_conf, 2),
        })

    labels = list(INTENTS)
    truth = [r["truth"] for r in rows]
    scores: dict[str, dict] = {}
    per_class_metrics: dict[str, tuple] = {}
    support: list[int] = []

    for model in ("rule", "trained"):
        predicted = [r[model] for r in rows]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=labels, zero_division=0, average=None
        )
        per_class_metrics[model] = (precision, recall)
        scores[model] = {
            "Accuracy": round(
                sum(1 for t, p in zip(truth, predicted) if t == p) / len(truth), 3
            ),
            "Macro precision": round(float(precision.mean()), 3),
            "Macro recall": round(float(recall.mean()), 3),
            "Macro F1": round(float(f1.mean()), 3),
            "Unknown rate": round(
                sum(1 for p in predicted if p == UNKNOWN) / len(predicted), 3
            ),
            "Out-of-scope declined": _unknown_on_queries(
                out_scope, rule if model == "rule" else trained
            ),
        }

    rule_p, rule_r = per_class_metrics["rule"]
    trained_p, trained_r = per_class_metrics["trained"]
    per_class = [
        {
            "intent": intent,
            "label": INTENTS[intent],
            "support": int(support[i]),
            "rule P": round(float(rule_p[i]), 3),
            "rule R": round(float(rule_r[i]), 3),
            "trained P": round(float(trained_p[i]), 3),
            "trained R": round(float(trained_r[i]), 3),
        }
        for i, intent in enumerate(labels)
    ]

    return {
        "rows": rows,
        "scores": scores,
        "per_class": per_class,
        "in_scope": len(in_scope),
        "out_scope": len(out_scope),
    }


def _unknown_on_queries(queries: Sequence[dict], model) -> float:
    if not queries:
        return float("nan")
    hits = sum(1 for q in queries if model.predict(q["query"])[0] == UNKNOWN)
    return round(hits / len(queries), 3)

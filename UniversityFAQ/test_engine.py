"""Plain-assert checks for faq_engine and intent.

Run: python test_engine.py

No pytest dependency, so it works inside the project venv as shipped. The
thresholds encode measured behaviour of the current corpus, not aspirations:
if a change moves the numbers, these checks should fail loudly.
"""

import json
import sys
from pathlib import Path

import faq_engine as fe
import intent as it

BASE = Path(__file__).resolve().parent
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("PASS  " if condition else "FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


# -- tokenisation ----------------------------------------------------------
check("english words tokenised", fe.tokens("How do I apply?")[:3] == ["how", "do", "i"])
zh = fe.tokens("延期申请")
check("chinese unigrams and bigrams", zh == ["延", "期", "申", "请", "延期", "期申", "申请"], str(zh))
check("language detection", fe.lang("作业延期") == "zh" and fe.lang("extension") == "en")

# -- content terms / stem ---------------------------------------------------
check("stop words removed", fe.content_terms("What is the weather tomorrow?") == ["weather", "tomorrow"],
      str(fe.content_terms("What is the weather tomorrow?")))
check("light stemming", fe._stem("examination") == fe._stem("exam") == "exam")

# -- corpus shape -----------------------------------------------------------
faq = fe.load_faq()
tests = fe.load_tests()
doc_tokens = [fe.tokens("\n".join(x[k] for k in fe.REQUIRED_FIELDS[1:5])) for x in faq]
check("corpus has 48 entries", len(faq) == 48, str(len(faq)))
check("every entry carries a known intent",
      all(x.get("intent") in it.INTENTS for x in faq))
check("twelve intents, four entries each",
      len({x["intent"] for x in faq}) == 12, str(len({x["intent"] for x in faq})))
check("twelve entries per intent",
      all(list(x["intent"] for x in faq).count(k) == 4 for k in it.INTENTS))
demo_entries = [x for x in faq if "demo" in x["source"].lower()]
check("provenance split is 21 official-only and 27 with a demo clause",
      (len(faq) - len(demo_entries), len(demo_entries)) == (21, 27),
      f"{len(faq) - len(demo_entries)}/{len(demo_entries)}")
check("test set has 96 answerable and 12 out-of-scope queries",
      sum(1 for t in tests if t["relevant"]) == 96
      and sum(1 for t in tests if not t["relevant"]) == 12, str(len(tests)))

# -- coverage separation ----------------------------------------------------
engine = fe.Engine(faq)
positive = []
for t in tests:
    if not t["relevant"]:
        continue
    hits, verdict, _ = engine.retrieve(t["query"], "TF-IDF", 5, fe.GuardPolicy(min_coverage=0.0))
    positive.append(verdict.coverage)
negative = []
for t in tests:
    if t["relevant"]:
        continue
    hits, verdict, _ = engine.retrieve(t["query"], "TF-IDF", 5, fe.GuardPolicy(min_coverage=0.0))
    negative.append(verdict.coverage)

share_above = sum(1 for c in positive if c >= fe.GuardPolicy().min_coverage) / len(positive)
check("most answerable queries clear the guard", share_above >= 0.70, f"{share_above:.0%}")
check("median answerable coverage is healthy",
      sorted(positive)[len(positive) // 2] >= 0.45,
      f"median={sorted(positive)[len(positive) // 2]:.2f}")
check("ten of twelve out-of-scope queries stay below the guard",
      sum(1 for c in negative if c < fe.GuardPolicy().min_coverage) >= 10,
      f"max={max(negative):.2f}")

# -- validation -------------------------------------------------------------
good = dict(faq[0])
for label, payload in (
    ("invalid id rejected", dict(good, id="001")),
    ("duplicate id rejected", [dict(good), dict(good)]),
    ("empty field rejected", dict(good, a_en="  ")),
    ("invalid intent rejected", dict(good, intent="not-an-intent")),
):
    try:
        fe.validate(payload if isinstance(payload, list) else [payload])
        check(label, False)
    except ValueError:
        check(label, True)

# -- engine + guard ---------------------------------------------------------
hits, verdict, ms = engine.retrieve("How do I apply for an assignment extension?", "TF-IDF", 3)
check("tf-idf retrieves the gold entry", hits[0]["id"] == "FAQ_001", hits[0]["id"])
check("in-scope question is answered", verdict.accepted)
check("latency measured", ms >= 0)

hits, verdict, _ = engine.retrieve("What is tomorrow's weather?", "TF-IDF", 3)
check("out-of-scope question is refused", not verdict.accepted)
check("refusal names the missing terms",
      {"weather", "tomorrow"} <= set(verdict.missing), str(verdict.missing))
check("refusal text is actionable", "contact" in fe.refusal_text(verdict).lower())

# Known limitation: a short Chinese negative can clear the guard on single
# characters (明 / 天 both occur in "考试当天生病"). Assert the documented
# behaviour so the weakness cannot grow silently.
hits, verdict, _ = engine.retrieve("明天会下雨吗？", "BM25", 3)
check("single-character false accept stays documented",
      verdict.coverage <= 0.45,
      f"coverage={verdict.coverage:.2f} accepted={verdict.accepted}")

# -- evidence selection -----------------------------------------------------
hits, verdict, _ = engine.retrieve("I want to borrow a book from the library", "TF-IDF", 3)
check("answer comes from the best-supported hit",
      hits[verdict.answer_index]["coverage"] >= max(h["coverage"] for h in hits),
      f"chosen={hits[verdict.answer_index]['id']}")

# -- intent routing ---------------------------------------------------------
allowed = engine.intent_index["LIBRARY"]
hits, verdict, _ = engine.retrieve("Can I renew a book?", "TF-IDF", 3, allowed=allowed)
check("filtered retrieval stays inside the intent",
      all(h["intent"] == "LIBRARY" for h in hits), str([h["id"] for h in hits]))
hits, verdict, _ = engine.retrieve("Can I renew a book?", "TF-IDF", 3, allowed=[])
check("empty candidate pool refuses without touching the index",
      not verdict.accepted and not hits, str(verdict.reasons))

# -- uploads: JSON Lines and CSV --------------------------------------------
import csv
import io

faq_lines = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in faq[:4]).encode()
check("json lines knowledge base parses", len(fe.parse_faq(faq_lines, "kb.jsonl")) == 4)

csv_buffer = io.StringIO()
writer = csv.DictWriter(
    csv_buffer,
    fieldnames=["id", "q_zh", "q_en", "a_zh", "a_en", "source", "intent"],
)
writer.writeheader()
for record in faq[:4]:
    writer.writerow(record)
csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")
check("csv knowledge base parses", len(fe.parse_faq(csv_bytes, "kb.csv")) == 4)
check("renamed json still parses", len(fe.parse_faq(faq_lines, "kb.json")) == 4)
check("csv keeps the intent column",
      fe.parse_faq(csv_bytes, "kb.csv")[0]["intent"] == faq[0]["intent"])

tests_buffer = io.StringIO()
writer = csv.DictWriter(tests_buffer, fieldnames=["id", "query", "relevant"])
writer.writeheader()
for record in tests[:3]:
    writer.writerow({**record, "relevant": "|".join(record["relevant"])})
parsed_tests = fe.parse_tests(tests_buffer.getvalue().encode("utf-8-sig"), "tests.csv")
check("csv test set splits relevant ids on a pipe",
      parsed_tests[0]["relevant"] == tests[0]["relevant"], str(parsed_tests[0]["relevant"]))

# -- uploads: document and photo routes -------------------------------------
import zipfile

qa_text = (
    "Q: How do I renew a library book?\n"
    "A: Renew it in your Library account before the due date.\n\n"
    "Q: \u7e8c\u501f\u8981\u5728\u4f55\u6642\u8fa6\u7406\uff1f\n"
    "A: \u8abf\u65bc\u5230\u671f\u65e5\u524d\u81ea\u884c\u7e8c\u501f\u3002\n"
).encode("utf-8")
doc_records = fe.parse_faq(qa_text, "library_faq.txt")
check("text file yields one record per Q/A pair", len(doc_records) == 2)
check("document records get generated ids",
      doc_records[0]["id"].startswith("FAQ_LIBRARYFAQ_"))
check("document keeps the chinese question",
      doc_records[1]["q_zh"].startswith("\u7e8c\u501f"), doc_records[1]["q_zh"][:12])
check("document source names the file",
      "library_faq.txt" in doc_records[0]["source"])

docx_buffer = io.BytesIO()
with zipfile.ZipFile(docx_buffer, "w") as archive:
    archive.writestr(
        "word/document.xml",
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Q: Can I pay by instalments?</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>A: Yes, where the debit note is issued in instalments.</w:t></w:r></w:p>"
        "</w:body></w:document>",
    )
docx_records = fe.parse_faq(docx_buffer.getvalue(), "fees.docx")
check("docx yields a record", len(docx_records) == 1)
check("docx text keeps the question",
      docx_records[0]["q_en"] == "Can I pay by instalments?")

for bad_name, bad_raw in (
    ("faq.xlsx", b"anything"),
    ("notes.txt", b"prose with no question or answer markers"),
):
    try:
        fe.parse_faq(bad_raw, bad_name)
    except ValueError:
        check(f"rejects unusable upload {bad_name}", True)
    else:
        check(f"rejects unusable upload {bad_name}", False)

try:
    fe.parse_tests(qa_text, "tests.docx")
except ValueError:
    check("test set refuses documents", True)
else:
    check("test set refuses documents", False)

# -- bilingual strings ------------------------------------------------------
zh_refusal = fe.refusal_text(
    fe.Verdict(accepted=False, coverage=0.2,
               reasons=[{"code": "coverage", "value": 0.2}], missing=["weather"]),
    lang="zh",
)
check("chinese refusal text is chinese", any("\u4e00" <= c <= "\u9fff" for c in zh_refusal))
check("chinese answer comes from a_zh",
      fe.answer_from_evidence(faq[0], "zh").startswith(faq[0]["a_zh"][:12]))
check("chinese intent label resolved", it.label_text("LIBRARY", "zh") == "圖書館")
check("english intent label still resolved", it.label_text("LIBRARY", "en") == "Library")
check("guard reason renders in chinese",
      "涵蓋" in fe.format_reason({"code": "coverage", "value": 0.2}, "zh"))

# -- evaluation -------------------------------------------------------------
rows, summary = fe.evaluate(engine, tests, ["TF-IDF", "BM25"], k=5)
check("evaluation covers every test query per method",
      len(rows) == len(tests) * 2, str(len(rows)))
check("summary has both measures", {"Recall@1", "Refusal acc."} <= set(summary[0]))
check("summary carries the answer-overlap and end-to-end measures",
      {"BLEU", "ROUGE-L", "Task success"} <= set(summary[0]))

# -- answer overlap metrics -------------------------------------------------
check("identical answers score perfect BLEU", fe.bleu("Submit the form to the registry.",
                                                      "Submit the form to the registry.") == 1.0)
check("unrelated answers score zero BLEU",
      fe.bleu("Submit the form to the registry.", "Renew the book at the counter.") == 0.0)
check("identical answers score perfect ROUGE-L",
      fe.rouge_l("Submit the form to the registry.", "Submit the form to the registry.") == 1.0)
partial = fe.rouge_l("Submit the form to the registry.", "Submit the form.")
check("a partial answer scores between zero and one", 0.0 < partial < 1.0, f"{partial:.3f}")
check("chinese answers are scored by character overlap",
      fe.rouge_l("請於五日內提交表格。", "請於五日內提交") > 0.5)
check("task success stays a probability",
      all(0.0 <= s["Task success"] <= 1.0 for s in summary),
      str([s["Task success"] for s in summary]))
check("refusal accuracy stays high on the demo negatives",
      all(s["Refusal acc."] >= 0.80 for s in summary),
      str([s["Refusal acc."] for s in summary]))
check("recall is no longer saturated", all(s["Recall@1"] < 0.95 for s in summary),
      str([s["Recall@1"] for s in summary]))

routers = [("None", None), ("Rule", it.RuleIntent()), ("Trained", it.TrainedIntent().fit(faq))]
rows, summary = fe.evaluate(engine, tests, ["TF-IDF"], k=5, routers=routers)
check("every router and method pair is summarised", len(summary) == 3, str(len(summary)))
check("routing is recorded per query",
      {"None", "Rule", "Trained"} <= {r["router"] for r in rows})
trained_row = next(s for s in summary if s["router"] == "Trained")
none_row = next(s for s in summary if s["router"] == "None")
check("hard filtering costs recall, as measured",
      trained_row["Recall@1"] <= none_row["Recall@1"] + 1e-9,
      f"{trained_row['Recall@1']:.3f} vs {none_row['Recall@1']:.3f}")

# -- intent classifiers -----------------------------------------------------
rule = it.RuleIntent()
check("rule matcher finds the extension intent",
      rule.predict("I need more time for my coursework")[0] == "ASSESSMENT_EXTENSION")
check("rule matcher declines an unrelated question",
      rule.predict("What is tomorrow's weather?")[0] == it.UNKNOWN)
check("every intent has triggers", all(rule.triggers(k) for k in it.INTENTS))

result = it.evaluate_intent(faq, tests)
check("intent evaluation covers every answerable query",
      result["in_scope"] == 96, str(result["in_scope"]))
check("both classifiers are scored",
      set(result["scores"]) == {"rule", "trained"}, str(set(result["scores"])))
check("trained classifier beats the rule matcher on accuracy",
      result["scores"]["trained"]["Accuracy"] > result["scores"]["rule"]["Accuracy"],
      f"{result['scores']['trained']['Accuracy']:.3f} vs "
      f"{result['scores']['rule']['Accuracy']:.3f}")
check("rule matcher is the more precise of the two",
      result["scores"]["rule"]["Macro precision"]
      > result["scores"]["trained"]["Macro precision"],
      f"{result['scores']['rule']['Macro precision']:.3f} vs "
      f"{result['scores']['trained']['Macro precision']:.3f}")

print("\nmethod  R@1    R@3    R@5    MRR@5  answer  refusal  latency")
rows, summary = fe.evaluate(engine, tests, ["TF-IDF", "BM25"], k=5)
for s in summary:
    print(f"{s['method']:7s} {s['Recall@1']:.3f}  {s['Recall@3']:.3f}  "
          f"{s['Recall@5']:.3f}  {s['MRR@5']:.3f}  {s['Answer rate']:.3f}   "
          f"{s['Refusal acc.']:.3f}    {s['Latency (ms)']:.2f}")

print()
print("intent  accuracy  macro P  macro R  macro F1  unknown  oos declined")
for name, values in it.evaluate_intent(faq, tests)["scores"].items():
    print(f"{name:8s}{values['Accuracy']:8.3f}{values['Macro precision']:9.3f}"
          f"{values['Macro recall']:9.3f}{values['Macro F1']:10.3f}"
          f"{values['Unknown rate']:9.3f}{values['Out-of-scope declined']:12.3f}")

print()
print(f"{len(failures)} failure(s)" if failures else "All checks passed.")
sys.exit(1 if failures else 0)

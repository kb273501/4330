"""Streamlit front-end for the University FAQ assistant.

UI only: retrieval, guard, evaluation and intent logic live in faq_engine.py and
intent.py; interface strings live in i18n.py. The knowledge base can be loaded
as JSON, JSON Lines or CSV.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from faq_engine import (
    DATA_DIR,
    FAQ_FILE_TYPES,
    METHODS,
    TEST_FILE_TYPES,
    Engine,
    GuardPolicy,
    answer_from_evidence,
    evaluate,
    format_reason,
    parse_faq,
    parse_tests,
    refusal_text,
)
from i18n import LANGUAGES, pack
from intent import (
    INTENTS,
    UNKNOWN,
    RuleIntent,
    TrainedIntent,
    evaluate_intent,
    label_text,
)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"

st.set_page_config(
    page_title="University FAQ Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1200px;}
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6ebf1;}
h1, h2, h3 {color: #12212f; letter-spacing: -0.01em;}
p, li {color: #24333f;}
.tag {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.72rem;
      font-weight: 600; letter-spacing: 0.02em; margin: 0 5px 6px 0;
      background: #eef3f7; color: #31465a; border: 1px solid #dbe4ec;}
.tag.ok {background: #e8f5ef; color: #0b6b5f; border-color: #bfe3d4;}
.tag.warn {background: #fdf1e3; color: #8a5300; border-color: #f3d8b4;}
.tag.bad {background: #fdeaea; color: #9b2226; border-color: #f6c9ca;}
.kv {color: #5b6b7b; font-size: 0.79rem; line-height: 1.5;}
.card {border: 1px solid #e6ebf1; border-radius: 14px; padding: 1rem 1.15rem;
       background: #ffffff; box-shadow: 0 1px 2px rgba(16, 32, 48, 0.04);}
.rule {border: none; border-top: 2px solid #0b6b5f; margin: 0.2rem 0 1.1rem 0; width: 46px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

OUTCOME_KEYS = {
    "correct": "out.correct",
    "correct in top-3": "out.top3",
    "miss": "out.miss",
    "refused (should answer)": "out.refused",
    "answered out-of-scope": "out.oos",
    "correctly refused": "out.refused_ok",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tag(text: str, kind: str = "") -> str:
    return f'<span class="tag {kind}">{text}</span>'


def badges(items: list[tuple[str, str]]) -> None:
    st.markdown("".join(tag(t, k) for t, k in items), unsafe_allow_html=True)


def safe_csv(frame: pd.DataFrame) -> bytes:
    """Neutralise spreadsheet formula injection before export."""
    frame = frame.copy()
    for col in frame.select_dtypes(include=["object"]).columns:
        frame[col] = frame[col].map(
            lambda x: "'" + x
            if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@"))
            else x
        )
    return frame.to_csv(index=False).encode("utf-8-sig")


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def classifiers(
    digest: str, _data: list[dict]
) -> tuple[RuleIntent, TrainedIntent | None]:
    """The trained model needs intent labels; a plain document upload has none."""
    trained: TrainedIntent | None
    try:
        trained = TrainedIntent().fit(_data)
    except ValueError:
        trained = None
    return RuleIntent(), trained


def rag_answer(query: str, hits: list[dict], model: str, lang: str, t) -> tuple[str, bool]:
    """Generate from retrieved evidence, then verify language and citations."""
    evidence = [
        {
            "id": h["id"],
            "question": h["q_zh"] if lang == "zh" else h["q_en"],
            "answer": h["a_zh"] if lang == "zh" else h["a_en"],
            "source": h["source"],
        }
        for h in hits
    ]
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": t("rag.system")},
            {
                "role": "user",
                "content": t("rag.evidence_label") + "\n"
                + json.dumps(evidence, ensure_ascii=False, indent=1)
                + "\n" + t("rag.question_label") + "\n" + query,
            },
        ],
        "options": {"temperature": 0, "num_predict": 400},
    }
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(10, 180))
    response.raise_for_status()
    answer = (response.json().get("message", {}).get("content") or "").strip()
    if not answer:
        raise ValueError(t("rag.empty"))

    allowed = {h["id"] for h in hits}
    cited = set(re.findall(r"\[(FAQ_[A-Za-z0-9_-]+)\]", answer))
    if cited - allowed:
        return t("rag.withheld_citation"), True
    if lang == "zh" and not has_cjk(answer):
        return t("rag.withheld_language"), True
    if lang == "en" and has_cjk(answer):
        return t("rag.withheld_language"), True
    return answer, False


def run_turn(
    query: str,
    engine: Engine,
    method: str,
    mode: str,
    k: int,
    policy: GuardPolicy,
    model: str,
    lang: str,
    t,
    allowed: list[int] | None = None,
) -> dict:
    started = time.perf_counter()
    hits, verdict, retrieval_ms = engine.retrieve(query, method, k, policy, allowed)
    guarded = False

    if not verdict.accepted:
        answer = refusal_text(verdict, lang)
    elif mode == "Retrieval-only":
        answer = answer_from_evidence(hits[verdict.answer_index], lang)
    else:
        answer, guarded = rag_answer(query, hits, model, lang, t)

    return {
        "role": "assistant",
        "content": answer,
        "method": method,
        "mode": mode,
        "refused": not verdict.accepted,
        "withheld": guarded,
        "coverage": verdict.coverage,
        "reasons": verdict.reasons,
        "hits": hits,
        "answer_index": verdict.answer_index,
        "retrieval_ms": retrieval_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
        "at": datetime.now().strftime("%H:%M:%S"),
    }


def render_turn(message: dict, lang: str, t) -> None:
    st.markdown(message["content"])

    if message["role"] != "assistant":
        return

    kind = "bad" if message["refused"] else ("warn" if message["withheld"] else "ok")
    state = t("badge.refused") if message["refused"] else t("badge.answered")
    percent = f"{message['coverage']:.0%}"
    chips = [
        (message["method"], ""),
        (message["mode"], ""),
        (f"{state} · {t('badge.coverage', p=percent)}", kind),
        (t("badge.retrieval", ms=f"{message['retrieval_ms']:.0f}"), ""),
    ]
    if message.get("intent"):
        intent_kind = "warn" if message["intent"] == UNKNOWN else ""
        chips.insert(0, (
            f"{label_text(message['intent'], lang)} {message['intent_conf']:.0%}",
            intent_kind,
        ))
    badges(chips)

    if message["withheld"]:
        st.warning(t("rag.failed_banner"))

    with st.expander(t("ask.evidence", n=len(message["hits"]))):
        for reason in message.get("reasons", []):
            st.caption(t("ask.guard_reason", text=format_reason(reason, lang)))
        if not message["hits"]:
            st.caption(t("ask.no_candidates"))
        else:
            chosen = message.get("answer_index", 0)
            table = pd.DataFrame([
                {
                    t("kb.col_id"): h["id"],
                    t("ask.used"): t("ask.used") if i == chosen else "",
                    t("col.score"): round(h["score"], 4),
                    t("col.coverage"): round(h["coverage"], 3),
                    t("kb.col_source"): h["source"],
                    t("kb.col_question"): h["q_zh"] if lang == "zh" else h["q_en"],
                }
                for i, h in enumerate(message["hits"])
            ])
            st.dataframe(table, hide_index=True, width="stretch")
            for i, h in enumerate(message["hits"]):
                marker = t("ask.used_marker") if i == chosen else ""
                st.markdown(f"**[{h['id']}]** "
                            f"{h['q_zh'] if lang == 'zh' else h['q_en']}{marker}")
                body = h["a_zh"] if lang == "zh" else h["a_en"]
                st.markdown(f'<div class="kv">{body}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: language first, then configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    picked = st.radio(
        "Language / 語言",
        list(LANGUAGES.values()),
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
        horizontal=True,
        help="Interface and answer language / 介面與回答語言",
    )
    lang = "en" if picked == LANGUAGES["en"] else "zh"
    st.session_state.lang = lang
    t = pack(lang)

with st.sidebar:
    st.markdown(f"### 🎓 {t('app.name')}")
    st.markdown(f'<div class="kv">{t("app.tagline")}</div>', unsafe_allow_html=True)
    st.markdown("---")

uploaded = st.sidebar.file_uploader(
    t("upload.faq_label"),
    type=FAQ_FILE_TYPES,
    help=t("upload.faq_help"),
)
try:
    if uploaded is not None:
        raw = uploaded.getvalue()
        data = parse_faq(raw, uploaded.name)
    else:
        raw = (DATA_DIR / "faq.json").read_bytes()
        data = parse_faq(raw, "faq.json")
except Exception as error:  # noqa: BLE001 - surfaced to the user
    st.error(t("err.kb", msg=error))
    st.stop()

digest = hashlib.sha256(raw).hexdigest() + (uploaded.name if uploaded else "")
if st.session_state.get("digest") != digest:
    st.session_state.engine = Engine(data)
    st.session_state.digest = digest
    st.session_state.messages = []
    st.session_state.pop("evaluation", None)
    st.session_state.pop("intent_evaluation", None)
engine: Engine = st.session_state.engine
if "messages" not in st.session_state:
    st.session_state.messages = []

dense_ok = Engine.dense_installed()
intent_count = len({x["intent"] for x in data if x.get("intent")})

with st.sidebar:
    st.markdown(f"#### {t('sec.retrieval')}")
    options = [m for m in METHODS if m != "Dense" or dense_ok]
    method = st.selectbox(
        t("retriever.label"),
        options,
        format_func=lambda m: {
            "TF-IDF": "TF-IDF", "BM25": "BM25",
            "Dense": "Dense (E5)", "Hybrid": "Hybrid (RRF)",
        }[m],
        help=t("retriever.help"),
    )
    if not dense_ok:
        st.caption(t("retriever.dense_hidden"))
    mode_options = ["Retrieval-only", "RAG"]
    mode = st.radio(
        t("mode.label"),
        mode_options,
        format_func=lambda m: t("mode.retrieval") if m == "Retrieval-only" else t("mode.rag"),
        horizontal=True,
        help=t("mode.help"),
    )
    k = st.slider(t("topk.label"), 1, min(10, len(data)), min(3, len(data)))

    st.markdown(f"#### {t('sec.intent')}")
    if intent_count:
        model_options = ["Rule-based", "Trained"]
        intent_model = st.selectbox(
            t("intent.model_label"),
            model_options,
            format_func=lambda m: t("intent.rule") if m == "Rule-based" else t("intent.trained"),
        )
        routing_options = ["Off", "Advisory", "Filter"]
        routing = st.selectbox(
            t("intent.routing_label"),
            routing_options,
            format_func=lambda m: {
                "Off": t("intent.off"),
                "Advisory": t("intent.advisory"),
                "Filter": t("intent.filter"),
            }[m],
            help=t("intent.help"),
        )
    else:
        intent_model, routing = "Rule-based", "Off"
        st.caption(t("intent.none"))

    st.markdown(f"#### {t('sec.guard')}")
    st.markdown(f'<div class="kv">{t("guard.note")}</div>', unsafe_allow_html=True)
    min_coverage = st.slider(t("guard.coverage"), 0.10, 0.80, 0.35, 0.05)
    min_dense = st.slider(t("guard.dense"), 0.20, 0.90, 0.42, 0.02)
    policy = GuardPolicy(min_coverage=min_coverage, min_dense=min_dense)

    st.markdown(f"#### {t('sec.models')}")
    models = ollama_models()
    if models:
        st.markdown(
            tag(t("ollama.ok"), "ok") + tag(t("ollama.count", n=len(models)), ""),
            unsafe_allow_html=True,
        )
        model = st.selectbox(t("ollama.model"), models, index=0)
    else:
        st.markdown(tag(t("ollama.down"), "warn"), unsafe_allow_html=True)
        st.caption(t("ollama.hint"))
        model = st.text_input(t("ollama.model"), DEFAULT_MODEL)

    st.markdown("---")
    st.markdown(f'<div class="kv">{t("sidebar.footer")}</div>', unsafe_allow_html=True)

rule_clf, trained_clf = classifiers(digest, data)
active_clf = rule_clf if (intent_model == "Rule-based" or trained_clf is None) else trained_clf

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(t("app.name"))
st.markdown('<hr class="rule">', unsafe_allow_html=True)
demo = any("demo" in x["source"].lower() for x in data)
badges([
    (t("chip.entries", n=len(data)), ""),
    (
        t("chip.intents", n=intent_count) if intent_count else t("chip.no_intents"),
        "" if intent_count else "warn",
    ),
    (t("chip.retrievers", n=len(METHODS)), ""),
    (t("chip.guard", p=f"{min_coverage:.0%}"), ""),
])
if demo:
    st.markdown(f'<div class="kv">{t("header.demo")}</div>', unsafe_allow_html=True)
st.write("")

ask_tab, kb_tab, intent_tab, eval_tab, help_tab = st.tabs(
    [t("tab.ask"), t("tab.kb"), t("tab.intents"), t("tab.eval"), t("tab.help")]
)

# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

with ask_tab:
    if not st.session_state.messages:
        st.markdown(t("ask.intro"))
        examples = [t(f"ask.ex{i}") for i in (1, 2, 3, 4)]
        cols = st.columns(len(examples))
        for i, (col, example) in enumerate(zip(cols, examples), 1):
            if col.button(example, width="stretch", key=f"example_{lang}_{i}"):
                st.session_state.pending_query = example
                st.rerun()

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            render_turn(message, lang, t)
            if message["role"] == "assistant":
                if hasattr(st, "feedback"):
                    rating = st.feedback("stars", key=f"rate_{index}")
                else:
                    rating = st.select_slider(
                        t("ask.rating"), options=[1, 2, 3, 4, 5],
                        value=3, key=f"rate_{index}",
                    )
                if rating:
                    message["rating"] = int(rating)

    query = st.chat_input(t("ask.placeholder"))
    if not query:
        query = st.session_state.pop("pending_query", None)

    if query:
        if len(query) > 4000:
            st.error(t("ask.too_long"))
        elif mode == "RAG" and not models:
            st.error(t("ask.rag_offline"))
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner(t("ask.spinner")):
                try:
                    intent, confidence = active_clf.predict(query)
                    allowed = (
                        engine.intent_index.get(intent, [])
                        if routing == "Filter" and intent_count
                        else None
                    )
                    answer = run_turn(
                        query, engine, method, mode, k, policy, model, lang, t, allowed
                    )
                    answer["intent"] = intent
                    answer["intent_conf"] = confidence
                except Exception as error:  # noqa: BLE001 - surfaced to the user
                    st.session_state.messages.pop()
                    st.error(t("ask.error", msg=error))
                else:
                    st.session_state.messages.append(answer)
            st.rerun()

    if st.session_state.messages:
        st.markdown("")
        col_a, col_b, col_c, _ = st.columns([1, 1, 1, 1.4])
        if col_a.button(t("ask.clear"), width="stretch"):
            for widget_key in [w for w in st.session_state if w.startswith("rate_")]:
                del st.session_state[widget_key]
            st.session_state.messages = []
            st.rerun()
        export = [
            {key: m[key] for key in
             ("role", "content", "method", "mode", "intent", "at", "rating")
             if key in m}
            for m in st.session_state.messages
        ]
        col_b.download_button(
            t("ask.export"),
            json.dumps(export, ensure_ascii=False, indent=2),
            file_name="conversation.json",
            mime="application/json",
            width="stretch",
        )
        rated = [
            {
                "at": m.get("at"),
                "question": st.session_state.messages[i - 1]["content"] if i else "",
                "method": m.get("method"),
                "mode": m.get("mode"),
                "intent": m.get("intent"),
                "refused": m.get("refused"),
                "coverage": m.get("coverage"),
                "rating": m["rating"],
            }
            for i, m in enumerate(st.session_state.messages)
            if m["role"] == "assistant" and "rating" in m
        ]
        if rated:
            col_c.download_button(
                t("ask.export_ratings"),
                safe_csv(pd.DataFrame(rated)),
                file_name="user_study.csv",
                mime="text/csv",
                width="stretch",
            )

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

with kb_tab:
    st.markdown(t("kb.intro"))
    search = st.text_input(
        t("kb.filter"), placeholder=t("kb.filter_hint"), key=f"kb_filter_{lang}"
    )
    rows = [
        {
            t("kb.col_id"): x["id"],
            t("kb.col_intent"): label_text(x["intent"], lang) if x.get("intent") else "—",
            t("kb.col_question"): x["q_zh"] if lang == "zh" else x["q_en"],
            t("kb.col_answer"): x["a_zh"] if lang == "zh" else x["a_en"],
            t("kb.col_source"): x["source"],
        }
        for x in data
    ]
    frame = pd.DataFrame(rows)
    if search.strip():
        mask = frame.apply(
            lambda r: search.lower() in " ".join(map(str, r.values)).lower(), axis=1
        )
        frame = frame[mask]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(t("kb.count", shown=len(frame), total=len(data)))
    with st.expander(t("kb.bilingual")):
        st.dataframe(
            pd.DataFrame(
                [{"id": x["id"], "q_zh": x["q_zh"], "q_en": x["q_en"], "a_zh": x["a_zh"]}
                 for x in data]
            ),
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        t("kb.download"),
        json.dumps(data, ensure_ascii=False, indent=2),
        file_name="faq.json",
        mime="application/json",
    )
    st.caption(t("kb.note"))

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

with intent_tab:
    st.markdown(t("intents.intro"))
    if not intent_count:
        st.warning(t("intents.no_intents"))
    else:
        st.caption(t("intents.counts", intents=intent_count, entries=len(data)))
        if st.button(t("intents.run"), type="primary"):
            try:
                tests = parse_tests((DATA_DIR / "tests.json").read_bytes(), "tests.json")
                with st.spinner(t("intents.spinner")):
                    st.session_state.intent_evaluation = evaluate_intent(data, tests)
            except Exception as error:  # noqa: BLE001
                st.error(str(error))

        if "intent_evaluation" in st.session_state:
            result = st.session_state.intent_evaluation
            scores = pd.DataFrame([
                {"model": name, **values}
                for name, values in result["scores"].items()
            ])
            scores["model"] = scores["model"].map(
                lambda m: t("intent.rule") if m == "rule" else t("intent.trained")
            )
            display = scores.rename(columns={
                "model": t("col.model"),
                "Accuracy": t("m.accuracy"),
                "Macro precision": t("m.precision"),
                "Macro recall": t("m.recall"),
                "Macro F1": t("m.f1"),
                "Unknown rate": t("m.unknown"),
                "Out-of-scope declined": t("m.oos"),
            })
            st.dataframe(display, hide_index=True, width="stretch")
            st.caption(t("intents.caption", in_scope=result["in_scope"],
                         out_scope=result["out_scope"]))

            per_class = pd.DataFrame(result["per_class"])
            per_class["label"] = per_class.apply(
                lambda r: label_text(r["intent"], lang), axis=1
            )
            st.dataframe(
                per_class.drop(columns=["intent"]).rename(columns={
                    "label": t("m.label"),
                    "support": t("m.support"),
                    "rule P": t("m.rule_p"),
                    "rule R": t("m.rule_r"),
                    "trained P": t("m.trained_p"),
                    "trained R": t("m.trained_r"),
                }),
                hide_index=True,
                width="stretch",
            )

            errors = [r for r in result["rows"]
                      if r["rule"] != r["truth"] or r["trained"] != r["truth"]]
            with st.expander(t("intents.mislabelled", n=len(errors))):
                error_frame = pd.DataFrame(errors)[["query", "truth", "rule", "trained"]]
                for column in ("truth", "rule", "trained"):
                    error_frame[column] = error_frame[column].map(
                        lambda v: label_text(v, lang)
                    )
                st.dataframe(
                    error_frame.rename(columns={
                        "query": t("col.query"),
                        "truth": t("intents.col_truth"),
                        "rule": t("intent.rule"),
                        "trained": t("intent.trained"),
                    }),
                    hide_index=True,
                    width="stretch",
                )
            st.download_button(
                t("intents.download"),
                safe_csv(scores),
                file_name="intent_metrics.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

with eval_tab:
    st.markdown(t("eval.intro"))
    test_upload = st.file_uploader(
        t("eval.tests_label"), type=TEST_FILE_TYPES, key=f"tests_{lang}"
    )
    eval_methods = [m for m in METHODS if m != "Dense" or dense_ok]
    selected = st.multiselect(
        t("eval.methods_label"),
        eval_methods,
        default=[m for m in ("TF-IDF", "BM25") if m in eval_methods],
        help=None if dense_ok else t("eval.need_dense"),
    )
    router_choice = st.multiselect(
        t("eval.routers_label"),
        ["none", "rule", "trained"],
        default=["none"],
        format_func=lambda r: {
            "none": t("eval.no_routing"),
            "rule": t("intent.rule"),
            "trained": t("intent.trained"),
        }[r],
        help=t("eval.routers_help"),
    )
    run = st.button(t("eval.run"), type="primary")

    if run:
        try:
            if not selected:
                raise ValueError(t("eval.need_one"))
            if "Dense" in selected and not dense_ok:
                raise ValueError(t("eval.need_dense"))
            if test_upload is not None:
                tests_raw = test_upload.getvalue()
                tests = parse_tests(tests_raw, test_upload.name)
            else:
                tests = parse_tests((DATA_DIR / "tests.json").read_bytes(), "tests.json")
            available = {"none": None, "rule": rule_clf}
            if trained_clf is not None:
                available["trained"] = trained_clf
            chosen = [r for r in router_choice if r in available] or ["none"]
            with st.spinner(t("eval.spinner")):
                rows, summary = evaluate(
                    engine, tests, selected, k=5, policy=policy,
                    routers=[(r, available[r]) for r in chosen],
                )
            st.session_state.evaluation = (rows, summary, len(tests), chosen)
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.error(str(error))

    if "evaluation" in st.session_state:
        rows, summary, total, chosen = st.session_state.evaluation
        st.caption(t("eval.caption", total=total, settings=len(chosen)))
        router_names = {
            "none": t("eval.no_routing"),
            "rule": t("intent.rule"),
            "trained": t("intent.trained"),
        }
        summary_frame = pd.DataFrame(summary)
        display_summary = summary_frame.rename(columns={
            "router": t("col.router"),
            "method": t("col.method"),
            "Recall@1": t("m.recall1"),
            "Recall@3": t("m.recall3"),
            "Recall@5": t("m.recall5"),
            "MRR@5": t("m.mrr"),
            "Answer rate": t("m.answer"),
            "Refusal acc.": t("m.refusal"),
            "BLEU": t("m.bleu"),
            "ROUGE-L": t("m.rouge"),
            "Task success": t("m.success"),
            "Latency (ms)": t("col.latency"),
        })
        display_summary[t("col.router")] = display_summary[t("col.router")].map(router_names)
        display_summary = display_summary[[
            t("col.router"), t("col.method"), t("m.recall1"), t("m.recall3"),
            t("m.recall5"), t("m.mrr"), t("m.answer"), t("m.refusal"),
            t("m.bleu"), t("m.rouge"), t("m.success"), t("col.latency"),
        ]]
        st.dataframe(
            display_summary,
            hide_index=True,
            width="stretch",
            column_config={
                t("m.recall1"): st.column_config.NumberColumn(format="%.3f"),
                t("m.refusal"): st.column_config.NumberColumn(format="%.3f"),
                t("m.bleu"): st.column_config.NumberColumn(format="%.3f"),
                t("m.rouge"): st.column_config.NumberColumn(format="%.3f"),
                t("m.success"): st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(t("eval.metric_note"))
        chart = summary_frame.assign(
            setting=summary_frame["router"].map(router_names)
            + " · " + summary_frame["method"]
        ).set_index("setting")[["Recall@1", "MRR@5", "Refusal acc.", "Task success"]].fillna(0)
        chart.columns = [t("m.recall1"), t("m.mrr"), t("m.refusal"), t("m.success")]
        st.bar_chart(chart)

        detail = pd.DataFrame(rows)

        def colour(value: str) -> str:
            positive = (t("out.correct"), t("out.top3"), t("out.refused_ok"))
            neutral = (t("out.refused"), t("out.miss"))
            if value in positive:
                return "background-color: #e8f5ef; color: #0b6b5f"
            if value in neutral:
                return "background-color: #fdf1e3; color: #8a5300"
            return "background-color: #fdeaea; color: #9b2226"

        shown = pd.DataFrame({
            t("col.router"): detail["router"].map(router_names),
            t("col.method"): detail["method"],
            t("kb.col_id"): detail["query_id"],
            t("col.query"): detail["query"],
            t("col.intent"): detail["routed_intent"].map(
                lambda v: t("eval.no_routing") if v == "all" else label_text(str(v), lang)
            ),
            t("col.outcome"): detail["outcome"].map(lambda v: t(OUTCOME_KEYS.get(v, v))),
            t("col.coverage"): detail["coverage"],
            t("col.latency"): detail["latency_ms"],
        })
        st.dataframe(
            shown.style.map(colour, subset=[t("col.outcome")]),
            hide_index=True,
            width="stretch",
        )
        st.download_button(t("eval.download_metrics"), safe_csv(summary_frame),
                           file_name="metrics.csv", mime="text/csv")
        st.download_button(t("eval.download_rows"), safe_csv(detail),
                           file_name="predictions.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

HELP_EN = """
**Pipeline**

1. A question is classified into one of twelve intents, either by a keyword matcher
   or by a trained classifier. Routing can be off, advisory, or a hard filter.
2. The question is tokenised — ASCII words, plus Chinese characters and character
   bigrams — so one index serves both languages.
3. One of four retrievers ranks the FAQ entries: TF-IDF, BM25, dense multilingual-E5
   embeddings, or a reciprocal-rank fusion of dense and BM25.
4. A refusal guard checks term coverage of the best-supported candidate, and dense
   similarity when embeddings are used. If the evidence is thin, the assistant refuses
   and names the terms it could not find instead of guessing.
5. In Retrieval-only mode the curated answer is returned with its source. In RAG mode
   a local Ollama model writes from the retrieved evidence, and the reply is checked
   for out-of-set citations and for language before it is shown.

**Answers and sources**

Every answer follows the University's reply pattern: the office that decides, the
form to complete, the evidence required, the deadline, then the enquiry channel.
The `source` column names the policy page or form the answer rests on; entries whose
source starts with `Demo policy` are illustrative and not verified university
policy, and the page banner says so.

**Uploads**

The knowledge base accepts JSON, JSON Lines or CSV, or a PDF, Word, text or photo
file containing Q/A blocks. In a CSV the knowledge base needs the columns
`id, q_zh, q_en, a_zh, a_en, source` and optionally `intent`; the test set needs
`id, query, relevant`, where several relevant ids are separated by `|`. In a document
each pair starts with `Q:` and `A:` (or `問：` and `答：`); a photo needs OCR, so
install tesseract first. Uploads apply to this session only.

**Metrics**

- `Recall@k` — share of the labelled FAQ answers found in the top *k*.
- `MRR@5` — mean of 1/rank of the first relevant hit, capped at 5.
- `Answer rate` — share of answerable questions actually answered (over-refusal).
- `Refusal acc.` — share of out-of-scope questions correctly refused.
- `Accuracy`, macro `P/R/F1` — intent classification on paraphrase queries absent
  from training.
- `Latency` — retrieval time, excluding generation.

**Known limitations**

- Forty-eight synthetic entries, all policies fictional. Treat the scores as a
  protocol demonstration, not as findings about the real university.
- Four training questions per intent: the trained classifier generalises unevenly
  and has no abstain class.
- Hard intent filtering costs recall; evaluation is reported with and without routing.
- The coverage threshold was calibrated on the same queries it is reported on, so the
  figures are optimistic; a separate development split is still needed.
- Single-turn, no BLEU / ROUGE-L, no authentication, localhost only.
"""

HELP_ZH = """
**處理流程**

1. 問題先歸類到十二個意圖之一，可用關鍵詞比對或訓練式分類器。路由可設為關閉、提示或硬過濾。
2. 斷詞同時處理英文詞、中文字與中文二字組合，因此一個索引可同時服務兩種語言。
3. 四種檢索方法排序 FAQ 條目：TF-IDF、BM25、E5 向量，以及向量與 BM25 的倒數排名融合。
4. 拒答守衛檢查「支撐最強的候選」有多少問題實詞被涵蓋（使用向量時另檢查相似度）。證據不足就拒答，並列出缺少的詞，而不是猜。
5. 僅檢索模式回傳既有答案與來源；RAG 模式由本機 Ollama 依證據生成，生成後檢查引用與語言才顯示。

**答案與出處**

每條答案依大學回覆格式書寫：先寫負責部門，再寫須填的表格、所需證明、期限，最後是查詢渠道。`source` 欄列出答案所依據的政策頁或表格；`source` 以 `Demo policy` 開頭的條目屬示範內容，未經大學正式核實，頁首橫幅會標示。

**上傳格式**

知識庫接受 JSON、JSON Lines 或 CSV，亦可上傳含 Q/A 段落的 PDF、Word、文字檔或相片。CSV 知識庫需有 `id, q_zh, q_en, a_zh, a_en, source`，`intent` 選填；測試集需有 `id, query, relevant`，多個正確答案以 `|` 分隔。文件內每組以 `Q:`／`A:`（或`問：`／`答：`）開頭；相片須先安裝 tesseract 作 OCR。上傳只影響本次工作階段。

**指標**

- `Recall@k`：正確答案出現在前 k 名的比例。
- `MRR@5`：第一個正確答案排名倒數的平均，上限 5。
- `Answer rate`：可答題中實際作答的比例（衡量誤拒）。
- `Refusal acc.`：越界題中正確拒答的比例。
- `Accuracy`、macro `P/R/F1`：意圖分類在未參與訓練的改寫問句上的表現。
- `Latency`：檢索時間，不含生成。

**已知限制**

- 48 條條目全為虛構示範政策，分數只能視為流程示範，不能當作對真實大學的結論。
- 每個意圖只有 4 條訓練問句，訓練式模型泛化不均，且沒有棄答類別。
- 硬性意圖過濾會掉召回，評估同時報告有無路由的結果。
- 覆蓋率門檻是在同一批題目上校準的，數字偏樂觀，仍需要獨立驗證集。
- 單輪對話、無 BLEU / ROUGE-L、無身分驗證，僅限本機使用。
"""

with help_tab:
    st.markdown(HELP_ZH if lang == "zh" else HELP_EN)

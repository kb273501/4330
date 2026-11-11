"""Retrieval, refusal-guard and evaluation logic for the University FAQ assistant.

This module has no Streamlit imports, so the retrieval pipeline can be unit
tested and reused from a notebook or a command-line script.

Pipeline:  query -> tokenise -> score (TF-IDF | BM25 | Dense | Hybrid)
           -> rank -> coverage-based refusal guard -> answer or refusal.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

METHODS = ("TF-IDF", "BM25", "Dense", "Hybrid")
REQUIRED_FIELDS = ("id", "q_zh", "q_en", "a_zh", "a_en", "source")
EMBED_MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60

_CJK = re.compile(r"[\u3400-\u9fff]")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
_WORD = re.compile(r"[a-z0-9]+")
_FAQ_ID = re.compile(r"FAQ_[A-Za-z0-9_-]+")

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

STOP_EN = frozenset("""
a an the is are was were be been being am do does did done i my me mine you your
we our us they them it its this that these those how what when where which who
whom whose why can could should would may might must shall will to for of in on
at and or nor not no if then than so as by with from into over under about after
before between during through out up down have has had get got there here
s t d m ll re ve please
""".split())

STOP_ZH = frozenset(
    "的了是在我你他她它你们您这這那哪甚麼什么怎樣怎么怎样怎么為何为何啊呀吧呢嗎吗嗯哦"
    "请請帮幫能可否要不要没沒有有一二三四五六七八九十如何為为"
)


def lang(text: str) -> str:
    """Return 'zh' when the text contains any CJK character, else 'en'."""
    return "zh" if _CJK.search(text) else "en"


def tokens(text: str) -> list[str]:
    """Index-time tokens: ASCII words, plus CJK unigrams and bigrams."""
    low = text.lower()
    result = _WORD.findall(low)
    for part in _CJK_RUN.findall(text):
        result.extend(part)
        result.extend(part[i:i + 2] for i in range(len(part) - 1))
    return result


def content_terms(text: str) -> list[str]:
    """Query-time content terms, with function words removed.

    Used only by the refusal guard: it answers "does the knowledge base actually
    talk about what the user asked?", which raw similarity scores cannot.
    """
    terms = [w for w in _WORD.findall(text.lower()) if w not in STOP_EN]
    for part in _CJK_RUN.findall(text):
        terms.extend(ch for ch in part if ch not in STOP_ZH)
    return terms


def _stem(term: str) -> str:
    """Very light normalisation so exam/examination, date/dates still overlap."""
    if not term.isascii() or len(term) < 4:
        return term
    if term.endswith("s") and not term.endswith(("ss", "us", "is")):
        term = term[:-1]
    return term[:4]


def coverage(
    query: str,
    doc_tokens: Sequence[str],
    idf: dict[str, float] | None = None,
) -> float:
    """Share of the query's content terms that appear in one document.

    With an IDF table the share is weighted: a word that occurs in every entry
    (student, form, office) counts for little, a distinctive word counts for more.
    """
    terms = content_terms(query)
    if not terms:
        return 0.0
    keys = {_stem(t) for t in doc_tokens}
    if idf:
        weights = [idf.get(_stem(t), 1.0) for t in terms]
    else:
        weights = [1.0 for _ in terms]
    total = sum(weights)
    if total <= 0:
        return 0.0
    matched = sum(w for t, w in zip(terms, weights) if _stem(t) in keys)
    return matched / total


def missing_terms(query: str, doc_tokens: Sequence[str]) -> list[str]:
    keys = {_stem(t) for t in doc_tokens}
    return [t for t in content_terms(query) if _stem(t) not in keys]


# ---------------------------------------------------------------------------
# Data loading / validation
# ---------------------------------------------------------------------------


def validate(data: Any) -> list[dict]:
    if not isinstance(data, list) or not 1 <= len(data) <= 10000:
        raise ValueError("The FAQ file must contain between 1 and 10,000 records.")
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Invalid FAQ record: expected a JSON object.")
        for key in REQUIRED_FIELDS:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"FAQ record is missing a non-empty '{key}' field."
                )
        if not re.fullmatch(r"FAQ_[A-Za-z0-9_-]+", item["id"]):
            raise ValueError(
                f"Invalid FAQ id {item['id']!r}: ids must look like FAQ_001."
            )
        if item["id"] in seen:
            raise ValueError(f"Duplicate FAQ id: {item['id']}.")
        seen.add(item["id"])
        intent = item.get("intent")
        if intent is not None and (
            not isinstance(intent, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,39}", intent)
        ):
            raise ValueError(
                f"Invalid intent {intent!r}: use an upper-case label such as "
                "ASSESSMENT_EXTENSION, or omit the field."
            )
    return data


def load_faq(path: Path | None = None) -> list[dict]:
    raw = (path or DATA_DIR / "faq.json").read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"faq.json is not valid JSON: {error}") from error
    return validate(data)


def load_tests(path: Path | None = None) -> list[dict]:
    raw = (path or DATA_DIR / "tests.json").read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"tests.json is not valid JSON: {error}") from error
    if not isinstance(data, list) or not data:
        raise ValueError("The test file must be a non-empty JSON list.")
    return data


# ---------------------------------------------------------------------------
# Upload parsing: JSON, JSON Lines, CSV
# ---------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The uploaded file is not valid UTF-8 or UTF-16 text.")


def _split_records(text: str, filename: str) -> tuple[str, object]:
    """Return (kind, payload) where kind is 'json', 'jsonl' or 'csv'."""
    name = (filename or "").lower()
    if name.endswith((".jsonl", ".ndjson")) or name.endswith((".csv", ".tsv")):
        kind = "csv" if name.endswith((".csv", ".tsv")) else "jsonl"
    else:
        kind = "json"

    if kind == "json":
        try:
            return "json", json.loads(text)
        except json.JSONDecodeError as error:
            # Tolerate a file renamed .json that actually holds one object per line.
            try:
                records = [
                    json.loads(line) for line in text.splitlines() if line.strip()
                ]
            except json.JSONDecodeError:
                raise ValueError(f"The file is not valid JSON: {error}") from error
            if records:
                return "jsonl", records
            raise ValueError("The uploaded file is empty.") from error
    if kind == "jsonl":
        return "jsonl", [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    delimiter = "\t" if name.endswith(".tsv") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return "csv", rows


def _normalise_row(row: dict, array_fields: Sequence[str]) -> dict:
    clean: dict = {}
    for key, value in row.items():
        if key is None:
            continue
        field = str(key).strip().lower()
        if isinstance(value, str):
            value = value.strip()
        if field in array_fields:
            if isinstance(value, str):
                parts = [p.strip() for p in re.split(r"[|;、]", value) if p.strip()]
                clean[field] = parts
            elif isinstance(value, list):
                clean[field] = [str(v).strip() for v in value]
            elif value in (None, ""):
                clean[field] = []
        else:
            clean[field] = "" if value is None else value
    return clean


# Documents and images carry Q/A text rather than table rows.

DOC_EXTENSIONS = ("pdf", "doc", "docx", "txt", "md", "rtf")
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "tif", "tiff", "webp")
FAQ_FILE_TYPES = ["json", "jsonl", "ndjson", "csv", "tsv", *DOC_EXTENSIONS, *IMAGE_EXTENSIONS]
TEST_FILE_TYPES = ["json", "jsonl", "ndjson", "csv", "tsv"]

_QA_BLOCK = re.compile(
    r"(?:^|\n)\s*(?:Q|Question|問|問題)\s*(?:\d+)?\s*[::.]\s*(.+?)"
    r"\s*(?:A|Answer|答|答案)\s*(?:\d+)?\s*[::.]\s*(.+?)"
    r"(?=\n\s*(?:Q|Question|問|問題)\s*(?:\d+)?\s*[::.]|\Z)",
    re.S,
)


def _pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - depends on install
        raise ValueError(
            "Reading PDF uploads needs pypdf: pip install pypdf."
        ) from error
    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as error:  # noqa: BLE001 - any failure means locked
            raise ValueError("The PDF is password protected.") from error
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        try:
            document = archive.read("word/document.xml").decode("utf-8")
        except KeyError as error:
            raise ValueError("The .docx file has no word/document.xml.") from error
    document = document.replace("</w:p>", "\n").replace("</w:tr>", "\n")
    text = re.sub(r"<[^>]+>", "", document)
    return html.unescape(text)


def _legacy_doc_text(raw: bytes) -> str:
    if shutil.which("textutil") is None:
        raise ValueError(
            "Old .doc files need macOS textutil or LibreOffice; save as .docx first."
        )
    handle = tempfile.NamedTemporaryFile(suffix=".doc", delete=False)
    try:
        handle.write(raw)
        handle.close()
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", handle.name],
            capture_output=True,
            check=True,
        )
        return result.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "The .doc file could not be converted; save it as .docx or .txt."
        ) from error
    finally:
        os.unlink(handle.name)


def _image_text(raw: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise ValueError(
            "Reading photos needs OCR: pip install pytesseract pillow, then "
            "brew install tesseract."
        ) from error
    if shutil.which("tesseract") is None:
        raise ValueError(
            "The tesseract engine is not installed; run brew install tesseract."
        )
    return pytesseract.image_to_string(
        Image.open(io.BytesIO(raw)), lang="eng+chi_tra"
    )


def _extract_text(raw: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _pdf_text(raw)
    if name.endswith(".docx"):
        return _docx_text(raw)
    if name.endswith((".doc", ".rtf")):
        return _legacy_doc_text(raw)
    if name.endswith(tuple(f".{suffix}" for suffix in IMAGE_EXTENSIONS)):
        return _image_text(raw)
    if name.endswith((".txt", ".md")):
        return _decode(raw)
    raise ValueError(
        "Unsupported file type. Use JSON, JSON Lines or CSV, or a PDF, Word, "
        "text or photo file for the knowledge base."
    )


def _records_from_text(text: str, filename: str) -> list[dict]:
    """Turn document text into FAQ records by pairing Q/A blocks."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n")
    pairs = [
        (question.strip(), answer.strip())
        for question, answer in _QA_BLOCK.findall(lines)
        if question.strip() and answer.strip()
    ]
    if not pairs:
        # A two-column table pasted into a PDF or a slide also counts.
        for line in lines.splitlines():
            cells = [c.strip() for c in re.split(r"\t|\|", line) if c.strip()]
            if len(cells) >= 2 and len(cells[0]) > 6:
                pairs.append((cells[0], " ".join(cells[1:])))
    if not pairs:
        raise ValueError(
            "No question/answer pairs were found in this file. Write the content "
            "as blocks starting with Q: and A: (or 問：and 答：), one pair per "
            "question, and upload again."
        )
    stem = re.sub(r"[^A-Za-z0-9]+", "", Path(filename or "doc").stem)[:18] or "DOC"
    records = []
    for index, (question, answer) in enumerate(pairs, 1):
        records.append(
            {
                "id": f"FAQ_{stem.upper()}_{index:03d}",
                # A single-language document serves both query languages.
                "q_zh": question,
                "q_en": question,
                "a_zh": answer,
                "a_en": answer,
                "source": f"Uploaded file: {filename}",
            }
        )
    return records


def parse_faq(raw: bytes, filename: str = "faq.json") -> list[dict]:
    """Accept a FAQ knowledge base as JSON, JSONL, CSV or a Q/A document."""
    name = (filename or "").lower()
    if name.endswith(tuple(f".{suffix}" for suffix in DOC_EXTENSIONS + IMAGE_EXTENSIONS)):
        return validate(_records_from_text(_extract_text(raw, filename), filename))
    if name and not name.endswith((".json", ".jsonl", ".ndjson", ".csv", ".tsv")):
        raise ValueError(
            f"Unsupported knowledge base file: {filename}. Use JSON, JSON Lines, "
            "CSV, PDF, Word, text or a photo of the Q/A sheet."
        )
    kind, payload = _split_records(_decode(raw), filename)
    if kind == "csv":
        records = [_normalise_row(row, ()) for row in payload]
    elif isinstance(payload, list):
        records = [_normalise_row(row, ()) for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("faq"), list):
        records = [_normalise_row(row, ()) for row in payload["faq"]]
    else:
        raise ValueError(
            "The FAQ file must hold a list of records, a JSON Lines file, or a CSV "
            "with the columns id, q_zh, q_en, a_zh, a_en, source and intent."
        )
    if not records:
        raise ValueError("The uploaded FAQ file contains no records.")
    return validate(records)


def parse_tests(raw: bytes, filename: str = "tests.json") -> list[dict]:
    """Accept a test set as JSON, JSON Lines or CSV (relevant may be a|b)."""
    name = (filename or "").lower()
    if name and not name.endswith((".json", ".jsonl", ".ndjson", ".csv", ".tsv")):
        raise ValueError(
            "The test set must be JSON, JSON Lines or CSV; documents and photos "
            "are for the knowledge base only."
        )
    kind, payload = _split_records(_decode(raw), filename)
    if kind == "csv":
        records = [_normalise_row(row, ("relevant",)) for row in payload]
    elif isinstance(payload, list):
        records = [
            _normalise_row(row, ("relevant",)) for row in payload if isinstance(row, dict)
        ]
    elif isinstance(payload, dict) and isinstance(payload.get("tests"), list):
        records = [_normalise_row(row, ("relevant",)) for row in payload["tests"]]
    else:
        raise ValueError("The test file must hold a list of records.")
    if not records:
        raise ValueError("The uploaded test file contains no records.")
    return records


# ---------------------------------------------------------------------------
# Refusal guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardPolicy:
    """Decides when the evidence is too weak to answer."""

    min_coverage: float = 0.35
    min_dense: float = 0.42


@dataclass
class Verdict:
    accepted: bool
    coverage: float
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    answer_index: int = 0  # position of the best-supported hit within the returned list


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------


class DenseUnavailable(RuntimeError):
    pass


class Engine:
    def __init__(self, data: list[dict]):
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.data = validate(data)
        self.docs = [
            "\n".join(x[k] for k in REQUIRED_FIELDS[1:5]) for x in self.data
        ]
        self.doc_tokens = [tokens(d) for d in self.docs]
        self.question_tokens = [
            tokens(x["q_zh"] + "\n" + x["q_en"]) for x in self.data
        ]
        # Document frequency over the whole corpus, so the guard can discount
        # words that every entry carries.
        counts: dict[str, int] = {}
        for doc in self.doc_tokens:
            for stem in {_stem(t) for t in doc}:
                counts[stem] = counts.get(stem, 0) + 1
        total = len(self.doc_tokens)
        self.term_idf = {
            stem: math.log((1 + total) / (1 + df)) + 1.0 for stem, df in counts.items()
        }
        self.intent_index: dict[str, list[int]] = {}
        for index, record in enumerate(self.data):
            intent = record.get("intent")
            if isinstance(intent, str) and intent:
                self.intent_index.setdefault(intent, []).append(index)
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokens, token_pattern=None, lowercase=False
        )
        self.matrix = self.vectorizer.fit_transform(self.docs)
        self.bm25 = BM25Okapi(self.doc_tokens)
        self.model: Any = None
        self.embeddings: np.ndarray | None = None

    # -- dense ------------------------------------------------------------
    @property
    def dense_ready(self) -> bool:
        return self.embeddings is not None

    @staticmethod
    def dense_installed() -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def prepare(self) -> None:
        if self.embeddings is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise DenseUnavailable(
                "Dense retrieval needs the optional dependencies: "
                "run install_dense_unix.sh (or install_dense_windows.bat)."
            ) from error
        model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = model.encode(
            ["passage: " + d for d in self.docs],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.model = model

    # -- scoring ----------------------------------------------------------
    def scores(self, query: str, method: str) -> np.ndarray:
        if method == "TF-IDF":
            q = self.vectorizer.transform([query])
            return (self.matrix @ q.T).toarray().ravel()
        if method == "BM25":
            return np.asarray(self.bm25.get_scores(tokens(query)))
        if method == "Dense":
            self.prepare()
            q = self.model.encode(
                ["query: " + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            return self.embeddings @ q
        if method == "Hybrid":
            lists: list[np.ndarray] = []
            try:
                self.prepare()
                lists.append(self.scores(query, "Dense"))
            except DenseUnavailable:
                pass  # degrade to lexical fusion instead of failing the question
            lists.append(self.scores(query, "BM25"))
            fused = np.zeros(len(self.data))
            for scores in lists:
                for rank, index in enumerate(
                    np.argsort(-scores, kind="stable")[:50], 1
                ):
                    fused[index] += 1 / (RRF_K + rank)
            return fused
        raise ValueError(f"Unknown retrieval method: {method}")

    # -- retrieval + guard -------------------------------------------------
    def judge(
        self,
        query: str,
        hit: dict,
        method: str,
        policy: GuardPolicy,
        answer_index: int = 0,
    ) -> Verdict:
        cov = hit["coverage"]
        reasons: list[dict] = []
        if cov < policy.min_coverage:
            reasons.append({"code": "coverage", "value": cov})
        if method in ("Dense", "Hybrid") and hit.get("dense") is not None:
            if hit["dense"] < policy.min_dense:
                reasons.append(
                    {"code": "dense", "value": hit["dense"], "limit": policy.min_dense}
                )
        return Verdict(
            accepted=not reasons,
            coverage=cov,
            reasons=reasons,
            missing=missing_terms(query, self.doc_tokens[hit["index"]]),
            answer_index=answer_index,
        )

    def retrieve(
        self,
        query: str,
        method: str,
        k: int,
        policy: GuardPolicy | None = None,
        allowed: Sequence[int] | None = None,
    ) -> tuple[list[dict], Verdict, float]:
        if not query.strip():
            raise ValueError("Empty query.")
        policy = policy or GuardPolicy()
        start = time.perf_counter()
        scores = self.scores(query, method)

        pool = (
            np.arange(len(self.data))
            if allowed is None
            else np.asarray(list(allowed), dtype=int)
        )
        if pool.size == 0:
            verdict = Verdict(
                accepted=False,
                coverage=0.0,
                reasons=[{"code": "no_intent"}],
            )
            return [], verdict, (time.perf_counter() - start) * 1000

        indices = pool[np.argsort(-scores[pool], kind="stable")][:k]

        dense_scores: np.ndarray | None = None
        if method == "Dense":
            dense_scores = scores
        elif method == "Hybrid" and self.dense_ready:
            dense_scores = self.scores(query, "Dense")

        hits: list[dict] = []
        for index in indices:
            record = dict(self.data[index])
            record.update(
                index=int(index),
                score=float(scores[index]),
                coverage=coverage(query, self.doc_tokens[index]),
                dense=None if dense_scores is None else float(dense_scores[index]),
            )
            hits.append(record)

        # The guard judges the best-supported candidate, not the top-ranked one:
        # a high lexical score on one shared word should not outrank a passage
        # that actually covers the question.
        best = max(range(len(hits)), key=lambda i: (hits[i]["coverage"], hits[i]["score"]))
        verdict = self.judge(query, hits[best], method, policy, answer_index=best)
        return hits, verdict, (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Refusal / answering
# ---------------------------------------------------------------------------


def format_reason(reason: dict, lang: str = "en") -> str:
    """Render one guard reason for display."""
    code = reason.get("code")
    if code == "no_intent":
        return (
            "沒有條目屬於預測的意圖" if lang == "zh"
            else "no FAQ entry matches the predicted intent"
        )
    if code == "coverage":
        value = float(reason.get("value", 0.0))
        return (
            f"最佳證據只涵蓋問題 {value:.0%} 的實詞"
            if lang == "zh"
            else f"only {value:.0%} of the question's content terms occur in the best match"
        )
    if code == "dense":
        value = float(reason.get("value", 0.0))
        limit = float(reason.get("limit", 0.0))
        return (
            f"向量相似度 {value:.3f} 低於門檻 {limit:.2f}"
            if lang == "zh"
            else f"dense similarity {value:.3f} is below {limit:.2f}"
        )
    return str(reason.get("text", code or ""))


def refusal_text(verdict: Verdict, lang: str = "en") -> str:
    if lang == "zh":
        parts = ["知識庫現有的資料不足以安全地回答這個問題。"]
        if verdict.missing:
            parts.append("未能涵蓋：" + "、".join(verdict.missing[:6]) + "。")
        parts.append("請說明是哪一門課、哪個部門或哪項服務，或直接聯絡相關校方人員。")
        return "".join(parts)

    parts = [
        "The knowledge base does not hold enough evidence to answer this safely."
    ]
    if verdict.missing:
        parts.append(
            "It does not cover: " + ", ".join(verdict.missing[:6]) + "."
        )
    parts.append(
        "Please name the course, office or service you mean, or contact the "
        "relevant university staff directly."
    )
    return " ".join(parts)


def answer_from_evidence(hit: dict, lang: str = "en") -> str:
    if lang == "zh":
        return f"{hit['a_zh']}\n\n來源：[{hit['id']}] {hit['source']}"
    return f"{hit['a_en']}\n\nSource: [{hit['id']}] {hit['source']}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def check_tests(tests: list[dict], known_ids: Iterable[str]) -> None:
    known = set(known_ids)
    seen: set[str] = set()
    for item in tests:
        if not isinstance(item, dict):
            raise ValueError("Invalid test record: expected a JSON object.")
        qid, query, relevant = item.get("id"), item.get("query"), item.get("relevant")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Test ids must be unique non-empty strings.")
        seen.add(qid)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Test {qid} has an empty query.")
        if not isinstance(relevant, list) or not all(isinstance(x, str) for x in relevant):
            raise ValueError(f"Test {qid} has an invalid 'relevant' list.")
        if not set(relevant) <= known:
            raise ValueError(f"Test {qid} references an unknown FAQ id.")


def evaluate(
    engine: Engine,
    tests: list[dict],
    methods: Sequence[str],
    k: int = 5,
    policy: GuardPolicy | None = None,
    routers: Sequence[tuple[str, object]] = (("none", None),),
    answer_fn: Any = None,
    max_rows: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (per-query rows, summary rows).

    Retrieval metrics use answerable queries; refusal accuracy uses the
    out-of-scope ones. A router, when supplied, restricts each query to the
    entries of the predicted intent, which is how the NLU stage is measured
    against plain retrieval on the same protocol.

    BLEU and ROUGE-L compare the delivered answer with the curated answer of
    the gold entry, so they score answer selection when answers are curated and
    wording quality when ``answer_fn`` supplies generated text. Task success
    counts a query as successful when an answerable one is answered from the
    gold entry at rank one, or when an out-of-scope one is refused.
    """
    check_tests(tests, [x["id"] for x in engine.data])
    policy = policy or GuardPolicy()
    answerable = [t for t in tests if t["relevant"]]
    unanswerable = [t for t in tests if not t["relevant"]]
    if not answerable:
        raise ValueError("No answerable test queries: nothing to evaluate.")

    def route(router, query: str) -> tuple[object, list[int] | None]:
        if router is None:
            return "all", None
        intent = router.predict(query)[0]
        return intent, engine.intent_index.get(intent, [])

    rows: list[dict] = []
    for router_name, router in routers:
        for method in methods:
            if method == "Dense":
                engine.prepare()
            # warm-up, excluded from timings
            engine.retrieve(
                answerable[0]["query"], method, k, policy, route(router, answerable[0]["query"])[1]
            )
            for item in answerable:
                intent, allowed = route(router, item["query"])
                hits, verdict, latency = engine.retrieve(
                    item["query"], method, k, policy, allowed
                )
                ids = [h["id"] for h in hits]
                gold = set(item["relevant"])
                row = {
                    "router": router_name,
                    "method": method,
                    "query_id": item["id"],
                    "query": item["query"],
                    "language": lang(item["query"]),
                    "routed_intent": intent,
                    "retrieved": " | ".join(ids),
                    "answered": verdict.accepted,
                    "outcome": _outcome(ids, gold, verdict.accepted),
                    "coverage": round(verdict.coverage, 3),
                    "latency_ms": round(latency, 2),
                }
                for cutoff in (1, 3, 5):
                    row[f"Recall@{cutoff}"] = round(
                        len(set(ids[:cutoff]) & gold) / len(gold), 3
                    )
                row["MRR@5"] = round(
                    next((1 / r for r, fid in enumerate(ids, 1) if fid in gold), 0.0), 3
                )
                row["Top1"] = 1.0 if ids and ids[0] in gold else 0.0
                language = row["language"]
                gold_record = next(
                    (x for x in engine.data if x["id"] in gold), None
                )
                reference = record_answer(gold_record, language) if gold_record else ""
                if verdict.accepted and reference:
                    if answer_fn is None:
                        candidate = answer_from_evidence(
                            hits[verdict.answer_index], language
                        )
                    else:
                        candidate = answer_fn(
                            item["query"], hits, verdict, language
                        )
                    row["answer"] = candidate
                    row["bleu"] = bleu(reference, candidate)
                    row["rouge_l"] = rouge_l(reference, candidate)
                else:
                    row["answer"] = ""
                    row["bleu"] = None
                    row["rouge_l"] = None
                rows.append(row)

            for item in unanswerable:
                intent, allowed = route(router, item["query"])
                hits, verdict, latency = engine.retrieve(
                    item["query"], method, k, policy, allowed
                )
                rows.append({
                    "router": router_name,
                    "method": method,
                    "query_id": item["id"],
                    "query": item["query"],
                    "language": lang(item["query"]),
                    "routed_intent": intent,
                    "retrieved": " | ".join(h["id"] for h in hits),
                    "answered": verdict.accepted,
                    "outcome": "answered out-of-scope" if verdict.accepted
                    else "correctly refused",
                    "coverage": round(verdict.coverage, 3),
                    "latency_ms": round(latency, 2),
                    "out_of_scope": True,
                })

    summary: list[dict] = []
    for router_name, _router in routers:
        for method in methods:
            mine = [
                r for r in rows
                if r["method"] == method and r["router"] == router_name
                and "out_of_scope" not in r
            ]
            oos = [
                r for r in rows
                if r["method"] == method and r["router"] == router_name
                and r.get("out_of_scope")
            ]
            summary.append({
                "router": router_name,
                "method": method,
                "Recall@1": round(float(np.mean([r["Recall@1"] for r in mine])), 3),
                "Recall@3": round(float(np.mean([r["Recall@3"] for r in mine])), 3),
                "Recall@5": round(float(np.mean([r["Recall@5"] for r in mine])), 3),
                "MRR@5": round(float(np.mean([r["MRR@5"] for r in mine])), 3),
                "Answer rate": round(float(np.mean([r["answered"] for r in mine])), 3),
                "Refusal acc.": round(
                    float(np.mean([not r["answered"] for r in oos])), 3
                ) if oos else float("nan"),
                "Latency (ms)": round(float(np.mean([r["latency_ms"] for r in mine])), 2),
                "BLEU": round(float(np.mean([
                    r["bleu"] for r in mine if r.get("bleu") is not None
                ])), 3) if any(r.get("bleu") is not None for r in mine) else float("nan"),
                "ROUGE-L": round(float(np.mean([
                    r["rouge_l"] for r in mine if r.get("rouge_l") is not None
                ])), 3) if any(r.get("rouge_l") is not None for r in mine) else float("nan"),
                "Task success": round(float(np.mean([
                    (bool(r["answered"]) and r["Top1"] == 1.0)
                    for r in mine
                ] + [not r["answered"] for r in oos])), 3),
            })
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows, summary


# ---------------------------------------------------------------------------
# Answer overlap metrics
# ---------------------------------------------------------------------------

def answer_tokens(text: str) -> list[str]:
    """Tokens for overlap metrics: ASCII words plus Chinese bigrams."""
    low = text.lower()
    result = _WORD.findall(low)
    for part in _CJK_RUN.findall(text):
        result.extend(part[i:i + 2] for i in range(len(part) - 1)) if len(part) > 1 else result.extend(part)
    return result


def bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    """Sentence BLEU with a brevity penalty, no smoothing.

    Precision at order n is zero when nothing matches, and the geometric mean
    then makes the score zero; that keeps a wrong answer visibly wrong instead
    of rewarding it with a smoothed floor.
    """
    ref = answer_tokens(reference)
    cand = answer_tokens(candidate)
    if not ref or not cand:
        return 0.0
    order = min(max_order, len(ref))
    precisions = []
    for n in range(1, order + 1):
        ref_grams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        cand_grams = Counter(tuple(cand[i:i + n]) for i in range(len(cand) - n + 1))
        matched = sum(min(count, cand_grams[gram]) for gram, count in ref_grams.items())
        total = max(len(cand) - n + 1, 0)
        if total == 0 or matched == 0:
            return 0.0
        precisions.append(matched / total)
    score = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    if len(cand) < len(ref):
        score *= math.exp(1 - len(ref) / len(cand))
    return round(score, 4)


def rouge_l(reference: str, candidate: str, beta: float = 1.0) -> float:
    """ROUGE-L F-score from the longest common subsequence."""
    ref = answer_tokens(reference)
    cand = answer_tokens(candidate)
    if not ref or not cand:
        return 0.0
    lcs = 0
    previous = [0] * (len(ref) + 1)
    for token in cand:
        current = [0]
        for index, other in enumerate(ref, 1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    score = ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)
    return round(score, 4)


def record_answer(record: dict, language: str) -> str:
    """The curated answer of one record in the requested language."""
    key = f"a_{language}"
    return str(record.get(key) or record.get("a_en") or record.get("a_zh") or "")


def _outcome(ids: list[str], gold: set[str], answered: bool) -> str:
    if not answered:
        return "refused (should answer)"
    if ids[0] in gold:
        return "correct"
    if set(ids[:3]) & gold:
        return "correct in top-3"
    return "miss"

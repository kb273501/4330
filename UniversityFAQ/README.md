# University FAQ Assistant

A local, open-source FAQ chatbot for course and university enquiries, built for
the 4330 group project. It compares four retrieval methods and two answering
modes under one evaluation protocol, and measures a rule-based against a trained
intent classifier.

**The corpus is synthetic.** Every policy in `data/faq.json` is invented for
demonstration. The numbers below show the protocol working, not findings about a
real university.

---

## 1. What it does

```
question
   │
   ├─ intent classifier (rule-based or trained) ── optional routing
   │
   ├─ retriever: TF-IDF · BM25 · Dense (E5) · Hybrid (RRF)
   │
   ├─ refusal guard: term coverage of the best-supported candidate
   │      └─ refuse → say what is missing, point to a human
   │
   └─ answer: Retrieval-only (curated answer) │ RAG (local Ollama, evidence-bound)
```

Everything runs on the local machine. No question, answer or embedding leaves
the host.

## 2. Repository layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI only: five tabs, badges, evidence tables, exports |
| `faq_engine.py` | Retrieval, guard, validation, metrics — no Streamlit import |
| `intent.py` | Rule matcher, trained classifier, intent evaluation |
| `i18n.py` | Interface strings, English and Chinese |
| `data/faq.json` | 48 entries, 12 intents, answers in HKMU reply style |
| `data/tests.json` | 108 queries: 96 answerable, 12 out-of-scope |
| `test_engine.py` | 68 plain-assert checks, no pytest needed |
| `setup_*.sh` / `setup_windows.bat` | Creates `.venv` with Python 3.10–3.13 |
| `start_*.sh` / `start_windows.bat` | Launches the app |
| `install_dense_*.sh` | Optional sentence-transformers + E5 weights |
| `cli.py` | Terminal evaluation with the same protocol as the UI, CSV export |
| `results/` | `eval_summary.txt` and `eval_rows.csv` written by `cli.py` |

## 3. Quick start

```bash
./setup_unix.sh          # or setup_windows.bat  (needs Python 3.10–3.13)
./start_unix.sh          # opens http://localhost:8501
```

`setup_unix.sh` searches for `python3.13`, `3.12`, `3.11`, `3.10` and stops with
an install hint if none is present. Python 3.14 has no scikit-learn wheel yet,
so the guard is deliberate.

Dense retrieval is optional:

```bash
./install_dense_unix.sh  # adds sentence-transformers and downloads E5-small
```

### Language

The first control in the sidebar switches between **English** and **繁體中文**.
It changes the interface, the answer language, the refusal wording, the intent
labels and the RAG prompt together — a Chinese question is answered from `a_zh`,
and a generated answer is rejected if it comes back in the wrong language. The
retrieval corpus stays bilingual either way.

### Upload formats

The knowledge base loader accepts **JSON**, **JSON Lines** and **CSV**:

- JSON: a list of records, or an object with a `faq` list.
- JSON Lines: one record per line (`.jsonl`, `.ndjson`).
- CSV: header row `id, q_zh, q_en, a_zh, a_en, source`, with `intent` optional;
    a `.json` file that actually holds JSON Lines is detected automatically.
- PDF, Word (`.docx`, `.doc`), plain text (`.txt`, `.md`) and photos
  (`.jpg`, `.png`): the text is extracted and split into records at `Q:` / `A:`
  blocks, or `問：` / `答：` for Chinese. `.docx` is read directly, `.doc` goes
  through macOS `textutil`, PDF through `pypdf`, and a photo needs `tesseract`
  installed for OCR — without it the app says so instead of guessing.

A document upload produces one record per Q/A pair, with generated ids and a
`source` naming the file; a single-language document is used for both query
languages. Such records carry no intent label, so the intent tab shows no
classes, the filter and the trained classifier are switched off, and retrieval
runs over the whole uploaded set. The test set stays tabular (JSON, JSON Lines or CSV), with several
relevant ids separated by `|` (or `;` / `、`). Uploads apply to the running
session only.

### Answer style and sources

Answers are written the way the University answers an enquiry: the office that
decides, the form to complete, the evidence required, the deadline, then the
enquiry channel. Where a procedure could be verified against public HKMU
material it is cited in `source` — for example the FCS003 Course Issues
Application Form, the 5/7-calendar-day windows in the Student Guidelines on
Examination, the 1–11 September add/drop period in the 2026/27 academic calendar,
the Library Handbook renewal rule and the Regulations Governing Academic
Integrity and Student Discipline. Entries whose `source` names a `demo policy` clause
are partly illustrative: the wording reads like a university reply, but that
clause is invented and the page banner says so. Twenty-one entries cite only
official material; twenty-seven carry a demo clause, and eleven of those rest
on nothing else. That split is counted by the tests, so it cannot drift
silently.

## 4. Intents

Twelve labels, four entries each. The label also narrows retrieval when routing
is set to *Filter*.

| Label | Display name | Label | Display name |
| --- | --- | --- | --- |
| `ASSESSMENT_EXTENSION` | Assignment extension | `FEES_PAYMENT` | Tuition payment |
| `ASSESSMENT_GRADE` | Grades and appeals | `FEES_REFUND` | Refund |
| `EXAM_TIMETABLE` | Exam timetable | `SCHOLARSHIP` | Scholarship |
| `EXAM_ABSENCE` | Missed exam | `IT_ACCOUNT` | IT account |
| `ENROLMENT_CHANGE` | Add or drop a course | `LIBRARY` | Library |
| `ACADEMIC_INTEGRITY` | Academic integrity | `STUDENT_SUPPORT` | Student support |

Two classifiers produce the label:

- `RuleIntent` — keyword triggers, English and Chinese, no training. Scores one
  point per distinct trigger, abstains when nothing fires.
- `TrainedIntent` — TF-IDF over the retrieval tokeniser (word 1–2 grams, plus
  character 2–4 grams) with a logistic-regression head. It abstains only when
  the top probability falls below the uniform prior `1/12`.

Routing has three settings in the sidebar: *Off*, *Advisory* (label shown,
retrieval unchanged), *Filter* (retrieval restricted to the routed intent).

## 5. Retrievers

| Method | Mechanism | Notes |
| --- | --- | --- |
| TF-IDF | sublinear term weighting, cosine | baseline, very fast |
| BM25 | Okapi, k1 = 1.5, b = 0.75 | strong on exact wording |
| Dense | `intfloat/multilingual-e5-small`, normalised cosine | needs the optional install |
| Hybrid | reciprocal-rank fusion of Dense and BM25, k = 60 | falls back to BM25 when Dense is absent |

The tokeniser is shared by every lexical method and by the intent model: ASCII
words, Chinese characters, and Chinese character bigrams. One index therefore
serves both languages, and a Chinese question can match a Chinese paraphrase
while the answer is delivered in English.

## 6. Refusal guard

Answers are withheld when the question's content terms (stop words removed,
light stemming applied) are not covered by the best-supported retrieved
candidate — the candidate with the highest term coverage among the top-k, not
the top-scored one:

```
coverage = |content terms of query ∩ terms of candidate| / |content terms of query|
```

The guard is scale-free, so the same threshold works for all four retrievers,
and it names the terms it could not find. On the current corpus (TF-IDF, k = 5,
judged on the best-supported candidate):

| Threshold | Answer rate | Refusal accuracy |
| --- | --- | --- |
| 0.30 | 0.865 | 0.750 |
| **0.35** | **0.729** | **0.833** |
| 0.40 | 0.688 | 0.917 |
| 0.45 | 0.604 | 1.000 |

Answerable queries have median coverage 0.56 (10th percentile 0.25); the
out-of-scope set tops out at 0.40. There is no clean gap at this corpus size, so
0.35 is a deliberate balance rather than a natural threshold — the slider is
exposed in the UI so the trade-off can be inspected directly. Ten of the twelve
negatives are refused; the two survivors are 明天会下雨吗？, which overlaps the
single characters 明 and 天 in "考试当天生病", and a flight-booking question that
shares the generic word "book". This is documented rather than hidden.

The threshold was recalibrated when the answers were rewritten in HKMU style: the
guard reads the whole entry, so longer answers raise coverage across the board and
separate the two sets less cleanly than the short demo answers did.

## 7. Answering modes

- **Retrieval-only** returns the curated answer of the best-supported candidate
  with its source id.
- **RAG** prompts a local Ollama model (`qwen2.5:3b` by default) with the
  retrieved evidence plus the question, at temperature 0, and then checks the
  reply: citations must come from the retrieved set, and the reply must be in
  English. Failing either check, the answer is withheld and the user is told why.

## 8. Metrics

| Level | Measure | Definition |
| --- | --- | --- |
| Retrieval | Recall@1/3/5, MRR@5 | over the 96 answerable queries, k = 5 |
| Guard | Answer rate | share of answerable queries actually answered |
| Guard | Refusal acc. | share of the 12 out-of-scope queries correctly refused |
| Response | BLEU, ROUGE-L | delivered answer against the curated answer of the gold entry |
| End-to-end | Task success | correct top-1 answer or correct refusal, over all 108 queries |
| Intent | Accuracy, macro P/R/F1 | over paraphrase queries absent from training |
| Speed | Latency (ms) | retrieval only, generation excluded |

`bleu` is a sentence BLEU with a brevity penalty and no smoothing, so an
unrelated answer scores zero instead of a smoothed floor. `rouge_l` is the
LCS F-score (beta = 1). Both tokenise English into words and Chinese into
character bigrams, so the same scorer works for either language.

BLEU and ROUGE-L are computed only on answered queries: a refusal has no text
to compare, and scoring it as zero would mix coverage into a wording measure.
With retrieval-only answers the pair behaves as a wording-sensitive check of
answer selection (identical text when the gold entry is returned); pass
`answer_fn` to `evaluate()` and it scores generated text instead, which is how
the RAG mode is meant to be measured.

Out-of-scope queries are scored, not discarded, which is what makes over-refusal
and over-answering both visible.

## 9. Current results

Measured with `python cli.py --methods TF-IDF BM25 Dense Hybrid --routers none
rule trained` on this machine (macOS, Python 3.13, scikit-learn 1.9,
sentence-transformers 6.2 with `intfloat/multilingual-e5-small`). Retrieval,
k = 5, 96 answerable plus 12 out-of-scope queries. Full output:
`results/eval_summary.txt`; per-query rows: `results/eval_rows.csv`.

| Routing | Method | R@1 | R@3 | R@5 | MRR@5 | Answer rate | Refusal acc. | BLEU | ROUGE-L | Task success | ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | TF-IDF | 0.531 | 0.750 | 0.802 | 0.646 | 0.729 | 0.833 | 0.497 | 0.582 | 0.444 | 0.38 |
| None | BM25 | 0.562 | 0.781 | 0.812 | 0.670 | 0.729 | 0.833 | 0.521 | 0.604 | 0.463 | 0.25 |
| None | Dense | 0.792 | 0.948 | 0.969 | 0.868 | 0.708 | 0.917 | 0.609 | 0.696 | 0.593 | 10.9 |
| None | Hybrid | 0.729 | 0.885 | 0.927 | 0.808 | 0.750 | 0.917 | 0.540 | 0.626 | 0.583 | 21.3 |
| Rule | TF-IDF | 0.438 | 0.583 | 0.615 | 0.516 | 0.448 | 1.000 | 0.591 | 0.670 | 0.380 | 0.34 |
| Rule | BM25 | 0.417 | 0.615 | 0.615 | 0.512 | 0.448 | 1.000 | 0.610 | 0.691 | 0.380 | 0.20 |
| Rule | Dense | 0.500 | 0.594 | 0.615 | 0.552 | 0.448 | 1.000 | 0.631 | 0.713 | 0.435 | 10.1 |
| Rule | Hybrid | 0.490 | 0.615 | 0.615 | 0.550 | 0.448 | 1.000 | 0.610 | 0.691 | 0.426 | 17.4 |
| Trained | TF-IDF | 0.500 | 0.646 | 0.688 | 0.583 | 0.594 | 1.000 | 0.520 | 0.599 | 0.435 | 0.35 |
| Trained | BM25 | 0.490 | 0.667 | 0.688 | 0.583 | 0.594 | 1.000 | 0.534 | 0.615 | 0.426 | 0.23 |
| Trained | Dense | 0.594 | 0.677 | 0.688 | 0.638 | 0.594 | 1.000 | 0.579 | 0.659 | 0.491 | 10.3 |
| Trained | Hybrid | 0.583 | 0.667 | 0.688 | 0.628 | 0.594 | 1.000 | 0.548 | 0.628 | 0.491 | 20.5 |

Every figure is copied from `results/eval_summary.txt`; the CSV also carries
per-query coverage, outcome, latency and answer text. The latency column is
warm: `evaluate()` runs one discarded query per setting first, because the
first dense call loads the model (about 4.8 s on this machine) and would
otherwise dominate a 108-query mean.

Intent classification, 96 paraphrase queries, 12 out-of-scope:

| Model | Accuracy | Macro P | Macro R | Macro F1 | Abstain rate | OOS declined |
| --- | --- | --- | --- | --- | --- | --- |
| Rule | 0.615 | 0.892 | 0.615 | 0.700 | 0.302 | 1.000 |
| Trained | 0.688 | 0.731 | 0.688 | 0.687 | 0.000 | 0.000 |

Five readings worth carrying into the report:

1. **Semantic retrieval wins on this corpus.** Dense reaches R@1 0.792 against
   0.531 for TF-IDF and 0.562 for BM25, and it lifts task success from 0.44 to
   0.59. The corpus is bilingual and paraphrased, which is exactly the case that
   bag-of-words scoring cannot cover: 考試改期 and 測驗延期 share no term but
   land in the same place in embedding space.
2. **Hybrid fusion loses to its better half.** RRF over dense and BM25 scores
   0.729 R@1, below dense alone at 0.792, because it re-admits lexical
   candidates that dense had already ranked worse. Fusion helps latency
   tolerance, not accuracy; a weighted sum or a tuned RRF constant might fix it,
   and that experiment has not been run.
3. **Hard filtering costs recall.** Every routed setting loses 7–11 points of
   R@1 against unrouted retrieval, because a misrouted query is answered from
   the wrong slice or refused outright. Routing buys refusal accuracy
   (0.833 → 1.000) and pays for it in coverage: answer rate falls from 0.729 to
   0.448 with the rule matcher.
4. **The two classifiers fail in opposite directions.** The rule matcher is
   precise (0.892) and abstains on 30% of answerable queries; the trained model
   answers everything (0.688 accuracy, zero abstention), including every
   out-of-scope query. Neither is usable as a hard filter without the other.
5. **Lexical recall is no longer saturated.** On the earlier six-entry corpus
   every retriever scored 1.000; at 48 entries TF-IDF and BM25 separate by
   R@3/R@5 and by latency, which is what the protocol was built to expose.

Why the trained model is weak: four training questions per intent cannot cover
the vocabulary of the paraphrases, and character n-grams carry most of the
Chinese signal. Both are reported rather than tuned away.

## 10. User study

Each answer carries a 1–5 star rating widget (with a slider fallback). Rated
answers export to `user_study.csv` with the question, method, mode, routed
intent, refusal flag and coverage, ready for the questionnaire analysis in the
report.

## 11. Tests

```bash
python test_engine.py
```

68 checks, all passing: tokenisation, stop words, stemming, corpus shape
(48 entries / 12 intents / 108 queries), guard separation, validation rejections,
retrieval and refusal behaviour, evidence selection, intent filtering including
the empty-candidate case, evaluation shape with routers, both intent classifiers
against the measured numbers above, JSON Lines / CSV parsing for the knowledge
base and the test set, the Chinese rendering of refusals, answers, intent
labels and guard reasons, Q/A extraction from text, Word and photo routes
including the rejection of a file with no Q/A markers, and the metric
definitions: identical text scores 1.000 on both BLEU and ROUGE-L, unrelated
text scores 0.000 on BLEU, a partial answer stays strictly between 0 and 1,
Chinese answers are scored through character overlap, and task success remains
a probability; the provenance split (21 official-only, 27 with a demo clause)
is pinned too. The checks encode current
behaviour, including the known single-character false accept, so a regression
cannot pass silently.

## 12. Limitations

- Synthetic corpus, 48 entries, all policies fictional; no real university
  source has been consulted.
- Small training set (four questions per intent); the trained classifier
  generalises unevenly and has no abstain class.
- The guard threshold was calibrated on the same 108 queries it is reported on;
  the figures in §6 are therefore optimistic, and a separate development split
  is needed before quoting them.
- Answers are paraphrases written to the University's reply pattern, not quoted
  policy. Twenty-seven of the forty-eight entries carry invented detail and are
  marked demo in `source`; the remaining twenty-one rest on a cited form or page,
  but each still needs checking against the current edition before release.
- Photo upload depends on tesseract, which is not installed here, so only the
  text and Word routes were exercised end to end.
- Dense and Hybrid are measured with `intfloat/multilingual-e5-small`, a small
  multilingual model; a larger encoder or a fine-tuned one would likely move
  the numbers, and none of the tuning was searched.
- Single-turn: history is displayed but not used as context.
- Switching language does not retranslate earlier answers; they keep the language
  they were produced in.
- The Chinese guard is weaker than the English one: a short Chinese negative such
  as 明天會下雨嗎？ gets through on single-character overlap with 考試當天生病.
- BLEU and ROUGE-L score answer selection here, not generation: retrieval-only
  answers are curated text, so an exact hit scores 1.000 by construction. They
  become a wording measure only when `answer_fn` supplies generated answers.
- User study is a rating widget, not a controlled study; no participants have
  been recruited and no responses are stored yet.
- RAG replies can still be wrong; the citation and language checks are not fact
  checking.
- Localhost only, no authentication, no rate limiting.

## 13. Changelog

- Corpus 6 → 48 entries, 12 intents, answers and sources in English only.
- Test set 14 → 108 queries (96 answerable, 12 out-of-scope).
- Added `intent.py`: rule-based and trained classifiers with shared evaluation.
- Added intent routing (off / advisory / filter) to retrieval and evaluation.
- Guard rewritten from score threshold to term coverage; re-calibrated to 0.30
  on the enlarged corpus.
- Guard now judges the best-supported candidate among top-k, and the answer is
  drawn from that candidate.
- Evaluation reports per-router rows, so routing is measured, not assumed.
- Removed Chinese and demo-policy prefixes from answer text; provenance lives in
  `source` and in the page banner.
- `app.py` split into UI-only; retrieval, guard and metrics moved to
  `faq_engine.py`.
- `setup_*` scripts pin Python 3.10–3.13; `requirements.txt` requires
  Streamlit ≥ 1.50.
- Added an English / 繁體中文 switch covering the interface, answers, refusals,
  intent labels and the RAG prompt (`i18n.py`).
- Knowledge base and test set can be uploaded as JSON, JSON Lines or CSV; the
  demo-policy answer prefix is gone.
- All 48 answers rewritten in the University's reply pattern (office, form,
  evidence, deadline, enquiry channel), 21 grounded in public HKMU material and
  27 marked as demo policy.
- Guard threshold recalibrated from 0.30 to 0.35 after the rewording changed
  coverage; answer rate 0.729, refusal accuracy 0.833.
- Knowledge base also accepts PDF, Word, text and photo uploads, split into
  records at Q/A blocks; the test set stays tabular.
- Added `bleu`, `rouge_l`, `answer_tokens` and `record_answer` to
  `faq_engine.py`; BLEU, ROUGE-L and task success are now in every summary row
  and in the evaluation table and chart.
- Dense retrieval made real: `sentence-transformers` 6.2 with
  `intfloat/multilingual-e5-small`, so the four-retriever comparison in Table 1
  is measured rather than asserted (dense R@1 0.792, hybrid 0.729, BM25 0.562,
  TF-IDF 0.531).
- Added `cli.py` for reproducible evaluation from the terminal, with CSV export
  of per-query rows; results stored in `results/`.
- Test suite 60 → 68 checks, adding the metric definitions and the
  provenance split.


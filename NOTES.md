# Design notes

Why the service is built the way it is, what I considered instead, what I learned from the sample documents, and what I'd build next. The README covers setup and the API.

## Guiding principle

This service answers security questionnaires on behalf of customers. In that setting, a fluent answer that the document doesn't support is the worst possible output. It's worse than "Not found in document", because a human reviewer is likely to trust it. So wherever there was a trade-off between answer coverage and grounding, I chose grounding, and I tried to make each check verifiable rather than relying on prompt wording.

## Decisions

### 1. Two-layer faithfulness verification

**What:** The model never states page numbers. It cites sources by label (`S1`, `S2`...) with a verbatim quote, and page, section and record id come from our own chunk metadata. Then:

- **Layer 1 (deterministic, free):** each quote must actually appear in the chunk it cites. Matching folds case, quotes and whitespace, then falls back to a fuzzy in-order match (≥90%) that tolerates small copying slips but rejects scattered fragments. The excerpt returned to the client is the span from the document, not the model's copy.
- **Layer 2 (one gpt-4o-mini call):** an LLM judge checks whether the answer's claims are entailed by the verified excerpts.

If either layer fails, the answer becomes "Not found in document".

**Why both:** They catch different failures. Layer 1 catches invented quotes, which is cheap to detect. Layer 2 catches the subtler case: a real quote attached to a claim it doesn't support ("GCP hosts the service" is quoted; the answer adds "with 99.99% uptime in three regions"). `tests/unit/test_pipeline.py` has a test for each.

**Alternatives:**
- *Prompt-only grounding* ("only use the sources"). This is necessary but can't be verified.
- *Sampling the judge* (run it on a percentage of answers, as I do in production for monitoring). This is fine for measuring drift, but here the judge is a gate, not a metric.
- *NLI cross-encoder instead of an LLM judge.* It's cheaper, but weaker on the multi-clause answers questionnaires need, and it adds a third local model.

**Cost:** Normally two LLM calls per answered question instead of one. Answers that fail Layer 1, and questions gated out by retrieval, skip the judge.

### 2. Hybrid retrieval, cross-encoder reranking, and RRF

**What:** FAISS (dense) and BM25 (lexical) each nominate 20 candidates. Scores are min-max normalized per list and fused with weights 0.7/0.3. A cross-encoder scores the fused shortlist. The final order is the reciprocal-rank fusion of the hybrid order and the cross-encoder order.

**Why hybrid:** Questionnaire answers often hinge on exact tokens: vendor names, "AES-256", "TLS 1.2", control IDs like "CC6.1". Dense embeddings blur those. The tokenizer keeps compound identifiers intact so BM25 can match them.

**Why RRF on top of the reranker, a decision based on data:** On the sample report, reranking alone sometimes buried chunks both retrievers agreed on. For "Is personal information... disclosed to third parties?", the cross-encoder pushed the Third-Party vendor table (p.17) out of the top 4. The model is trained on short web queries, and SOC 2 questions are long and multi-clause. RRF of both orders kept the best of each on all five sample questions:

| Question | Rerank only (top 3) | RRF (top 3) |
|---|---|---|
| Personal info / third parties | p40, p14, p41 | **p14, p17**, p67 |
| Monitoring checklist | p25, p29, p46 | **p25, p21**, p29 |
| Incident notification | p78, p42, p75 | p75, **p17**, p42 |

**The gate:** The cross-encoder's absolute score is still the best "is anything relevant at all?" signal. On the sample report, unanswerable probes ("capital of France", "annual revenue", "Hypervisor account lockout") all scored below 1e-4, and answerable questions scored above 2e-3. `MIN_RELEVANCE = 5e-4` sits in that gap. Below it, the question gets "Not found" with no LLM call.

### 3. Local embeddings and reranker

The challenge says to use `gpt-4o-mini` only, and the provided key may not allow embedding models. So `BAAI/bge-small-en-v1.5` (embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking) run locally on CPU. `sentence-transformers` was already needed for the reranker, so this added no new dependency.

- Costs: a larger Docker image (~2.5 GB with CPU torch and about 200 MB of weights baked in), and indexing time the first time a new document is seen. For the 84-page sample that's about 10 s natively on an Apple Silicon Mac and about 50 s in Docker on the same machine. After that, the index cache makes repeats instant.
- Benefits: no per-token embedding cost, no rate limits, no network dependency for retrieval, and the key budget goes only to generation.

**A bug worth knowing about:** The cross-encoder returned NaN for every input. I traced it to the safetensors checkpoint storing tensors at offsets that aren't float-aligned. transformers memory-maps them, and the CPU BLAS kernel silently produces NaN on misaligned data. NumPy gave correct results on the very same arrays. The fix (`app/services/model_utils.py`) copies any misaligned parameter into fresh aligned memory after loading. Downgrading torch didn't help, which is how I ruled out a regression.

### 4. Question routing

**What:** Each question is classified as boolean, factual, explanatory or checklist. Heuristics handle it when they can (auxiliary-verb openers, wh-words, "describe/if yes", multiple question marks, "which of the following" plus enumerated options). gpt-4o-mini is called only when no rule matches. If that call fails, the fallback is "explanatory", the safest default.

**What the type changes:**
- the answer format instruction (yes/no first, concise facts, or a multi-part description)
- retrieval depth (k=5, or 8 for explanatory)
- for checklists, the pipeline shape itself: it retrieves for the whole question and for each option, and interleaves the hits so every option has its evidence in context. The model answers per option, and each option is verified independently. A single fabricated option is dropped without discarding the others.

All five spec sample questions are classified by heuristics alone.

### 5. Document handling, based on the real files

Inspecting the sample files changed several early assumptions:

- **pypdf extraction mode.** The default mode emitted one word per line, which I fixed with whitespace normalization. Layout mode keeps lines, but interleaves text across columns on the 4-column test matrices (p.27–84). Default mode keeps each cell's text together. So content uses default mode, and layout mode is used only to parse the table of contents.
- **Running header.** Every page starts with "A Type 2 Independent Service Auditor's Report on Controls Relevant To Security". It's detected generically, as the longest word prefix shared by most pages, and stripped along with footer page numbers.
- **The ToC is useful, but its page numbers are wrong.** They're offset by one from the physical pages. So titles come from the ToC, and each title's real position is found in the body text, in order, with a small lookahead. 48 of 50 ToC titles are located; the other two ("SYSTEM DESCRIPTION", "TESTING MATRICES") appear only on divider slides that are skipped. Matching is case-sensitive: in an earlier version, the cover's "Independent Service Auditor's Report" and prose like "express an opinion" were mistaken for headings. Chunks are split at section boundaries so each one belongs to exactly one section. The section title is prepended to the text used for embedding (a contextual chunk header), and citations report it.
- **Front matter.** The ToC pages and blank "SECTION N" divider slides are skipped. The cover page is kept on purpose, because it holds the audit period.
- **The JSON sample is a Q&A knowledge base, not prose.** Each record (question, answer, details, confidence) is one retrieval unit, cited by record id, since JSON has no pages. The loader also accepts pandas-style `columns`, `index` and `split` layouts, wrapped lists, lists of strings, and page-shaped JSON (`{"page", "text"}`), because I don't control what shape a customer's export takes.

### 6. Avoiding redundant work

| Layer | Key | Saves |
|---|---|---|
| Index cache (LRU + TTL, single-flight) | SHA-256 of document bytes | Re-parsing and re-embedding; concurrent requests for a new document build it once |
| Question dedup | exact string, within a request | Duplicate questions answered once |
| Semantic answer cache | (document hash, question) | Every LLM call for repeat or reworded questions |
| Retrieval gate | cross-encoder score | LLM calls for questions the document can't answer |
| Heuristic classifier | — | A classification call for almost every question |

The semantic cache threshold (0.97) is deliberately strict. "Which cloud providers do you rely on?" and "Which cloud provider is primary?" are close in embedding space, but they're different questions, and a wrong cache hit returns a confident answer to the wrong question. Only verified answers are cached, never errors.

### 7. Concurrency model

Questions run concurrently under an `asyncio.Semaphore` (default 5) inside an `asyncio.TaskGroup`. CPU-bound work (PDF parsing, embedding, BM25, cross-encoder) runs in worker threads so the event loop stays responsive. All question embeddings are computed in one batch.

Failures are split by kind:
- **Systemic** (missing or rejected key): the TaskGroup cancels the remaining work and the request fails fast with 503.
- **Transient per-question** (timeout after retries): only that question gets `results[].error`. If every question fails, the whole request returns 502.

Retries and per-call timeouts come from the OpenAI client. An overall `asyncio.timeout` bounds the request (504).

### 8. HTTP-layer robustness

- Uploads are validated by content sniffing (`%PDF-` magic, JSON first byte), and the extension must agree with the content.
- Size is enforced at two levels. An ASGI middleware rejects oversized bodies before Starlette spools them to disk, checking both Content-Length and a streaming byte count. Per-file limits then give precise messages.
- Services raise domain exceptions, and one module maps them to HTTP statuses. Services never import FastAPI.
- One error shape everywhere, including 404/405 and framework validation errors, and every error carries the request id.

### 9. Smaller choices

- **FAISS through LangChain.** The FAISS integration still lives in `langchain-community`, which is being sunset. A `langchain-faiss` package exists on PyPI, but it's an empty package from an unofficial author, so I didn't use it. Its deprecation warning is filtered in pytest config with a pointer here. Swapping to a maintained store (pgvector, or a standalone package once it exists) only touches `indexing.py`.
- **Strict JSON-schema structured output** (`method="json_schema", strict=True`) instead of parsing free text. Schema violations become upstream errors rather than silent garbage.
- **Prompt-injection hygiene.** Sources are wrapped in `<sources>` tags, and the model is told they are untrusted data. Layer 1 also means injected text can't make the model cite something that isn't in the document.
- **Pinned dependencies, including torch.** The Docker image installs CPU-only torch, because the default Linux wheel bundles CUDA (several GB).

## Evals: what I'd build next

I deliberately didn't build an eval harness for this exercise. It's the next thing I'd add, and several thresholds above are waiting on it.

One distinction matters first: **the Layer 2 judge is a guard, not an eval.** It blocks individual bad answers at request time, but it doesn't tell you whether the system is getting better or worse, how often it wrongly says "Not found", or whether a prompt change helped.

**1. Golden set.** About 60–100 questions over the Nave report and the knowledge base, each labelled with the expected answer, the supporting page(s) or record id(s), and the question type. It should include:
- deliberate "Not found" cases, including near-misses where the document discusses the topic without answering the specific question
- attribution traps (GCP's responsibilities vs. Nave's)
- checklist questions with mixed answers

**2. Retrieval metrics,** which are cheap and need no LLM: recall@k and MRR of the expected page. Use them to tune `k`, fusion weights, chunk size, and the RRF-vs-rerank choice with numbers instead of five examples.

**3. Answer metrics:**
- "Not found" precision and recall. This is the key compliance metric: false "found" is dangerous, and false "not found" is costly.
- citation precision (does the cited excerpt support the answer?)
- answer correctness against the label (exact for boolean and checklist, LLM-graded for free text, with a sample of grades checked by hand)

**4. Offline faithfulness scoring** with a stronger judge model than the runtime one, to measure how often the runtime judge is wrong in each direction.

**5. Threshold tuning from data:** `MIN_RELEVANCE`, the semantic-cache similarity, and the fuzzy-quote ratio. Each is a precision/recall knob currently set from a handful of observations.

**6. CI regression gate.** Run the retrieval metrics on every push (fast and free). Run the answer metrics on prompt or model changes, and fail the build if "Not found" precision or citation precision drops beyond a tolerance.

**7. Production feedback loop.** Log reviewer edits and overrides as labelled data. Track the judge's rejection rate and the retrieval-gate rate per customer document type as drift signals.

## Other things I'd do with more time

- Persist indexes (pgvector) and caches (Redis) so they survive restarts and can be shared across workers.
- Table-aware extraction for SOC 2 test matrices (control ID → description → test → result as structured fields), so "were there exceptions for CC6.1?" becomes a lookup.
- OCR fallback for scanned reports.
- Stream results per question (SSE) so the UI can show answers as they complete.
- A `/metrics` endpoint (Prometheus) for latency histograms, LLM calls and cache hit rates. The same data is in the JSON logs today.

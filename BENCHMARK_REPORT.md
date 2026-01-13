# Benchmark Report: PII Masking Context Retention

> **Seed:** 42 | **Evaluator:** GPT-4o-mini | **Data:** 100% Synthetic

---

## Summary

| Benchmark | Generic Redaction | Semantic Masking |
|-----------|-------------------|------------------|
| **Coreference** (entity tracking) | 27% retained | 100% retained |
| **RAG QA** (relationship questions) | 17% retained | 92% retained |

**Baseline:** Original unmasked text = 100%

---

## 1. Coreference Resolution

**Question:** Can the LLM track multiple people through a document?

**Example:**
```
"Emma Roberts is the team lead. Emma presented the proposal."
Q: Who presented the proposal?
```

| Strategy | Result |
|----------|--------|
| Original | ✅ "Emma Roberts" |
| `<PERSON>` tags | ❌ Ambiguous (both are `<PERSON>`) |
| `{Name_hash}` | ✅ Returns correct hash → unmasks to "Emma Roberts" |

**Key Challenge Solved:** The "Emma Problem" — when "Emma Roberts" and "Emma" appear in the same text, semantic masking links them to the same hash.

---

## 2. RAG Question Answering

**Question:** After retrieval, can the LLM answer "who did what"?

**Corpus:** 5 enterprise documents (HR, Support, Security logs)  
**Questions:** 12 relationship queries

| Strategy | Retention | Sample Failure |
|----------|-----------|----------------|
| Original | 100% | — |
| Generic | 17% | Q: "Who approved?" A: `<PERSON>` ❌ |
| Semantic | 92% | Rare format hallucination |

---

## Methodology

1. Generate synthetic documents with known PII
2. Mask with each strategy
3. Ask LLM relationship questions
4. Compare answers to ground truth
5. For semantic masking: unmask answer before comparison

**Scripts:**
- [`context_research/01_coreference_benchmark.py`](context_research/01_coreference_benchmark.py)
- [`context_research/02_rag_qa_benchmark.py`](context_research/02_rag_qa_benchmark.py)

---

## Reproduce

```bash
pip install privalyse-mask presidio-analyzer presidio-anonymizer openai
export OPENAI_API_KEY="sk-..."
python context_research/01_coreference_benchmark.py
python context_research/02_rag_qa_benchmark.py
```

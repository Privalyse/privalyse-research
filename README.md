# PII Masking Research: Context Retention Benchmarks

> **Question:** How much reasoning capability do LLMs lose when you mask PII?  
> **Answer:** With generic redaction (`<PERSON>`), they lose ~90%. With semantic masking (`{Name_hash}`), they retain ~100%.

[![Seed](https://img.shields.io/badge/seed-42-blue)](#reproducibility)
[![Data](https://img.shields.io/badge/data-100%25%20synthetic-green)](#data)

---

## The Problem

You want to use LLMs on sensitive documents (HR files, support tickets, medical records). Compliance says you can't send raw PII. So you mask it.

**But masking destroys context:**
```
Original: "John's manager Sarah approved the request."
Masked:   "<PERSON>'s manager <PERSON> approved the request."
```

Now the LLM can't answer "Who approved the request?" — everyone is `<PERSON>`.

---

## Our Finding: Semantic Masking Preserves Context

Replace entities with **distinguishable placeholders**:
```
Semantic: "{Name_a3f2}'s manager {Name_b7c9} approved the request."
```

The LLM answers `{Name_b7c9}`. We unmask it → `Sarah`. ✅

---

## Benchmark Results

### 1. Coreference Resolution (Entity Tracking)

**Test:** Can the LLM track "who did what" across a document with multiple people?

| Strategy | Context Retention |
|----------|-------------------|
| Original (baseline) | 100% |
| Generic Redaction (`<PERSON>`) | **~30%** |
| Semantic Masking (`{Name_hash}`) | **~100%** |

**Script:** [`context_research/01_coreference_benchmark.py`](context_research/01_coreference_benchmark.py)
**Results:** [`results/coref_benchmark_20260111_232715.json`](results/coref_benchmark_20260111_232715.json)

### 2. RAG Question Answering

**Test:** After retrieving a masked document, can the LLM answer relationship questions?

| Strategy | Context Retention |
|----------|-------------------|
| Original (baseline) | 100% |
| Generic Redaction | **~10%** |
| Semantic Masking | **92-100%** |

**Script:** [`context_research/02_rag_qa_benchmark.py`](context_research/02_rag_qa_benchmark.py)  
**Results:** [`results/rag_qa_benchmark.json`](results/rag_qa_benchmark.json)

---

## Quick Start

```bash
# Install
pip install privalyse-mask presidio-analyzer presidio-anonymizer openai

# Set API key (for LLM evaluation)
export OPENAI_API_KEY="sk-..."

# Run Coreference Benchmark
python context_research/01_coreference_benchmark.py

# Run RAG QA Benchmark
python context_research/02_rag_qa_benchmark.py
```

---

## Reproducibility

- **Seed:** 42 (all randomness is seeded)
- **Data:** 100% synthetic (no real PII)
- **Evaluator:** GPT-4o-mini (temperature=0)
- **Embedding:** text-embedding-3-small

---

## Repository Structure

```
privalyse-research/
├── README.md                      # This file
├── context_research/
│   ├── 01_coreference_benchmark.py   # Entity tracking test
│   └── 02_rag_qa_benchmark.py        # RAG QA test
├── results/
│   └── rag_qa_benchmark.json      # Latest results
└── _archive/                      # Old experiments (for reference)
```

---

## Key Insight

> **The LLM doesn't need to know WHO the person is.**  
> **It just needs to know that Person A ≠ Person B.**

Semantic placeholders preserve the **relationship graph** while removing the actual identities.

---

## License

MIT

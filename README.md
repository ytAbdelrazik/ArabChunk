# Arabic Ontology Chunker

> Semantically-aware text chunking for Arabic, built around Arabic morphology — not adapted from English tools.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Offline](https://img.shields.io/badge/runs-100%25%20offline-brightgreen)](docs/)

---

## Motivation

Modern RAG (Retrieval-Augmented Generation) pipelines chunk documents into pieces before embedding them for search.  Generic chunkers — fixed windows, recursive character splits — were designed for English.  Applied to Arabic they produce:

- **Broken morphological units**: Arabic clitics (`و`, `ال`, `ب`) attach to stems; whitespace tokenisation severs meaning.
- **Mixed-topic chunks**: A fixed 500-token window may span a politics paragraph and an economics paragraph, producing an incoherent embedding that retrieves poorly.
- **No explainability**: You cannot tell *why* a boundary was placed where it is.

This library addresses all three by building a pipeline that is:

| Property | How |
|---|---|
| **Offline** | Zero API keys, zero network calls |
| **Reproducible** | Same input → same output, always |
| **Extendable** | Add a domain by dropping a JSON file |
| **Explainable** | Every boundary has a `boundary_reason` field |
| **Arabic-first** | CAMeL Tools morphology, Arabic punctuation, Arabic domain ontologies |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Input Arabic Text                     │
└───────────────────────────┬────────────────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │    Normalizer      │  Remove diacritics, tatweel,
                  │  (normalizer.py)   │  normalise alef/yeh variants
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │   Sentence Split   │  Detect Arabic sentence
                  │  (tokenizer.py)    │  boundaries (. ؟ ! ، \n)
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  Morphology Layer  │  CAMeL Tools root extraction
                  │  (morphology.py)   │  + heuristic fallback
                  └─────────┬─────────┘
                            │
              ┌─────────────▼─────────────┐
              │     Sliding Window         │  window=3 sentences, step=1
              │     (OntologyChunker)      │
              │                           │
              │  for each window:         │
              │  ┌───────────────────┐    │
              │  │  ConceptTagger    │    │  Match roots/keywords
              │  │  (offline)        │    │  against domain JSONs
              │  │                   │    │  Score: |match| / √|vocab|
              │  └────────┬──────────┘    │
              │           │               │
              │  ┌────────▼──────────┐    │
              │  │ detect_shift()    │    │  concept differs AND
              │  │ threshold=0.7     │    │  confidence ≥ threshold
              │  └────────┬──────────┘    │
              └───────────┼───────────────┘
                          │
                ┌─────────▼─────────┐
                │  Size Enforcement  │  merge(min=3 sents)
                │                   │  split(max=15 sents)
                └─────────┬─────────┘
                          │
          ┌───────────────▼───────────────────────┐
          │           OntologyChunk                │
          │  {                                     │
          │    text, concept, concept_en,          │
          │    confidence, keywords,               │
          │    sentence_count, chunk_index,        │
          │    boundary_reason                     │  ← explainability
          │  }                                     │
          └───────────────────────────────────────┘
```

### Domain ontology files (`data/domains/*.json`)

The concept vocabulary lives in plain JSON files — one file per domain.
Adding a new domain requires zero code changes:

```json
{
  "concept":    "رياضة",
  "concept_en": "Sports",
  "keywords":   ["كرة", "ملعب", "لاعب", "بطولة", "هدف"],
  "roots":      ["كرر", "لعب", "طول", "هدف"]
}
```

Drop the file into `data/domains/` and re-instantiate `ConceptTagger`.

---

## Installation

```bash
git clone https://github.com/your-org/arabic-ontology-chunker
cd arabic-ontology-chunker

pip install -r requirements.txt

# Optional: download CAMeL Tools morphology database for better root extraction
camel_data -i morphology-db-msa-r13
```

---

## Quick Start

```python
from src.chunker.ontology_chunker import OntologyChunker

chunker = OntologyChunker()
chunks  = chunker.chunk_rich("النص العربي الذي تريد تقطيعه يذهب هنا ...")

for c in chunks:
    print(c.chunk_index, c.concept, c.confidence, c.boundary_reason)
```

**Using the standard dict interface** (same schema across all chunkers):

```python
chunks = chunker.chunk_dicts(text)
# Each chunk: {"text", "concept", "concept_en", "keywords",
#              "sentence_count", "chunk_index", "boundary_reason"}
```

**Chunking a file:**

```python
chunks = chunker.chunk_file("data/news/article_1.txt")
```

---

## How It Works

### Offline ontology matching

Rather than calling an LLM, the `ConceptTagger` compares the roots and keywords in a text window against pre-built domain vocabularies stored as JSON files.

**Scoring formula:**

```
score(domain) = |matched_unique_terms| / sqrt(|domain_vocabulary|)
```

Dividing by the square root of vocabulary size rewards *tight, specific* domains over large catch-all ones — a domain with 20 keywords that matches 5 of them scores higher than a domain with 2000 keywords that matches the same 5.

Confidence is mapped to [0, 1] via a soft cap:

```
confidence = min(1.0, score / 0.5)
```

### Root-based matching

Arabic `كَتَبَ` (wrote), `كِتَاب` (book), `مَكْتَبَة` (library) all share the root `كتب`.  The tagger extracts roots with CAMeL Tools (falling back to a heuristic prefix/suffix stripper) and matches them against root vocabularies in the domain JSONs.  This means a domain file does not need to enumerate every surface form of every word.

### Boundary detection

```
detect_shift(concept_a, concept_b, threshold=0.7)
  → True  iff  concept_a ≠ concept_b  AND  concept_b.confidence ≥ threshold
```

No boundary is placed unless the incoming concept has enough evidence.  This avoids spurious splits caused by a single off-topic sentence.

---

## Benchmark Results

> Run `python -m benchmarks.run_benchmark` to reproduce.  Fill in after running.

| Chunker | P@1 | P@3 | Coherence | Purity |
|---|---|---|---|---|
| **OntologyChunker** | — | — | — | — |
| FixedChunker | — | — | — | — |
| RecursiveChunker | — | — | — | — |
| SemanticChunker | — | — | — | — |

*Evaluated on 5 Arabic documents (news + Islamic text), 20 QA pairs, embedded with `aubmindlab/bert-base-arabertv02`, retrieved via FAISS flat inner-product index.*

---

## Comparison: Approaches to Arabic Chunking

| Feature | OntologyChunker | FixedChunker | RecursiveChunker | SemanticChunker |
|---|---|---|---|---|
| Topic-aware splits | ✅ | ❌ | ❌ | Partial |
| Fully offline | ✅ | ✅ | ✅ | ✅* |
| Arabic morphology | ✅ | ❌ | ❌ | ❌ |
| Explainable boundaries | ✅ | ❌ | Partial | ❌ |
| Extendable domains | ✅ (JSON) | ❌ | ❌ | ❌ |
| Speed | Fast | Fastest | Fast | Slow** |

*\* Requires model download on first run.*
*\*\* Embedding each sentence is slow on CPU.*

---

## Project Structure

```
arabic-ontology-chunker/
├── src/
│   ├── preprocessing/
│   │   ├── normalizer.py       Arabic text normalisation
│   │   ├── tokenizer.py        CAMeL Tools wrapper + sentence splitter
│   │   └── morphology.py       Root extraction, lemmatisation
│   ├── chunker/
│   │   ├── base_chunker.py     Abstract base + shared chunk_dicts() schema
│   │   ├── concept_tagger.py   Offline domain-JSON tagger with caching
│   │   └── ontology_chunker.py Full pipeline with boundary explainability
│   ├── baselines/
│   │   ├── fixed_chunker.py    Fixed token windows
│   │   ├── recursive_chunker.py Hierarchical separator splitting
│   │   └── semantic_chunker.py Cosine-similarity sentence splits
│   └── evaluation/
│       └── metrics.py          P@K, coherence, purity, boundary F1
├── data/
│   ├── domains/                Domain ontology JSON files (add here)
│   ├── news/                   3 Arabic news samples
│   ├── islamic/                Quran tafsir + hadith samples
│   └── qa_pairs.json           20 QA pairs for RAG evaluation
├── benchmarks/
│   └── run_benchmark.py        Full benchmark pipeline
├── tests/
│   ├── test_concept_tagger.py
│   └── test_ontology_chunker.py
└── notebooks/
    └── demo.ipynb
```

---

## Roadmap

- [ ] **True OWL/RDF ontology integration** — link domains to Wikidata/DBpedia concepts for cross-lingual alignment
- [ ] **Dialect support** — Egyptian, Levantine, Gulf Arabic domain variants
- [ ] **CAMeL Tools full pipeline** — POS tagging and named-entity features as additional boundary signals
- [ ] **Pip package** — `pip install arabic-ontology-chunker`
- [ ] **Streaming API** — chunk very large documents without loading them fully into memory
- [ ] **Domain auto-discovery** — unsupervised extraction of domain keywords from a seed corpus
- [ ] **Evaluation dataset** — human-annotated Arabic text with gold-standard chunk boundaries

---

## Contributing

1. Fork the repo and create a feature branch.
2. To **add a new domain**: create `data/domains/<name>.json` following the existing schema — no code changes required.
3. To **improve the tagger**: edit `src/chunker/concept_tagger.py` — the test suite in `tests/test_concept_tagger.py` covers scoring, caching, and shift detection.
4. Run the test suite: `pytest tests/ -v`
5. Open a pull request with a description of the change and, if adding a domain, sample sentences that triggered correct classification.

---

## License

MIT License — see [LICENSE](LICENSE).

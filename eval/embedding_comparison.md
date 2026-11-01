# Embedding model comparison: bge-small vs bge-base

Run on 30 October 2026 with `compare_retrieval.py`.

| | bge-small | bge-base |
|---|---|---|
| Model | `BAAI/bge-small-en-v1.5` | `BAAI/bge-base-en-v1.5` |
| Dimensions | 384 | 768 |
| Collections | `arla_recursive`, `arla_parent` | `arla_recursive_bge-base-en-v1.5`, `arla_parent_bge-base-en-v1.5` |

## Setup

- 20 retrieval questions (`eval/questions.csv`): 10 policy, 4 multi-year, 6 numeric. Top 5 results.
- Both models use the same LLM rewrites, cached in `eval/rewrites.json`, so only the embedding model differs.
- Chunking: recursive (800 / 150) and parent-child (parent 1,500 / 200, child 300 / 50). Chunk counts: 4,789 recursive and 13,647 parent-child.
- Modes: `none` raw question; `query` rewritten query; `year` raw question + year filter; `qyear` rewritten query + year filter; `full` rewritten query + year + doc_type filter.
- Metrics: hit@k (expected file and an accepted page in the top k), MRR, kw@5 (expected keyword in the top 5 texts).

## Results

Each cell is hit@1 / hit@3 / hit@5 / MRR / kw@5.

| Strategy | Mode | bge-small | bge-base |
|---|---|---|---|
| recursive | none | .30 / .50 / .70 / .44 / .55 | .50 / .60 / .65 / .55 / .65 |
| recursive | query | .40 / .50 / .60 / .46 / .60 | .45 / .50 / .60 / .49 / .60 |
| recursive | year | .55 / .80 / .85 / .67 / .70 | .65 / .75 / .75 / .69 / .75 |
| recursive | qyear | .55 / .80 / .90 / .70 / .80 | .55 / .75 / .85 / .66 / .80 |
| recursive | full | .50 / .70 / .75 / .61 / .65 | .45 / .65 / .70 / .55 / .65 |
| parent | none | .50 / .60 / .60 / .54 / .60 | .55 / .60 / .70 / .59 / .70 |
| parent | query | .35 / .60 / .60 / .47 / .60 | .35 / .60 / .65 / .48 / .65 |
| parent | year | .65 / .80 / .85 / .72 / .80 | .70 / .90 / .90 / .78 / .85 |
| **parent** | **qyear** | .65 / .90 / .90 / .76 / .85 | .60 / .95 / .95 / .76 / .90 |
| parent | full | .55 / .75 / .75 / .64 / .70 | .55 / .80 / .80 / .66 / .75 |

### Chosen configuration (parent-child, rewritten query + year filter), by question type

| Type | n | bge-small hit@5 / MRR | bge-base hit@5 / MRR |
|---|---|---|---|
| policy | 10 | 0.80 / 0.68 | 0.90 / 0.80 |
| multi_year | 4 | 1.00 / 0.71 | 1.00 / 0.71 |
| numeric | 6 | 1.00 / 0.92 | 1.00 / 0.72 |

### Cost

| | bge-small | bge-base |
|---|---|---|
| Retrieval time per query (parent, qyear) | about 15 ms | about 25 ms |
| Vector size per chunk | 384 floats | 768 floats |

The whole `chroma_db` folder (both models, both strategies) is 356 MB. Index build time was not measured.

## Conclusion

- **No reasonable improvement.** In the chosen configuration MRR is equal (0.76). `bge-base` gains one question at hit@3, hit@5 and kw@5 and loses one at hit@1. On the recursive baseline it is slightly worse.
- **By type:** `bge-base` is better on policy questions (it fixed the facilitation-payments question but missed two others, child labour and the German complaints procedure). It is worse at ranking numeric questions (MRR 0.92 to 0.72).
- **The differences are within noise.** One question is 5 points, and the 95% interval on a score of 0.80 is about ±0.18.
- **Run-to-run variation is as large as the model effect.** The earlier recorded run (`DESIGN.md` section 5.5, different rewrites) scored 0.70 / 0.95 / 0.95 / 0.81 for bge-small in this configuration, against 0.65 / 0.90 / 0.90 / 0.76 here. That shift comes from LLM rewriting alone.
- **Shared miss:** "What share of Arla's group revenue does the UK contribute?" misses in both runs. The retrieval gold page is the 2022 UK tax strategy, while the system retrieves a newer annual report (the label issue described in `DESIGN.md` section 12.4).
- **Decision:** keep `bge-small`. A model three times larger gives no measurable gain, has a larger index and slower ingest, and the grading thresholds (0.30 and 0.45) and the policy limit (0.35) were tuned on bge-small distances. A larger evaluation set would be needed to detect a small gain.

## Reproduce

```
set EMBED_MODEL=BAAI/bge-base-en-v1.5
python ingest.py
python compare_retrieval.py
```

Without `EMBED_MODEL`, the scripts use bge-small. Raw outputs: `eval/compare_small.txt`, `eval/compare_base.txt`, `eval/ingest_base.txt`.

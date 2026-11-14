# Retrieval: full corpus, annotated evidence

100 questions; 6040 searchable chunks; no chapter filter.
Queries use annotated textbook evidence. Any-evidence Hit/MRR differ from complete evidence coverage.

| Split | Language | Method | Cases | Hit@5 | MRR@5 | All evidence@5 | Coverage@5 | Candidate all@20 | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | en | hybrid | 30 | 0.833 | 0.706 | 0.767 | 0.800 | 0.900 | 0.08 |
| dev | en | hybrid_rerank | 30 | 0.900 | 0.844 | 0.867 | 0.883 | 0.900 | 14.27 |
| dev | vi | hybrid | 30 | 0.367 | 0.252 | 0.333 | 0.350 | 0.667 | 0.08 |
| dev | vi | hybrid_rerank | 30 | 0.667 | 0.611 | 0.633 | 0.650 | 0.667 | 15.53 |
| eval | en | hybrid | 20 | 0.900 | 0.860 | 0.750 | 0.825 | 0.800 | 0.08 |
| eval | en | hybrid_rerank | 20 | 0.950 | 0.925 | 0.800 | 0.875 | 0.800 | 11.83 |
| eval | vi | hybrid | 20 | 0.350 | 0.183 | 0.300 | 0.325 | 0.450 | 0.08 |
| eval | vi | hybrid_rerank | 20 | 0.500 | 0.500 | 0.400 | 0.450 | 0.450 | 13.78 |

Evidence labels were fixed before retrieval and were not expanded after inspecting results.
Annotations are not exhaustive relevance qrels; another valid passage can be scored as a miss.
Paired translations share a split. Authoring and scoring use the coding agent, without independent human review.
Topics overlap earlier project experiments; this is not a wholly unseen final benchmark.
Timing includes live hybrid retrieval plus reranking, with candidates shared across methods.
Model/index setup, token preflight and one warm-up are excluded. One pass; no p95 claim.
This run evaluates retrieval only; no LLM answers or API calls.

Files: queries.jsonl, frozen_config.json, per_query.csv, topk.jsonl, summary.csv, overall.csv, evidence_misses.csv.

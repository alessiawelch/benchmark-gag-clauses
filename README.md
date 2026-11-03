# Benchmark Gag Clauses: Data and Code

Data and code behind the paper's tables and figures.

## Setup

```
pip install pandas numpy scipy statsmodels matplotlib
```

## Scripts

Each script corresponds to one of the paper's tables or figures. Run using python.

| Script | Output | Files in `outputs/` |
|---|---|---|
| `table_1_model_performance.py` | Model performance on the 123-document validation set, with statistical tests | `table_1_model_performance.csv` |
| `table_2_scope_prevalence.py` | Prevalence of the five clauses by document scope, for documents and for firms | `table_2_scope_prevalence.csv` |
| `table_3_industry_prevalence.py` | Clause prevalence by industry (17,637 product terms) | `table_3_industry_prevalence.csv` |
| `figure_2_growth.py` | Growth of the five clauses, panels A and B | `figure_2_growth.{pdf,png}` |
| `figure_3_industry_revenue.py` | Clause prevalence by industry over time and by company revenue | `figure_3_industry_revenue.{pdf,png}` |
| `figures_4-8_clause_adoption.py` | Clause adoption over time, one figure per clause | `figure_4_clause_adoption_benchmarking` to `figure_8_clause_adoption_conditional` (`.pdf`, `.png`) |
| `figure_9_window_coverage.py` | Product-term documents observed in the start and end year of each window | `figure_9_window_coverage.{pdf,png}` |

`_common.py` holds the helpers the scripts share.

## Data

**`data/documents.jsonl`**: one line per document in the paper's universe (30,003).

```json
{"document_id": 2197, "name": "Snyk", "company_domain": "snyk.io", "spciq_id": "...", "url": "https://snyk.io/policies/terms-of-service/",
 "scope": "Service", "product_terms": true, "version_id": 981141, "version_date": "2026-08-13",
 "clauses": [{"label": "Benchmarking and Results Restrictions", "text": "...", "run": "full_corpus_run"}],
 "text_zip": "part-003.zip", "text_path": "2197/termtrove_981141_2026-08-13.txt"}
```

- `scope` is the document's scope from Harvey (Service, Mixed, Site-only, Sales terms). `product_terms` = Service or Mixed.
- `clauses` lists the clauses Harvey found in the document's latest version (TermTrove copy as of September 22, 2026).
- Clause and scope labels come from Claude Opus 4.7 on Harvey, using the prompts in the paper's Appendix D (clauses) and Appendix E (scope).

**`data/history.jsonl`**: one line per archived observation, dated on or before September 22, 2026.

```json
{"document_id": 2197, "source": "wayback", "id": 1379070, "date": "2025-01-03",
 "labels": ["Benchmarking and Results Restrictions", "Evaluation-Related Confidentiality Restrictions"],
 "text_zip": "...", "text_path": "2197/wayback_1379070_2025-01-03.txt"}
```

- `source`: `wayback` (a Wayback Machine snapshot) or `termtrove` (a TermTrove version).

**`data/text/part-001.zip` to `part-040.zip`**: the full text of all 229,368 versions (59,273 TermTrove versions and 170,095
Wayback pages), one folder per document: `<document_id>/termtrove_<version_id>_<date>.txt` and
`<document_id>/wayback_<snapshot_id>_<date>.txt`. `text_zip` and `text_path` in the files above point to them.

**`data/validation/`**: the hand-labeled samples.

- `validation_labels.csv`: the 123-document clause validation set (read by `table_1_model_performance.py`).
- `dev_sample_clause_detection.csv` (98 documents) and `test_sample_clause_detection.csv` (the same 123 documents): one row
  per hand-labelled clause, with its text and label. `None` = a candidate clause that is not a restriction. 
- `dev_sample_doc_scope.csv` (documents 1-50) and `test_sample_doc_scope.csv` (51-100): the 100-document sample of
  document scope, with the final hand label (Service, Mixed, Site-only, Sales terms).
- `validation_extra.zip`: the text of the 18 clause-sample documents outside the 30,003-document universe.

Each sample row gives `document_id`, `version_id`, `url`, and `text_zip` / `text_path`.
`text_zip` is a `data/text/part-NNN.zip` shard, or `validation_extra.zip` for the 18 documents above.

**Capital IQ Pro Data**: `table_3_industry_prevalence.py` and `figure_3_industry_revenue.py` read S&P Capital IQ
company fields from `data/capiq/capiq_company_info.csv`. Users with Capital IQ
access can build the file with the columns `spciq_id, iq_industry, iq_total_rev` (`iq_total_rev` is total revenue in
thousands of US dollars).

# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that turns messy order exports into a trusted database (Part A), segments customers and predicts lapse (Part B), triages support tickets with privacy and cost guardrails (Part C), and answers policy questions and decides refunds with a tool-calling agent (Part D).

All data is synthetic and fictional (made by `generate_data.py`); all money is in INR. Every language-model call goes through the offline `mock_llm.py`: no API key, account or network is used anywhere. `generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. Libraries: Python standard library, pandas, numpy, scikit-learn, joblib.

Layout: `src/a*_*.py` (Part A), `src/b*_*.py` (Part B), `src/c_triage.py` (Part C), `src/d1_rag.py` and `src/d2_agent.py` (Part D), `sql/queries.sql`, `data/`, `outputs/`, `state/`. All scripts are run from the repository root.

Versions used for the numbers below: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0, SQLite 3.53.

## Reproduce

```bash
git clone <repo-url> instamart-copilot && cd instamart-copilot
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py              # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/a1_audit.py               # Part A: audit of raw data
python src/a2_clean.py               # Part A: clean + write data/clean/*.csv and data/instamart.db
python src/a3_validate.py            # Part A: prints ALL CHECKPOINTS PASSED
python src/a4_sql.py                 # Part A: runs the 7 queries in sql/queries.sql
python src/a5_pandas.py              # Part A: pandas cross-check and EDA
python src/b1_segments.py            # Part B: RFM, K-means, PCA, outputs/customer_segments.csv
python src/b2_lapse.py               # Part B: lapse model, outputs/lapse_model.joblib, outputs/retention_list.csv
python src/c_triage.py               # Part C: outputs/triage_budget_demo.csv, outputs/triage_results.csv
python src/d1_rag.py                 # Part D: chunking, retrieval, evaluation
python src/d2_agent.py               # Part D: 7 scenarios, demos, outputs/agent_runs.jsonl, refund_ledger.jsonl, state/
```

`requirements.txt` pins the five libraries and their dependencies. Order matters: Part B needs Part A, Part C needs `outputs/customer_segments.csv` from Part B, and Part D needs `data/instamart.db` from Part A. `d2_agent.py` deletes and rebuilds `outputs/agent_runs.jsonl`, `outputs/agent_demos.jsonl`, `outputs/refund_ledger.jsonl` and `state/*.json` at the start of each run, so repeated runs do not duplicate ledger lines.

## Part A

### Audit (raw)

| Table | Shape | Columns with missing values (count, %) |
|---|---|---|
| customers.csv | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%); customer_id, city, signup_date 0 |
| products.csv | 40 x 4 | none |
| orders.csv | 13084 x 9 | delivery_minutes 2541 (19.42%); all other columns 0 |

Data types (pandas): customers: customer_id/city/signup_date/gender str, age float64; products: product_id/product_name/category str, price_inr int64; orders: order_id/customer_id/product_id/order_ts/status/order_amount str, quantity and discount_pct int64, delivery_minutes float64. Note: `order_amount` is a string column because it holds the planted text values `N/A` and `error`. The audit reads files with `keep_default_na=False, na_values=[""]` so that the text `N/A` is counted as a bad value, not silently turned into a blank.

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount (53 `N/A` + 67 `error`) | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

All twelve equal the reference values.

### Cleaning rules and how many rows/values each changed

Customers (1,200 rows):
1. City: strip spaces and title-case, 24 spellings become 6. Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. Gender: 10 spellings plus 30 blanks mapped to male / female / unknown. Result: female 610, male 560, unknown 30.
3. Age: 6 values outside 10-100 set to missing (67 missing in total with the 61 blanks), then all 67 filled with the city median using `groupby().transform("median")`. Result: 0 missing, all ages 18-60.
4. `signup_date` converted to datetime (all 1,200 rows).

Orders (13,084 raw rows), applied in this order:
1. `order_ts` converted to datetime (all rows).
2. Exact duplicates removed: 119 rows.
3. For each `order_id` keep only the record with the latest `order_ts`: 656 earlier `pending` records removed. Result: 12,309 rows, one per order.
4. `order_amount` converted with `pd.to_numeric(errors="coerce")`: 112 unparseable values (the other 8 of the 120 were in rows already removed) rebuilt as `round(price_inr x quantity x (1 - discount_pct/100), 2)`. 0 missing remain.
5. Negative `delivery_minutes` set to missing: 105 values (the 106th was in a removed row). Then 774 missing delivery times of completed orders (the 105 negatives and the blanks) filled with the median of the product category among completed orders (`groupby().transform`). The 1,199 cancelled orders keep a missing delivery time because they were never delivered.

Result: 12,309 rows; 11,110 completed, 1,199 cancelled, 0 pending; 0 missing `order_amount`; 0 missing delivery among completed, 1,199 among cancelled; completed revenue INR 2,998,668.80. All reference checkpoints match.

### Store and validate

Clean tables are in `data/clean/{customers,products,orders}.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text). `src/a3_validate.py` asserts 17 checkpoints (row counts, status counts, missing counts, revenue, city counts, gender counts, age range, database table sizes) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result | Reference |
|---|---|---|
| Q1 WHERE then HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows; revenue sums to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 | match |
| Q3 scalar subquery | 11 products above the average catalogue price | 11 |
| Q4 LEFT JOIN | 25 customers without a completed order | 25 |
| Q5 CTEs | 550 customers above the average spend of INR 2,552.06 | match |
| Q6 window ranking | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK() != ROW_NUMBER() | match |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 | match |

Why ties cause the RANK() / ROW_NUMBER() difference: customers in the same city with identical total spend share one RANK(), but ROW_NUMBER() still gives them different consecutive numbers, so the second customer of each tied pair differs (8 tied pairs here).

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` grand total is INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Distribution of completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30; 0 outliers below the lower fence and 897 above the upper fence.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**What the data shows (under 200 words).** This is six months (April-September 2025) of 11,110 completed grocery, dairy, snack, personal-care and electronics orders from six cities. Order amounts are strongly right-skewed: mean INR 269.91 against median 135.00, skewness 4.61, quartiles 64.80 and 297.00. There are 897 outliers above the upper fence of INR 645.30 and none below; 635 of them are electronics, so they look like genuine expensive baskets rather than errors, and I kept them. The real anomalies were in the raw export (duplicates, negative delivery times, unparseable amounts) and are already cleaned. Relationships are weak. The correlation largest in absolute value is order_amount with discount_pct at -0.095; order_amount with quantity is +0.087, and delivery time is almost uncorrelated with everything (|r| at most 0.013). Correlations this small are negligible, and even a strong one would not establish causation: the -0.095 does not show that discounts lower amounts, it may simply reflect which items are discounted.

## Part B

Everything here uses only completed orders for customer behaviour, unless stated.

### RFM table (snapshot 2025-10-01 00:00:00)

1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55. All match the references.

### K-means (StandardScaler on the three RFM columns, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

All values match the references. **Choice: K = 4.** The elbow is at K=4: inertia drops by 41% from K=3 to K=4 (1268 to 743) but only 17% from K=4 to K=5. K=4 also has the highest silhouette (0.526), and its four groups are easy for the business to act on (see below).

Why inertia of K=8 is lower than K=2, and why that cannot pick K: inertia is the sum of squared distances to the nearest centroid, so adding centroids can only keep it equal or lower, and it reaches 0 when every customer is its own cluster. Lower inertia therefore always rewards more clusters, so only the *rate* of improvement (the elbow), the silhouette and interpretability can choose K.

### Segment profile (K = 4)

| Segment (cluster id) | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Business name | Recommended action |
|---|---|---|---|---|---|---|
| 1 | 138 | 24.1 | 6.77 | 7,129.70 | **Premium Big Spenders** (`is_premium` = true) | Early access to electronics drops and a premium support desk |
| 2 | 419 | 8.8 | 19.40 | 3,410.02 | Loyal Regulars | Subscription / bundle offers on groceries and dairy |
| 0 | 436 | 28.9 | 3.77 | 1,057.62 | Occasional Buyers | Cross-sell nudges and a free-delivery threshold offer |
| 3 | 182 | 109.0 | 2.20 | 685.99 | Lapsing / Dormant | Win-back coupon and a reminder push |

Cluster numbers are only labels with no ordering; names are assigned from the profile (highest mean monetary becomes Premium), never from the id. PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852 (match). `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (1,175 rows, 138 premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with at least one completed order before the cutoff. Label `lapsed` = no completed order from 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Features (history strictly before the cutoff plus the customer table): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 train / 232 test (test: 89 lapsed).

Baseline "always not lapsed" on the test set: accuracy 61.6%, lapsed-class F1 0.000.

**No leakage.** No order dated on or after 2025-09-01 entered the features (asserted in code), and the segment labels from the RFM clustering were not used as features. Every encoder and scaler sits inside a scikit-learn `Pipeline` (`StandardScaler` on numeric columns, `OneHotEncoder(drop="first")` on city, gender and favourite category), so it was fitted on training rows only, including inside every cross-validation fold; no encoder, scaler or resampler saw test rows. No feature column has a missing value (checked in code). The fitted scaler's mean of frequency is 8.1147 on the training rows versus 8.0285 for the full cohort; they differ because the scaler learned from the 924 training rows only, not from the test rows too.

### Tuning and test results

| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100}, cv=5 | C = 100 | 0.6506 |
| RandomForest(n_estimators=100, random_state=42) | max_depth {3,5,8} x min_samples_leaf {5,10,20}, cv=5 (45 fits) | max_depth 8, min_samples_leaf 10 | 0.6811 |

The logistic regression best C differs from the reference (1), which is fine because the F1 differences between C values are small; the random forest best parameters equal the reference.

Test set (lapsed = positive class; confusion matrix is [[TN, FP], [FN, TP]]):

| Model | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Baseline (never lapsed) | 0.000 | 0.000 | 0.000 | [[143, 0], [89, 0]] |
| Logistic regression | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest | 0.659 | 0.652 | 0.655 | [[113, 30], [31, 58]] |

**Selected model: random forest** (selected on cross-validated F1, 0.681 against 0.651, never on the test set; test F1 0.655, above the 0.55 requirement). Missing a customer who is about to lapse costs the whole future revenue of that customer, while wrongly sending a discount to a loyal customer costs only one coupon, so recall matters at least as much as precision. Both models catch the same 58 of 89 lapsing customers, so the forest's slightly higher precision (fewer wasted coupons, 30 against 32 false alarms) and its higher cross-validated F1 decide the choice. The forest can also capture non-linear effects of recency and frequency.

### Class-balance experiment

The 354 lapsed training rows were over-sampled with replacement (`random_state=42`, 216 duplicates added) to 570 per class, training rows only. The best logistic regression refit on it, evaluated on the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20), precision fell slightly (-0.055). For the retention team this means the balanced model catches 76 of 89 lapsing customers (up from 58) at the price of 53 false alarms instead of 32: it is the right choice if a coupon is cheap relative to a lost customer, and the un-balanced model is better if the campaign budget is tight.

### Interpretation, persistence and retention list

Top five random-forest features: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.052. Business reading: how often and how much a customer has bought dominate the prediction, and slow delivery and cancellations matter much less.

Logistic regression coefficients (standardised features):

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | -1.698 | the more orders so far, the less likely to lapse (habit protects) |
| recency | +0.040 | the longer since the last order, the slightly more likely to lapse |
| monetary | +0.285 | at equal frequency, big-ticket customers are a bit more likely to lapse (they buy rarely and are not habitual) |

The selected model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions and probabilities exactly (checked in code).

`outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`): of the 40 Premium customers in the test set, the 10 with the highest predicted lapse probability are listed (probabilities 0.452 to 0.491); **1 of the 10 truly lapsed**. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes September, the month that defines the label.

## Part C

### Guardrails and redaction

Input guardrails: 4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, 734 estimated tokens each, limit `MAX_INPUT_TOKENS = 400`); none of these 8 ever reaches the model. Redaction (email, then phone, then order id, with the exact regular expressions and placeholders of the brief):

| Type | Substitutions |
|---|---|
| email -> `[EMAIL]` | 55 |
| phone -> `[PHONE]` | 91 |
| order id -> `[ORDER_ID]` | 132 |
| total | 278 |

Re-applying the three patterns to the redacted texts finds 0. The user message sent to the model is exactly the redacted text, and an assertion before every call checks that no email, phone or order id is in any message.

### Prompts

Zero-shot: `You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. Reply with JSON only, in the form {"category": "...", "reason": "..."}.` (33 estimated tokens, no `EXAMPLES:`). Few-shot: the same text plus a line `EXAMPLES:` with four labelled examples (billing, delivery, technical and one fraud example), 124 estimated tokens.

### Retries and validation

Every call goes through a retry function (at most 3 retries, wait `0.05 x 2**n x uniform(0.9, 1.15)` seconds, one log line per retry, `mock_llm.reset()` right before each run). A full run over the 156 eligible tickets had 19 TransientLLMErrors, all recovered by a retry (19 log lines, all at retry number 0, waits between 0.0481 and 0.0573 s, inside the 0.045-0.0575 s range); no ticket ended with a permanent error. Example log line: `RETRY ticket=T0002 retry_number=0 wait=0.0556s`.

Validation: valid JSON with an allowed category becomes `classified_json`; non-JSON replies are parsed with `re.findall(r"[a-z]+", text.lower())` and accepted as `classified_keyword` if exactly one allowed category word appears; anything else (including JSON with category `shipping`) becomes `flagged_invalid_output` and goes to human review, never guessed.

| Status (full run, both prompts) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot (no token budget)

| | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 9,556 | 65.01 |
| Few-shot | 147 | 100.0% (147/147) | 23,752 | 161.58 |

Total tokens count all model calls including the 9 answers later flagged invalid. Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10). The few-shot prompt costs about 96.6 more tokens per classified ticket (2.5 times the total tokens: 23,752 against 9,556) and removes all 21 errors. It is worth paying for when a wrong category is expensive: here the 11 fraud tickets sent to billing would miss the 2-hour security-team deadline of the escalation policy. It is not worth paying for if tickets are only roughly sorted by humans afterwards and cost per ticket is the main concern; then the cheaper zero-shot prompt with its 85.7% accuracy is enough.

### Token budget, routing and outputs

Budget demo (zero-shot, `TOKEN_BUDGET = 6000`): **58 tickets ended `budget_exhausted`** (the system prompt is about 33 estimated tokens); saved to `outputs/triage_budget_demo.csv`. Final `outputs/triage_results.csv` (few-shot, `TOKEN_BUDGET = 100000`, 164 rows, 0 budget_exhausted), routed with `outputs/customer_segments.csv` (customers without a segment row are treated as not premium):

| Route | Tickets | Priority |
|---|---|---|
| support_queue | 102 | NORMAL |
| security_team | 34 | HIGH |
| premium_desk | 11 | HIGH |
| human_review | 17 | NORMAL |

**Rule-validated versus human:** 147 tickets were classified and passed the output validation (135 by JSON schema check, 12 by keyword extraction) and were routed automatically. 17 tickets went to human review: 4 empty, 4 too long, and 9 whose model answer was invalid.

## Part D

### Chunking and retrieval

40-word chunks with 10-word overlap give 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2 (match). `retrieve(query, k=3)` uses `TfidfVectorizer()` fitted on the chunk texts and cosine similarity. For the query `refund policy late delivery` the top three chunks are all from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, distinctive words count more. A limitation is that TF-IDF matches words, not meaning: a question phrased as "money back" would not match a chunk that says "refund".

`answer(question)` retrieves three chunks and sends a system message (contains `POLICY ASSISTANT`, answer only from the context, otherwise say the information was not found) and a user message `CONTEXT:\n[file]\nchunk...\nQUESTION:\nquestion` to `mock_llm.chat`; it returns the text and the sources.

### Retrieval and grounding evaluation (`policy_eval.csv`, 14 questions)

| Setting | Answerable: expected_doc in top 3 | Answerable: answer contains expected_fact | Unanswerable: fallback `I could not find...` |
|---|---|---|---|
| 40 words, overlap 10 (15 chunks) | 12/12 | 12/12 | 2/2 |
| 40 words, overlap 0 (13 chunks) | 12/12 | 10/12 | 2/2 |

With overlap 0, two questions break because the sentence holding the answer is cut across a chunk boundary. "What delivery time makes a completed order eligible for a full refund?" returns "receives a full refund of the order amount." without `more than 45 minutes`, which sits in the previous chunk. "How soon is money returned when a payment fails but is deducted?" returns the first half of the sentence without `within 48 hours`, which fell into the next chunk. The 10-word overlap repeats the boundary words so the sentence stays whole in at least one chunk.

### Tools

Five tools in `TOOLS_SPEC` (name, one-sentence description, JSON-style parameter spec) and `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (returns retrieve output under `results`), `calculate_refund` (100% of the order amount if delivery took more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (each append one JSON line with timestamp and tool name to `outputs/refund_ledger.jsonl`). Two safety details: `issue_refund` refuses amounts above INR 1000, and both ledger tools are idempotent (the same tool and order is never recorded twice), so a resumed or repeated run cannot pay out twice.

### ReAct agent: seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)`. Results (also in `outputs/agent_runs.jsonl`, one line per run with every step):

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,027 |
| S2 | REFUND | 135.00 | 5 | 1,027 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,014 |
| S4 | DENIED | 0.00 | 4 | 709 |
| S5 | DENIED | 0.00 | 3 | 415 |
| S6 | NOT_FOUND | 0.00 | 2 | 124 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,156 |

Refund amounts equal `expected_refund_inr` within 0.01 (asserted in code). S7 uses all six model calls under the default `token_limit=6000`.

### Stop-safely demonstrations (S1; logged in `outputs/agent_demos.jsonl`)

| Demonstration | Status | Detail |
|---|---|---|
| `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped after 3 model calls and 432 tokens, with 2 tools executed; the tool requested by the third call was not executed |
| `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (0 calls, 0 steps) |

Both fallbacks return the steps so far and a handoff text in `final_answer`, for example: `HANDOFF TO HUMAN: automated run stopped (time limit 0s reached) after 0 tool step(s) [] for scenario S1; please review the case manually.` S7 ends with FALLBACK_MAX_ITERATIONS after 6 calls.

### Crash and resume (S1)

After every tool observation a checkpoint is written to `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed, plus steps, call count and limits). A simulated crash was raised right after the second tool execution and `resume_agent("demo-S1-crash")` continued from the checkpoint (the time limit restarts on resume). Tools executed across the crash and the resume, in order:

1. `get_order`
2. `search_policy` (crash here, after its checkpoint was saved)
3. `calculate_refund`
4. `issue_refund`

Four executions, none repeated; final status REFUND with INR 128.25. The ledger holds 3 lines (S1 and S2 issued, S3 escalated); the resumed `issue_refund` found the S1 line already there and did not duplicate it. `state/` holds 9 checkpoints (7 scenario runs and 2 demo runs that executed tools).

### ReAct versus plan-and-execute (about 210 words)

Measured ReAct cost: 5 model calls for S1 (1,027 tokens), 3 for S5 (415), 2 for S6 (124). Every call re-sends the growing history, including the roughly 250-token policy search result, so tokens grow faster than calls. A plan-and-execute design asks for the whole step list once, so the happy path (S1) needs about 1 to 2 model calls and perhaps 250 to 300 tokens; that is an estimate, I did not build it. Unexpected observations are where it is weaker. If `get_order` returns NOT_FOUND or a cancelled order, the remaining planned steps (calculate, issue) are invalid, so the executor must detect the error and ask for a new plan, costing an extra call, and any step run before the check is wasted. ReAct reacts after every observation: S6 stops after 2 calls and S5 after 3, without planning steps that will never run. For this refund workflow I would choose ReAct, because money moves and the next action depends on the order's status, amount and delivery time, so each tool result should be inspected before the next step, and the hard limits (6 calls, token limit, timeout) keep its cost bounded. Plan-and-execute would be a reasonable optimisation for high-volume, uniform cases, with a replan on any error.

## Limitations and Next Steps

- All data is synthetic and the language model is a deterministic keyword mock, so the Part C and Part D accuracies (100% few-shot, 12/12 retrieval) say little about a real model; the mock's failures (every 9th call, chatty or invalid answers) are simulated.
- The Premium flag is a post-hoc whole-window label (its RFM includes September, the month that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation: only 1 of its 10 customers truly lapsed. A proper evaluation would compute the segment as of the cutoff.
- The lapse model uses one temporal cut-off and 1,156 customers; a rolling-window backtest and probability calibration are next steps, and the test set (232 rows) is small, so F1 differences of 0.01 are noise. Over-sampling raised recall at the price of precision; the right operating point depends on the real cost of a coupon versus a lost customer.
- Median imputation of delivery times and rebuilding of unparseable amounts from the price list are assumptions; both were documented with counts, but they could hide real data problems at the source.
- TF-IDF retrieval is lexical only; embeddings would handle paraphrases. The refund agent trusts the database and the simulated 7-day window; a real system needs authentication of the requester, the delivery date check, and human approval logs.
- `requirements.txt` was written from the installed versions because the build machine had no network for `pip freeze` in a fresh virtual environment; the pins are the versions the numbers above were produced with.

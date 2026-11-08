"""Part C: support-ticket triage with guardrails (redaction, retries, validation, budget, routing)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm

import json
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
PATTERNS = [("email", EMAIL_RE, "[EMAIL]"), ("phone", PHONE_RE, "[PHONE]"), ("order_id", ORDER_RE, "[ORDER_ID]")]

ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
            'Reply with JSON only, in the form {"category": "...", "reason": "..."}.\n'
            'EXAMPLES:\n'
            'Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The rider marked [ORDER_ID] as delivered but I never received it. -> {"category": "delivery", "reason": "missing package"}\n'
            'Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app error"}\n'
            'Ticket: There is an unauthorized transaction on my card linked to [ORDER_ID]. -> {"category": "fraud", "reason": "unauthorized transaction"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

RETRY_LOG, WAITS = [], []
STATS = {"transient": 0}
SENT_PROMPTS = []


def redact(text):
    counts = {}
    for name, rx, ph in PATTERNS:
        text, n = rx.subn(ph, text)
        counts[name] = n
    return text, counts


def llm_with_retry(messages, ticket_id):
    """Call the mock; on TransientLLMError retry up to MAX_RETRIES times with exponential backoff + jitter."""
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            STATS["transient"] += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            line = f"RETRY ticket={ticket_id} retry_number={retry} wait={wait:.4f}s"
            RETRY_LOG.append(line); WAITS.append((retry, wait))
            time.sleep(wait)


def validate(content):
    """Returns (status, category). Never guesses."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        pass
    words = re.findall(r"[a-z]+", (content or "").lower())
    found = {w for w in words if w in ALLOWED}
    if len(found) == 1:
        return "classified_keyword", next(iter(found))
    return "flagged_invalid_output", ""


def run_pipeline(tickets, system_prompt, token_budget=None):
    """Process tickets in file order. Returns a dataframe with one row per ticket."""
    mock_llm.reset()                       # immediately before each full run
    RETRY_LOG.clear(); WAITS.clear(); STATS["transient"] = 0; SENT_PROMPTS.clear()
    random.seed(0)
    total_tokens, out = 0, []
    for t in tickets.itertuples(index=False):
        rec = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            red, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            for m in messages:  # privacy assertion: nothing identifying reaches the model
                assert not any(rx.search(m["content"]) for _, rx, _ in PATTERNS), "PII in prompt!"
            SENT_PROMPTS.append(messages)
            resp = llm_with_retry(messages, t.ticket_id)
            used = resp.usage.total_tokens
            total_tokens += used
            rec["tokens_used"] = used
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        out.append(rec)
    return pd.DataFrame(out)


def route(df, seg_path="outputs/customer_segments.csv"):
    seg = pd.read_csv(seg_path).set_index("customer_id")
    rows = []
    for r in df.itertuples(index=False):
        in_seg = r.customer_id in seg.index
        seg_name = seg.loc[r.customer_id, "segment_name"] if in_seg else ""
        premium = bool(seg.loc[r.customer_id, "is_premium"]) if in_seg else False
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            rt, pr = "human_review", "NORMAL"
        elif r.category == "fraud":
            rt, pr = "security_team", "HIGH"
        elif premium:
            rt, pr = "premium_desk", "HIGH"
        else:
            rt, pr = "support_queue", "NORMAL"
        rows.append({"ticket_id": r.ticket_id, "customer_id": r.customer_id, "status": r.status, "category": r.category,
                     "segment_name": seg_name, "priority": pr, "route": rt, "tokens_used": r.tokens_used})
    return pd.DataFrame(rows)


def summarize(name, df):
    cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (cls["category"] == cls["true_category"]).mean()
    tot = int(df["tokens_used"].sum())
    print(f"\n[{name}] statuses: {df['status'].value_counts().to_dict()}")
    print(f"[{name}] classified {len(cls)} | accuracy {acc*100:.1f}% ({int((cls['category']==cls['true_category']).sum())}/{len(cls)}) "
          f"| total tokens {tot} | tokens per classified ticket {tot/len(cls):.2f}")
    print(f"[{name}] TransientLLMErrors: {STATS['transient']} | retry log lines: {len(RETRY_LOG)}")
    return cls, acc, tot


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

    # --- task 1/2: guardrail counts and redaction counts
    toks = tickets["text"].map(est_tokens)
    print("empty tickets:", int((~tickets['text'].str.strip().astype(bool)).sum()),
          "| long tickets (>400 est tokens):", int((toks > MAX_INPUT_TOKENS).sum()),
          "| long ticket est tokens:", sorted(set(toks[toks > MAX_INPUT_TOKENS])))
    tot = {"email": 0, "phone": 0, "order_id": 0}
    redacted_all = []
    for txt in tickets["text"]:
        r, cnt = redact(txt)
        redacted_all.append(r)
        for k, v in cnt.items():
            tot[k] += v
    print("substitutions:", tot, "| total:", sum(tot.values()))
    left = sum(len(rx.findall(r)) for r in redacted_all for _, rx, _ in PATTERNS)
    print("PII patterns found after redaction:", left)
    assert left == 0

    # --- task 4/5/6: full runs with each prompt (no budget)
    zero = run_pipeline(tickets, ZERO_SHOT)
    print("\n".join(RETRY_LOG[:3]), "\n...")
    assert all(retry == 0 for retry, _ in WAITS) and all(0.045 <= w <= 0.0575 for _, w in WAITS)
    print("all waits within [0.045, 0.0575]:", True, "| min %.4f max %.4f" % (min(w for _, w in WAITS), max(w for _, w in WAITS)))
    z_cls, z_acc, z_tot = summarize("zero-shot", zero)
    print("zero-shot confusion (rows=true, cols=predicted):")
    print(pd.crosstab(z_cls["true_category"], z_cls["category"]).to_string())
    print("zero-shot system prompt est tokens:", est_tokens(ZERO_SHOT))

    few = run_pipeline(tickets, FEW_SHOT)
    f_cls, f_acc, f_tot = summarize("few-shot", few)
    print("few-shot system prompt est tokens:", est_tokens(FEW_SHOT))
    print("extra tokens per classified ticket (few - zero): %.2f" % (f_tot / len(f_cls) - z_tot / len(z_cls)))

    # --- task 7a: budget demo
    BUDGET = 6000
    demo = run_pipeline(tickets, ZERO_SHOT, token_budget=BUDGET)
    n_ex = int((demo["status"] == "budget_exhausted").sum())
    print(f"\nbudget demo TOKEN_BUDGET={BUDGET}: budget_exhausted = {n_ex} | statuses {demo['status'].value_counts().to_dict()}")
    assert n_ex >= 1
    route(demo).to_csv("outputs/triage_budget_demo.csv", index=False)

    # --- task 7b: final results (few-shot, budget 100000, routed with Part B segments)
    final = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    res = route(final)
    res.to_csv("outputs/triage_results.csv", index=False)
    print("\nfinal run statuses:", final["status"].value_counts().to_dict(), "| rows:", len(res))
    print("routes:", res["route"].value_counts().to_dict())
    print("priorities:", res["priority"].value_counts().to_dict())
    print("final-run budget_exhausted:", int((res['status'] == 'budget_exhausted').sum()))

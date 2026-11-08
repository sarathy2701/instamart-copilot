"""Part D, tasks 5-9: tools, ReAct refund agent, stop-safely limits, resumable runs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm

import glob
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import mock_llm
from d1_rag import retrieve

DB = os.path.join("data", "instamart.db")
OUT = "outputs"
STATE_DIR = "state"
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
DEMOS = os.path.join(OUT, "agent_demos.jsonl")

AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, "
                "calculate_refund, issue_refund and escalate_to_human. Finish with a final answer that starts with "
                "REFUND:, ESCALATE:, DENIED: or NOT_FOUND:.")


# ----------------------------------------------------------------- tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    if dm is not None and not (isinstance(dm, float) and dm != dm):
        dm = int(dm) if float(dm).is_integer() else float(dm)
    else:
        dm = None
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]), "delivery_minutes": dm}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger_write(tool, order_id, refund_inr):
    """Append one JSON line. Idempotent: the same tool+order is never paid out / escalated twice."""
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            rec = json.loads(line)
            if rec["tool"] == tool and rec["order_id"] == order_id:
                return {"refund_inr": rec["refund_inr"]}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    if refund_inr > 1000:  # safety net: the policy says refunds above INR 1000 need a human
        return {"error": "refund above INR 1000 must be escalated to a human"}
    return _ledger_write("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger_write("escalate_to_human", order_id, refund_inr)


def _spec(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the database by its order id.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant chunks.",
          {"query": {"type": "string", "description": "Free-text search query"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}},
          ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ----------------------------------------------------------------- agent loop
class SimulatedCrash(Exception):
    pass


def _save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)


def _refund_of(st):
    for s in reversed(st["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return s["observation"]["refund_inr"]
    return 0


def _finish(st, status, final):
    return {"run_id": st["run_id"], "status": status, "final_answer": final, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "refund_inr": _refund_of(st)}


def _fallback(st, status, reason):
    done = [s["tool"] for s in st["steps"]]
    summary = (f"HANDOFF TO HUMAN: automated run stopped ({reason}) after {len(done)} tool step(s) "
               f"{done} for scenario {st['scenario_id']}; please review the case manually.")
    return _finish(st, status, summary)


def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit {time_limit_seconds}s reached")
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"{max_iterations} model calls without a final answer")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:  # stop before executing the tool that was just requested
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded ({st['tokens_used']})")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _finish(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        try:
            obs = TOOL_MAP[tc.name](**tc.arguments)
        except Exception as e:  # unknown tool / bad arguments -> observation, never a crash
            obs = {"error": f"{type(e).__name__}: {e}"}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        st["step"] = len(st["steps"])
        _save_checkpoint(st)
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"run-{scenario_id}", "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                     "time_limit_seconds": time_limit_seconds}}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id):
    """Load state/<run_id>.json and continue from the last saved tool observation (no tool is re-executed)."""
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    lim = st["limits"]
    return _loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"])


def _record(path, scenario_id, res, extra=None):
    rec = {"scenario_id": scenario_id, "status": res["status"], "final_answer": res["final_answer"],
           "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "refund_inr": res["refund_inr"],
           "steps": res["steps"]}
    rec.update(extra or {})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for p in [LEDGER, RUNS, DEMOS] + glob.glob(os.path.join(STATE_DIR, "*.json")):
        if os.path.exists(p):
            os.remove(p)

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("| scenario | status | refund_inr | model calls | tokens used | expected |")
    print("|---|---|---|---|---|---|")
    results = {}
    for r in sc.itertuples(index=False):
        mock_llm.reset()
        res = run_agent(r.message, r.scenario_id)
        results[r.scenario_id] = res
        _record(RUNS, r.scenario_id, res)
        exp = r.expected_outcome
        ok_status = res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp
        ok_amt = abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
        assert ok_status and ok_amt, (r.scenario_id, res["status"], res["refund_inr"], exp, r.expected_refund_inr)
        print(f"| {r.scenario_id} | {res['status']} | {res['refund_inr']:.2f} | {res['llm_calls']} | {res['tokens_used']} | {exp} {float(r.expected_refund_inr):.2f} |")
    assert results["S1"]["llm_calls"] == 5 and results["S7"]["llm_calls"] == 6
    assert results["S7"]["status"] == "FALLBACK_MAX_ITERATIONS"

    s1 = sc[sc.scenario_id == "S1"].iloc[0]
    print("\n-- stop-safely demos (S1) --")
    mock_llm.reset()
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="demo-S1-tokenlimit")
    _record(DEMOS, "S1", d1, {"demo": "token_limit=200"})
    print("token_limit=200 ->", d1["status"], "| llm_calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| steps", len(d1["steps"]))
    print("   final_answer:", d1["final_answer"])
    mock_llm.reset()
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="demo-S1-timeout")
    _record(DEMOS, "S1", d2, {"demo": "time_limit_seconds=0"})
    print("time_limit_seconds=0 ->", d2["status"], "| llm_calls", d2["llm_calls"], "| steps", len(d2["steps"]))
    print("   final_answer:", d2["final_answer"])

    print("\n-- crash and resume (S1) --")
    mock_llm.reset()
    try:
        run_agent(s1.message, "S1", run_id="demo-S1-crash", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crashed:", e)
    ck = json.load(open(os.path.join(STATE_DIR, "demo-S1-crash.json")))
    print("tools executed before crash:", ck["tools_executed"])
    fin = resume_agent("demo-S1-crash")
    ck = json.load(open(os.path.join(STATE_DIR, "demo-S1-crash.json")))
    print("tools executed across crash + resume:", ck["tools_executed"])
    print("final:", fin["status"], "%.2f" % fin["refund_inr"], "| llm_calls", fin["llm_calls"])
    assert ck["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert fin["status"] == "REFUND" and abs(fin["refund_inr"] - 128.25) <= 0.01
    _record(DEMOS, "S1", fin, {"demo": "crash_and_resume", "tools_executed": ck["tools_executed"]})

    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| agent_runs lines:", sum(1 for _ in open(RUNS)),
          "| checkpoints:", len(glob.glob(os.path.join(STATE_DIR, '*.json'))))

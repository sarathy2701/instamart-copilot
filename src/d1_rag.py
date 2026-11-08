"""Part D, tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm

import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = os.path.join("data", "policies")

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied context. "
                 "If the answer is not in the context, say that you could not find the information.")


def chunk_words(text, size=40, overlap=10):
    """Word windows: chunk 1 = words 0..size-1, next starts size-overlap later; stop at the first chunk reaching the end."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_index(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        for ch in chunk_words(text, size, overlap):
            chunks.append({"source": os.path.basename(path), "text": ch})
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vectorizer": vec, "matrix": mat}


INDEX = build_index()  # default: 40-word chunks, 10-word overlap


def retrieve(query, k=3, index=None):
    index = index or INDEX
    sims = cosine_similarity(index["vectorizer"].transform([query]), index["matrix"])[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": index["chunks"][i]["source"], "score": float(sims[i]), "text": index["chunks"][i]["text"]} for i in order]


def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


def evaluate(index, label):
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = n_ans = n_none = 0
    failures = []
    for r in qa.itertuples(index=False):
        text, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_none += 1
            ok = text.startswith("I could not find")
            fallback += ok
            if not ok:
                failures.append((r.question, "should have fallen back", text))
        else:
            n_ans += 1
            d = r.expected_doc in sources
            f = r.expected_fact in text
            doc_hit += d; fact_hit += f
            if not (d and f):
                failures.append((r.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    print(f"\n[{label}] answerable={n_ans}: expected_doc in top-3 {doc_hit}/{n_ans} | expected_fact in answer {fact_hit}/{n_ans} "
          f"| unanswerable={n_none}: fallback {fallback}/{n_none}")
    for q, why, t in failures:
        print("  FAIL:", q, "|", why, "| answer:", t)
    return failures


if __name__ == "__main__":
    counts = {}
    for c in INDEX["chunks"]:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    print("chunks total:", len(INDEX["chunks"]), counts)

    print("\nretrieve('refund policy late delivery'):")
    for h in retrieve("refund policy late delivery"):
        print("  %-22s score %.3f" % (h["source"], h["score"]))

    ans, src = answer("Who must approve a refund above INR 1000?")
    print("\nanswer demo:", ans, "| sources:", src)

    evaluate(INDEX, "overlap 10")
    idx0 = build_index(40, 0)
    c0 = {}
    for c in idx0["chunks"]:
        c0[c["source"]] = c0.get(c["source"], 0) + 1
    print("\noverlap 0 chunks total:", len(idx0["chunks"]), c0)
    evaluate(idx0, "overlap 0")

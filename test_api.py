"""
test_api.py — Smoke test for the v2 FAQ Mining API.
Requires the server to be running (python backend/main.py --serve).
"""
import json
import time
import urllib.request

base = "http://localhost:8000"


def _get(path):
    r = urllib.request.urlopen(base + path)
    return json.loads(r.read())


def _post(path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req)
    return json.loads(r.read())


# ── /groups (primary new endpoint) ──────────────────────────────────────────
data = _get("/groups")
print(f"/groups: total_groups={data['total_groups']}")
if data["total_groups"] > 0:
    g = data["groups"][0]
    print(f"  group_id   : {g['group_id']}")
    print(f"  group_name : {g['group_name']}")
    print(f"  total_q    : {g['total_questions']}")
    print(f"  rep_qs     : {g['representative_questions'][:2]}")
    print(f"  admin_reply: {g['suggested_admin_reply'][:80]}")

# ── /faqs (backwards compat) ─────────────────────────────────────────────────
faqs = _get("/faqs")
print(f"\n/faqs (compat): total={faqs['total']}")

# ── /search  ─────────────────────────────────────────────────────────────────
t0 = time.time()
res = _post("/search", {"query": "เข้าระบบไม่ได้", "top_k": 3})
ms = (time.time() - t0) * 1000
print(f"\n/search ({ms:.1f}ms): {res['count']} results")
if res["results"]:
    top = res["results"][0]
    print(f"  top group_name : {top.get('group_name', top.get('faq_question', '?'))}")
    print(f"  similarity     : {top['similarity_score']}")

# ── /clusters ────────────────────────────────────────────────────────────────
cl = _get("/clusters")
print(f"\n/clusters: total_clusters={cl['total_clusters']}")

# ── /analytics ───────────────────────────────────────────────────────────────
an = _get("/analytics")
s = an.get("summary", an)
print(f"\n/analytics: total={s.get('total_conversations', '?')}, "
      f"faqs={s.get('total_faqs_generated', '?')}, "
      f"noise={s.get('noise_ratio_percent', '?')}%")

print("\nALL ENDPOINTS OK ✓")

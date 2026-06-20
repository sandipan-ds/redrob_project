# GLM Critic v3 — Recheck of the Revised Execution Plan

> **Reviewer:** GLM-5.2 (opencode-go/glm-5.2)
> **Reviewed artifact:** `docs/project_docs/EXCEUTION_PLAN.md` (now 781 lines, with §2.5.j added and
> §5.1/§7.1/SYSTEM_DESIGN/criteria_map synced since v2).
> **Reference:** `docs/project_docs/GLM_CRITIC_v2.md` (the 14-item audit this rechecks).
> **Cross-checked against actual repo state:** `config/scoring_config.yaml`,
> `config/jd_intent_embeddings.npy` (verified shape/norm), `src/jd_embedding.py`,
> `tests/test_p1.py`, `docs/project_docs/criteria_map.md`, `docs/project_docs/SYSTEM_DESIGN.md`,
> `docs/project_docs/PHASED_BUILD_PLAN.md`, `docs/project_docs/P1_TEST_REPORT.md`, `git log`.
> **Question answered:** of the 14 v2 items, how many are fixed and what still needs addressing.

---

## 1. Tally — 12 of 14 addressed, 2 outstanding

| # | v2 item | Status | Evidence |
|---|---|---|---|
| 1 | Education weight too high | ✅ Fixed (as knob) | §5.1 now lists `w_edu` as a calibrated macro knob "which the JD's anti-credentialist stance suggests may want to drop toward ~0.05"; `SYSTEM_DESIGN.md` §4 row notes "`w_edu` is a calibrated knob (anti-credentialist JD)." Prior still 0.10, but the ask (make it calibratable) is met. |
| 2 | `consulting_only` name list | 🟡 Plan-fixed, pending P2 | New §2.5.j records the decision: generalize via `current_industry == "IT Services"` + `company_size` bands, keep the name list as a high-precision booster, real "prior product-co" exemption check. **`config.penalties.consulting_only.consulting_companies` is still the same 10 names** — config not yet updated. |
| 3 | `langchain_only_junior` gate | 🟡 Plan-fixed, pending P2 | §2.5.j: "demote from hard ×0.40 gate to mild soft feature unless conjunctive conditions hold (junior exp AND no pre-2022 ML AND LangChain is the only AI signal)." **`config.penalties.langchain_only_junior.score` still `0.40`** — config not yet updated. |
| 4 | Skill synonym table | 🟡 Partial — residual bug | `config.skills.skill_synonyms` now exists (RAG / Vector DB / Embeddings / Transformers / Fine-tuning / LLM collapse map). **But `jd_core_skills` still lists `Milvus`, `Pinecone`, `Weaviate`, `Faiss` (each w=7) AND `Hugging Face` (w=7) as standalone entries** while the synonym map collapses all of them into `Vector Database` (w=8) / `Transformers` (w=7). If synonyms are applied first, those entries are dead; if not, double-counting. Remove the subsumed entries. |
| 5 | `profile.summary` unused | ✅ Fixed | `criteria_map.md` §C now "Use (low-weight)"; §2.5.j commits it as role-fit supplementary input + thin-desc fallback source. No longer the ambiguous "keep as proxy." |
| 6 | OSS proxy = github only | ✅ Fixed | `criteria_map.md` §B "Open-source contributions" now proxies via description-text (papers / talks / OSS projects), explicitly "NOT `github_activity_score` — dropped." §B "Writes production code" likewise moved off github. |
| 7 | Disk budget unsized | ✅ Fixed | §7.1 has a new "Disk sizing" table: vendored model ~90 MB + vectors ~461 MB + parquet ~50–150 MB + 487 MB input ≈ **0.6–0.7 GB ≪ 5 GB ✅**; `SYSTEM_DESIGN.md` §8 table mirrors it. |
| 8 | Git history | ❌ Not fixed | `git log` unchanged: 9 commits, two `Initial commit`s, P0 unlabeled (bundled into `first commit`). **The one Stage-4-penalized item still in motion and uncorrected.** |
| 9 | SYSTEM_DESIGN max-pool | ✅ Fixed | §4 table row + §4.1 rewritten to "top-K-mean pooled, recency-weighted" + new §4.1.1 "How duration and recency compose" composition rule. |
| 10 | criteria_map §E github | ✅ Fixed | §E now "`github_activity_score` ❌ dropped — Low coverage"; §D sentinel note updated to say it's no longer a feature at all. |
| 11 | criteria_map §F stacking | ✅ Fixed | §F now carries the `honeypot? × min(non_hp) × Π(others)^0.5` formula with the `0.15 × √0.20 ≈ 0.067` example, explicitly "NOT a raw product" and "calibrated via `p_scale` (P5), not frozen by assertion." |
| 12 | Stale P1 embedding | ✅ Fixed (well) | `config/jd_intent_embeddings.npy` exists, verified shape `(4, 384)`, unit-norm rows. `src/jd_embedding.py` adds `generate_jd_intent_set` / `load_jd_intent_set` / `max_query_similarity`; `tests/test_p1.py` adds `TestJdIntentSet` (+4 tests); P1 report updated to 24/24. **P3 is now unblocked.** Legacy single vector kept for back-compat. |
| 13 | P2–P8 prose only | ❌ Not fixed (expected) | `src/` still only `config_loader.py`, `data_loader.py`, `jd_embedding.py`. No `honeypot.py` / `disqualifiers.py` / `features/` / `scoring.py` / `retrieve.py` / `rank.py` / `reasoning.py` / `eval/`. Expected — P2 hasn't started. |
| 14 | duration × recency composition | ✅ Fixed | §2.5.f pins `w_d = duration_norm(d) × recency_decay(d)` with exact formulas (`duration_norm = min(duration_months/24, 1.0)`, `recency_decay = 0.5 ** (months_since_end / recency_half_life_months)`); `SYSTEM_DESIGN.md` §4.1.1 mirrors it. Exactly one per-description weight, not two stacked reweightings. |

**Count: 9 fully fixed, 2 plan-fixed-and-deferred-to-P2 (#2, #3), 1 partial-with-residual-bug (#4),
2 not fixed (#8 git, #13 P2–P8 prose — the latter expected).**

---

## 2. What still needs to be addressed

In priority order:

### 1. Git history (v2 #8) — now the most urgent unaddressed item
No per-phase commits have started; P0 is still buried in `first commit`, P1 in `added phase P1 of
the development`. Every phase that lands without a labeled commit compounds the Stage-4 "flat
history with no iteration" penalty — a named, avoidable scoring loss. **Start now:** the next
commit should be `P2: honeypot + disqualifier gates`, and going forward one labeled commit per
phase with the documented prefix. Optionally a single retroactive `docs: document P0/P1 split in
CHANGELOG` commit (no history rewrite) to make the early phases legible.

### 2. Residual synonym-config inconsistency (v2 #4) — a real latent double-count
`config.skills.skill_synonyms` was added (good), but `config.skills.jd_core_skills` still contains
the entries the synonym map subsumes:

- `Milvus`, `Pinecone`, `Weaviate`, `Faiss` (each w=7) — all collapse to `Vector Database` (w=8).
- `Hugging Face` (w=7) — collapses to `Transformers` (w=7).

If the scorer applies synonyms first, these standalone entries are dead config; if it doesn't,
a candidate listing both "RAG" and "Retrieval-Augmented Generation", or "Pinecone" and "Vector
Database", scores 2×. **Fix:** delete the five subsumed entries from `jd_core_skills` so the
canonical-name list and the synonym map agree. This is the exact double-counting bug the synonym
table was added to prevent — it's only half-wired.

### 3. Config drift vs §2.5.j (v2 #2, #3) — plan moved, config didn't
The plan now says (a) consulting detection generalizes to `industry == "IT Services"` + size bands
with a real prior-product-co exemption, and (b) `langchain_only_junior` demotes from a hard ×0.40
gate to a soft feature unless the conjunctive conditions hold. But `config/scoring_config.yaml`
still reflects the old design:

- `penalties.consulting_only.consulting_companies` — still the same 10-name list, no industry/size
  fields, no exemption detector config.
- `penalties.langchain_only_junior.score` — still `0.40`, still framed as a gate.

A reviewer comparing `EXCEUTION_PLAN §2.5.j` against the config sees a contradiction now. Either
update the config in the same commit that lands P2 (so plan and config move together), or add a
`# TODO §2.5.j` marker in the config so the drift is explicit rather than silent. Silent drift is
the failure mode that caused the v2 doc-contradiction bucket in the first place.

### 4. P2–P8 code (v2 #13) — not started, expected
With the stale-embedding blocker (#12) resolved and the composition rule (#14) pinned, P3 is
unblocked and P2 can start. Nothing to fix here — just the next thing to build. `PHASED_BUILD_PLAN`
P2/P3 specs are already aligned with §2.5 (conjunctive `research_only`, softened penalty
combination, `github` dropped from behavior, top-K-mean + recency role-fit).

---

## 3. Net assessment

The v2 concerns are almost fully absorbed. The plan, config, sibling docs, and P1 artifacts are
now internally consistent on **11 of 14** items; the multi-query embedding is generated and tested
(24/24); the duration×recency ambiguity is pinned; and the three doc-contradictions that would have
read badly at Stage-4 are gone. The remaining work is small and concrete:

- **one process item** — start per-phase labeled commits (the only Stage-4-penalized item still
  uncorrected);
- **one config cleanup** — delete the five subsumed skill entries so the synonym map actually
  prevents double-counting;
- **one config↔plan sync** — bring `consulting_only` and `langchain_only_junior` in config up to
  §2.5.j, or mark the drift explicitly;
- **and starting P2** — the design is ready.

No re-architecture, no further plan revisions needed. The plan is now in a buildable state; the
constraint is execution, not design.

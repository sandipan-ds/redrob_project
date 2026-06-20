# GLM Critic v2 — Post-Revision Audit of the Redrob Execution Plan

> **Reviewer:** GLM-5.2 (opencode-go/glm-5.2)
> **Reviewed artifacts:** the three critic docs (`GLM_CRITIC.md`, `MIMO_CRITIC.md`,
> `MINIMAX_CRITIC.md`) and the revised `docs/project_docs/EXCEUTION_PLAN.md` (with the new §2.5
> "Post-review refinements" block, now 743 lines).
> **Cross-checked against actual repo state:** `config/scoring_config.yaml`,
> `docs/project_docs/PHASED_BUILD_PLAN.md`, `docs/project_docs/criteria_map.md`,
> `docs/project_docs/SYSTEM_DESIGN.md`, `src/**/*.py`, `tests/**/*.py`, `git log`.
> **Purpose:** answer two questions — (1) what has been *implemented* from the three critics'
> concerns, and (2) what still needs to change. "Implemented" is taken to mean wired into the
> config / build plan / code, not merely restated in plan prose.

---

## 1. What's been implemented (in the revised plan + config + build plan)

The new §2.5 block folds in most of the three critics' concerns. Mapping each raised concern to
its disposition:

| Concern (raised by) | Disposition | Where it landed |
|---|---|---|
| Single-vector role signal too blunt (GLM#1, MINIMAX#1) | ✅ Added `s_role_fit = w_dense·s_dense + w_lex·s_lex`, multi-query intents + lexical blend | §2.5.a; `config.role_fit` block (`w_dense 0.7`, `w_lex 0.3`, 4 `intent_queries`); P3 role_fit spec |
| Max-pool failure mode (MINIMAX#2, MIMO#1) | ✅ Switched to top-K mean (K=2) + duration weighting | §2.5.b; `config.role_fit.pool: "topk_mean"`, `pool_k: 2`, `duration_weight: true` |
| `research_only` keyword-absence trap (GLM#2) | ✅ Now conjunctive (3 conditions: no prod-lexicon match AND no product-co tenure AND explicit research framing) | §2.5.c; P2 disqualifiers spec |
| Penalty stacking too aggressive (MINIMAX#3) | ✅ Worst-gate-full + geometric softening of secondaries | §2.5.d; P2 combination formula |
| Gate magnitudes frozen by assertion (GLM#2) | ✅ `p_scale` global scale added as a calibrated knob (replaces 6 hidden gate constants) | §2.5.e; `config.penalties.p_scale: 1.0`; P5 calibrate |
| Career-recency weighting (MINIMAX#5) | ✅ Recency decay on role-fit contributions | §2.5.f; `config.role_fit.recency_half_life_months: 36` |
| Drop `github_activity_score` (MINIMAX#4) | ✅ Dropped from `M_behavior` in the plan | §2.5.g; P3 behavior spec |
| Thin/empty-description fallback (MIMO#3, MINIMAX#6) | ✅ Title-prior fallback, the single justified use of the otherwise-demoted title | §2.5.h; `config.role_fit.min_desc_chars: 40` |
| Fewer-than-100-fits handling (MIMO#7) | ✅ Honest low scores + stable secondary ordering (exp band, then `candidate_id` asc) | §2.5.i; P4 rank.py step 5 |
| Anti-keyword regression test (GLM#5) | ✅ `tests/test_anti_keyword.py` vs `sample_submission.csv` canonical-bad output | §5.3; P5 |
| Proxy self-label circularity (GLM#3) | ✅ Adversarial near-miss decoys / second-labeler independence guard | P5 proxy_labels "Independence guard" |
| Two-band retrieval (MINIMAX#7) | ✅ Second-chance band for high-skill/moderate-cosine candidates | P4 retrieve.py |
| Expand calibrated knobs (MINIMAX#8) | ✅ `w_dense`/`w_lex`, `p_scale`, recency added to the macro-knob budget | P5 calibrate.py |
| JD-intent sanity check vs ground truth (MINIMAX#9) | ✅ Embed clearly-good samples, assert high cosine before locking | §2.5.a; P3 |
| Reasoning whitelist not substring (MINIMAX#10) | ✅ Entity-whitelist hallucination test | §6; P6 exit test #4 |
| `sample_submission.csv` as cautionary trap (MIMO#5) | ✅ Explicitly called out as the canonical bad output | §5.3 |

So ~16 of the ~22 distinct critic concerns are addressed in the revised plan and wired into the
config/build-plan layer. That's a real revision, not just prose.

---

## 2. What still needs to change

### A. Concerns raised but not addressed at all

1. **Education weight 0.10 still too high (GLM#6)** — `config.weights.education: 0.10` is
   unchanged. The JD is explicitly anti-credentialist ("5–9 is a range, not a requirement,"
   title-chaser warning, focus on shipped systems) and never mentions CGPA; `CAND_0000001` is
   LPU/tier_3 yet a reasonable data-eng-adjacent fit. At minimum make `w_edu` a calibrated knob;
   I'd cut it to ~0.05 and move the freed weight to role-fit (→ 0.50) or a production-evidence
   feature.

2. **`consulting_only` 10-company name list vs the JD's "etc." (GLM#7b, MINIMAX#15)** —
   `config.penalties.consulting_only.consulting_companies` is the same 10 names (TCS, Infosys,
   Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mindtree, Mphasis). Hidden-pool
   variants (Genpact, LTIMindtree, IBM India…) escape; the "prior product-company experience"
   exemption still has no real detector. Needs an industry/size-based product-co detector
   (`current_industry != "IT Services"` + `company_size` bands), not a name list.

3. **`langchain_only_junior` gate (GLM#7c)** — still a ×0.40 gate, detection-hard, with false-fire
   risk on senior ML engineers who recently *added* LangChain. Consider dropping it as a gate
   entirely: a LangChain-only junior's descriptions won't embed near "built retrieval/ranking in
   production," so role-fit + the exp band already demote them without a brittle rule.

4. **Skill synonym-collapse table (MIMO#2, MINIMAX#12)** — `config.skills.jd_core_skills` still has
   40 separately-weighted entries with no synonym map. `RAG` and `Retrieval-Augmented Generation`
   are both listed at weight 8 as *separate* entries; `Vector Database` coexists with
   `Milvus`/`Pinecone`/`Weaviate`/`Faiss`. The plan says "collapse synonyms" (§5.1) but no table
   exists in config. This is the §5.1 overfit risk in concrete form — and a real double-counting
   bug waiting to happen (a candidate listing both "RAG" and "Retrieval-Augmented Generation"
   scores 16, not 8).

5. **`profile.summary` "kept as proxy" but unused (MINIMAX#11)** — `criteria_map.md` §C still says
   "Keep as proxy" with no role in the formula. Commit to it (low weight, behind career
   descriptions) or drop it. The current middle state is what causes drift later.

6. **Open-source proxy = github only (MINIMAX#14)** — `criteria_map.md` §B still proxies "Open-source
   contributions" via `github_activity_score`, which §2.5.g just dropped from `M_behavior`. So the
   OSS proxy now points at a signal the plan removed — nowhere left. Needs a description-text proxy
   (mentions of papers, talks, OSS projects in `career_history[].description`).

7. **≤5 GB disk budget never sized (GLM#7a)** — §7.1 still sizes only RAM (154 MB / 461 MB / 487 MB
   JSONL). Disk is a hard Stage-3 constraint (spec §3) and should be in the measured-budget table.
   The arithmetic is fine (vendored MiniLM ~90 MB + 461 MB un-pooled vectors + parquet + 487 MB raw
   JSONL ≪ 5 GB), but the table should say so.

8. **Git history (GLM#4)** — still 9 commits, two `Initial commit`s, P0 unlabeled (bundled into
   `first commit`). No per-phase commits have started, and the plan explicitly promises them
   (PHASED_BUILD_PLAN §0). This is the one Stage-4-penalized item already in motion and uncorrected;
   cheap to fix now, expensive once P2–P8 land in big dumps.

### B. Doc inconsistencies introduced by the revision (must fix — these will hurt at Stage-4 review)

The execution plan moved, but two sibling docs were **not** updated and now contradict it. A
reviewer reading both will see self-contradictions.

9. **`SYSTEM_DESIGN.md` §4 / §4.1 still says "max-pool"** as the headline pooling decision —
   directly contradicts §2.5.b's top-K mean. The whole §4.1 "Why max-pool for role-fit (key
   decision)" block needs rewriting to reflect top-K mean + duration weighting + recency decay.

10. **`criteria_map.md` §E still marks `github_activity_score` as "✅ (weak)"** — contradicts §2.5.g
    (drop it). The build plan even says "Update `criteria_map.md` §E to ❌ dropped — low coverage"
    (P3 behavior.py note) but it wasn't done.

11. **`criteria_map.md` §F still says "Penalties are multiplicative and stackable —
    consulting-only × research-only = 0.03"** — contradicts §2.5.d's softened stacking. Under the
    new formula that product is `0.15 × sqrt(0.20) ≈ 0.067`, not 0.03, and §F's framing ("stackable")
    no longer matches §2.5.d's "worst-gate-full + softened secondaries."

### C. Implemented in prose only — not yet in code/artifacts

12. **The P1 frozen embedding is now stale.** `config/jd_intent_embedding.npy` is a **single**
    1,200-char vector (per `config/jd_embedding_meta.yaml`: `jd_intent_text_chars: 1200`), but §2.5.a
    requires a **multi-query set** — the 4 `intent_queries` strings in `config.role_fit` have no
    corresponding `.npy` files. P1's "frozen" artifact needs regenerating before P3 can build the
    blended role signal, and `tests/test_p1.py` will need updating to assert on the query set, not
    one vector. (P1's test report currently celebrates a single-vector freeze — that artifact is now
    a blocker for the revised design.)

13. **P2–P8 are entirely prose.** `src/` has only `config_loader.py`, `data_loader.py`,
    `jd_embedding.py`; no `honeypot.py`, `disqualifiers.py`, `features/`, `scoring.py`,
    `retrieve.py`, `rank.py`, `reasoning.py`, `eval/`. The refined design exists only as spec.
    (Expected at this stage — flagging it so the "implemented" claim isn't over-read: §2.5 is
    *design- implemented*, not *code-implemented*.)

### D. One design worry about the revised plan itself

14. **§2.5.b duration-weighting + top-K mean interacts with §2.5.f recency weighting in an
    unspecified way.** Both reweight per-description contributions — is the final weight
    `duration × recency`, or one then the other, or combined into a single per-description score
    before pooling? `config.role_fit` has both flags (`duration_weight: true`,
    `recency_half_life_months: 36`) but no stated composition rule. Worth pinning down before P3
    code, or calibration (P5) will conflate two effects moving together — exactly the
    "interdependent knobs" failure mode MINIMAX#8 warned about, reintroduced inside a single
    feature.

---

## 3. Net assessment

The revision is genuine and substantial: ~16 of ~22 distinct critic concerns are addressed and
wired into config/build-plan (not just plan prose). The spine — offline precompute + NumPy rerank,
distrust titles, multiplicative gates, sentinels-neutral, few-knob calibration — is intact and
strengthened. The remaining gaps split cleanly into four buckets, none requiring re-architecture:

- **8 unaddressed concerns** (education weight, consulting-list, langchain gate, synonym table,
  summary, OSS proxy, disk budget, git history) — fixes for P2/P3 and one doc table.
- **3 doc-contradictions** (`SYSTEM_DESIGN.md` max-pool, `criteria_map.md` §E github,
  `criteria_map.md` §F stacking) — a one-pass doc sync; these will read badly at Stage-4 if left.
- **The stale P1 embedding artifact** — regenerate the multi-query intent set and update `test_p1.py`;
  this is the only item that blocks the next phase from starting correctly.
- **One weighting-composition ambiguity** (duration × recency) — pin down the composition rule in
  §2.5 before P3 code.

Priority order for the next working session: (i) regenerate the multi-query JD-intent vectors and
update P1's test (unblocks P3); (ii) sync `SYSTEM_DESIGN.md` and `criteria_map.md` with §2.5
(eliminates self-contradictions); (iii) build the skill synonym-collapse table (prevents a real
double-counting bug); (iv) start per-phase labeled commits (the only Stage-4-penalized item already
in motion). Everything else can ride with P3/P5.

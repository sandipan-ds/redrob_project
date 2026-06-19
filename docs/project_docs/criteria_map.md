# P1 Criteria Map — JD ↔ Signals ↔ Candidate Schema

**Phase:** P1  
**Status:** Frozen (dev-time, human-reviewed)  
**Sources:**
- `docs/reference_docs/job_description.docx` — primary fit reference
- `docs/reference_docs/redrob_signals_doc.docx` — behavioral signal definitions
- `data/samples/candidate_schema.json` — field inventory

---

## Section A — Criteria Common to All Three Sources

These criteria appear in the JD (as requirements), have a corresponding signal in the signals doc, and map directly to a field in the candidate schema. These are the **highest-confidence scoring inputs** — all three sources agree they matter.

| JD Requirement | Signal / Schema Field | Scoring Role |
|---|---|---|
| Active, available candidate | `last_active_date`, `open_to_work_flag`, `recruiter_response_rate` | Behavior multiplier |
| Responds to recruiters | `recruiter_response_rate`, `avg_response_time_hours` | Behavior multiplier |
| Completes interview cycles | `interview_completion_rate` | Behavior multiplier |
| Sub-30-day notice preferred | `notice_period_days` | Soft tie-breaker |
| Noida / Pune preferred | `profile.location`, `willing_to_relocate` | Location score |
| Skills in embeddings, retrieval, ranking | `skills[].name`, `skills[].proficiency`, `skills[].endorsements`, `skills[].duration_months` | Skill score |
| Skill assessment scores (platform-verified) | `skill_assessment_scores` (dict) | Skill trust modifier |
| Years of experience (5–9 yr range) | `profile.years_of_experience` | Experience band score |
| Production ML at product companies | `career_history[].description`, `career_history[].industry`, `career_history[].company_size` | Role-fit score (dominant) |

---

## Section B — JD-Only Criteria (no direct schema field)

These are requirements stated in the JD that **cannot be read directly from a schema field**. They require inference from text or structural patterns. The approach for each is noted.

| JD Criterion | Why No Direct Field | Approach |
|---|---|---|
| "Built a ranking / recsys / search system at a product company" | No boolean field; must be read from `career_history[].description` | Semantic cosine similarity to JD-intent embedding (dominant `s_role_fit` feature) |
| "Pre-LLM ML production experience" | No date-tagged skill history | Proxy: `career_history` entries pre-2022 with ML-relevant descriptions |
| "Understands retrieval/ranking evaluation (NDCG, MRR, MAP)" | Not a schema field | Proxy: skill name match + career description text |
| "Shipped to real users at meaningful scale" | No scale field | Proxy: `career_history[].company_size` ≥ 201-500 + product industry |
| "Has strong opinions / can defend with reference to systems built" | Subjective | Not scored — Stage 4 interview criterion only |
| "Writes production code (not just architecture)" | No field | Proxy: `github_activity_score` > 0 + recent `career_history` descriptions |
| "3+ year tenure intent" | Future intent, not observable | Not scored — no reliable proxy |
| "Open-source contributions" | No field | Proxy: `github_activity_score` |
| "Async-first, writes well" | Subjective | Not scored — noise, not derivable from JSON |
| "Not a title-chaser (1.5yr job-hopping)" | No explicit field | Proxy: average `duration_months` across `career_history` entries |

---

## Section C — Schema Fields Not in JD (keep / drop decision)

These fields exist in the candidate schema but are not mentioned in the JD. Each is assessed for predictive value.

| Schema Field | JD Mention | Decision | Rationale |
|---|---|---|---|
| `profile.summary` | No | **Keep as proxy** | Useful for role-fit text matching when `career_history` descriptions are thin |
| `profile.headline` | No | **Keep as proxy** | Quick title signal; used in role affinity lookup |
| `profile.current_company` | No | **Keep** | Used for consulting-company penalty detection |
| `profile.current_company_size` | No | **Keep** | Product-company proxy (small/mid-size + non-IT-services = product signal) |
| `profile.current_industry` | No | **Keep** | IT Services = consulting signal; Software/Fintech/etc. = product signal |
| `connection_count` | No | **Drop** | No predictive signal for technical fit or availability |
| `endorsements_received` (total) | No | **Drop** | Redundant with per-skill `endorsements`; too noisy at aggregate level |
| `profile_views_received_30d` | No | **Drop** | Passive metric; not controlled by candidate; not predictive |
| `search_appearance_30d` | No | **Drop** | Platform artifact; not predictive of fit |
| `saved_by_recruiters_30d` | No | **Weak keep** | Mild signal of market interest; used only as a very weak tie-breaker |
| `applications_submitted_30d` | No | **Drop** | Desperation signal as much as availability signal; too noisy |
| `expected_salary_range_inr_lpa` | No | **Drop** | Not a fit criterion per JD; comp is handled separately |
| `verified_email`, `verified_phone` | No | **Drop** | Platform hygiene, not fit signal |
| `linkedin_connected` | No | **Drop** | Platform hygiene, not fit signal |
| `signup_date` | No | **Drop** | No predictive value for fit |
| `profile_completeness_score` | No | **Weak keep** | Very low completeness (<40) may indicate low engagement; used as a floor check only |
| `certifications` | No | **Weak keep** | Relevant certs (AWS ML, GCP ML) add minor skill signal |
| `languages` | No | **Drop** | Not a JD criterion |
| `preferred_work_mode` | No | **Drop** | JD is flexible (hybrid); not a differentiator |

---

## Section D — Signals Doc Architecture Decision

The signals doc states explicitly:

> *"These behavioral signals are often more predictive of whether a candidate can actually be hired than their static profile... This dataset includes these signals so that ranking systems can incorporate them as a **multiplier or modifier on top of skill-match scoring**."*

This is honored literally in the scoring formula:

```
final = fit_score × M_behavior × P_penalty
```

The signals doc does **not** say signals replace fit scoring — they modulate it. A candidate with a perfect fit score but zero availability is still ranked lower. A candidate with moderate fit but strong availability signals is not artificially boosted above a clearly better fit.

**Sentinel handling (from signals doc):**
- `github_activity_score = -1` → no GitHub linked → treated as neutral (0 contribution to multiplier)
- `offer_acceptance_rate = -1` → no prior offers → treated as neutral
- `skill_assessment_scores = {}` → no assessments taken → treated as neutral

Sentinels are **"unknown," never "bad."** They do not penalize the candidate.

---

## Section E — Signals Selected for M_behavior Multiplier

Of the 23 signals, the following are used in scoring. The rest are dropped per Section C.

| Signal | Used | Role in M_behavior |
|---|---|---|
| `last_active_date` | ✅ | Recency score — primary availability signal |
| `open_to_work_flag` | ✅ | Bonus if true |
| `recruiter_response_rate` | ✅ | Weighted contribution to multiplier |
| `interview_completion_rate` | ✅ | Weighted contribution to multiplier |
| `notice_period_days` | ✅ | Small bonus (<30d) / small penalty (>90d) |
| `avg_response_time_hours` | ✅ | Minor modifier (fast response = slight bonus) |
| `github_activity_score` | ✅ (weak) | Proxy for active coding; sentinel-safe |
| `skill_assessment_scores` | ✅ | Trust modifier on skill proficiency claims |
| `saved_by_recruiters_30d` | ✅ (weak) | Very minor market-interest signal |
| `profile_completeness_score` | ✅ (floor only) | Penalizes <40 as low-engagement signal |
| `profile_views_received_30d` | ❌ | Passive; dropped |
| `applications_submitted_30d` | ❌ | Noisy; dropped |
| `connection_count` | ❌ | Not predictive; dropped |
| `endorsements_received` (total) | ❌ | Redundant; dropped |
| `search_appearance_30d` | ❌ | Platform artifact; dropped |
| `expected_salary_range_inr_lpa` | ❌ | Not a fit criterion; dropped |
| `verified_email` / `verified_phone` | ❌ | Platform hygiene; dropped |
| `linkedin_connected` | ❌ | Platform hygiene; dropped |
| `signup_date` | ❌ | Not predictive; dropped |
| `preferred_work_mode` | ❌ | JD is flexible; dropped |

---

## Section F — Hard Disqualifier Gates (P_penalty)

These are explicitly stated in the JD as rejection criteria. They are applied as multiplicative penalties **before** ranking, not as scoring components.

| Disqualifier | JD Source | Penalty Applied |
|---|---|---|
| Honeypot candidate (fabricated profile) | Implicit in challenge spec | `× 0.01` (effectively removed) |
| Pure research career, no production deployment | "We will not move forward" | `× 0.20` |
| LangChain-only AI experience < 12 months, no pre-LLM ML | "We will probably not move forward" | `× 0.40` |
| No production code in last 18 months | "We will probably not move forward" | `× 0.25` |
| Consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/etc.) with no prior product-company experience | "We've had bad fit experiences" | `× 0.15` |
| CV / speech / robotics primary expertise, no NLP/IR | "You'd be re-learning fundamentals" | `× 0.30` |

Penalties are **multiplicative and stackable** — a consulting-only candidate who is also research-only gets `0.15 × 0.20 = 0.03`.

---

## Section G — What Is Deliberately NOT Scored

The following appear in the JD but are explicitly excluded from scoring because they are either not derivable from the candidate JSON or are noise.

| JD Element | Reason Excluded |
|---|---|
| "Async-first, writes well" | Subjective; not in schema |
| "Disagrees openly, decides quickly" | Subjective; not in schema |
| "Plans to stay 3+ years" | Future intent; not observable |
| "Comp / salary logistics" | Not a fit criterion |
| "Work visa sponsorship" | Not a fit criterion |
| "Quarterly travel for offsites" | Not a fit criterion |
| "Culture / vibe check" | Stage 4 interview only |
| Company-stage storytelling | Noise per execution plan |

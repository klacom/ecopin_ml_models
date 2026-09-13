# EcoPin ML — Experiment 06
## TARGETED POLLUTION ↔ WASTE TEST ERROR ANALYSIS
### Protected 12-Candidate Benchmark Audit

**Status:** Audit only — no retraining, no relabeling, no dataset changes.
**Date:** 2026-03-09
**Scope:** 12 protected pollution-labeled test images flagged RECLASSIFY_TO_WASTE by Experiment 04.

---

## 1. Objective

Determine WHY Experiment 05's protected benchmark shows 12/25 true-pollution images predicted as waste. For each of the 12 protected candidates, classify the evidence into one of:

- **A. TRUE POLLUTION** — Current pollution label is appropriate.
- **B. TRUE WASTE** — Image belongs to waste under project taxonomy.
- **C. TAXONOMY-AMBIGUOUS** — Both classes defensible.
- **D. MODEL ERROR** — Correctly labeled pollution; model incorrectly predicted waste.
- **E. BENCHMARK LABEL NOISE** — Current pollution benchmark label inconsistent with project taxonomy.

Additionally: compare Exp01/02/03/05 predictions on these candidates, detect systematic patterns, and assess whether Exp05's poor pollution recall (32%) stems from label noise, taxonomy overlap, model behaviour, or a combination.

---

## 2. Background

EcoPin ML classifies environmental report images into four classes:

| Class | Intended Scope (per Exp04/Exp05 + metadata CSV) |
|---|---|
| `flooding` | Standing water, inundation, flood waters |
| `non_environmental` | Generic scenes, animals, food, selfies, UI |
| `pollution` | Chemical/industrial contamination, acid mine drainage, smoke/air pollution, water with chemical discoloration, **ACTIVE construction with workers/machinery** |
| `waste` | **Solid waste**: garbage_accumulation, roadside_dumping, litter_accumulation, waterway_waste, public_area_waste, **fly-tipped construction scrap** (WST_001 precedent), illegal tire dumps |

**Key Exp05 results (official benchmark):**
- Accuracy 64.00%, Macro F1 0.6203
- Pollution: P=0.5714  R=0.3200  F1=0.4103
- Waste:     P=0.6154  R=0.9600  F1=0.7500
- Confusion boundary: **true pollution → waste = 12**, waste → pollution = 0

Exp04 audited 101 land_pollution + construction_debris images and found **90.10% should be RECLASSIFY_TO_WASTE**. Exp05 sanitized train/val accordingly but **protected the 12 matching test candidates unchanged** to preserve benchmark comparability. Those 12 are the subject of this report.

---

## 3. Sources Consulted

| Source | Path | Notes |
|---|---|---|
| SKILLS.md | `c:\dev\ecopin_ml_models\image_validation\my-skills\SKILLS.md` | Generic dev skills only; **contains no EcoPin taxonomy definitions** (contrary to initial assumption in experiment spec). Taxonomy authority therefore proxied via Exp04/Exp05 reports + metadata CSV. |
| Research paper (PDF) | `c:\dev\ecopin_ml_models\image_validation\my-skills\references\EcoPin A.I._ Crowdsourced Geospatial Platform for Transparent Environmental Reporting and Rapid Institutional Detection for the Pasig City Solid Waste Management Office.pdf` | Referenced; PDF binary not machine-parsed. |
| Exp04 audit report | `..\experiment_04_pollution_waste_audit.md` | 101-image audit with per-candidate subcategory + decision + reasoning |
| Exp05 report | `..\experiment_05_sanitized_taxonomy\EXPERIMENT_05_REPORT.md` | Protected filenames list (§8 L55); confusion matrix (§19 L169); boundary trends §20; comparative table §21 |
| Exp05 cross-exp predictions | `..\experiment_05_sanitized_taxonomy\protected_test_candidate_analysis.json` | Full 4-class probabilities for Exp01 / Exp02 / Exp03 / Exp05 on each of 12 candidates |
| Exp05 full test preds | `..\experiment_05_sanitized_taxonomy\test_predictions.json` | 100-image Exp05 predictions; confirms 8/12 protected = waste |
| Dataset metadata CSV | `C:\dev\datasets\ecopin_dataset\metadata\ecopin_dataset.csv` | WST_001–WST_099 subcategories establish waste precedent. **POL_113 through POL_246 have NO CSV rows** (discovered via grep; CSV ends at ~POL_030 for pollution entries). |
| Excluded rows CSV | `C:\dev\datasets\ecopin_dataset\reports\excluded_rows.csv` | Consulted for excluded-pattern precedent |
| Exp01 / 02 / 03 reports | Respective experiment dirs | Historical baseline, Exp03 sanitized non_environmental ↔ waste labels with same test-protection pattern |
| 12 image files | `C:\dev\datasets\ecopin_dataset\split\test\pollution\` | Visual taxonomy audit (all 12 inspected; paths verified §6) |

---

## 4. Exact 12 Protected Candidates

All 12 filenames verified to exist in `C:\dev\datasets\ecopin_dataset\split\test\pollution\`:

| # | File | Exp04 subcategory |
|---|---|---|
| 1 | POL_113.jpg | land_pollution |
| 2 | POL_121.jpg | land_pollution |
| 3 | POL_126.jpg | land_pollution |
| 4 | POL_135.jpg | land_pollution |
| 5 | POL_143.jpg | land_pollution |
| 6 | POL_145.jpg | land_pollution |
| 7 | POL_202.jpg | construction_debris |
| 8 | POL_209.jpg | construction_debris |
| 9 | POL_214.jpg | land_pollution |
| 10 | POL_222.jpg | construction_debris |
| 11 | POL_223.jpg | construction_debris |
| 12 | POL_246.jpg | construction_debris |

Breakdown by Exp04 source subcategory: 8 land_pollution, 4 construction_debris.

---

## 5–11. Per-Image Analysis (Consolidated)

The table below unifies Sections 5–11 for each candidate. Taxonomy reasoning uses project definitions (§2 table): pollution = chemical/industrial/atmospheric contamination OR active construction WITH workers/machinery; waste = solid waste dumping / litter / fly-tipped scrap / illegal tires.

| # | Image | Visual content | Dominant phenomenon | Exp04 decision | **Independent decision** | Exp05 pred | Exp05 probs (F / N / P / W) | p→w? | Cross-exp preds & waste-prob trend |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **POL_113** | Sandy beach shore + scattered plastic bottles/bags/IPA Andina bin/bamboo | Coastal litter accumulation | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | non_env | 0.01 / 0.65 / 0.00 / 0.34 | ❌ | 01:w(0.98)→02:p(0.07)→03:n(0.33)→05:n(0.34) — INCONSISTENT (scene-context: sandy beach overwhelms litter signal) |
| 2 | **POL_121** | Forest dirt road + 4 concrete culvert rings + household junk dump + NO-ENTRY barrier | Roadside waste dumping | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | flooding | 0.59 / 0.04 / 0.06 / 0.31 | ❌ | 01:w(0.51)→02:p(0.15)→03:p(0.27)→05:f(0.31) — INCONSISTENT (Exp05-only flooding false-positive; confuses wet dirt for flooding) |
| 3 | **POL_126** | Close-up ravine slope dense garbage: Pepsi bottles, sacks, shoes, barrels | Dense garbage dump | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.00 / **0.9997** | ✅ | 01:w(0.9999)→02:w(0.9998)→03:w(0.9998)→05:w(0.9997) — ALWAYS_WASTE (strongest label-noise signal) |
| 4 | **POL_135** | Grassy hillside + pile of CRT monitors / keyboards / yellow crates / tires | E-waste (hazardous) dump | RECLASSIFY_TO_WASTE | **A — TRUE_POLLUTION** (DISAGREEMENT) | pollution | 0.00 / 0.00 / **0.87** / 0.12 | ❌ | 01:p(0.03)→02:p(0.00)→03:p(0.00)→05:p(0.12) — NEVER_WASTE (all 4 experiments correctly retain as pollution; e-waste = toxic = pollution per project chemical-contamination KEEP rule) |
| 5 | **POL_143** | Alleyway dirt ditch + dense domestic trash food pack / bottles / cans / black bags by stone drain | Domestic litter accumulation | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.02 / 0.01 / **0.97** | ✅ | 01:p(0.19)→02:w(0.997)→03:w(0.984)→05:w(0.969) — NEWLY_WASTE (Exp02 onwards learned signal) |
| 6 | **POL_145** | Wooded hillside + dozens of discarded black rubber tires among ivy/trees | Illegal tire dump | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.02 / **0.98** | ✅ | 01:p(0.06)→02:w(0.949)→03:w(0.853)→05:w(0.977) — NEWLY_WASTE |
| 7 | **POL_202** | Open park-like field + weathered mossy concrete slabs pile + embedded red bricks + bare branches | Fly-tipped concrete rubble | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | non_env | 0.02 / 0.56 / 0.19 / 0.24 | ❌ | 01:p(0.03)→02:p(0.00)→03:p(0.06)→05:n(0.24) — NEVER_WASTE (scene-context bias: open-field/park setting overwhelms inert-concrete-rubble waste signal; no visual garbage-shortcut trigger) |
| 8 | **POL_209** | Fenced commercial yard + dumped white building panels/tarps/rusted pipes/ladder wood on gravel | Fly-tipped commercial construction scrap | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.00 / **0.996** | ✅ | 01:p(0.15)→02:w(0.998)→03:w(0.960)→05:w(0.996) — NEWLY_WASTE |
| 9 | **POL_214** | Rural dirt road + massive overflowing roadside dump: drywall/furniture/bags/cabinet drawers + 3 cleanup workers + white truck | Roadside dumping (cleanup in progress) | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.00 / **0.996** | ✅ | 01:w(0.960)→02:w(0.999)→03:w(0.999)→05:w(0.996) — ALWAYS_WASTE (cleanup workers = NOT active construction crew; they are remediating a dump) |
| 10 | **POL_222** | Extreme close-up: jumbled broken terracotta roof tiles + concrete chunks w/ mortar + red brick | Fly-tipped demolition rubble | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.48 / **0.52** | ✅ | 01:p(0.03)→02:p(0.00)→03:p(0.03)→05:w(0.52) — NEWLY_WASTE (Exp05 training distribution shift: model now classifies tile rubble as waste after sanitized train/val taught pattern; borderline at 51.85%) |
| 11 | **POL_223** | Urban active construction site (blue tarps) + multi-story buildings + pile of rubble-filled plastic sacks (Y/G/W/B) + broken tricycle on top + muddy puddles | Fly-tipped construction waste bags curbside | RECLASSIFY_TO_WASTE | **E — BENCHMARK_LABEL_NOISE** | waste | 0.00 / 0.00 / 0.01 / **0.985** | ✅ | 01:w(0.698)→02:w(0.960)→03:w(0.976)→05:w(0.985) — ALWAYS_WASTE (sacks foreground + tricycle = clear garbage-shortcut trigger; active construction is background, not subject) |
| 12 | **POL_246** | Urban street + building with fish murals + mixed curb pile: plaster/concrete rubble + cardboard sacks + torn white bags + black garbage bags + food-orange debris | Mixed: construction dust + municipal waste (ambiguous) | RECLASSIFY_TO_WASTE | **C — TAXONOMY-AMBIGUOUS** | waste | 0.00 / 0.00 / 0.35 / **0.645** | ✅ | 01:p(0.06)→02:w(0.813)→03:w(0.843)→05:w(0.645) — NEWLY_WASTE (Exp03 correctly predicted pollution at 56.31% while Exp02/05 predict waste; visual genuinely contains both construction-rubble (pollution) and garbage-bag (waste) signals — no forced call) |

### §11 Exp04 vs. Independent Decision Matrix

| Alignment | Count | Candidates |
|---|---|---|
| **CONFIRMED** (both agree RECLASSIFY_TO_WASTE) | 10 | POL_113, 121, 126, 143, 145, 202, 209, 214, 222, 223 |
| **DISAGREEMENT** (Exp04: waste; Independent: pollution) | 1 | **POL_135** — e-waste contains hazardous/heavy-metal components, meeting Exp05's "chemical/toxic contamination" pollution-KEEP rule. Exp04 was too broad in equating all dumped solid items with waste. |
| **AMBIGUOUS** | 1 | **POL_246** — both interpretations defensible |

**Exp04 findings 83.3% confirmed, 8.3% disagreed, 8.3% ambiguous.**

---

## 12. Systematic Error Patterns

### 12A. Dominant Pattern: Model Shortcut = Dense Garbage-Pile Detector

Visual pattern shared by the **8 candidates Exp05 predicts waste with ≥96% confidence**:

- POL_126 (99.97%), POL_143 (96.92%), POL_145 (97.71%), POL_209 (99.57%), POL_214 (99.64%), POL_222 (51.85%), POL_223 (98.47%), POL_246 (64.47%)

All contain a **dense, frame-filling, colorful, amorphous concentrated pile** whose texture the model has learned to map to `waste`. The 5 strongest waste predictions (≥98%) are all tight close-ups on recognizable consumer garbage (plastic sacks, bottles, food packaging, garbage bags, furniture). This is consistent with **shortcut learning: dense_amorphous_pile + colorful_packaging / garbage_bag → waste**, without considering the semantic pollution/waste distinction.

### 12B. Scene-Context-Overwhelms-Subject Pattern

The **4 candidates NOT consistently predicted waste despite being solid-waste**:

| Candidate | Inconsistency cause |
|---|---|
| POL_113 | Sandy beach / sea-shore context triggers `non_environmental` shortcut instead of litter-dump detector (beach = "scenic/non-env" association). Second-top class is waste at 34.06%. |
| POL_121 | Forested wet dirt road triggers `flooding` shortcut in Exp05 (59.40% — Exp05-only error; exp02/03 correctly predict pollution). Concrete culvert rings visually resemble flood-drain infrastructure → flooding association. |
| POL_202 | Weathered concrete in open park-like field → model sees stone/landscape texture → `non_environmental` (56.33%). No plastic/garbage-color cues → garbage-shortcut does NOT fire, even though subject = inert fly-tipped rubble waste. |
| POL_246 | Mixed signals plaster rubble + cardboard sacks → borderline. Exp03 correctly predicted pollution; Exp02/05 predict waste. |

### 12C. Exp05 Training-Sanitization Effect

Comparing waste-prediction counts across experiments on 12 protected candidates:

| Experiment | # predicted waste (of 12) | Trend |
|---|---|---|
| Exp01 (fresh pretrained) | 6 | Baseline: learned garbage shortcut from general pretrained features + small waste set |
| Exp02 (baseline finetune) | 7 | +1 (POL_143 joins waste) |
| Exp03 (sanitized labels: non_env↔waste) | 7 | Stable, same as Exp02 |
| **Exp05 (sanitized pollution↔waste)** | **8** | **+1 (POL_222 newly waste at 51.85%)** — Exp05's train/val taught the model that previously-pollution-labeled tile rubble is now waste. **The model's p→w boundary got MORE aggressive after train/val sanitization, not less.** |

This training-shift explains Exp05's **worse** pollution→waste boundary (12) vs Exp01 (7), Exp02 (11), Exp03 (10). The model correctly learned the train/val ground truth and applied the same reasoning to the protected benchmark. Because the benchmark is inconsistent with the train/val sanitized taxonomy, the gap *widened*.

---

## 13. Confirmed Benchmark Label Noise

**10 of 12 protected candidates (83.3%) confirmed as BENCHMARK_LABEL_NOISE** — their images fall squarely in waste under the project's solid-waste-vs-pollution distinction:

1. POL_113 — beach litter
2. POL_121 — roadside dumping (concrete rings + junk)
3. POL_126 — ravine garbage dump (strongest signal)
4. POL_143 — alleyway domestic trash
5. POL_145 — illegal tire dump
6. POL_202 — inert concrete/brick rubble in field
7. POL_209 — commercial construction scrap behind fence
8. POL_214 — massive roadside dump (cleanup in progress)
9. POL_222 — terracotta roof-tile demolition rubble
10. POL_223 — curbside rubble-filled plastic sacks near construction

*All 10 lack chemical/industrial/atmospheric pollution and lack active construction workers/machinery as the subject.*

---

## 14. Confirmed Genuine Model Errors

**0 confirmed MODEL-ERROR cases on the protected set.** Reason:

- POL_135 is the only confirmed TRUE_POLLUTION case, and Exp01/02/03/05 ALL correctly predict it as pollution (86.92% confidence Exp05). So the model handles e-waste correctly.
- POL_246 is ambiguous and predicted waste 64.47% → at worst a genuine model error IF we enforce pollution on the ambiguous case, but since we treat it as ambiguous (not forced pollution), we don't count it.
- The 4 non-p→w wrong predictions (POL_113→non_env, POL_121→flooding, POL_202→non_env, POL_246→waste borderline) are scene-context failures, NOT pollution-vs-waste confusions.

However, **Exp05 full confusion matrix reports 12 total pollution→waste errors. The 12 protected candidates contribute 8 of those 12. The remaining 4 (POL_042, POL_165, POL_169, POL_199) are NON-protected pollution images NOT flagged by Exp04 as reclassify-to-waste.** Those 4 are **genuine pollution→waste model errors**. This audit does not analyze those 4; we note them for follow-up.

---

## 15. Ambiguous Cases

**1 of 12 = TAXONOMY-AMBIGUOUS: POL_246**

- Pro-pollution evidence: plaster rubble, tile fragments, construction dust, concrete chunks, urban street in front of a building — fits "active construction / construction pollution" pattern. Exp03 predicted pollution (56.31%).
- Pro-waste evidence: cardboard boxes stacked like discarded sacks, torn white/black garbage bags, food-orange trash on ground, curbside position (not inside active work site), no visible workers/machinery. Exp02/05 predict waste.
- **Verdict:** both classes defensible. Do not force into either class for benchmark purposes; document as ambiguous.

---

## 16. HYPOTHETICAL SECONDARY ANALYSIS

> **WARNING — THIS SECTION IS NOT OFFICIAL.** It calculates what Exp05 pollution/waste metrics WOULD look like IF the 10 independently-confirmed BENCHMARK-LABEL-NOISE candidates were reclassified as waste (removing them from pollution test support). The physical benchmark (`split/test/`) is UNCHANGED. This is an analytic sensitivity study for planning purposes only.

### Inputs

| Parameter | Value |
|---|---|
| Original true-pollution support (Exp05 benchmark) | 25 |
| Benchmark-label-noise candidates (§13) | 10 |
| After correction: new true-pollution support | 25 − 10 = **15** |
| Among the 10 label-noise: how many predicted waste by Exp05? | **6** (POL_126, 143, 145, 209, 214, 223) + borderline POL_222 (51.85% waste) ≈ 7 effectively-correct-as-waste now, vs 10 total moved to waste ground-truth |
| Original total p→w errors (benchmark) | 12 |
| Original 12 p→w errors breakdown: 8 protected + 4 non-protected | POL_042, 165, 169, 199 (genuine model errors) remain p→w errors under corrected taxonomy |

### Corrected (Hypothetical) Pollution Metrics

| Metric | Original (benchmark) | **Hypothetical corrected** | Δ |
|---|---|---|---|
| Support | 25 | 15 | −10 |
| Correct (TP) | 8 | 15 − (4 non-prot model errors + POL_246 waste) ≈ 15 − 5 = **10** | +2 |
| Recall | 8/25 = **32.00%** | 10/15 = **66.67%** | **+34.67 pp** |
| Precision (orig) | 0.5714 | Depends on new waste→p errors (unchanged 0) + corrected TP vs p total |
| F1 (orig) | 0.4103 | ≈ **0.61–0.66** (depends on exact precision calc) |

### Interpretation

If the 10 confirmed label-noise candidates are treated as waste per project taxonomy, pollution recall jumps from **32.00% → ~66.67%**. The 12-candidate protected benchmark accounts for roughly **34.67 percentage points of the apparent pollution-recall collapse**.

Remaining genuine pollution→waste errors (~4–5) are:

- 4 non-protected (POL_042, 165, 169, 199) — model failures on true-pollution images
- 1 ambiguous (POL_246) — borderline

**Bottom line: most of Exp05's apparent pollution-recall crisis is benchmark label noise, NOT model failure.**

---

## 17. Limitations

1. **SKILLS.md taxonomy absence.** The specified file contained generic dev skills only. A proxy taxonomy was derived from Exp04 §5 definitions, Exp05 KEEP precedent, and metadata CSV subcategory names. If SKILLS.md or the referenced PDF actually contained stricter/different taxonomy rules than the project reports, this analysis may differ.
2. **Missing metadata CSV rows.** POL_113–POL_246 have no rows in `ecopin_dataset.csv`. We therefore could not cross-reference source_name, source_url, location, event_group, or notes fields. Subcategory membership for these 12 is derived solely from Exp04 audit table rows.
3. **PDF unreadability.** The referenced Pasig-SWMO research PDF was not text-parsed; reliance was entirely on project reports.
4. **Visual-only taxonomy call.** For each image we relied on visible content. A single still image cannot capture olfactory, aqueous-contamination, or time-series evidence that might elevate a solid-waste scene into pollution. This biases §13 calls slightly conservative (fewer label-noise claims).
5. **Non-protected p→w errors not audited.** 4 of 12 total p→w errors (POL_042, POL_165, POL_169, POL_199) are outside this audit's scope. §14 count of "0 confirmed model errors" only covers the 12 protected candidates; genuine model errors exist among non-protected pollution images.
6. **Label-noise vs. taxonomy overlap conflation.** "Land pollution" as a phrase inherently overlaps with "solid waste dumped on land" at the semantic level. To the extent the original dataset's intent was broader than Exp04's refined distinction, the benchmark may be internally consistent with its own (different) taxonomy — but then it contradicts Exp05's train/val sanitization, leading to the train/test-skew Exp05 experienced.

---

## 18. Recommendations for the Next Experiment

Based on §12–§17 evidence, the priority stack for Exp07:

### Priority 1 (Do first) — RESOLVE BENCHMARK LABEL NOISE → Create a Manually-Verified Clean Test Benchmark

The current 100-image protected benchmark contains at minimum **10 confirmed label-noise pollution→waste candidates** and likely additional noise in pollution and other classes. Continuing to run training experiments against the current benchmark produces misleadingly low pollution recall and an unstable optimization target. Recommend:

- **Create `experiment_07_clean_benchmark`:** A new 100–120 image benchmark, manually verified *blind* by 2 independent reviewers against the project's refined pollution/waste taxonomy (Exp04 §5 rules).
- Include audit trails (reviewer A/B decisions per image, ambiguity notes).
- Do NOT overwrite `split/test/`; store the clean benchmark as a separate manifest that indexes into the existing files without moving them (e.g., `clean_benchmark_manifest.json`).
- Run Exp01–05 checkpoints against the clean benchmark retroactively to obtain apples-to-apples historical comparison.

### Priority 2 — If Retraining Continues Against Existing Benchmark: Targeted Hard-Negative Sampling

If the original benchmark must be retained for comparability, Exp07 should offset the model's garbage-shortcut tendency by:

- Augmenting pollution-class training with **hard negatives**: images of dense garbage piles *that have also been relabeled pollution* (rare — only e-waste POL_135 style cases exist; or alternatively synthetic mixes of chemical-pollution + garbage-foreground).
- **Loss weighting**: upweight pollution class by factor ~1.5–2.0× to counteract the shortcut toward waste caused by 10 label-noise pollution examples teaching the model that *land_pollution-labeled garbage = actually waste*.
- Adding **scene-context augmentation** for POL_202-style concrete-rubble-in-fields (crop-only policy, cutmix/mixup with chemical-pollution scenes) to break "inert stone = non_environmental" shortcuts.

### Priority 3 — Low-Utility Actions

- **Do NOT run more generic finetuning / longer training runs / different seeds alone.** These will not fix label noise and will likely widen the p→w boundary further (as Exp05 vs Exp01 demonstrated).
- **Do NOT swap architecture alone.** The shortcut is data-driven, not capacity-limited; EfficientNet-B0 already correctly handles POL_135 (e-waste) and POL_126 (garbage) at 99%+ confidence, showing sufficient visual capacity given consistent labels.

### Priority 4 — Follow-Up Audit (Non-Protected p→w Errors)

Audit the 4 remaining pollution→waste errors (POL_042, POL_165, POL_169, POL_199) with the same §5–§11 framework; they are genuine model errors per Exp04's lack of reclassify flag. Understanding their visual pattern will inform hard-negative augmentation design.

---

*End of Experiment 06 Error Analysis. No dataset files, labels, checkpoints, or prior experiments were modified for this report.*

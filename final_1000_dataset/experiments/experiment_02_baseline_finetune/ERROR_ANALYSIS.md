# Experiment 02 Error Analysis

## 1. Executive finding

The misclassification of obvious garbage images as `non_environmental` is primarily caused by a **dataset label contradiction and structural class overlap** within the `non_environmental` training dataset. Specifically, 125 images in `non_environmental` are designated as `hard_negative` samples, of which at least 15 explicitly depict **trash cans, waste containers, trash collection rickshaws, garbage trucks, and rubbish removal vehicles**. Because the model was trained to identify urban public settings containing trash containers and collection vehicles as `non_environmental`, it heavily activates `non_environmental` feature detectors when shown dumped garbage in urban public areas.

---

## 2. Waste → Non-environmental confusion

Based on existing Experiment 02 validation and test artifacts:

- **Validation Set (100 samples total, 25 per class)**:
  - **3 out of 25 true `waste` images (12.0%)** were misclassified as `non_environmental`.
  - Notable examples:
    - `WST_180.jpg` ("Close-up of soda cans, face masks and other trash"): Predicted as `non_environmental` with **94.77% confidence** (only 4.38% `waste`).
    - `WST_154.jpg` (Public area waste): Predicted as `non_environmental` with **67.68% confidence**.
    - `WST_171.jpg` ("Streets of Baguio waste"): Predicted as `non_environmental` with **63.03% confidence**.
  - **Waste Validation Metrics**: Recall = **68.00%** (17/25), Precision = **77.27%** (17/22), F1 = **0.7234**.

- **Test Set (100 samples total, 25 per class)**:
  - **1 out of 25 true `waste` images (4.0%)** was misclassified as `non_environmental`: `WST_095.jpg` (predicted as `non_environmental` with **62.63% confidence**, 35.92% `waste`).
  - **Waste Test Metrics**: Recall = **88.00%** (22/25), Precision = **59.46%** (22/37), F1 = **0.7097**.

---

## 3. Non-environmental → Waste confusion

- **Validation Set (25 true `non_environmental` samples)**:
  - **2 out of 25 true `non_environmental` images (8.0%)** were misclassified as `waste`.
  - Specific instances:
    - `NEG_075.jpg`: Predicted as `waste` with **92.12% confidence**.
    - `NEG_203.jpg` ("Sunset in Cebu"): Predicted as `waste` with **95.05% confidence**.
  - **Non-Environmental Validation Metrics**: Recall = **68.00%** (17/25), Precision = **65.38%** (17/26), F1 = **0.6667**.

- **Test Set (25 true `non_environmental` samples)**:
  - **4 out of 25 true `non_environmental` images (16.0%)** were misclassified as `waste`.
  - Specific instances:
    - `NEG_060.jpg` (Sculpture): Predicted as `waste` with **97.56% confidence**.
    - `NEG_220.jpg` (Iloilo City Public Market): Predicted as `waste` with **92.98% confidence**.
    - `NEG_222.jpg` (Tondo life urban scene): Predicted as `waste` with **70.50% confidence**.
    - `NEG_224.jpg`: Predicted as `waste` with **50.76% confidence**.
  - **Non-Environmental Test Metrics**: Recall = **56.00%** (14/25), Precision = **53.85%** (14/26), F1 = **0.5490**.
  - **Over-prediction of Waste**: In the test set, `waste` was predicted **37 times** across 100 images, receiving 11 false positives from `pollution` and 4 from `non_environmental`, driving `waste` test precision down to **59.46%**.

---

## 4. Waste class characteristics

The `waste` class comprises **250 total images** across the final dataset (200 train / 25 val / 25 test):

- **Subcategory Distribution**:
  - `garbage_accumulation`: 91 images (36.4%)
  - `litter_accumulation`: 76 images (30.4%)
  - `roadside_dumping`: 35 images (14.0%)
  - `public_area_waste`: 21 images (8.4%)
  - `other`: 18 images (7.2%)
  - `waterway_waste`: 8 images (3.2%)
  - `liter_accumulation`: 1 image (0.4%)
- **Visual Features**: Piles of uncollected trash, discarded plastic bags on road curbs, street litter in public spaces/markets/slums, roadside waste dumps, and overflowing trash receptacles.

---

## 5. Non-environmental class characteristics

The `non_environmental` class comprises **250 total images** (200 train / 25 val / 25 test):

- **Subcategory Distribution**:
  - `easy_negative`: 125 images (50.0%) — clear non-environmental subjects such as indoor items, domestic cats, historic architecture/monuments, clean highway pavements, metro stations, and scenic mountains.
  - `hard_negative`: 125 images (50.0%) — complex urban environments, crowded city markets, roadside ditches, puddles, **AND trash management infrastructure (trash cans, waste containers, garbage trucks, and rubbish pick-up vehicles)**.
- **Visual Features**: Urban outdoor spaces, street asphalt, concrete sidewalks, public area infrastructure, and street scenes containing trash bins or collection vehicles.

---

## 6. Potential label contradictions

Direct inspection of `C:\dev\datasets\ecopin_google_sheets.xlsx` metadata reveals severe label contradictions within `non_environmental` (`hard_negative`):

- **Concrete Examples of Waste / Trash Labeled as `non_environmental`**:
  - `NEG_126`: "Trash collection Rickshaw in Banani, Dhaka"
  - `NEG_127`: "Trash can, Beyoğlu 7May23"
  - `NEG_128`: "Trash cans spot"
  - `NEG_129`: "Trash can, Oak & Dante Streets, New Orleans"
  - `NEG_130`: "Trash can, Huntley Meadows Park"
  - `NEG_131`: "Trash, recycling and compost bins in Sonoma, Calif."
  - `NEG_132`: "Trash Truck"
  - `NEG_133`: "Sorted waste containers in the Philippines"
  - `NEG_134`: "Trash collectors on truck in Leh / Ladakh, India"
  - `NEG_135`: "Trash collection Rickshaw in Rajshahi, Bangladesh"
  - `NEG_136`: "Rid My Rubbish Darlington Rubbish Removal Loaded Pick-Up"
  - `NEG_141`: "Garbage Aboboyaa.jpg" (Ghanaian garbage collection vehicle)
  - `NEG_142`: "Waste containers in Dolný Kubín, Slovakia"
  - `NEG_248`: Explicit note: `"Non-environmental subject: Garbage...NO!!!"`
  - `NEG_249`: Explicit note: `"Non-environmental subject: brazilian garbage."`

- **Impact**: The training dataset explicitly teaches the neural network that scenes containing trash cans, waste containers, garbage trucks, and rubbish collection rickshaws are `non_environmental`. When an urban garbage scene is evaluated during inference, the combination of urban street background and trash/container visual features triggers strong `non_environmental` activations.

---

## 7. Replacement-data impact

The final 1,000-image dataset contains **96 leakage replacements** (replacing images by blacklisted author `Judgefloro` and near-duplicates):

- **Replacement Breakdown**:
  - `flooding`: 37 replacements
  - `non_environmental`: 28 replacements
  - `waste`: 19 replacements
  - `pollution`: 12 replacements
- **Impact on `waste` vs `non_environmental`**:
  - `non_environmental` replacements added modern urban street environments and public transit stations (e.g., NYC street scenes, Lisbon street photography, Istanbul urban scenes, Brazil metro stations).
  - `waste` replacements added structured waste containers and Paris street litter bins (`WST_017`, `WST_018`, `WST_020`, `WST_031`).
  - This design choice further heightened the visual feature overlap between structured trash receptacles in `waste` vs. structured trash receptacles in `non_environmental`.

---

## 8. Exp 8 vs Exp 02

- **Historical Exp 8 (Original ~350-image dataset)**:
  - User's target garbage image output: `non_environmental`: **99.98%**, `waste`: **~0.00%**.
- **Experiment 02 (Final 1,000-image dataset)**:
  - Same target garbage image output: `non_environmental`: **88.48%**, `waste`: **10.75%**, `flooding`: **0.56%**, `pollution`: **0.21%**.
- **Comparison Summary**:
  - **What Improved**: Exp 02 detected waste signal in the target image (increasing `waste` probability from ~0% to 10.75% and reducing `non_environmental` from 99.98% to 88.48%).
  - **What Did Not Improve**: Exp 02 still misclassified the target image as `non_environmental` (88.48%) because the underlying dataset label ambiguity (trash containers/vehicles included in `non_environmental` hard negatives) was preserved and scaled up in the 1,000-image dataset.

---

## 9. Diagnosis

### Why does an obvious garbage image produce `non_environmental` = 88.48%, `waste` = 10.75%?

1. **Confirmed Evidence**:
   - `non_environmental` contains 125 `hard_negative` images, including at least 15 explicit images of trash cans, waste containers, recycling bins, garbage trucks, and rubbish collection rickshaws (`NEG_126`–`NEG_142`, `NEG_248`, `NEG_249`).
   - EfficientNet-B0 extracts both global background context (urban road pavement, buildings, public area layout) and localized object patterns (plastic bags, bins, trash objects).
   - Because the training set explicitly conditions `non_environmental` on urban street scenes containing trash containers and collection vehicles, the model's highest-weighted output branch for urban public waste settings is `non_environmental`.
2. **Hypotheses**:
   - The global urban pavement/background features in the target image strongly match `hard_negative` urban non-environmental patterns, causing `non_environmental` logits to heavily dominate `waste` logits.

---

## 10. Recommended next step

**MINIMUM RECOMMENDED ACTION**:

1. **Audit and Sanitize `non_environmental` Hard Negatives**:
   - Inspect all 125 `hard_negative` samples in `non_environmental`.
   - Remove or reclassify images containing trash cans, waste containers, recycling bins, garbage trucks, rubbish collection vehicles, and street litter (specifically `NEG_126` through `NEG_142`, `NEG_248`, `NEG_249`).
2. **Clarify Class Taxonomy**:
   - Establish a unambiguous definition for `waste`: any visible environmental waste, uncollected garbage, or public trash containers with waste.
   - Restrict `non_environmental` strictly to clean outdoor/indoor scenes, urban infrastructure without waste/trash objects, animals, and non-polluted landscapes.
3. **Do NOT Retrain Immediately**:
   - Postpone retraining until `non_environmental` hard negatives are sanitized to prevent perpetuating class ambiguity.

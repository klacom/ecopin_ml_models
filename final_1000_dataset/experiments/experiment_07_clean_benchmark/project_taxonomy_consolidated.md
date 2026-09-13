# EcoPin Project Taxonomy - Consolidated for Experiment 07

## Taxonomy Sources
1. Dataset metadata CSV (ecopin_dataset.csv) - subcategory definitions
2. Experiment 04 audit report - pollution vs waste distinction
3. Experiment 05 report - KEEP precedent for construction pollution
4. Experiment 06 error analysis - e-waste classification rules
5. Research paper (PDF) - could not be read as text

## Class Definitions

### 1. Flooding
**Definition**: Standing water, inundation, flood waters

**Subcategories** (from metadata):
- General flooding scenes

**Key Characteristics**:
- Visible standing water
- Inundated areas
- Flood waters
- Water overflow

---

### 2. Non_Environmental
**Definition**: Generic scenes, animals, food, selfies, UI, anything not related to environmental reporting

**Subcategories** (from metadata):
- Generic scenes
- Animals
- Food
- Selfies
- UI elements

**Key Characteristics**:
- No environmental phenomenon
- Generic everyday scenes
- Non-environmental content

---

### 3. Pollution
**Definition**: Chemical/industrial contamination, acid mine drainage, smoke/air pollution, water with chemical discoloration, ACTIVE construction with workers/machinery

**Subcategories** (from metadata):
- land_pollution
- construction_debris
- air_pollution
- water_pollution
- industrial_contamination

**Key Characteristics**:
- Chemical/toxic contamination
- Industrial contamination
- Acid mine drainage
- Smoke/air emissions
- Water with chemical discoloration
- **ACTIVE construction with workers/machinery** (KEEP precedent from Exp05)

**Special Rules**:
- **E-waste**: Electronic waste containing hazardous/toxic components (lead, mercury, cadmium, heavy metals, brominated flame retardants) qualifies as pollution due to toxic contamination (Exp06 POL_135 precedent)
- **Active construction**: Construction sites with visible workers, machinery, or active work qualify as pollution (Exp05 KEEP precedent: POL_215, POL_224, POL_225, POL_238, POL_242, POL_247, POL_249)
- **Inert construction debris**: Dumped/fly-tipped construction rubble without active workers/machinery does NOT qualify as pollution (classifies as waste)

**Pollution KEEP Examples** (from Exp04):
- POL_148: Acid mine drainage and severely contaminated toxic chemical orange mud/soil runoff
- POL_215: Active municipal street excavation site with urban roadwork barriers
- POL_224: Active road construction site with 'MCD MEN AT WORK' sign
- POL_225: Active street utility excavation with dirt mounds and 'MEN AT WORK' sign
- POL_238: Active sidewalk trenching and pipe utility construction site
- POL_242: Active municipal road repaving work site
- POL_247: Active construction worker shoveling inside structural building renovation
- POL_249: Active CAT excavator operating at heavy building foundation excavation
- POL_135: E-waste (CRT monitors, keyboards) - toxic contamination

---

### 4. Waste
**Definition**: Solid waste accumulation, illegal dumping, litter, fly-tipped construction scrap

**Subcategories** (from metadata):
- garbage_accumulation
- roadside_dumping
- litter_accumulation
- waterway_waste
- public_area_waste
- other

**Key Characteristics**:
- Solid waste accumulation
- Illegal dumping
- Litter accumulation
- Roadside dumping
- Waterway waste
- Public area waste
- **Fly-tipped construction scrap** (WST_001 precedent: "Fly-tipping construction waste")

**Special Rules**:
- **Construction waste**: Dumped/fly-tipped construction debris (concrete rubble, bricks, tiles, drywall, timber scrap) without active workers/machinery qualifies as waste (WST_001 precedent)
- **Tire dumps**: Illegal tire dumping qualifies as waste (solid waste mass)
- **E-waste context**: While e-waste has toxic components, dumped consumer electronics without visible toxic contamination may be classified as waste unless toxic aspect is dominant (Exp06 POL_135 was classified as pollution due to explicit toxic contamination reasoning)

**Waste Precedent Examples** (from metadata):
- WST_001: Fly-tipping construction waste in Parco Alto Milanese
- WST_002: Mixed-waste plastic packaging
- WST_003: Various packaging waste in water systems
- WST_011: Refuse dump at river side
- WST_012: Electrical garbage thrown at road side

---

## Critical Decision Boundaries

### Pollution vs Waste

**Construction Context**:
- **Pollution**: Active construction with workers, machinery, active work (MEN AT WORK signs, excavators, construction crews)
- **Waste**: Inert/fly-tipped construction debris (rubble, bricks, tiles, drywall) without active workers/machinery

**E-Waste**:
- **Pollution**: E-waste with visible toxic/hazardous contamination aspect (CRT monitors, electronics with heavy metals)
- **Waste**: General electronic items without dominant toxic contamination (context-dependent)

**Chemical/Toxic**:
- **Pollution**: Visible chemical contamination, acid mine drainage, toxic runoff, industrial emissions
- **Waste**: Solid waste accumulation even if decomposition could cause contamination (focus on visible solid waste aspect)

**Scene Context**:
- **Pollution**: Active industrial activity, smoke emissions, chemical runoff
- **Waste**: Solid waste piles, garbage accumulation, litter, illegal dumping

### Ambiguous Cases

**Mixed Scenes**:
- Images containing both pollution and waste elements require determining the DOMINANT reportable phenomenon
- Both interpretations may be defensible
- Mark as AMBIGUOUS if reasonable interpretations remain

**Insufficient Evidence**:
- Images where context is insufficient to determine category
- Mark as INSUFFICIENT_EVIDENCE if image does not provide enough information

---

## Verification Methodology

**Step 1**: Inspect image visually
**Step 2**: Identify dominant environmental phenomenon
**Step 3**: Apply taxonomy rules above
**Step 4**: Assign verification status:
- VERIFIED_CORRECT_LABEL
- VERIFIED_MISLABELED
- AMBIGUOUS
- INSUFFICIENT_EVIDENCE

**Step 5**: Document reasoning citing specific taxonomy rules

---

## Taxonomy Limitations

1. **Research paper**: PDF could not be text-parsed; taxonomy derived from experiment reports and metadata CSV only
2. **Metadata coverage**: POL_113 through POL_246 have no CSV rows; subcategory information derived from Exp04 audit
3. **Visual-only taxonomy**: Single still images cannot capture olfactory, aqueous-contamination, or time-series evidence
4. **E-waste boundary**: Toxic contamination vs solid waste distinction requires careful visual analysis

---

## References

- Dataset metadata: C:\dev\datasets\ecopin_dataset\metadata\ecopin_dataset.csv
- Exp04 audit: experiment_04_pollution_waste_audit.md
- Exp05 report: experiment_05_sanitized_taxonomy\EXPERIMENT_05_REPORT.md
- Exp06 analysis: experiment_06_error_analysis\EXPERIMENT_06_ERROR_ANALYSIS.md
- Research paper: EcoPin A.I._ Crowdsourced Geospatial Platform... (PDF - not text-parseable)
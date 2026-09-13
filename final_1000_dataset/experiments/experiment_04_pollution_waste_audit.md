# EXPERIMENT 04 — POLLUTION ↔ WASTE LABEL AUDIT REPORT

## 1. Objective

In Experiment 03, after resolving `nature` vs `pollution` label noise, the overall test accuracy reached 62.00% and test macro F1 reached 0.6137. However, the error analysis identified a single dominant remaining error boundary: **POLLUTION ↔ WASTE**.

Specifically, 10 out of 25 true `pollution` test images were misclassified as `waste` by the model. The objective of this audit is to conduct a systematic, image-level visual audit of target `pollution` subcategories (`construction_debris` and `land_pollution`) to evaluate whether dataset label noise or taxonomy ambiguity explains this performance bottleneck.

## 2. Dataset Scope

- **construction_debris images audited**: 51
- **land_pollution images audited**: 50
- **Total images audited**: 101

## 3. Full Image-Level Audit

| ID | Filename | Subcategory | Current Label | Decision | Reason |
|---|---|---|---|---|---|
| POL_101 | POL_101.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Mass dumping of discarded clothing and textile waste in Atacama desert. |
| POL_102 | POL_102.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Urban roadside refuse accumulation and scattered food packaging waste near wall. |
| POL_103 | POL_103.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Scattered plastic wrappers, bottles, and litter accumulation in grassy vegetation. |
| POL_104 | POL_104.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Massive open solid waste landfill dumping site with livestock present. |
| POL_105 | POL_105.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Coastal shoreline littered with plastic debris, bottles, and discarded items. |
| POL_106 | POL_106.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Concrete drainage ditch choked with floating plastic bottles and debris. |
| POL_107 | POL_107.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Extensive open garbage dumping site in front of residential buildings. |
| POL_108 | POL_108.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Uncontrolled roadside refuse pile on unpaved dirt ground. |
| POL_109 | POL_109.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Natural stream completely choked with hundreds of floating plastic bottles. |
| POL_110 | POL_110.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Roadside solid waste dumping pile on bare dirt soil. |
| POL_111 | POL_111.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Illegal roadside trash dumping pile along paved roadway. |
| POL_112 | POL_112.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Accumulated household garbage dumped in rural grass clearing. |
| POL_113 | POL_113.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Sandy beach shore heavily littered with plastic bottles, bags, and debris. |
| POL_114 | POL_114.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Shoreline littered with discarded foam, plastic trash, and wood waste. |
| POL_115 | POL_115.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Paper packaging and plastic trash scattered on red soil slope. |
| POL_116 | POL_116.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dense accumulation of plastic cups, wrappers, and food packaging litter. |
| POL_117 | POL_117.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Volunteers clearing massive solid waste and litter accumulation in ditch. |
| POL_118 | POL_118.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Large stack of discarded newspapers dumped on street sidewalk. |
| POL_119 | POL_119.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Hillside ravine used for illegal dumping of appliances, tires, and junk. |
| POL_120 | POL_120.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Desert open dumping site with plastic bags, discarded furniture, and rubbish. |
| POL_121 | POL_121.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Roadside illegal dumping of furniture scrap, concrete rings, and household junk. |
| POL_122 | POL_122.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dumped porcelain coffee cups and sanitaryware scrap on forest dirt path. |
| POL_123 | POL_123.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Tile rubble and demolition rubbish dumped in overgrown weeds. |
| POL_124 | POL_124.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Forest clearing filled with plastic crates, boxes, and household garbage dump. |
| POL_125 | POL_125.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Rusted metal scrap, old tires, and trash bags dumped in forest. |
| POL_126 | POL_126.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Ravine slope filled with discarded plastic sacks, packaging, and rubbish. |
| POL_127 | POL_127.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dry brush land littered with plastic bottles, bags, and solid waste. |
| POL_128 | POL_128.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Logging clearing covered with large black trash bags and household refuse piles. |
| POL_129 | POL_129.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Forest slope used as illegal dump for hundreds of discarded tires and trash. |
| POL_130 | POL_130.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Roadside open dumping of garbage, cardboard, and litter along rural road. |
| POL_131 | POL_131.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Roadside dumping site with wooden pallets, plastic bags, and rubbish. |
| POL_132 | POL_132.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Field clearing with dumped household furniture, trash bags, and junk. |
| POL_133 | POL_133.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Coastal beach field littered with white plastic foam, demolition trash, and waste. |
| POL_134 | POL_134.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dirt clearing dumped with appliance scrap, furniture, and plastic trash. |
| POL_135 | POL_135.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Hillside slope dumped with discarded computer monitors and e-waste junk. |
| POL_136 | POL_136.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Ground covered with hundreds of colorful plastic shotgun shell casings. |
| POL_137 | POL_137.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Underpass dirt patch used for dumping cardboard boxes and plastic trash. |
| POL_138 | POL_138.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Forest floor strewn with discarded tin sheets, metal scrap, and plastic waste. |
| POL_139 | POL_139.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Roadside path entrance dumped with old furniture, buckets, and trash bags. |
| POL_140 | POL_140.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Disposable face mask litter thrown on autumn leaf ground. |
| POL_141 | POL_141.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Woods floor littered with discarded plastic toys, cups, and food packaging. |
| POL_142 | POL_142.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Municipal solid waste landfill dumping site with bulldozer compacting garbage. |
| POL_143 | POL_143.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Alleyway ditch littered with food wrappers, plastic bags, and packaging trash. |
| POL_144 | POL_144.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Drainage ditch choked with plastic bottles, food packaging, and rotting waste. |
| POL_145 | POL_145.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Woods hillside used as illegal dumping site for discarded vehicle tires. |
| POL_146 | POL_146.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Paved brick walkway littered with soda cans, plastic bottles, and wrappers. |
| POL_147 | POL_147.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dirt ground strewn with discarded glass bottles, plastic waste, and rubbish. |
| POL_148 | POL_148.jpg | land_pollution | pollution | KEEP | Acid mine drainage and severely contaminated toxic chemical orange mud/soil runoff from drainage pipe. |
| POL_149 | POL_149.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Dirt patch outside fence littered with plastic food packaging wrappers. |
| POL_150 | POL_150.jpg | land_pollution | pollution | RECLASSIFY_TO_WASTE | Pasture field with plastic bags and household trash scattered across grass. |
| POL_200 | POL_200.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Curbside pile of discarded building renovation waste (drywall, timber scrap, mirror). |
| POL_201 | POL_201.jpg | construction_debris | pollution | AMBIGUOUS | Structural building collapse disaster scene with exposed rebar and emergency responder present; boundary between structural disaster and debris. |
| POL_202 | POL_202.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Dumped pile of weathered concrete blocks and bricks on overgrown dirt lot. |
| POL_203 | POL_203.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Mound of dumped demolition rubble and dirt in vacant field lot. |
| POL_204 | POL_204.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Demolition waste dumping pile consisting of concrete slabs and crushed bricks. |
| POL_205 | POL_205.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Large pile of broken clay roof tiles and red bricks dumped next to wall. |
| POL_206 | POL_206.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Fly-tipped construction waste pile explicitly labeled 'CONST DEBRIS'. |
| POL_207 | POL_207.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Curbside residential renovation waste pile dumped outside homes. |
| POL_208 | POL_208.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Remnants of burnt cardboard, plastic trash, and ashes on dirt ground. |
| POL_209 | POL_209.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Commercial building boards, plastic tarps, and metal poles dumped behind building. |
| POL_210 | POL_210.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Overgrown yard with discarded excavator bucket, mattress, trailer, and scrap metal junk. |
| POL_211 | POL_211.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Illegal curbside dumping of discarded refrigerator, wood scrap, and drywall. |
| POL_212 | POL_212.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Dirt ditch packed with plastic bags, crushed rubble, and paper waste. |
| POL_213 | POL_213.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Waterfront area filled with piles of dumped timber planks and window frames. |
| POL_214 | POL_214.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Illegal roadside dumping of black garbage bags, furniture, drywall, and boards. |
| POL_215 | POL_215.jpg | construction_debris | pollution | KEEP | Active municipal street excavation site with urban roadwork barriers and scattered foam. |
| POL_216 | POL_216.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of discarded wooden scaffolding planks dumped on sidewalk near trash bins. |
| POL_217 | POL_217.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Illegal dump containing tree branches, discarded mattress, carpet rolls, and furniture. |
| POL_218 | POL_218.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of dumped mortar, concrete rubble, and broken bricks against wall. |
| POL_219 | POL_219.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Demolished building plot strewn with broken roof tiles, concrete rubble, and litter. |
| POL_220 | POL_220.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Brush field with dumped concrete slabs, plaster scrap, and plastic bottles. |
| POL_221 | POL_221.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Close-up stack of dumped broken hollow bricks and concrete chunks. |
| POL_222 | POL_222.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of dumped broken terracotta roof tiles and concrete slabs. |
| POL_223 | POL_223.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Roadside pile of plastic sacks containing rubble and construction scrap. |
| POL_224 | POL_224.jpg | construction_debris | pollution | KEEP | Active road construction site with 'MCD MEN AT WORK' sign, sand mounds, and trenching. |
| POL_225 | POL_225.jpg | construction_debris | pollution | KEEP | Active street utility excavation with dirt mounds, traffic cones, and 'MEN AT WORK' sign. |
| POL_226 | POL_226.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of broken brick masonry rubble dumped in front of damaged building wall. |
| POL_227 | POL_227.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Large pile of tree branches mixed with rolled carpets, furniture scrap, and cardboard. |
| POL_228 | POL_228.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of discarded red clay bricks and mortar rubble stacked against building. |
| POL_229 | POL_229.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Discarded drywall panels, plaster scrap, and insulation sacks dumped outdoors. |
| POL_230 | POL_230.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Demolished wall rubble pile spilling onto outdoor sidewalk. |
| POL_231 | POL_231.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Renovation rubble (bricks, plaster) dumped outside residential doorway. |
| POL_232 | POL_232.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Stacked slate roofing tiles dumped on forest floor under trees. |
| POL_233 | POL_233.jpg | construction_debris | pollution | AMBIGUOUS | Overgrown dirt excavation mound mixed with embedded rock rubble; boundary between soil excavation mound and demolition waste. |
| POL_234 | POL_234.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Pile of discarded clay bricks and rubble dumped in dirt field. |
| POL_235 | POL_235.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Slope dumped with sacks of cement plaster, concrete rubble, and plastic waste. |
| POL_236 | POL_236.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Courtyard dumped with piles of concrete rubble and crushed bricks. |
| POL_237 | POL_237.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Yard filled with discarded wooden framing lumber, drywall scrap, and building waste. |
| POL_238 | POL_238.jpg | construction_debris | pollution | KEEP | Active sidewalk trenching and pipe utility construction site with caution tape. |
| POL_239 | POL_239.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Scrap metal pile with rebar, steel beams, and wires near concrete debris. |
| POL_240 | POL_240.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Demolition scrap yard with piled steel beams and metal frames. |
| POL_241 | POL_241.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Demolition rubble pile with rebar, concrete pieces, and discarded tires. |
| POL_242 | POL_242.jpg | construction_debris | pollution | KEEP | Active municipal road repaving work site with roadbed soil excavated and lined with stones. |
| POL_243 | POL_243.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Large mound of crushed concrete and brick demolition rubble in open field. |
| POL_244 | POL_244.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Stack of crushed concrete slabs with exposed twisted rebar mesh. |
| POL_245 | POL_245.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Sidewalk pile of white woven sacks containing plaster waste and renovation scrap. |
| POL_246 | POL_246.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Curbside street pile of cardboard, white sacks, and plaster rubble. |
| POL_247 | POL_247.jpg | construction_debris | pollution | KEEP | Active construction worker shoveling inside structural building renovation work site. |
| POL_248 | POL_248.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Mountain pile of crushed concrete and rebar demolition rubble. |
| POL_249 | POL_249.jpg | construction_debris | pollution | KEEP | Active CAT excavator operating at heavy building foundation excavation and structural demolition site. |
| POL_250 | POL_250.jpg | construction_debris | pollution | RECLASSIFY_TO_WASTE | Close-up pile of crushed concrete chunks, green painted plaster scrap, and rock rubble. |

## 4. Summary

### Overall Audit Summary (101 Target Images)

- **KEEP**: 8 (7.92%)
- **RECLASSIFY_TO_WASTE**: 91 (90.10%)
- **AMBIGUOUS**: 2 (1.98%)

### Breakdown by Subcategory

#### `land_pollution` (50 images)
- **KEEP**: 1 (2.00%)
- **RECLASSIFY_TO_WASTE**: 49 (98.00%)
- **AMBIGUOUS**: 0 (0.00%)

#### `construction_debris` (51 images)
- **KEEP**: 7 (13.73%)
- **RECLASSIFY_TO_WASTE**: 42 (82.35%)
- **AMBIGUOUS**: 2 (3.92%)

## 5. Taxonomy Findings

Visual audit of all 101 target images revealed major systematic taxonomy overlap between `pollution` and `waste`:

1. **Land Pollution Subcategory (98.00% Reclassification Rate)**:
   - `land_pollution` in the original dataset consists almost entirely of solid waste accumulation, roadside refuse piles, illegal garbage dumping, plastic litter in vegetation, and clogged plastic bottles in waterways.
   - Under the project's existing taxonomy, solid waste dumping, litter accumulation, and plastic refuse are explicitly defined under `waste` (`garbage_accumulation`, `roadside_dumping`, `litter_accumulation`, `waterway_waste`).
   - Only 1 image (`POL_148`: Acid mine drainage / chemical sludge) represents genuine non-solid chemical/soil pollution.

2. **Construction Debris Subcategory (82.35% Reclassification Rate)**:
   - 42 out of 51 `construction_debris` images depict dumped concrete rubble, fly-tipped renovation scrap (drywall, timber, bricks), drywall bags, or scrap metal piles.
   - Crucially, the original `waste` dataset contains `WST_001` explicitly titled 'Fly-tipping construction waste', establishing that dumped construction waste belongs under `waste` in the project taxonomy.
   - 7 images (`POL_215`, `POL_224`, `POL_225`, `POL_238`, `POL_242`, `POL_247`, `POL_249`) depict active urban roadwork/trenching, heavy excavation machinery (CAT excavator), or active structural building renovation workers. These represent active construction/industrial activity rather than solid waste dumping.

## 6. Relation to Experiment 03

- **OBSERVED DATASET ISSUE**: 91 out of 101 target `pollution` images (90.10%) depict conditions that visually match the project's `waste` subcategories (solid garbage accumulation, illegal dumping, plastic litter, construction demolition scrap).
- **MODEL BEHAVIOR**: In Experiment 03, the model misclassified 10/25 true `pollution` test images as `waste`. Visual inspection confirms test images like `POL_113` (plastic bottle/bag litter on beach shore) and `POL_222` (demolition roof tile pile) visually feature solid waste, causing the vision model to predict `waste`.
- **HYPOTHESIS**: The model's `pollution` -> `waste` confusion is largely driven by severe label taxonomy overlap in the dataset, where solid waste dumping was assigned to `pollution` instead of `waste` during dataset creation.

## 7. Recommended Next Experiment

Based on this audit, **Experiment 05 (Sanitized Taxonomy & Relabeling)** is strongly recommended:
1. Relabel solid waste accumulation / illegal dumping images currently in `land_pollution` and `construction_debris` to `waste` (or merge solid waste into `waste`).
2. Restrict `pollution` to genuine chemical/industrial contamination, air emissions, water pollution, and active industrial/construction activity.
3. Evaluate fine-tuned model performance on the re-sanitized dataset to verify if resolving the `pollution` ↔ `waste` taxonomy collision eliminates the dominant error boundary.

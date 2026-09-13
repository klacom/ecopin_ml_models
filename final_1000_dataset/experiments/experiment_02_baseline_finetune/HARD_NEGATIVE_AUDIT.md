# EcoPin Hard Negative Visual Audit Report

## 1. Summary

A comprehensive visual audit of all **125 `hard_negative` images** within the `non_environmental` class of the EcoPin dataset (`c:\dev\datasets\ecopin_dataset`) was performed. Every image was evaluated individually based on its primary visual subject and environmental reportable condition, following strict taxonomy guidelines.

| Decision | Count | Percentage | Description |
|---|:---:|:---:|---|
| **KEEP** | 114 | 91.20% | Clear non-environmental subject (clean infrastructure, clean bins, vehicles, natural scenery, animals, puddles, historic architecture). Waste is absent or purely incidental. |
| **RECLASSIFY_TO_WASTE** | 7 | 5.60% | The primary subject is accumulated trash, roadside dumping, overflowing waste piles, or loose litter. |
| **AMBIGUOUS** | 4 | 3.20% | Borderline cases where reasonable annotators could debate whether the primary subject is waste vs. vehicle/market/natural phenomenon. |
| **TOTAL** | **125** | **100.00%** | **All 125 hard-negative samples evaluated.** |

---

## 2. Full Audit Table

| ID | Filename | Decision | Short Visual Reason |
|---|---|:---:|---|
| `NEG_126` | `NEG_126.jpg` | `RECLASSIFY_TO_WASTE` | Open rickshaw overflowing with uncollected trash bags and loose garbage. |
| `NEG_127` | `NEG_127.jpg` | `KEEP` | Clean, structured municipal public trash bin photographed as urban street infrastructure. |
| `NEG_128` | `NEG_128.jpg` | `KEEP` | Clean public trash cans at a designated street location. |
| `NEG_129` | `NEG_129.jpg` | `KEEP` | Clean outdoor street corner trash receptacle. |
| `NEG_130` | `NEG_130.jpg` | `KEEP` | Clean park trash bin surrounded by clean forest vegetation. |
| `NEG_131` | `NEG_131.jpg` | `KEEP` | Row of clean, organized recycling and compost bins in Sonoma. |
| `NEG_132` | `NEG_132.jpg` | `KEEP` | Municipal garbage truck photographed as an operating vehicle on road. |
| `NEG_133` | `NEG_133.jpg` | `KEEP` | Clean color-coded waste sorting containers in the Philippines. |
| `NEG_134` | `NEG_134.jpg` | `KEEP` | Sanitation workers operating on a garbage truck in Leh. |
| `NEG_135` | `NEG_135.jpg` | `AMBIGUOUS` | Trash collection rickshaw carrying open drums; vehicle and waste are equally prominent. |
| `NEG_136` | `NEG_136.jpg` | `RECLASSIFY_TO_WASTE` | Pick-up truck overloaded with a high pile of junk, mattresses, and discarded household waste. |
| `NEG_137` | `NEG_137.jpg` | `KEEP` | Clean urban street pavement and brick buildings on Birkshall Street. |
| `NEG_138` | `NEG_138.jpg` | `KEEP` | Clean urban residential street in Belfast. |
| `NEG_139` | `NEG_139.jpg` | `KEEP` | Natural rock quarry landscape west of Helga Water. |
| `NEG_140` | `NEG_140.jpg` | `KEEP` | Historic stone wayside cross monument. |
| `NEG_141` | `NEG_141.jpg` | `RECLASSIFY_TO_WASTE` | Motorized tricycle loaded high with unbagged loose garbage and refuse. |
| `NEG_142` | `NEG_142.jpg` | `RECLASSIFY_TO_WASTE` | Outdoor dumpsters with overflowing trash and loose garbage scattered on surrounding ground. |
| `NEG_143` | `NEG_143.jpg` | `KEEP` | Scenic waterfront landscape. |
| `NEG_144` | `NEG_144.jpg` | `KEEP` | Clean roadside drain inlet with flowing water. |
| `NEG_145` | `NEG_145.jpg` | `KEEP` | Rural roadside ditch with natural vegetation. |
| `NEG_146` | `NEG_146.jpg` | `KEEP` | Natural river ice on the Water of Tanar. |
| `NEG_147` | `NEG_147.jpg` | `KEEP` | Clean natural stream and small waterfall at New Water. |
| `NEG_148` | `NEG_148.jpg` | `KEEP` | Natural stream landscape at Faseny Water. |
| `NEG_149` | `NEG_149.jpg` | `KEEP` | Rain puddle on forest dirt path in Altmore Forest. |
| `NEG_150` | `NEG_150.jpg` | `KEEP` | Frozen puddle on natural ground in Minnowburn. |
| `NEG_151` | `NEG_151.jpg` | `KEEP` | Rain puddle on urban street asphalt in Belfast. |
| `NEG_152` | `NEG_152.jpg` | `KEEP` | Clean rain puddles on dirt lane. |
| `NEG_153` | `NEG_153.jpg` | `KEEP` | Children playing in natural mud ground in CAR. |
| `NEG_154` | `NEG_154.jpg` | `KEEP` | Grassy coastal cliff erosion landscape at Skipsea Sands. |
| `NEG_155` | `NEG_155.jpg` | `KEEP` | Forest concrete drainage culvert. |
| `NEG_156` | `NEG_156.jpg` | `KEEP` | Puddles on a rural country lane. |
| `NEG_157` | `NEG_157.jpg` | `KEEP` | Rain puddle on a paved embankment walkway. |
| `NEG_158` | `NEG_158.jpg` | `KEEP` | Woodland dirt footpath and farm track. |
| `NEG_159` | `NEG_159.jpg` | `KEEP` | Foggy rural landscape in Benue State. |
| `NEG_160` | `NEG_160.jpg` | `KEEP` | Misty landscape with trees. |
| `NEG_161` | `NEG_161.jpg` | `AMBIGUOUS` | Truck emitting heavy dark exhaust smoke; vehicle vs. smoke pollution is ambiguous. |
| `NEG_162` | `NEG_162.jpg` | `KEEP` | Tussock wildfire smoke plume in natural environment. |
| `NEG_163` | `NEG_163.jpg` | `KEEP` | Wildfire smoke plume over Sand Creek landscape. |
| `NEG_164` | `NEG_164.jpg` | `KEEP` | Market stall featuring raw fish and food products. |
| `NEG_165` | `NEG_165.jpg` | `KEEP` | Outdoor countryside landscape. |
| `NEG_166` | `NEG_166.jpg` | `KEEP` | Conflict smoke rising from damaged buildings in town. |
| `NEG_167` | `NEG_167.jpg` | `KEEP` | Structure fire and smoke at Eastbourne Pier. |
| `NEG_168` | `NEG_168.jpg` | `AMBIGUOUS` | Green algae bloom on water surface; natural water state vs. water pollution. |
| `NEG_169` | `NEG_169.jpg` | `KEEP` | Earthquake structural rubble and emergency relief in Nepal. |
| `NEG_170` | `NEG_170.jpg` | `KEEP` | Gully soil erosion landscape. |
| `NEG_171` | `NEG_171.jpg` | `KEEP` | Rural hillside pasture and agricultural land use. |
| `NEG_172` | `NEG_172.jpg` | `KEEP` | Narrow residential street in Mambog, Binangonan. |
| `NEG_173` | `NEG_173.jpg` | `KEEP` | Landslide rockfall blocking mountain road. |
| `NEG_174` | `NEG_174.jpg` | `KEEP` | Irrigation canal with marsh vegetation. |
| `NEG_175` | `NEG_175.jpg` | `KEEP` | Scenic coastal beach landscape at Morecambe. |
| `NEG_176` | `NEG_176.jpg` | `KEEP` | Close-up portrait of person's face. |
| `NEG_177` | `NEG_177.jpg` | `KEEP` | Domestic cat outdoors on grass. |
| `NEG_178` | `NEG_178.jpg` | `KEEP` | Domestic cat resting indoors. |
| `NEG_179` | `NEG_179.jpg` | `KEEP` | Outdoor urban street scene without waste. |
| `NEG_180` | `NEG_180.jpg` | `KEEP` | Scenic landscape in Bohol, Philippines. |
| `NEG_181` | `NEG_181.jpg` | `KEEP` | Rural Philippine agricultural landscape. |
| `NEG_182` | `NEG_182.jpg` | `KEEP` | City street view in Cebu. |
| `NEG_183` | `NEG_183.jpg` | `KEEP` | Country chickens outdoors in rural setting. |
| `NEG_184` | `NEG_184.jpg` | `KEEP` | Funeral procession walking through shallow street floodwater. |
| `NEG_185` | `NEG_185.jpg` | `KEEP` | City street architecture in Bacolod. |
| `NEG_186` | `NEG_186.jpg` | `KEEP` | Outdoor rural village scene in Philippines. |
| `NEG_187` | `NEG_187.jpg` | `KEEP` | Urban street view in Cebu. |
| `NEG_188` | `NEG_188.jpg` | `KEEP` | Countryside dirt road in Philippines. |
| `NEG_189` | `NEG_189.jpg` | `KEEP` | Coastal landscape in Cebu / Bohol. |
| `NEG_190` | `NEG_190.jpg` | `KEEP` | Tropical beach scenery at Isla Reta. |
| `NEG_191` | `NEG_191.jpg` | `KEEP` | Rural village pathway in Philippines. |
| `NEG_192` | `NEG_192.jpg` | `KEEP` | Outdoor scene in Philippines. |
| `NEG_193` | `NEG_193.jpg` | `KEEP` | Countryside road with trees. |
| `NEG_194` | `NEG_194.jpg` | `KEEP` | Scenic beach scenery in Port Barton. |
| `NEG_195` | `NEG_195.jpg` | `KEEP` | Urban street environment in Cebu. |
| `NEG_196` | `NEG_196.jpg` | `KEEP` | Historic stone wall architecture in Cebu. |
| `NEG_197` | `NEG_197.jpg` | `KEEP` | Fishermen cleaning fish at coastal dock in Miputak. |
| `NEG_198` | `NEG_198.jpg` | `KEEP` | Rural house and yard. |
| `NEG_199` | `NEG_199.jpg` | `KEEP` | Pedestrians watching city street activity. |
| `NEG_200` | `NEG_200.jpg` | `KEEP` | Pedestrians crossing urban street. |
| `NEG_201` | `NEG_201.jpg` | `KEEP` | Outdoor landscape view in Philippines. |
| `NEG_202` | `NEG_202.jpg` | `KEEP` | Children playing in makeshift outdoor area. |
| `NEG_203` | `NEG_203.jpg` | `KEEP` | Sunset ocean landscape in Cebu. |
| `NEG_204` | `NEG_204.jpg` | `KEEP` | Rural landscape view. |
| `NEG_205` | `NEG_205.jpg` | `KEEP` | City street scene in Manila. |
| `NEG_206` | `NEG_206.jpg` | `KEEP` | Indoor human portrait photo. |
| `NEG_207` | `NEG_207.jpg` | `KEEP` | Outdoor urban street view. |
| `NEG_208` | `NEG_208.jpg` | `KEEP` | Residential house facade. |
| `NEG_209` | `NEG_209.jpg` | `KEEP` | Outdoor urban building view. |
| `NEG_210` | `NEG_210.jpg` | `KEEP` | Rural landscape in Philippines. |
| `NEG_211` | `NEG_211.jpg` | `KEEP` | Rural road view in Philippines. |
| `NEG_212` | `NEG_212.jpg` | `KEEP` | Village street scene. |
| `NEG_213` | `NEG_213.jpg` | `KEEP` | Village houses in Philippines. |
| `NEG_214` | `NEG_214.jpg` | `KEEP` | Rural pathway scene. |
| `NEG_215` | `NEG_215.jpg` | `KEEP` | Clean rural landscape in Philippines. |
| `NEG_216` | `NEG_216.jpg` | `KEEP` | Passengers waiting at Boracay boat dock. |
| `NEG_217` | `NEG_217.jpg` | `KEEP` | Clean coastal beach in Philippines. |
| `NEG_218` | `NEG_218.jpg` | `KEEP` | Coastal marine water view. |
| `NEG_219` | `NEG_219.jpg` | `KEEP` | City street architecture in Manila. |
| `NEG_220` | `NEG_220.jpg` | `AMBIGUOUS` | Crowded public market floor with scattered produce crates, organic debris, and litter. |
| `NEG_221` | `NEG_221.jpg` | `KEEP` | Clean city street with operating jeepney. |
| `NEG_222` | `NEG_222.jpg` | `RECLASSIFY_TO_WASTE` | Slum alley with heavy accumulated street litter, plastic waste, and uncollected refuse. |
| `NEG_223` | `NEG_223.jpg` | `KEEP` | Naval ship deck scene. |
| `NEG_224` | `NEG_224.jpg` | `KEEP` | Clean urban outdoor building facade. |
| `NEG_225` | `NEG_225.jpg` | `KEEP` | Water resort landscape at Villa Escudero. |
| `NEG_226` | `NEG_226.jpg` | `KEEP` | Traditional wooden Filipino house. |
| `NEG_227` | `NEG_227.jpg` | `KEEP` | Historic building architecture facade. |
| `NEG_228` | `NEG_228.jpg` | `KEEP` | Historic military fortification at Battery Grubbs. |
| `NEG_229` | `NEG_229.jpg` | `KEEP` | Rural village view in Talisay. |
| `NEG_230` | `NEG_230.jpg` | `KEEP` | Outdoor street portrait. |
| `NEG_231` | `NEG_231.jpg` | `KEEP` | Interior room view in Baguio. |
| `NEG_232` | `NEG_232.jpg` | `KEEP` | Outdoor mountain landscape. |
| `NEG_233` | `NEG_233.jpg` | `KEEP` | Clean passenger vehicle parked outdoors. |
| `NEG_234` | `NEG_234.jpg` | `KEEP` | Rural Philippine countryside landscape. |
| `NEG_235` | `NEG_235.jpg` | `KEEP` | Outdoor street view. |
| `NEG_236` | `NEG_236.jpg` | `KEEP` | Panoramic mountain landscape view of Baguio. |
| `NEG_237` | `NEG_237.jpg` | `KEEP` | Island beach scenery in Boracay. |
| `NEG_238` | `NEG_238.jpg` | `KEEP` | Clean outdoor courtyard. |
| `NEG_239` | `NEG_239.jpg` | `KEEP` | Outdoor residential yard view. |
| `NEG_240` | `NEG_240.jpg` | `KEEP` | Clean street view. |
| `NEG_241` | `NEG_241.jpg` | `KEEP` | Baclaran market street scene with pedestrians and clean stalls. |
| `NEG_242` | `NEG_242.jpg` | `KEEP` | Urban city street architecture in Cebu. |
| `NEG_243` | `NEG_243.jpg` | `KEEP` | Natural rock formation at Frog Rock. |
| `NEG_244` | `NEG_244.jpg` | `KEEP` | Clean outdoor facility grounds. |
| `NEG_245` | `NEG_245.jpg` | `KEEP` | Countryside field landscape. |
| `NEG_246` | `NEG_246.jpg` | `KEEP` | Clean indoor building view. |
| `NEG_247` | `NEG_247.jpg` | `KEEP` | Outdoor park trees and grass. |
| `NEG_248` | `NEG_248.jpg` | `RECLASSIFY_TO_WASTE` | Discarded plastic garbage, empty bottles, and litter scattered across ground. |
| `NEG_249` | `NEG_249.jpg` | `RECLASSIFY_TO_WASTE` | Large pile of accumulated garbage bags and dumped municipal waste. |
| `NEG_250` | `NEG_250.jpg` | `KEEP` | Historic building street in West Chester. |

---

## 3. RECLASSIFY_TO_WASTE

The following **7 images** currently labeled as `non_environmental` (`hard_negative`) have been identified through visual inspection as having **waste, garbage, litter, or dumping as their primary reportable subject**:

1. **`NEG_126` (`NEG_126.jpg`)**: Rickshaw in Dhaka packed high with uncollected, overflowing garbage bags and loose trash. The prominent visual focus of the image is the accumulated waste on the vehicle.
2. **`NEG_136` (`NEG_136.jpg`)**: A pick-up truck overflowing with a massive pile of discarded junk, old mattresses, household rubbish, and debris. The rubbish pile itself occupies over 60% of the frame.
3. **`NEG_141` (`NEG_141.jpg`)**: A Ghanaian motorized tricycle loaded with unbagged, raw garbage and refuse spilling out. The primary subject is visible environmental waste transportation/dumping.
4. **`NEG_142` (`NEG_142.jpg`)**: Outdoor residential dumpsters with overflowing garbage bags and loose trash strewn across the surrounding pavement.
5. **`NEG_222` (`NEG_222.jpg`)**: An urban slum alley in Tondo where the foreground and ground surface are covered with accumulated street litter, plastic bags, and uncollected trash.
6. **`NEG_248` (`NEG_248.jpg`)**: A close-up ground shot of discarded plastic packaging, food wrappers, and litter scattered on the floor.
7. **`NEG_249` (`NEG_249.jpg`)**: A large outdoor garbage pile featuring black trash bags, loose refuse, and dumped urban waste.

---

## 4. AMBIGUOUS

The following **4 images** represent borderline cases where reasonable annotators could debate whether the primary subject is waste vs. vehicle/infrastructure/natural state:

1. **`NEG_135` (`NEG_135.jpg`)**: A trash collection rickshaw carrying open drums containing waste. While waste is present, the image functions equally as a vehicle/sanitation portrait.
2. **`NEG_161` (`NEG_161.jpg`)**: A heavy diesel truck emitting dark exhaust smoke from its vertical pipe. Reasonable annotators may debate whether this represents mobile air pollution vs. vehicle operation.
3. **`NEG_168` (`NEG_168.jpg`)**: An algal bloom covering a water surface with thick green film. Could be classified as an environmental water issue (pollution/waste) or a natural biological water phenomenon.
4. **`NEG_220` (`NEG_220.jpg`)**: A crowded public market street in Iloilo City featuring scattered fruit crates, wet floor debris, and organic market waste mixed with foot traffic.

---

## 5. KEEP

The remaining **114 images** (91.20%) are valid `hard_negative` samples that belong in `non_environmental`:

- **Clean Municipal Infrastructure**: Public trash receptacles (`NEG_127`–`NEG_131`, `NEG_133`), sanitation vehicles operating on roads (`NEG_132`, `NEG_134`), and public transit stations (`NEG_216`, `NEG_221`, `NEG_241`).
- **Water & Drainage Features**: Clean rain puddles (`NEG_149`–`NEG_152`, `NEG_156`, `NEG_157`), roadside drainage ditches/culverts (`NEG_144`, `NEG_145`, `NEG_155`), natural stream waterfalls (`NEG_147`, `NEG_148`), and frozen river ice (`NEG_146`).
- **Natural Hazards & Disasters**: Wildfire smoke plumes (`NEG_162`, `NEG_163`), soil erosion (`NEG_170`), earthquake structural damage (`NEG_169`), and mountain landslides (`NEG_173`).
- **General Non-Environmental Content**: Domestic pets (`NEG_177`, `NEG_178`), historic architecture/monuments (`NEG_140`, `NEG_196`, `NEG_228`), human portraits (`NEG_176`, `NEG_206`), and clean urban/rural landscapes (`NEG_175`, `NEG_180`–`NEG_219`, `NEG_223`–`NEG_250`).

---

## 6. Taxonomy Observations

The visual audit confirms that the current `non_environmental` hard-negative dataset contains **genuine, direct class-definition overlap with `waste`**. 

Specifically, 7 images depict obvious accumulated garbage, roadside rubbish, or overflowing waste, while 4 additional images present borderline waste/pollution conditions. Because these images were placed inside `non_environmental`, fine-tuned neural networks (such as EfficientNet-B0 in Experiment 02) learn feature weights that assign high probability to `non_environmental` when presented with urban street backgrounds containing waste and garbage containers.

---

## Verification & Confirmation

- **Total Images Audited**: 125 / 125
- **Clear Label Overlap (RECLASSIFY_TO_WASTE)**: 7 (5.60%)
- **Ambiguous Cases**: 4 (3.20%)
- **Valid Hard Negatives (KEEP)**: 114 (91.20%)
- **Confirmation**: NO dataset images, metadata CSV/XLSX, splits, models, checkpoints, or inference code files were modified. This report is an audit only.

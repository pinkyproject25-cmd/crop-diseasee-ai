# Stage 2 knowledge prototype

Status: all 26 of Candidate-v2's diseased labels have source-linked English
entries. This is experimental editorial content, not production agricultural
approval. Source review was performed on 2026-10-07; regional review by a
qualified agricultural adviser remains required.

The backend selects an entry by the exact classifier label only after an
accepted diseased classification. Unknown, blurry, and healthy results receive
no disease-specific knowledge. Entries describe *typical symptoms to check*
and general next steps, not observations extracted from the uploaded image.
The UI links directly to the reference for every populated entry.

## Covered exact labels

| Exact label | Authoritative reference |
| --- | --- |
| `Apple___Apple_scab` | [UC IPM](https://ipm.ucanr.edu/agriculture/apple/apple-scab/) |
| `Apple___Black_rot` | [University of New Hampshire Extension](https://extension.unh.edu/resource/frogeye-leaf-spot-black-rot-apple-0) |
| `Apple___Cedar_apple_rust` | [University of Minnesota Extension](https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/cedar-apple-rust) |
| `Cherry_(including_sour)___Powdery_mildew` | [Utah State University Extension](https://extension.usu.edu/planthealth/research/cherry-powdery-mildew-in-utah) |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | [Purdue Extension](https://www.extension.purdue.edu/extmedia/BP/BP-56.html) |
| `Corn_(maize)___Common_rust_` | [Purdue Extension](https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf) |
| `Corn_(maize)___Northern_Leaf_Blight` | [Purdue Extension](https://www.extension.purdue.edu/extmedia/BP/BP-84-W.pdf) |
| `Grape___Black_rot` | [Cornell CALS](https://cals.cornell.edu/news/2014/03/grapes-101-managing-black-rot) |
| `Grape___Esca_(Black_Measles)` | [UC IPM](https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/) |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | [University of Kentucky](https://fruitscout.mgcafe.uky.edu/leaf-spots-grapes) |
| `Orange___Haunglongbing_(Citrus_greening)` | [USDA APHIS](https://www.aphis.usda.gov/plant-pests-diseases/citrus-diseases/citrus-greening-and-asian-citrus-psyllid) |
| `Peach___Bacterial_spot` | [University of Georgia CAES](https://peaches.caes.uga.edu/research/diseases.html) |
| `Pepper,_bell___Bacterial_spot` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/bacterial-spot-of-tomato-and-pepper) |
| `Potato___Early_blight` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato) |
| `Potato___Late_blight` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight) |
| `Squash___Powdery_mildew` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/powdery-mildew-of-cucurbits) |
| `Strawberry___Leaf_scorch` | [University of Minnesota Extension](https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/growing-strawberries-in-the-home-garden) |
| `Tomato___Bacterial_spot` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/bacterial-spot-of-tomato-and-pepper) |
| `Tomato___Early_blight` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato) |
| `Tomato___Late_blight` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight) |
| `Tomato___Leaf_Mold` | [University of Minnesota Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold) |
| `Tomato___Septoria_leaf_spot` | [University of Maryland Extension](https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes) |
| `Tomato___Spider_mites Two-spotted_spider_mite` | [University of Minnesota Extension](https://extension.umn.edu/garden-and-home/yard-and-garden/yard-and-garden-insects/spider-mites) |
| `Tomato___Target_Spot` | [UF/IFAS Extension](https://edis.ifas.ufl.edu/publication/PP351) |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | [UF/IFAS Extension](https://ask.ifas.ufl.edu/publication/IN1430) |
| `Tomato___Tomato_mosaic_virus` | [UF/IFAS Extension](https://blogs.ifas.ufl.edu/stlucieco/2023/03/03/tomato-mosaic-virus-tomv-and-its-management/) |

The recommendations intentionally avoid pesticide product, dose, and schedule
instructions. References from US institutions are evidence sources, not
location-specific prescriptions for India or other regions. A local adviser
must interpret actions for the user's crop, regulations, and season.

Healthy and Unknown reports must continue to receive no disease-specific
knowledge. Translation and speech require separate provider credentials and
end-to-end validation. Affected area, severity, and health score remain wholly
separate from this catalogue.

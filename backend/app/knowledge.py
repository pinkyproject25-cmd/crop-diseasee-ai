"""Source-linked general disease facts; never observations from the uploaded photo.

This reviewed catalogue is an editorial prototype. Recommendations are broad
non-chemical checks, not prescriptions or region-specific treatment schedules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeEntry:
    typical_symptoms: tuple[str, ...]
    causes: tuple[str, ...]
    recommendations: tuple[str, ...]
    source_title: str
    source_url: str


ENTRIES: dict[str, KnowledgeEntry] = {
    "Apple___Apple_scab": KnowledgeEntry(
        typical_symptoms=(
            "Check for velvety olive-to-black spots on leaves; fruit may develop scab-like spots.",
        ),
        causes=(
            "Apple scab is caused by Venturia inaequalis; fallen infected leaves and prolonged wet conditions can contribute to its spread.",
        ),
        recommendations=(
            "Inspect other leaves and fruit for similar signs, and seek local confirmation before making treatment decisions.",
            "Ask a local agricultural adviser about orchard sanitation and prevention suited to your region and season.",
        ),
        source_title="UC IPM: Apple Scab",
        source_url="https://ipm.ucanr.edu/agriculture/apple/apple-scab/",
    ),
    "Apple___Black_rot": KnowledgeEntry(
        typical_symptoms=(
            "Check for small circular brown leaf spots with darker concentric rings; older spots may contain tiny black fruiting bodies.",
            "Fruit lesions can darken, form concentric rings, and eventually mummify, but a leaf photo cannot establish fruit infection.",
        ),
        causes=(
            "Apple black rot and frogeye leaf spot are caused by the fungus Diplodia seriata (also reported as Botryosphaeria obtusa), which can persist in cankers, fallen leaves, and mummified fruit.",
        ),
        recommendations=(
            "Inspect leaves, fruit, and branches for related signs and obtain local confirmation because other disorders can resemble these spots.",
            "Ask a local orchard adviser about removing mummified fruit and dead or cankered wood, and about sanitation appropriate for the season.",
        ),
        source_title="University of New Hampshire Extension: Frogeye leaf spot and black rot on apple",
        source_url="https://extension.unh.edu/resource/frogeye-leaf-spot-black-rot-apple-0",
    ),
    "Apple___Cedar_apple_rust": KnowledgeEntry(
        typical_symptoms=(
            "Check for yellow leaf spots that become bright orange-red, often with a red border and small raised black dots in mature spots.",
            "Short finger-like fungal tubes can develop on the lower leaf surface directly beneath a spot.",
        ),
        causes=(
            "Cedar-apple rust is caused by a Gymnosporangium rust fungus that requires both apple-family hosts and juniper or red cedar hosts to complete its life cycle; wet spring weather supports spore release and infection.",
        ),
        recommendations=(
            "Inspect both leaf surfaces and nearby apple leaves; seek local confirmation because several rust diseases produce similar signs.",
            "Ask a local orchard adviser about resistant varieties and management of nearby alternate hosts that is suitable for your location.",
        ),
        source_title="University of Minnesota Extension: Cedar-apple rust and related rust diseases",
        source_url="https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/cedar-apple-rust",
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": KnowledgeEntry(
        typical_symptoms=(
            "Check mature leaves for long, narrow, rectangular tan-to-brown lesions whose parallel sides are limited by leaf veins.",
            "Early spots can look dark and water-soaked with a narrow yellow halo; mature lesions may become grayish.",
        ),
        causes=(
            "Gray leaf spot is caused by Cercospora fungi; infected corn residue can supply spores, and prolonged warm, humid leaf-wetness periods favor disease development.",
        ),
        recommendations=(
            "Inspect several leaves and request local confirmation because gray leaf spot can resemble other corn leaf blights.",
            "Discuss resistant hybrids, crop rotation, residue management, and field-specific risk with a local crop adviser.",
        ),
        source_title="Purdue Extension: Gray Leaf Spot",
        source_url="https://www.extension.purdue.edu/extmedia/BP/BP-56.html",
    ),
    "Corn_(maize)___Common_rust_": KnowledgeEntry(
        typical_symptoms=(
            "Check both leaf surfaces for scattered, elongated pustules that are brown to reddish-brown.",
            "Rust-colored spores may rub from a pustule, but mixed rust infections and other leaf diseases can complicate visual diagnosis.",
        ),
        causes=(
            "Common rust is caused by Puccinia sorghi; windborne spores infect living corn, and cool conditions with leaf wetness or dew favor infection.",
        ),
        recommendations=(
            "Inspect both sides of multiple leaves and seek local confirmation to distinguish common rust from southern rust and other lesions.",
            "Ask a local crop adviser whether hybrid resistance, crop stage, and current regional disease pressure require any action.",
        ),
        source_title="Purdue Extension: Common and Southern Rusts",
        source_url="https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": KnowledgeEntry(
        typical_symptoms=(
            "Check for long, oblong or cigar-shaped tan-to-gray lesions; early lesions can be narrow and run parallel to the leaf margin.",
            "Under high humidity, lesions may look dark or dirty as olive-green to black fungal spores develop.",
        ),
        causes=(
            "Northern corn leaf blight is caused by Exserohilum turcicum; the fungus can survive in infected corn residue and spreads by splashing or windborne spores during moderate, wet, humid weather.",
        ),
        recommendations=(
            "Inspect several leaves and obtain local confirmation because other foliar diseases and bacterial wilts can produce similar lesions.",
            "Discuss resistant hybrids, crop residue, rotation, crop stage, and current weather risk with a local crop adviser.",
        ),
        source_title="Purdue Extension: Northern Corn Leaf Blight",
        source_url="https://www.extension.purdue.edu/extmedia/BP/BP-84-W.pdf",
    ),
    "Cherry_(including_sour)___Powdery_mildew": KnowledgeEntry(
        typical_symptoms=(
            "Check young leaves for small white powdery patches, often beginning on the underside, and for lighter green areas above the infection.",
            "More advanced signs can include curled, puckered, brittle, bronzed, or dead leaf tissue.",
        ),
        causes=(
            "Cherry powdery mildew is caused by the fungus Podosphaera cerasi; susceptible young tissue and a dense canopy can support disease development.",
        ),
        recommendations=(
            "Inspect both sides of young leaves with good light and seek local confirmation because early infections can be difficult to see.",
            "Ask a local orchard adviser about canopy airflow, monitoring, sanitation, and region-appropriate prevention.",
        ),
        source_title="Utah State University Extension: Cherry Powdery Mildew in Utah",
        source_url="https://extension.usu.edu/planthealth/research/cherry-powdery-mildew-in-utah",
    ),
    "Grape___Black_rot": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for circular tan spots with darker edges and tiny black dots within the spots.",
            "Affected berries can darken and shrivel into hard mummified fruit.",
        ),
        causes=(
            "Grape black rot is a fungal disease; infected mummified fruit and cane lesions can carry inoculum into the next season.",
        ),
        recommendations=(
            "Inspect nearby leaves and clusters; a leaf photo alone cannot establish whether fruit is infected.",
            "Discuss removal of infected fruit and dormant-season sanitation with a local grape adviser.",
        ),
        source_title="Cornell CALS: Managing Black Rot",
        source_url="https://cals.cornell.edu/news/2014/03/grapes-101-managing-black-rot",
    ),
    "Grape___Esca_(Black_Measles)": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for interveinal striping that begins red in red cultivars or yellow in white cultivars, then dries and becomes dead tissue.",
            "Fruit can develop small round dark spots with brown-purple borders, but a leaf photo cannot establish fruit or internal wood symptoms.",
        ),
        causes=(
            "Esca is a grapevine trunk-disease complex caused by wood-infecting fungi; spores can be released from diseased wood during wet periods and enter pruning wounds.",
        ),
        recommendations=(
            "Inspect the distribution of symptoms across shoots and seek expert confirmation because several trunk diseases can look similar.",
            "Ask a local grape adviser about pruning-wound protection, sanitation, and management suited to the vineyard and season.",
        ),
        source_title="UC IPM: Esca (Black Measles)",
        source_url="https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": KnowledgeEntry(
        typical_symptoms=(
            "Check multiple leaves for fungal leaf spots and compare them with other grape leaf-spot diseases; image appearance alone may not distinguish the causal fungus.",
        ),
        causes=(
            "Isariopsis or Pseudocercospora leaf spot is one of several fungal leaf-spot diseases of grape.",
        ),
        recommendations=(
            "Seek local diagnostic confirmation before acting because Cercospora, Septoria, Isariopsis or Pseudocercospora, and Pestalotia leaf spots can be confused.",
            "Promote vine vigor and airflow that helps foliage dry, and discuss local vineyard management with a grape adviser.",
        ),
        source_title="University of Kentucky: Leaf Spots of Grapes",
        source_url="https://fruitscout.mgcafe.uky.edu/leaf-spots-grapes",
    ),
    "Orange___Haunglongbing_(Citrus_greening)": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for asymmetrical blotchy mottling and the tree for twig dieback or premature fruit drop.",
            "Affected fruit can be smaller, irregularly shaped, poorly colored, and bitter, but a leaf photo cannot verify fruit symptoms.",
        ),
        causes=(
            "Huanglongbing (citrus greening) is a bacterial citrus disease that is spread by infected Asian citrus psyllids.",
        ),
        recommendations=(
            "Treat this image result only as a screening signal and contact the appropriate local plant-health or agricultural authority for confirmation.",
            "Avoid moving suspect citrus plants or propagation material until you have checked the quarantine and movement guidance for your region.",
        ),
        source_title="USDA APHIS: Citrus Greening and Asian Citrus Psyllid",
        source_url="https://www.aphis.usda.gov/plant-pests-diseases/citrus-diseases/citrus-greening-and-asian-citrus-psyllid",
    ),
    "Pepper,_bell___Bacterial_spot": KnowledgeEntry(
        typical_symptoms=(
            "Check pepper leaves for small circular brown spots; unlike tomato bacterial-spot lesions, they often lack a yellow halo and their centers usually remain intact.",
            "Fruit may develop slightly raised brown scabby spots, often near the stem end, but a leaf photo cannot verify fruit symptoms.",
        ),
        causes=(
            "Bacterial spot of pepper is caused by closely related Xanthomonas bacteria; infected seed or transplants can introduce it, and splashing water, hands, and tools can spread it.",
        ),
        recommendations=(
            "Inspect several plants and seek local confirmation because other problems can produce similar leaf spots.",
            "Keep foliage dry where practical, avoid handling wet plants, and ask a local adviser about sanitation, rotation, and resistant varieties.",
        ),
        source_title="University of Minnesota Extension: Bacterial spot of tomato and pepper",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/bacterial-spot-of-tomato-and-pepper",
    ),
    "Peach___Bacterial_spot": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for water-soaked spots that darken and become dead, often with angular margins; affected leaves can yellow.",
            "Fruit can also develop angular spots, but a leaf photo cannot establish fruit or twig infection.",
        ),
        causes=(
            "Peach bacterial spot is caused by Xanthomonas bacteria and can affect foliage, twigs, and fruit.",
        ),
        recommendations=(
            "Inspect several leaves, fruit, and twigs and seek local confirmation because other disorders can cause similar spots.",
            "Ask a local orchard adviser about resistant cultivars, sanitation, and weather-related risk appropriate for your region.",
        ),
        source_title="University of Georgia CAES: Peach Diseases — Bacterial Spot",
        source_url="https://peaches.caes.uga.edu/research/diseases.html",
    ),
    "Potato___Early_blight": KnowledgeEntry(
        typical_symptoms=(
            "Check older lower leaves for round brown spots with target-like concentric rings and yellowing around the spots.",
            "Severely affected leaves can turn brown and fall, while stems may develop dry brown lesions with darker rings.",
        ),
        causes=(
            "Early blight is caused by Alternaria fungi that can persist in infected debris and soil; wet weather, heavy dew, or high humidity supports infection and spread.",
        ),
        recommendations=(
            "Inspect the lower canopy and seek local confirmation because early blight can resemble other potato leaf damage.",
            "Ask a local crop adviser about scouting, rotation, sanitation, airflow, and irrigation practices suited to your field.",
        ),
        source_title="University of Minnesota Extension: Early blight in tomato and potato",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato",
    ),
    "Potato___Late_blight": KnowledgeEntry(
        typical_symptoms=(
            "Check for large dark-brown leaf blotches with green-gray edges and for dark stem lesions.",
            "Whitish growth can appear on infected tissue during humid weather.",
        ),
        causes=(
            "Late blight is caused by the water mold Phytophthora infestans and can spread rapidly in cool, damp conditions.",
        ),
        recommendations=(
            "Inspect the crop promptly and seek confirmation from a local agricultural adviser; similar leaf damage can have other causes.",
            "Ask a local adviser about scouting and management of infected plants to limit spread.",
        ),
        source_title="University of Minnesota Extension: Late blight of tomato and potato",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight",
    ),
    "Squash___Powdery_mildew": KnowledgeEntry(
        typical_symptoms=(
            "Check older leaves first for white, powdery fungal growth that can expand across leaf surfaces and vines.",
            "Heavy infection can lead to leaf decline; squash fruit is not directly infected by powdery mildew.",
        ),
        causes=(
            "Cucurbit powdery mildew is mainly caused by Podosphaera xanthii; humid conditions favor infection and wind spreads spores to other leaves.",
        ),
        recommendations=(
            "Inspect mature leaves across the plant and seek local confirmation because natural leaf markings and other mildews can look similar.",
            "Ask a local adviser about resistant varieties, spacing, airflow, and a scouting plan appropriate for your crop and season.",
        ),
        source_title="University of Minnesota Extension: Powdery mildew of cucurbits",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/powdery-mildew-of-cucurbits",
    ),
    "Strawberry___Leaf_scorch": KnowledgeEntry(
        typical_symptoms=(
            "Check upper leaf surfaces for dark purple angular-to-round spots that remain reddish-purple instead of developing pale centers.",
            "As infection advances, surrounding tissue can turn red or purple and dry tan areas may make the leaf curl upward and appear scorched.",
        ),
        causes=(
            "Strawberry leaf scorch is caused by Diplocarpon earlianum; it can persist in infected leaves and debris, and spores spread in moisture by wind or splashing water.",
        ),
        recommendations=(
            "Inspect multiple leaves and seek local confirmation because strawberry leaf spot and leaf blight can appear similar.",
            "Ask a local adviser about clean planting material, removal of old infected foliage, airflow, and watering practices suited to the crop.",
        ),
        source_title="University of Minnesota Extension: Growing strawberries — Leaf scorch",
        source_url="https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/growing-strawberries-in-the-home-garden",
    ),
    "Tomato___Bacterial_spot": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for small circular brown spots with yellow halos; the centers can fall out and leave small holes.",
            "Unlike early blight, bacterial-spot leaf lesions generally do not have concentric rings.",
        ),
        causes=(
            "Bacterial spot of tomato is caused by closely related Xanthomonas bacteria; infected seed or transplants can introduce it, and splashing water, hands, and tools can spread it.",
        ),
        recommendations=(
            "Inspect several plants and seek local confirmation because bacterial spot can resemble other tomato diseases.",
            "Keep foliage dry where practical, avoid handling wet plants, and ask a local adviser about sanitation, rotation, and resistant varieties.",
        ),
        source_title="University of Minnesota Extension: Bacterial spot of tomato and pepper",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/bacterial-spot-of-tomato-and-pepper",
    ),
    "Tomato___Early_blight": KnowledgeEntry(
        typical_symptoms=(
            "Check older lower leaves for round brown spots with target-like concentric rings and yellow tissue around the spots.",
            "Severely affected leaves can turn brown and fall; stems may develop dry brown lesions with darker rings.",
        ),
        causes=(
            "Early blight is caused by Alternaria fungi that can persist in infected debris and soil; wet weather, heavy dew, or high humidity supports infection and spread.",
        ),
        recommendations=(
            "Inspect the lower canopy and seek local confirmation because early blight can resemble Septoria leaf spot and other damage.",
            "Ask a local adviser about resistant varieties, keeping leaves dry, airflow, sanitation, and rotation suited to your crop.",
        ),
        source_title="University of Minnesota Extension: Early blight in tomato and potato",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato",
    ),
    "Tomato___Late_blight": KnowledgeEntry(
        typical_symptoms=(
            "Check leaves for rapidly enlarging dark-brown blotches, often with a green-gray edge, and stems for firm dark-brown lesions.",
            "Thin white growth can appear on infected tissue in high humidity; fruit may develop firm dark-brown spots, but a leaf photo cannot verify fruit symptoms.",
        ),
        causes=(
            "Late blight is caused by the water mold Phytophthora infestans and is favored by cool, damp conditions; infected planting material and windborne spores can introduce it.",
        ),
        recommendations=(
            "Inspect the crop promptly and seek local confirmation because late blight can spread quickly and similar leaf damage has other causes.",
            "Ask a local agricultural adviser about current regional outbreaks and appropriate steps for isolating or managing suspect plants.",
        ),
        source_title="University of Minnesota Extension: Late blight of tomato and potato",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight",
    ),
    "Tomato___Leaf_Mold": KnowledgeEntry(
        typical_symptoms=(
            "Check older leaves for pale green-to-yellow upper-surface spots with corresponding olive-green to brown velvety growth underneath.",
            "Spots can merge, turn brown, and cause leaves to wither while remaining attached.",
        ),
        causes=(
            "Tomato leaf mold is caused by Passalora fulva and is strongly favored by high relative humidity, especially in greenhouses and high tunnels.",
        ),
        recommendations=(
            "Inspect both sides of older leaves and seek local confirmation because other tomato problems can produce yellow or brown patches.",
            "Reduce prolonged humidity and leaf wetness where practical, and ask a local adviser about airflow, sanitation, and resistant cultivars.",
        ),
        source_title="University of Minnesota Extension: Tomato leaf mold",
        source_url="https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold",
    ),
    "Tomato___Septoria_leaf_spot": KnowledgeEntry(
        typical_symptoms=(
            "Check lower leaves for small circular gray spots with dark borders; older spots may contain tiny black fungal fruiting bodies.",
            "Spots can merge as leaves yellow and die, especially during wet weather.",
        ),
        causes=(
            "Septoria leaf spot is a fungal tomato disease that is favored by wet conditions and commonly begins on lower leaves.",
        ),
        recommendations=(
            "Inspect lower leaves across the plant and seek local confirmation because Septoria can resemble early blight and bacterial spot.",
            "Keep foliage dry where practical and ask a local adviser about spacing, mulching, sanitation, and removal of infected material.",
        ),
        source_title="University of Maryland Extension: Septoria Leaf Spot of Tomatoes",
        source_url="https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": KnowledgeEntry(
        typical_symptoms=(
            "Check leaf undersides for very small yellow-orange mites with two dark spots and for fine webbing when infestations are heavy.",
            "Feeding produces tiny white or yellow stippling; severe damage can make leaves look bronzed or bleached and may cause leaf drop.",
        ),
        causes=(
            "Two-spotted spider-mite damage is caused by Tetranychus urticae feeding on plant cells, and populations can increase quickly in hot, dry conditions.",
        ),
        recommendations=(
            "Use a hand lens to inspect leaf undersides and seek local confirmation because drought stress and other pests can look similar.",
            "Support plant health and natural enemies, and ask a local adviser about locally appropriate mite management before using any treatment.",
        ),
        source_title="University of Minnesota Extension: Twospotted spider mites",
        source_url="https://extension.umn.edu/garden-and-home/yard-and-garden/yard-and-garden-insects/spider-mites",
    ),
    "Tomato___Target_Spot": KnowledgeEntry(
        typical_symptoms=(
            "Check for small dark-brown leaf spots that enlarge to light-brown or gray centers with darker outer rings, sometimes with yellowing.",
            "Lesions can merge and cause blighting and early leaf drop; similar spots can be caused by bacterial spot, early blight, or gray leaf spot.",
        ),
        causes=(
            "Tomato target spot is caused by the fungus Corynespora cassiicola; wind and rain spread spores, while high humidity and prolonged leaf wetness favor infection.",
        ),
        recommendations=(
            "Inspect the inner canopy and seek diagnostic confirmation because target spot is readily confused with other tomato leaf diseases.",
            "Ask a local adviser about canopy airflow, sanitation, crop monitoring, and region-appropriate management.",
        ),
        source_title="UF/IFAS Extension: Target Spot of Tomato in Florida",
        source_url="https://edis.ifas.ufl.edu/publication/PP351",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": KnowledgeEntry(
        typical_symptoms=(
            "Check for stunted growth, reduced leaf size, and bright yellow leaf margins across the plant.",
            "A photograph may suggest a viral pattern, but laboratory testing is needed for definitive virus identification.",
        ),
        causes=(
            "Tomato yellow leaf curl virus is transmitted persistently by Bemisia tabaci whiteflies; early infection can have a greater effect on yield.",
        ),
        recommendations=(
            "Inspect for whiteflies and similar symptoms on other plants, and seek local laboratory or agricultural confirmation.",
            "Use virus-free transplants and ask a local adviser about tolerant varieties, sanitation, and whitefly management suitable for your region.",
        ),
        source_title="UF/IFAS Extension: Managing whitefly and tomato yellow leaf curl virus",
        source_url="https://ask.ifas.ufl.edu/publication/IN1430",
    ),
    "Tomato___Tomato_mosaic_virus": KnowledgeEntry(
        typical_symptoms=(
            "Check for dark-and-light green leaf mottling, yellowing or stunting, and twisted, deformed, or unusually small leaves.",
            "Visual symptoms can overlap with other viruses and stresses; only laboratory testing can definitively identify a tomato virus.",
        ),
        causes=(
            "Tomato mosaic virus can be introduced in infected seed or plant material and spread mechanically on hands, clothing, and tools during plant handling.",
        ),
        recommendations=(
            "Avoid handling healthy plants after suspect plants, clean tools, and seek laboratory or agricultural confirmation before deciding on removal.",
            "Ask a local adviser about resistant varieties, clean transplants, sanitation, and disposal rules appropriate for your region.",
        ),
        source_title="UF/IFAS Extension: Tomato Mosaic Virus and its Management",
        source_url="https://blogs.ifas.ufl.edu/stlucieco/2023/03/03/tomato-mosaic-virus-tomv-and-its-management/",
    ),
}


def get_knowledge(label: str) -> KnowledgeEntry | None:
    return ENTRIES.get(label)

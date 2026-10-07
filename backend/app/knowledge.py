"""Source-linked general disease facts; never observations from the uploaded photo.

This limited catalogue is an editorial prototype. Recommendations are broad
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
}


def get_knowledge(label: str) -> KnowledgeEntry | None:
    return ENTRIES.get(label)

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedLabel:
    crop: str
    condition: str
    disease: str | None
    state: str


_CROP_NAMES = {
    "Cherry_(including_sour)": "Cherry (including sour)",
    "Corn_(maize)": "Corn (maize)",
    "Pepper,_bell": "Bell pepper",
}

_DISEASE_NAMES = {
    "Cercospora_leaf_spot Gray_leaf_spot": "Cercospora leaf spot / Gray leaf spot",
    "Common_rust_": "Common rust",
    "Esca_(Black_Measles)": "Esca (Black Measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis Leaf Spot)",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (Citrus greening)",
    "Spider_mites Two-spotted_spider_mite": "Spider mites / Two-spotted spider mite",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato_mosaic_virus": "Tomato mosaic virus",
}


def _readable(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def parse_combined_label(label: str) -> ParsedLabel:
    parts = label.split("___")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid combined crop-condition label: {label!r}")
    crop_key, condition_key = parts
    crop = _CROP_NAMES.get(crop_key, _readable(crop_key))
    if condition_key.lower() == "healthy":
        return ParsedLabel(crop=crop, condition="Healthy", disease=None, state="healthy")
    disease = _DISEASE_NAMES.get(condition_key, _readable(condition_key))
    return ParsedLabel(crop=crop, condition="Diseased", disease=disease, state="diseased")

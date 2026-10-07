"""Experimental discoloration estimate for one leaf on a plain light background.

This is not a trained lesion mask or a disease diagnosis.
"""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class VisualEstimate:
    affected_percent: float
    severity: str
    visible_health_percent: float
    observations: list[str]


def estimate_visible_area(image_bytes: bytes) -> VisualEstimate | None:
    """Return None when the leaf or background cannot be isolated reliably."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((256, 256))
    except (OSError, ValueError):
        return None

    pixels = np.asarray(image.filter(ImageFilter.MedianFilter(3)), dtype=np.float32) / 255.0
    border = np.concatenate((
        pixels[:12].reshape(-1, 3),
        pixels[-12:].reshape(-1, 3),
        pixels[12:-12, :12].reshape(-1, 3),
        pixels[12:-12, -12:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    border_distance = np.sqrt(np.sum((border - background) ** 2, axis=1))

    # Accept only a mostly uniform, light, neutral backdrop.
    if background.min() < .72 or background.max() - background.min() > .18:
        return None
    if float(np.mean(border_distance < .12)) < .92:
        return None

    foreground = np.sqrt(np.sum((pixels - background) ** 2, axis=2)) > .22
    foreground = np.asarray(
        Image.fromarray((foreground * 255).astype("uint8"))
        .filter(ImageFilter.MedianFilter(5))
    ) > 0

    h, w = foreground.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for y, x in zip(*np.where(foreground)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(int(y), int(x))]
        component = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in (
                (cy - 1, cx), (cy + 1, cx),
                (cy, cx - 1), (cy, cx + 1),
            ):
                if 0 <= ny < h and 0 <= nx < w and foreground[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)

    if not components:
        return None

    biggest = max(components, key=len)
    total = int(foreground.sum())

    # Reject clutter, cropped leaves, and very small objects.
    if not .10 <= len(biggest) / (h * w) <= .70:
        return None
    if len(biggest) / total < .90:
        return None

    leaf = np.zeros((h, w), dtype=bool)
    yy, xx = np.array(biggest).T
    leaf[yy, xx] = True

    if leaf[:3].any() or leaf[-3:].any() or leaf[:, :3].any() or leaf[:, -3:].any():
        return None
    if leaf[32:-32, 32:-32].sum() < len(biggest) * .55:
        return None

    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    green = (g > r * 1.12) & (g > b * 1.08) & (g > .19)
    if (green & leaf).sum() < len(biggest) * .30:
        return None

    # Exclude the leaf outline, where background blending resembles damage.
    interior = np.asarray(
        Image.fromarray((leaf * 255).astype("uint8"))
        .filter(ImageFilter.MinFilter(5))
    ) > 0
    if interior.sum() < len(biggest) * .75:
        return None

    yellow_brown = (r >= g * .96) & (g > b * 1.16) & (b < .67)
    dark_spot = (np.maximum.reduce((r, g, b)) < .42) & ~green
    discoloured = interior & (yellow_brown | dark_spot)
    proportion = float(discoloured.sum() / interior.sum())

    # Abstain on tiny colour differences or almost entirely discoloured objects.
    if not .01 <= proportion <= .60:
        return None

    percent = round(proportion * 100, 1)
    severity = "Low" if percent < 10 else "Medium" if percent < 30 else "High"

    return VisualEstimate(
        affected_percent=percent,
        severity=severity,
        visible_health_percent=round(100 - percent, 1),
        observations=[
            "Yellow-brown or dark discoloration is visible within the isolated leaf region."
        ],
    )

import unittest
from io import BytesIO

from PIL import Image, ImageDraw

from app.visual_area import estimate_visible_area


def photo(background="white", lesion=True):
    image = Image.new("RGB", (512, 512), background)
    draw = ImageDraw.Draw(image)
    draw.ellipse((90, 70, 420, 450), fill=(42, 134, 49))
    if lesion:
        draw.ellipse((210, 190, 300, 270), fill=(133, 76, 32))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class VisualAreaTests(unittest.TestCase):
    def test_visible_patch(self):
        result = estimate_visible_area(photo())
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.affected_percent, 6.1, delta=1)
        self.assertEqual(result.severity, "Low")
        self.assertAlmostEqual(
            result.affected_percent + result.visible_health_percent, 100
        )

    def test_no_discoloration_abstains(self):
        self.assertIsNone(estimate_visible_area(photo(lesion=False)))

    def test_dark_background_abstains(self):
        self.assertIsNone(estimate_visible_area(photo(background="#204522")))

    def test_invalid_image_abstains(self):
        self.assertIsNone(estimate_visible_area(b"not an image"))

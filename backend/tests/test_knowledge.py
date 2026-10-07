"""Run with: python -m unittest discover -s backend/tests -p test_knowledge.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.knowledge import ENTRIES, get_knowledge


class KnowledgeTests(unittest.TestCase):
    def test_only_reviewed_exact_labels_return_content(self):
        self.assertEqual(set(ENTRIES), {
            "Apple___Apple_scab",
            "Grape___Black_rot",
            "Potato___Late_blight",
        })
        self.assertIsNone(get_knowledge("Apple___healthy"))
        self.assertIsNone(get_knowledge("Tomato___Late_blight"))
        self.assertIsNone(get_knowledge("Apple___Apple_scab_other"))

    def test_every_entry_has_traceable_general_content(self):
        for label, entry in ENTRIES.items():
            with self.subTest(label=label):
                self.assertTrue(entry.typical_symptoms)
                self.assertTrue(entry.causes)
                self.assertTrue(entry.recommendations)
                self.assertTrue(entry.source_url.startswith("https://"))


if __name__ == "__main__":
    unittest.main()

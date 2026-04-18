import sys
import types
import unittest


# Minimal stubs so module import does not require external runtime deps.
if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.float32 = float
    np_stub.zeros = lambda shape, dtype=None: None
    sys.modules["numpy"] = np_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.from_numpy = lambda arr: arr
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub

from pccs_color_nodes.pccs_nodes import color_to_hsv, find_first_color_token


class TestPCCSLogic(unittest.TestCase):
    def test_pale_purplish_red_hsv_anchor(self):
        color = find_first_color_token("{p:1:pR(紫みの赤)}")
        hue, sat, val = color_to_hsv(color)
        self.assertEqual(hue, 345.0)
        self.assertEqual(sat, 0.35)
        self.assertEqual(val, 0.98)

    def test_red_anchor(self):
        color = find_first_color_token("{p:2:R(赤)}")
        hue, sat, val = color_to_hsv(color)
        self.assertEqual(hue, 0.0)
        self.assertEqual(sat, 0.35)
        self.assertEqual(val, 0.98)

    def test_mismatch_id_code_jp_is_rejected(self):
        with self.assertRaises(ValueError):
            find_first_color_token("{1:R(赤)}")

    def test_consistent_token_is_accepted(self):
        color = find_first_color_token("{1:pR(紫みの赤)}")
        self.assertEqual(color["id"], 1)
        self.assertEqual(color["code"], "pR")
        self.assertEqual(color["jp"], "紫みの赤")


if __name__ == "__main__":
    unittest.main()

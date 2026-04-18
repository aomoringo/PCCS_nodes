import re
import colorsys
from typing import Dict, Callable, Match, Optional, List, Tuple
import numpy as np
import torch

# =========================================================
# PCCS 24 HUES (official wheel orientation)
# 24-step hue circle for ComfyUI processing
# 1 step = 15 degrees
# ID 14 is 0 degrees (right), counterclockwise is positive
# =========================================================

PCCS_HUE_STEP = 15.0
PCCS_ANGLE_ZERO_ID = 14

def pccs_id_to_angle_deg(color_id: int) -> float:
    return ((PCCS_ANGLE_ZERO_ID - color_id) % 24) * PCCS_HUE_STEP

def angle_deg_to_pccs_id(angle_deg: float) -> int:
    step = int(round(angle_deg / PCCS_HUE_STEP)) % 24
    return ((PCCS_ANGLE_ZERO_ID - 1 - step) % 24) + 1

PCCS_24_BASE = [
    {"id": 1,  "code": "pR",  "jp": "紫みの赤",   "en": "purplish_red"},
    {"id": 2,  "code": "R",   "jp": "赤",         "en": "red"},
    {"id": 3,  "code": "yR",  "jp": "黄みの赤",   "en": "yellowish_red"},
    {"id": 4,  "code": "rO",  "jp": "赤橙",       "en": "reddish_orange"},
    {"id": 5,  "code": "O",   "jp": "橙",         "en": "orange"},
    {"id": 6,  "code": "yO",  "jp": "黄橙",       "en": "yellowish_orange"},
    {"id": 7,  "code": "rY",  "jp": "赤みの黄",   "en": "reddish_yellow"},
    {"id": 8,  "code": "Y",   "jp": "黄",         "en": "yellow"},
    {"id": 9,  "code": "gY",  "jp": "緑みの黄",   "en": "greenish_yellow"},
    {"id": 10, "code": "YG",  "jp": "黄緑",       "en": "yellow_green"},
    {"id": 11, "code": "yG",  "jp": "黄みの緑",   "en": "yellowish_green"},
    {"id": 12, "code": "G",   "jp": "緑",         "en": "green"},
    {"id": 13, "code": "bG",  "jp": "青みの緑",   "en": "bluish_green"},
    {"id": 14, "code": "BG",  "jp": "青緑",       "en": "blue_green"},
    {"id": 15, "code": "BG2", "jp": "青緑(2)",    "en": "blue_green_2"},
    {"id": 16, "code": "gB",  "jp": "緑みの青",   "en": "greenish_blue"},
    {"id": 17, "code": "B",   "jp": "青",         "en": "blue"},
    {"id": 18, "code": "B2",  "jp": "青(2)",      "en": "blue_2"},
    {"id": 19, "code": "pB",  "jp": "紫みの青",   "en": "purplish_blue"},
    {"id": 20, "code": "V",   "jp": "青紫",       "en": "violet"},
    {"id": 21, "code": "bP",  "jp": "青みの紫",   "en": "bluish_purple"},
    {"id": 22, "code": "P",   "jp": "紫",         "en": "purple"},
    {"id": 23, "code": "rP",  "jp": "赤みの紫",   "en": "reddish_purple"},
    {"id": 24, "code": "RP",  "jp": "赤紫",       "en": "red_purple"},
]

PCCS_24 = [{**c, "h": pccs_id_to_angle_deg(c["id"])} for c in PCCS_24_BASE]

PCCS_BY_ID = {c["id"]: c for c in PCCS_24}

# =========================================================
# TOKEN FORMAT
#
# Base token:
#   {1:R(赤)}
#
# Toned token:
#   {p:1:R(赤)}
#   {dk:1:R(赤)}
#   {sf:1:R(赤)}
#   {g:1:R(赤)}
#   {ltg:1:R(赤)}
#   {d:1:R(赤)}
# =========================================================

COLOR_TOKEN_PATTERN = re.compile(
    r'\{(?:(?P<tone>[a-zA-Z]+):)?(?P<id>\d+):(?P<code>[A-Za-z0-9]+)\((?P<jp>.*?)\)\}'
)

# =========================================================
# TONE DEFINITIONS
# These are practical approximations, not strict full PCCS recreation.
# hsv: h(0-360), s(0-1), v(0-1)
# =========================================================

TONE_TO_PROMPT_PREFIX = {
    None: "",
    "": "",
    "p": "pale",
    "dk": "dark",
    "sf": "soft",
    "g": "grayish",
    "ltg": "light_grayish",
    "d": "dull",
}

# Preview-oriented HSV presets
TONE_TO_HSV = {
    None: {"s": 0.85, "v": 0.90},
    "":   {"s": 0.85, "v": 0.90},
    "p":  {"s": 0.35, "v": 0.98},
    "dk": {"s": 0.60, "v": 0.40},
    "sf": {"s": 0.45, "v": 0.85},
    "g":  {"s": 0.25, "v": 0.70},
    "ltg":{"s": 0.18, "v": 0.88},
    "d":  {"s": 0.38, "v": 0.55},
}

# PCCS vivid tone (v1-v24) practical hex references
PCCS_VIVID_HEX = {
    1:  "#D40045",
    2:  "#EE0026",
    3:  "#FD1A1C",
    4:  "#FE4118",
    5:  "#FF590B",
    6:  "#FF7F00",
    7:  "#FFCC00",
    8:  "#FFE600",
    9:  "#CCE700",
    10: "#99CF15",
    11: "#66B82B",
    12: "#33A23D",
    13: "#008F62",
    14: "#008678",
    15: "#007A87",
    16: "#055D87",
    17: "#093F86",
    18: "#0F218B",
    19: "#1D1A88",
    20: "#281285",
    21: "#340C81",
    22: "#56007D",
    23: "#770071",
    24: "#AF0065",
}

# =========================================================
# BASIC HELPERS
# =========================================================

def rotate_hue_id(hue_id: int, step: int) -> int:
    # +1 step = +15 degrees counterclockwise on official PCCS wheel
    return ((hue_id - 1 - step) % 24) + 1

def get_color_by_id(color_id: int) -> Dict:
    if color_id not in PCCS_BY_ID:
        raise ValueError(f"Invalid PCCS color id: {color_id}")
    return dict(PCCS_BY_ID[color_id])

def parse_match_to_color(match: Match[str]) -> Dict:
    color_id = int(match.group("id"))
    if color_id not in PCCS_BY_ID:
        raise ValueError(f"Invalid PCCS color id: {color_id}")

    color = get_color_by_id(color_id)
    color["tone"] = match.group("tone")
    color["matched_text"] = match.group(0)
    color["matched_code"] = match.group("code")
    color["matched_jp"] = match.group("jp")
    color["start"] = match.start()
    color["end"] = match.end()
    return color

def build_color_token(color: Dict, tone: Optional[str] = None) -> str:
    if tone:
        return f'{{{tone}:{color["id"]}:{color["code"]}({color["jp"]})}}'
    return f'{{{color["id"]}:{color["code"]}({color["jp"]})}}'

def normalize_color_tokens_in_text(text: str, replace_all: bool = False) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    def _repl(match: Match[str]) -> str:
        color = parse_match_to_color(match)
        return build_color_token(color, tone=color.get("tone"))

    if replace_all:
        return COLOR_TOKEN_PATTERN.sub(_repl, text)
    return COLOR_TOKEN_PATTERN.sub(_repl, text, count=1)

def find_first_color_token(text: str) -> Dict:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    m = COLOR_TOKEN_PATTERN.search(text)
    if not m:
        raise ValueError("No PCCS color token found. Expected like {1:R(赤)} or {p:1:R(赤)}")
    return parse_match_to_color(m)

def find_all_color_tokens(text: str) -> List[Dict]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [parse_match_to_color(m) for m in COLOR_TOKEN_PATTERN.finditer(text)]

def replace_color_tokens(
    text: str,
    transform_func: Callable[[Dict], str],
    replace_all: bool = False
) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    def _repl(match: Match[str]) -> str:
        color = parse_match_to_color(match)
        return transform_func(color)

    if replace_all:
        return COLOR_TOKEN_PATTERN.sub(_repl, text)
    return COLOR_TOKEN_PATTERN.sub(_repl, text, count=1)

def safe_run(fallback_count: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err = f"ERROR: {type(e).__name__}: {e}"
                return tuple([err] * fallback_count)
        return wrapper
    return decorator

# =========================================================
# COMMON TRANSFORMS
# =========================================================

def tone_transform(target_tone: str) -> Callable[[Dict], str]:
    def _transform(color: Dict) -> str:
        base = get_color_by_id(color["id"])
        return build_color_token(base, tone=target_tone)
    return _transform

def hue_shift_transform(step: int, preserve_tone: bool = True, force_tone: Optional[str] = None) -> Callable[[Dict], str]:
    def _transform(color: Dict) -> str:
        shifted = get_color_by_id(rotate_hue_id(color["id"], step))
        tone = force_tone if force_tone is not None else (color.get("tone") if preserve_tone else None)
        return build_color_token(shifted, tone=tone)
    return _transform

def same_hue_transform(force_tone: Optional[str] = None, preserve_tone: bool = False) -> Callable[[Dict], str]:
    def _transform(color: Dict) -> str:
        same = get_color_by_id(color["id"])
        tone = force_tone if force_tone is not None else (color.get("tone") if preserve_tone else None)
        return build_color_token(same, tone=tone)
    return _transform

# =========================================================
# PREVIEW HELPERS
# =========================================================

def color_to_hsv(color: Dict) -> Tuple[float, float, float]:
    hue = float(color["h"])
    tone = color.get("tone")
    tone_hsv = TONE_TO_HSV.get(tone, TONE_TO_HSV[None])
    return hue, float(tone_hsv["s"]), float(tone_hsv["v"])

def hsv_to_rgb01(h: float, s: float, v: float) -> Tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return float(r), float(g), float(b)

def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    hex_value = hex_color.strip().lstrip("#")
    if len(hex_value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_value[0:2], 16) / 255.0
    g = int(hex_value[2:4], 16) / 255.0
    b = int(hex_value[4:6], 16) / 255.0
    return float(r), float(g), float(b)

def color_to_rgb01(color: Dict) -> Tuple[Tuple[float, float, float], str, Optional[str], Tuple[float, float, float]]:
    """
    If tone is vivid-like (None, "", "v"), use PCCS_VIVID_HEX by color id.
    Otherwise, fall back to HSV approximation using hue + TONE_TO_HSV.
    Returns (rgb01, source, vivid_hex_or_none, hsv_triplet).
    """
    tone = color.get("tone")
    if tone in (None, "", "v"):
        hex_color = PCCS_VIVID_HEX.get(int(color["id"]))
        if hex_color:
            hue, sat, val = color_to_hsv(color)
            return hex_to_rgb01(hex_color), "hex", hex_color, (hue, sat, val)
    hue, sat, val = color_to_hsv(color)
    return hsv_to_rgb01(hue, sat, val), "hsv", None, (hue, sat, val)

def make_swatch_image(width: int, height: int, rgb: Tuple[float, float, float]) -> torch.Tensor:
    swatch = np.zeros((1, height, width, 3), dtype=np.float32)
    swatch[:, :, :, 0] = rgb[0]
    swatch[:, :, :, 1] = rgb[1]
    swatch[:, :, :, 2] = rgb[2]
    return torch.from_numpy(swatch)

# =========================================================
# NODE 1: PARSE / NORMALIZE
# =========================================================

class PCCSParseColorToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "1girl, dress, {1:R(赤)}, ribbon",
                    "multiline": True
                }),
                "normalize_all_tokens": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "normalized_text",
        "found_token",
        "color_id",
        "code",
        "jp_name",
        "en_name",
        "tone",
    )
    FUNCTION = "parse"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=7)
    def parse(self, text: str, normalize_all_tokens: bool):
        color = find_first_color_token(text)
        normalized_text = normalize_color_tokens_in_text(text, replace_all=normalize_all_tokens)
        return (
            normalized_text,
            build_color_token(color, tone=color.get("tone")),
            color["id"],
            color["code"],
            color["jp"],
            color["en"],
            color.get("tone") or "",
        )

# =========================================================
# NODE 2: TONE PALE
# =========================================================

class PCCSTonePaleToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=1)
    def convert(self, text: str, replace_all: bool):
        return (replace_color_tokens(text, tone_transform("p"), replace_all=replace_all),)

# =========================================================
# NODE 3: TONE DARK
# =========================================================

class PCCSToneDarkToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=1)
    def convert(self, text: str, replace_all: bool):
        return (replace_color_tokens(text, tone_transform("dk"), replace_all=replace_all),)

# =========================================================
# NODE 4: ANALOGOUS
# +30, -30, +60, -60
# =========================================================

class PCCSAnalogousHarmonyToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plus_30", "minus_30", "plus_60", "minus_60")
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=4)
    def convert(self, text: str, replace_all: bool):
        return (
            replace_color_tokens(text, hue_shift_transform(2), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(-2), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(4), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(-4), replace_all=replace_all),
        )

# =========================================================
# NODE 5: COMPLEMENTARY
# 180 degrees => +12
# =========================================================

class PCCSComplementaryHarmonyToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("complementary_text",)
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=1)
    def convert(self, text: str, replace_all: bool):
        return (replace_color_tokens(text, hue_shift_transform(12), replace_all=replace_all),)

# =========================================================
# NODE 6: CONTRAST / TRIAD
# +120, -120 => +8, -8
# =========================================================

class PCCSContrastHarmonyToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plus_120", "minus_120")
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=2)
    def convert(self, text: str, replace_all: bool):
        return (
            replace_color_tokens(text, hue_shift_transform(8), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(-8), replace_all=replace_all),
        )

# =========================================================
# NODE 7: NATURAL HARMONY
# Practical approximation:
# - main: same hue, soft
# - near_plus: +30, grayish
# - near_minus: -30, light_grayish
# =========================================================

class PCCSNaturalHarmonyToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("main_soft", "near_plus_grayish", "near_minus_light_grayish")
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=3)
    def convert(self, text: str, replace_all: bool):
        return (
            replace_color_tokens(text, same_hue_transform(force_tone="sf"), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(2, preserve_tone=False, force_tone="g"), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(-2, preserve_tone=False, force_tone="ltg"), replace_all=replace_all),
        )

# =========================================================
# NODE 8: COMPLEX HARMONY
# Practical approximation:
# - main: same hue, grayish
# - complement: 180, dull
# - analog_plus: +30, soft
# - analog_minus: -30, light_grayish
# =========================================================

class PCCSComplexHarmonyToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1girl, dress, {1:R(赤)}, ribbon", "multiline": True}),
                "replace_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("main_grayish", "complement_dull", "analog_plus_soft", "analog_minus_light_grayish")
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=4)
    def convert(self, text: str, replace_all: bool):
        return (
            replace_color_tokens(text, same_hue_transform(force_tone="g"), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(12, preserve_tone=False, force_tone="d"), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(2, preserve_tone=False, force_tone="sf"), replace_all=replace_all),
            replace_color_tokens(text, hue_shift_transform(-2, preserve_tone=False, force_tone="ltg"), replace_all=replace_all),
        )

# =========================================================
# NODE 9: TOKEN TO PROMPT
# {p:1:R(赤)} -> pale_red
# =========================================================

class PCCSTokenToPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "1girl, dress, {p:1:R(赤)}, ribbon",
                    "multiline": True
                }),
                "replace_all": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)
    FUNCTION = "convert"
    CATEGORY = "PCCS/Color"

    @safe_run(fallback_count=1)
    def convert(self, text: str, replace_all: bool):
        def _transform(color: Dict) -> str:
            tone = color.get("tone")
            prefix = TONE_TO_PROMPT_PREFIX.get(tone, tone or "")
            if prefix:
                return f"{prefix}_{color['en']}"
            return color["en"]

        return (replace_color_tokens(text, _transform, replace_all=replace_all),)

# =========================================================
# NODE 10: COLOR SWATCH PREVIEW
# =========================================================

class PCCSColorSwatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "1girl, dress, {1:R(赤)}, ribbon",
                    "multiline": True
                }),
                "width": ("INT", {"default": 256, "min": 8, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 256, "min": 8, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "info", "color_id", "hue", "saturation", "value")
    FUNCTION = "preview"
    CATEGORY = "PCCS/Preview"

    def preview(self, text: str, width: int, height: int):
        try:
            color = find_first_color_token(text)
            token = build_color_token(color, tone=color.get("tone"))
            rgb, source, hex_color, (hue, sat, val) = color_to_rgb01(color)
            image = make_swatch_image(width, height, rgb)
            tone = color.get("tone")
            tone_label = "vivid" if tone in (None, "", "v") else tone
            info = (
                f"PCCS Swatch | token={token} | en={color['en']} | tone={tone_label} "
            )
            if source == "hex":
                info += f"| source=hex | hex={hex_color}"
            else:
                info += f"| source=hsv | hsv=({hue:.1f},{sat:.2f},{val:.2f})"
            return (image, info, int(color["id"]), float(hue), float(sat), float(val))
        except Exception as e:
            err = f"ERROR: {type(e).__name__}: {e}"
            image = make_swatch_image(width, height, (0.0, 0.0, 0.0))
            return (image, err, 0, 0.0, 0.0, 0.0)

# =========================================================
# NODE CLASS MAPPINGS
# =========================================================

NODE_CLASS_MAPPINGS = {
    "PCCSParseColorToken": PCCSParseColorToken,
    "PCCSTonePaleToken": PCCSTonePaleToken,
    "PCCSToneDarkToken": PCCSToneDarkToken,
    "PCCSAnalogousHarmonyToken": PCCSAnalogousHarmonyToken,
    "PCCSComplementaryHarmonyToken": PCCSComplementaryHarmonyToken,
    "PCCSContrastHarmonyToken": PCCSContrastHarmonyToken,
    "PCCSNaturalHarmonyToken": PCCSNaturalHarmonyToken,
    "PCCSComplexHarmonyToken": PCCSComplexHarmonyToken,
    "PCCSTokenToPrompt": PCCSTokenToPrompt,
    "PCCSColorSwatch": PCCSColorSwatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PCCSParseColorToken": "PCCS Parse Color Token",
    "PCCSTonePaleToken": "PCCS Tone Token: Pale",
    "PCCSToneDarkToken": "PCCS Tone Token: Dark",
    "PCCSAnalogousHarmonyToken": "PCCS Harmony Token: Analogous",
    "PCCSComplementaryHarmonyToken": "PCCS Harmony Token: Complementary",
    "PCCSContrastHarmonyToken": "PCCS Harmony Token: Contrast (±120°)",
    "PCCSNaturalHarmonyToken": "PCCS Harmony Token: Natural",
    "PCCSComplexHarmonyToken": "PCCS Harmony Token: Complex",
    "PCCSTokenToPrompt": "PCCS Token To Prompt",
    "PCCSColorSwatch": "PCCS Color Swatch",
}

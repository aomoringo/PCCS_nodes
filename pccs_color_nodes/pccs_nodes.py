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

def pccs_id_to_hsv_hue_deg(color_id: int) -> float:
    """
    Convert PCCS hue id to HSV hue degree where 0° is red.
    id=2 (R) is anchored to 0°, and each neighboring id is 15° apart.
    """
    return ((color_id - 2) % 24) * PCCS_HUE_STEP

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
    r'\{(?:(?P<tone>[a-zA-Z+]+):)?(?P<id>\d+):(?P<code>[A-Za-z0-9]+)\((?P<jp>.*?)\)\}'
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

# PCCS fixed hex table by tone (24 colors each, id=1..24)
PCCS_FIXED_HEX = {
    "v": [
        "#D40045","#EE0026","#FD1A1C","#FE4118","#FF590B","#FF7F00","#FFCC00","#FFE600",
        "#CCE700","#99CF15","#66B82B","#33A23D","#008F62","#008678","#007A87","#055D87",
        "#093F86","#0F218B","#1D1A88","#281285","#340C81","#56007D","#770071","#AF0065",
    ],
    "b": [
        "#ED3B6B","#FA344D","#FC3633","#FC4E33","#FF6E2B","#FF9913","#FFCB1F","#FFF231",
        "#CDE52F","#99D02C","#55A73B","#32A65D","#2DA380","#1AA28E","#1FB3B3","#1C86AE",
        "#2B78B0","#396BB0","#5468AD","#6A64AE","#8561AB","#A459AB","#C75BB1","#DF4C93",
    ],
    "s": [
        "#B01040","#CA1028","#CC2211","#CC4613","#D45F10","#D97610","#D19711","#CCB914",
        "#B3B514","#8CA114","#41941E","#28853F","#287A52","#297364","#26707B","#205B85",
        "#224A87","#243B8B","#241F86","#3D1C84","#4E2283","#5F2883","#8C1D84","#9A0F50",
    ],
    "dp": [
        "#870042","#9D002B","#A20715","#A51200","#A42F03","#A24A02","#A46603","#A48204",
        "#949110","#518517","#307A25","#306F42","#186A53","#025865","#034F69","#04436E",
        "#05426F","#073E74","#152A6B","#232266","#3F1B63","#531560","#690C5C","#75004F",
    ],
    "lt": [
        "#EE7296","#FB7482","#FA7272","#FB8071","#FA996F","#FDB56D","#FCD474","#FEF27A",
        "#DDED71","#B3DE6A","#9AD47F","#7FC97E","#72C591","#66C1AF","#66C4C4","#67B1CA",
        "#67A9C9","#689ECA","#7288C2","#817DBA","#9678B8","#B173B6","#C972B6","#E170A4",
    ],
    "sf": [
        "#BD576F","#C95F6B","#CF5E5A","#D77957","#D6763A","#D89048","#D29F34","#CCBA4C",
        "#C0B647","#B3B140","#79B055","#66AC78","#5BA37E","#4E9B87","#4E9995","#4F8B96",
        "#4E7592","#516691","#535A90","#5C5791","#77568F","#8B5587","#9E5485","#B05076",
    ],
    "d": [
        "#8C355F","#994052","#A6424C","#B24443","#B34D3E","#B25939","#A66E3D","#997F42",
        "#8C8946","#757E47","#678049","#5A814C","#39764D","#2A6A69","#256B75","#1D6283",
        "#204F79","#214275","#2E3A76","#39367B","#493278","#5F3179","#772D7A","#802A69",
    ],
    "dk": [
        "#632534","#632A31","#6B2B29","#743526","#6E3D1F","#6B4919","#695018","#6A5B18",
        "#6E6E26","#56561A","#506B3E","#355935","#28523A","#1E4B44","#154D4E","#0E4250",
        "#123B4F","#163450","#222A4E","#312C4C","#3E2E49","#4A304B","#57304B","#643142",
    ],
    "p": [
        "#EEAFCE","#FBB4C4","#FAB6B5","#FDCDB7","#FBD8B0","#FEE6AA","#FCF1AF","#FEFFB3",
        "#EEFAB2","#E6F5B0","#D9F6C0","#CCEAC4","#C0EBCD","#B3E2D8","#B4DDDF","#B4D7DD",
        "#B5D2E0","#B3CEE3","#B4C2DD","#B2B6D9","#BCB2D5","#CAB2D6","#DAAFDC","#E4ADD5",
    ],
    "ltg": [
        "#C99FB3","#D7A4B5","#D6A9A4","#D7AFA7","#D9B59F","#D8BA96","#D9C098","#D9C69B",
        "#C5CB9B","#AAC09A","#A0BD9E","#9EBCA4","#99BAA7","#92B8AD","#91B8B7","#91AFBA",
        "#92A9B9","#91A4B5","#9199B0","#9191AD","#9C93AE","#A997B1","#B89AB6","#C09FB4",
    ],
    "g": [
        "#6B455A","#7D4F5A","#7C575E","#7D5F61","#7E6261","#7C6764","#7C6A5E","#7E6F5A",
        "#72755A","#636F5B","#586E57","#476C5B","#416863","#395B64","#38555D","#384E5C",
        "#38475A","#394158","#353654","#3F3051","#463353","#4A3753","#553857","#5B3A55",
    ],
    "dkg": [
        "#3C2D30","#3A2B2E","#3B2B2C","#3A2C2B","#40322F","#463B35","#453B31","#47402C",
        "#42412F","#3E3F31","#2C382A","#24332C","#23342E","#253532","#253535","#283639",
        "#232C33","#212832","#242331","#282530","#2A2730","#2D2A31","#362C34","#392D31",
    ],
}

PCCS_FIXED_HEX_BY_ID = {
    tone: {idx + 1: hex_color for idx, hex_color in enumerate(hex_list)}
    for tone, hex_list in PCCS_FIXED_HEX.items()
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
    matched_code = match.group("code")
    matched_jp = match.group("jp")
    if matched_code != color["code"] or matched_jp != color["jp"]:
        raise ValueError(
            "PCCS token mismatch for id="
            f"{color_id}: expected {color['code']}({color['jp']}), "
            f"got {matched_code}({matched_jp})"
        )
    color["tone"] = match.group("tone")
    color["matched_text"] = match.group(0)
    color["matched_code"] = matched_code
    color["matched_jp"] = matched_jp
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
    hue = float(pccs_id_to_hsv_hue_deg(int(color["id"])))
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

def normalize_tone_for_fixed_hex(tone: Optional[str]) -> str:
    if tone in (None, "", "V"):
        return "v"
    if tone in {"lt+", "p+"}:
        raise ValueError(f"Unsupported PCCS tone: {tone}")
    return tone

def color_to_hex(color: Dict) -> str:
    tone = normalize_tone_for_fixed_hex(color.get("tone"))
    color_id = int(color["id"])
    tone_table = PCCS_FIXED_HEX_BY_ID.get(tone)
    if tone_table is None:
        raise ValueError(f"Unsupported PCCS tone: {tone}")
    hex_color = tone_table.get(color_id)
    if hex_color is None:
        raise ValueError(f"Invalid PCCS color id for fixed hex: {color_id}")
    return hex_color

def color_to_rgb01(color: Dict) -> Tuple[Tuple[float, float, float], str, str, Tuple[float, float, float]]:
    """
    Resolve swatch color using fixed PCCS hex tables.
    Returns (rgb01, source, hex_color, hsv_triplet).
    """
    hex_color = color_to_hex(color)
    rgb = hex_to_rgb01(hex_color)
    hue_01, sat, val = colorsys.rgb_to_hsv(*rgb)
    return rgb, "fixed_hex", hex_color, (float(hue_01 * 360.0), float(sat), float(val))

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
            tone_label = normalize_tone_for_fixed_hex(color.get("tone"))
            info = (
                f"PCCS Swatch | token={token} | en={color['en']} | tone={tone_label} "
            )
            info += f"| color_id={int(color['id'])} | hex={hex_color} | source={source} | hsv=({hue:.1f},{sat:.2f},{val:.2f})"
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

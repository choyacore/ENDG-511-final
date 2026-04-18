"""
config.py – Central configuration for the Hair Analysis IoT System.
ENDG 511 – Team 1  |  Darren Taylor · Naishah Adetunji · Sehba Samman
"""
#https://pmc.ncbi.nlm.nih.gov/articles/PMC3254613/
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# ─── Camera & Display ─────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 15          # Minimum FPS per project spec (§ Performance Metrics)

# ─── Hair Length Thresholds ───────────────────────────────────────────────────
# ratio = (top-of-hair to chin) / (forehead to chin)
# ratio < SHORT_MAX  → short   (hair mostly above/at chin level)
# ratio < MEDIUM_MAX → medium  (hair around shoulder level)
# else               → long
LENGTH_SHORT_MAX  = 1.20
LENGTH_MEDIUM_MAX = 1.70

# ─── Hair Color Ranges (HSV, OpenCV convention) ───────────────────────────────
# H: 0–179,  S: 0–255,  V: 0–255
# Entry format:  (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
HSV_COLOR_RANGES = {
    "black":  (  0, 179,   0, 255,   0,  55),
    "gray":   (  0, 179,   0,  55,  55, 185),
    "white":  (  0, 179,   0,  35, 185, 255),
    "brown":  (  5,  22,  40, 230,  28, 140),
    "auburn": ( 10,  22,  80, 230,  55, 165),
    "blonde": ( 14,  36,  20, 190, 130, 245),
    "red":    (  0,  12,  80, 255,  55, 220),
}
COLOR_FALLBACK = "dark"

# ─── Recommendation Rules ─────────────────────────────────────────────────────
# Key: (color, length)  →  list of up to 3 styling tips
RECOMMENDATIONS: dict[tuple[str, str], list[str]] = {
    # ── BLACK ─────────────────────────────────────────────────────────────────
    #https://www.schwarzkopf.com/insider-tips/expert-tips/color-trends-for-black-hair.html
    ("black", "short"):  [
        "A fade or textured crop keeps black hair looking sharp.",
        "Visit the barber every 3–4 weeks to maintain clean edges.",
        "Light pomade adds definition without heaviness.",
    ],
    ("black", "medium"): [
        "Waves or twists look great at this length.",
        "Use a moisturising conditioner regularly to reduce frizz.",
        "A taper fade on the sides adds contrast.",
    ],
    ("black", "long"):   [
        "Deep-condition weekly to maintain lustre.",
        "Protective styles like braids reduce breakage significantly.",
        "Trim 1–2 cm every 8–10 weeks to prevent split ends.",
    ],
    # ── BROWN ─────────────────────────────────────────────────────────────────
    #https://www.glamour.com/gallery/dark-brown-hair-color-ideas
    ("brown", "short"):  [
        "A textured quiff or side part complements brown tones well.",
        "Light-hold wax keeps the style flexible all day.",
        "Touch up the cut every 4 weeks for a crisp look.",
    ],
    ("brown", "medium"): [
        "Layers add movement — ask for face-framing cuts.",
        "A gloss treatment enhances the natural warmth in brown hair.",
        "Blow-dry with a round brush for volume.",
    ],
    ("brown", "long"):   [
        "Balayage highlights complement brown hair beautifully.",
        "Trim 1–2 cm every 10 weeks to avoid split ends.",
        "Argan oil tames flyaways without weighing hair down.",
    ],
    # ── BLONDE ────────────────────────────────────────────────────────────────
    #https://www.thehairstyler.com/features/articles/hair-color/blonde-hair-color-tips-tricks-suggestions
    ("blonde", "short"): [
        "A pixie cut or buzz brings out the boldness of blonde.",
        "Purple shampoo once a week prevents brassiness.",
        "Light sea-salt spray adds texture and definition.",
    ],
    ("blonde", "medium"):[
        "Beach waves look stunning at this length.",
        "Tone every 6–8 weeks to keep blonde vibrant.",
        "Always use heat-protectant before styling.",
    ],
    ("blonde", "long"):  [
        "Layers prevent long blonde hair from looking flat.",
        "Bond-repair treatments (e.g., Olaplex) protect bleached strands.",
        "Limit heat styling to 3 times per week maximum.",
    ],
    # ── RED ───────────────────────────────────────────────────────────────────
    #https://www.lorealparisusa.com/beauty-magazine/hair-color/hair-color-tutorials/hair-color-tips-red-hair
    ("red", "short"):    [
        "Use colour-safe shampoo — red fades faster than any other shade.",
        "A sharp cut showcases the vivid hue at its best.",
        "Re-tone every 4–6 weeks to keep red intense.",
    ],
    ("red", "medium"):   [
        "Copper or strawberry-blonde lowlights add depth to red.",
        "Avoid sun exposure without UV-protect spray — it fades red quickly.",
        "A silk pillowcase reduces frizz and colour friction overnight.",
    ],
    ("red", "long"):     [
        "Gloss treatments boost shine on red and copper tones.",
        "Moisturising masks weekly keep colour-treated red hair vibrant.",
        "Braids are a protective style that preserve both length and colour.",
    ],
    # ── AUBURN ────────────────────────────────────────────────────────────────
    #https://www.lorealparisusa.com/beauty-magazine/hair-color/hair-color-tutorials/how-to-get-auburn-hair
    ("auburn", "short"): [
        "A tapered cut showcases warm auburn tones perfectly.",
        "Colour-safe shampoo preserves the warmth.",
        "A light oil adds shine without dulling the colour.",
    ],
    ("auburn", "medium"):[
        "Loose curls complement auburn beautifully.",
        "Highlights or lowlights add depth and dimension.",
        "Re-gloss every 8 weeks to maintain richness.",
    ],
    ("auburn", "long"):  [
        "Deep-condition weekly to maintain lustre in long auburn hair.",
        "Avoid excessive heat to preserve warm undertones.",
        "Braid at night to prevent tangles and breakage.",
    ],
    # ── GRAY ──────────────────────────────────────────────────────────────────
    #https://therighthairstyles.com/gray-blending-for-dark-hair/
    ("gray", "short"):   [
        "Embrace the silver — a clean fade looks polished.",
        "Blue-tinted shampoo neutralises yellow tones.",
        "Regular trims every 3–4 weeks maintain crispness.",
    ],
    ("gray", "medium"):  [
        "Layered bobs frame the face elegantly with gray hair.",
        "Deep-condition bi-weekly — gray hair tends to be drier.",
        "A glossing treatment adds shine and softens texture.",
    ],
    ("gray", "long"):    [
        "A toner keeps silver looking bright rather than yellow.",
        "Use a wide-tooth comb — gray strands are more fragile.",
        "Leave-in conditioner is a daily essential for long gray hair.",
    ],
    # ── WHITE ─────────────────────────────────────────────────────────────────
    ("white", "short"):  [
        "Precision cuts look striking with white hair.",
        "Blue or purple shampoo prevents yellowing.",
        "A light pomade adds polish and definition.",
    ],
    ("white", "medium"): [
        "Soft waves complement white hair beautifully.",
        "Moisture is key — use a hydrating mask weekly.",
        "Avoid chlorine pools as they cause yellowing in white hair.",
    ],
    ("white", "long"):   [
        "Long white hair is striking — commit to deep moisture routines.",
        "Trim every 8 weeks to prevent fragile ends from breaking.",
        "A silk hair wrap overnight prevents tangling.",
    ],
    # ── DARK (for colored shaded hair with complex gradient) ───────────────────────────────────────────────────────
    ("dark", "short"):   [
        "A textured crop or fade suits your hair tone well.",
        "Light-hold product adds definition.",
        "Trim every 4 weeks for a consistently clean look.",
    ],
    ("dark", "medium"):  [
        "Try highlights to add dimension to darker tones.",
        "Moisture masks help maintain softness.",
        "A blowout adds volume and body.",
    ],
    ("dark", "long"):    [
        "Deep-condition regularly for healthy, strong hair.",
        "Protective styles reduce breakage at length.",
        "Trim every 8–10 weeks to keep ends healthy.",
    ],
}


def get_recommendation(color: str, length: str) -> list[str]:
    """
    Look up styling tips for (color, length).
    Falls back to generic advice if the combination isn't in the table.
    """
    tips = RECOMMENDATIONS.get((color, length))
    if tips:
        return tips
    # Generic fallback
    return [
        f"Your {length} {color} hair has great potential!",
        "Regular trims every 6–8 weeks keep all hair types healthy.",
        "Use products suited to your specific hair type for best results.",
    ]

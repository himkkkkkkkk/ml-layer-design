"""
Pragmatic mode prompts — same fact, different speech acts.

Tests whether illocutionary force (statement vs question vs command/order)
affects the semantic representation.

8 facts × 3 modes × 4 variants per mode = 96 prompts, English only.
"""

PRAG_PROMPTS = [
    # ── 0: sun rises ──
    {
        "statement": [
            "the sun rises in the east and sets in the west",
            "the sun always rises in the east",
            "each day the sun comes up in the east",
            "the sun's path goes from east to west",
        ],
        "question": [
            "does the sun rise in the east?",
            "in which direction does the sun rise?",
            "where does the sun come up each morning?",
            "is it true that the sun rises in the east?",
        ],
        "order": [
            "tell me about the sun rising in the east",
            "explain where the sun rises",
            "give me a brief introduction to the sun's daily path",
            "please describe how the sun moves across the sky",
        ],
    },
    # ── 1: water boils ──
    {
        "statement": [
            "water boils at 100 degrees celsius",
            "water reaches its boiling point at 100 degrees",
            "at sea level water boils at 100 degrees celsius",
            "the boiling point of water is 100 degrees celsius",
        ],
        "question": [
            "does water boil at 100 degrees celsius?",
            "at what temperature does water boil?",
            "what is the boiling point of water?",
            "is it true that water boils at 100 degrees?",
        ],
        "order": [
            "tell me about the boiling point of water",
            "explain at what temperature water boils",
            "give me a brief introduction to water's boiling point",
            "please describe when water starts to boil",
        ],
    },
    # ── 2: earth orbits sun ──
    {
        "statement": [
            "the earth revolves around the sun",
            "the earth orbits the sun once per year",
            "our planet goes around the sun",
            "the earth's orbit is centered on the sun",
        ],
        "question": [
            "does the earth revolve around the sun?",
            "what does the earth orbit around?",
            "does the earth go around the sun?",
            "is it true that the earth orbits the sun?",
        ],
        "order": [
            "tell me about the earth's orbit around the sun",
            "explain how the earth moves relative to the sun",
            "give me a brief introduction to the earth's revolution",
            "please describe the earth's path around the sun",
        ],
    },
    # ── 3: plants need sunlight ──
    {
        "statement": [
            "plants need sunlight to grow",
            "sunlight is essential for plant growth",
            "plants require light for photosynthesis",
            "without sunlight plants cannot survive",
        ],
        "question": [
            "do plants need sunlight to grow?",
            "what do plants need for growth?",
            "is sunlight necessary for plants?",
            "do plants require light to survive?",
        ],
        "order": [
            "tell me about plants needing sunlight",
            "explain why plants need light to grow",
            "give me a brief introduction to photosynthesis",
            "please describe how plants use sunlight",
        ],
    },
    # ── 4: humans need oxygen ──
    {
        "statement": [
            "humans need oxygen to survive",
            "oxygen is essential for human life",
            "people cannot live without oxygen",
            "human survival depends on oxygen",
        ],
        "question": [
            "do humans need oxygen to survive?",
            "what do humans need to breathe?",
            "is oxygen necessary for human life?",
            "can humans live without oxygen?",
        ],
        "order": [
            "tell me about humans needing oxygen",
            "explain why oxygen is essential for humans",
            "give me a brief introduction to human respiration",
            "please describe how humans use oxygen",
        ],
    },
    # ── 5: ice melts ──
    {
        "statement": [
            "ice melts when heated",
            "ice turns into water when warmed",
            "solid water becomes liquid above zero degrees",
            "ice changes state when temperature rises",
        ],
        "question": [
            "does ice melt when heated?",
            "what happens to ice when it gets warm?",
            "does ice turn into water with heat?",
            "is it true that ice melts in warm conditions?",
        ],
        "order": [
            "tell me about ice melting",
            "explain what happens when ice is heated",
            "give me a brief introduction to phase changes of water",
            "please describe the melting process of ice",
        ],
    },
    # ── 6: birds fly ──
    {
        "statement": [
            "most birds can fly in the sky",
            "the majority of birds are capable of flight",
            "flying is common among bird species",
            "birds typically have the ability to fly",
        ],
        "question": [
            "can most birds fly in the sky?",
            "are most birds able to fly?",
            "do birds generally have the ability to fly?",
            "is flying common among birds?",
        ],
        "order": [
            "tell me about birds flying",
            "explain how birds are able to fly",
            "give me a brief introduction to bird flight",
            "please describe the flying ability of birds",
        ],
    },
    # ── 7: light faster than sound ──
    {
        "statement": [
            "light travels faster than sound",
            "the speed of light exceeds the speed of sound",
            "light moves more quickly than sound does",
            "sound is slower than light",
        ],
        "question": [
            "does light travel faster than sound?",
            "which is faster, light or sound?",
            "is light quicker than sound?",
            "does sound move slower than light?",
        ],
        "order": [
            "tell me about the speed of light versus sound",
            "explain why light is faster than sound",
            "give me a brief introduction to comparing light and sound speed",
            "please describe how light and sound speeds compare",
        ],
    },
]

TOPIC_NAMES = ["Sun", "Water", "Earth", "Plants", "Humans", "Ice", "Birds", "Light"]
MODE_NAMES = ["statement", "question", "order"]


def flatten() -> list[dict]:
    """Return flat list: {text, topic, mode, variant, name}"""
    result = []
    for t, topic in enumerate(PRAG_PROMPTS):
        for mode, variants in topic.items():
            for v, text in enumerate(variants):
                result.append({
                    "text": text,
                    "topic": t,
                    "mode": mode,
                    "variant": v,
                    "name": f"topic{t}_{mode}_v{v}",
                })
    return result

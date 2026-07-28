"""
Syntactic variation prompts — same fact, different grammar.

8 facts × ~6 syntactic forms = 48 prompts, English only.
Tests whether syntactic structure affects semantic clustering.
"""

SYN_PROMPTS = [
    # ── 0: sun rises ──
    {
        "plain": "the sun rises in the east and sets in the west",
        "inverted": "in the east the sun rises, and in the west it sets",
        "cleft": "it is the east where the sun rises and the west where it sets",
        "emphatic": "the sun does rise in the east and does set in the west",
        "impersonal": "it is well-known that the sun rises in the east and sets in the west",
        "topicalized": "as for the sun, it rises in the east and sets in the west",
    },
    # ── 1: water boils ──
    {
        "plain": "water boils at 100 degrees celsius",
        "inverted": "at 100 degrees celsius water boils",
        "cleft": "it is 100 degrees celsius at which water boils",
        "emphatic": "water does boil at 100 degrees celsius",
        "impersonal": "it is a fact that water boils at 100 degrees celsius",
        "topicalized": "as for water, it boils at 100 degrees celsius",
    },
    # ── 2: earth orbits sun ──
    {
        "plain": "the earth revolves around the sun",
        "inverted": "around the sun the earth revolves",
        "cleft": "it is the sun that the earth revolves around",
        "emphatic": "the earth does revolve around the sun",
        "impersonal": "it is established that the earth revolves around the sun",
        "topicalized": "as for the earth, it revolves around the sun",
    },
    # ── 3: plants need sunlight ──
    {
        "plain": "plants need sunlight to grow",
        "inverted": "to grow, plants need sunlight",
        "cleft": "it is sunlight that plants need to grow",
        "emphatic": "plants do need sunlight to grow",
        "impersonal": "it is known that plants need sunlight to grow",
        "topicalized": "as for plants, they need sunlight to grow",
    },
    # ── 4: humans need oxygen ──
    {
        "plain": "humans need oxygen to survive",
        "inverted": "to survive, humans need oxygen",
        "cleft": "it is oxygen that humans need to survive",
        "emphatic": "humans do need oxygen to survive",
        "impersonal": "it is true that humans need oxygen to survive",
        "topicalized": "as for humans, they need oxygen to survive",
    },
    # ── 5: ice melts ──
    {
        "plain": "ice melts when heated",
        "inverted": "when heated, ice melts",
        "cleft": "it is when heated that ice melts",
        "emphatic": "ice does melt when heated",
        "impersonal": "it is observed that ice melts when heated",
        "topicalized": "as for ice, it melts when heated",
    },
    # ── 6: birds fly ──
    {
        "plain": "most birds can fly in the sky",
        "inverted": "in the sky most birds can fly",
        "cleft": "it is in the sky that most birds can fly",
        "emphatic": "most birds do fly in the sky",
        "impersonal": "it is common knowledge that most birds can fly in the sky",
        "topicalized": "as for birds, most can fly in the sky",
    },
    # ── 7: light faster than sound ──
    {
        "plain": "light travels faster than sound",
        "inverted": "faster than sound, light travels",
        "cleft": "it is light that travels faster than sound",
        "emphatic": "light does travel faster than sound",
        "impersonal": "it is understood that light travels faster than sound",
        "topicalized": "as for light, it travels faster than sound",
    },
]

TOPIC_NAMES = ["Sun", "Water", "Earth", "Plants", "Humans", "Ice", "Birds", "Light"]
STRUCT_NAMES = ["plain", "inverted", "cleft", "emphatic", "impersonal", "topicalized"]


def flatten() -> list[dict]:
    """Return flat list: {text, topic, struct, name}"""
    result = []
    for t, topic in enumerate(SYN_PROMPTS):
        for struct, text in topic.items():
            result.append({
                "text": text,
                "topic": t,
                "struct": struct,
                "name": f"topic{t}_{struct}",
            })
    return result

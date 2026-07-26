"""
Prompt collection for multi-lingual layer analysis.

Each entry: (statement, question)
Multiple languages for the same set of facts, so we can compare:
  - language encoding (EN vs ZH vs DE vs FR vs JA)
  - statement vs question representation

Languages: en, zh, de, fr, ja
"""

PROMPTS = [
    # ── 1. sun rises in the east ──
    {
        "en": ("the sun rises in the east and sets in the west",
               "does the sun rise in the east and set in the west"),
        "zh": ("太阳从东边升起，西边落下",
               "太阳是从东边升起西边落下吗"),
        "de": ("die sonne geht im osten auf und im westen unter",
               "geht die sonne im osten auf und im westen unter"),
        "fr": ("le soleil se lève à l'est et se couche à l'ouest",
               "le soleil se lève-t-il à l'est et se couche-t-il à l'ouest"),
        "ja": ("太陽は東から昇り西に沈む",
               "太陽は東から昇り西に沈みますか"),
    },
    # ── 2. water boils at 100 degrees ──
    {
        "en": ("water boils at 100 degrees celsius",
               "does water boil at 100 degrees celsius"),
        "zh": ("水在100摄氏度沸腾",
               "水在100摄氏度沸腾吗"),
        "de": ("wasser kocht bei 100 grad celsius",
               "kocht wasser bei 100 grad celsius"),
        "fr": ("l'eau bout à 100 degrés celsius",
               "l'eau bout-elle à 100 degrés celsius"),
        "ja": ("水は100度で沸騰する",
               "水は100度で沸騰しますか"),
    },
    # ── 3. earth revolves around the sun ──
    {
        "en": ("the earth revolves around the sun",
               "does the earth revolve around the sun"),
        "zh": ("地球绕着太阳转",
               "地球是绕着太阳转吗"),
        "de": ("die erde dreht sich um die sonne",
               "dreht sich die erde um die sonne"),
        "fr": ("la terre tourne autour du soleil",
               "la terre tourne-t-elle autour du soleil"),
        "ja": ("地球は太陽の周りを回っている",
               "地球は太陽の周りを回っていますか"),
    },
    # ── 4. plants need sunlight ──
    {
        "en": ("plants need sunlight to grow",
               "do plants need sunlight to grow"),
        "zh": ("植物生长需要阳光",
               "植物生长需要阳光吗"),
        "de": ("pflanzen brauchen sonnenlicht zum wachsen",
               "brauchen pflanzen sonnenlicht zum wachsen"),
        "fr": ("les plantes ont besoin de soleil pour pousser",
               "les plantes ont-elles besoin de soleil pour pousser"),
        "ja": ("植物は成長するために日光を必要とする",
               "植物は成長するために日光を必要としますか"),
    },
    # ── 5. humans need oxygen ──
    {
        "en": ("humans need oxygen to survive",
               "do humans need oxygen to survive"),
        "zh": ("人类生存需要氧气",
               "人类生存需要氧气吗"),
        "de": ("menschen brauchen sauerstoff zum überleben",
               "brauchen menschen sauerstoff zum überleben"),
        "fr": ("les humains ont besoin d'oxygène pour survivre",
               "les humains ont-ils besoin d'oxygène pour survivre"),
        "ja": ("人間は生きるために酸素を必要とする",
               "人間は生きるために酸素を必要としますか"),
    },
    # ── 6. ice melts in heat ──
    {
        "en": ("ice melts when heated",
               "does ice melt when heated"),
        "zh": ("冰加热会融化",
               "冰加热会融化吗"),
        "de": ("eis schmilzt wenn es erhitzt wird",
               "schmilzt eis wenn es erhitzt wird"),
        "fr": ("la glace fond quand elle est chauffée",
               "la glace fond-elle quand elle est chauffée"),
        "ja": ("氷は加熱すると溶ける",
               "氷は加熱すると溶けますか"),
    },
    # ── 7. birds can fly ──
    {
        "en": ("most birds can fly in the sky",
               "can most birds fly in the sky"),
        "zh": ("大多数鸟类能在天空飞翔",
               "大多数鸟类能在天空飞翔吗"),
        "de": ("die meisten vögel können am himmel fliegen",
               "können die meisten vögel am himmel fliegen"),
        "fr": ("la plupart des oiseaux peuvent voler dans le ciel",
               "la plupart des oiseaux peuvent-ils voler dans le ciel"),
        "ja": ("ほとんどの鳥は空を飛べる",
               "ほとんどの鳥は空を飛べますか"),
    },
    # ── 8. light travels faster than sound ──
    {
        "en": ("light travels faster than sound",
               "does light travel faster than sound"),
        "zh": ("光比声音传播得快",
               "光比声音传播得快吗"),
        "de": ("licht bewegt sich schneller als schall",
               "bewegt sich licht schneller als schall"),
        "fr": ("la lumière voyage plus vite que le son",
               "la lumière voyage-t-elle plus vite que le son"),
        "ja": ("光は音より速く進む",
               "光は音より速く進みますか"),
    },
]


def flatten_prompts() -> list[dict]:
    """
    Return a flat list of all prompts with metadata.

    Each entry: {
        "text": str,
        "lang": "en"|"zh"|"de"|"fr"|"ja",
        "type": "S"|"Q",       # statement or question
        "topic": int,           # 0..N topic index
        "name": str,            # unique run name e.g. "topic0_en_S"
    }
    """
    result = []
    for topic_idx, topic in enumerate(PROMPTS):
        for lang in topic:
            stmt, ques = topic[lang]
            result.append({
                "text": stmt,
                "lang": lang,
                "type": "S",
                "topic": topic_idx,
                "name": f"topic{topic_idx}_{lang}_S",
            })
            result.append({
                "text": ques,
                "lang": lang,
                "type": "Q",
                "topic": topic_idx,
                "name": f"topic{topic_idx}_{lang}_Q",
            })
    return result


if __name__ == "__main__":
    flat = flatten_prompts()
    print(f"Total prompts: {len(flat)}")
    for p in flat:
        print(f"  [{p['name']:22s}] {p['lang']} {p['type']} | {p['text'][:50]}...")

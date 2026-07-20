import json
from bs4 import BeautifulSoup


def clean_html(raw_html: str | None) -> str:
    """Strip HTML tags and collapse messy whitespace/&nbsp; into plain text."""
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    text = text.replace("\xa0", " ")          # &nbsp; leftovers
    text = " ".join(text.split())              # collapse repeated whitespace/newlines
    return text.strip()


def crop_to_documents(crop: dict) -> list[dict]:
    """Turn one crop record into a list of small, focused documents.
    Each document = {"text": ..., "metadata": {...}} ready for embedding."""
    name = crop["crop_name"]
    bn_name = crop["crop_bangla_name"]
    docs = []

    def add(text, section: str, extra: dict | None = None):
        text = clean_html(text) if isinstance(text, str) else text
        if not text:
            return
        meta = {"crop_id": crop["id"], "crop_name": name, "crop_bangla_name": bn_name, "section": section}
        if extra:
            meta.update(extra)
        docs.append({"text": f"[{name} / {bn_name}] ({section}) {text}", "metadata": meta})

    # 1. simple description sections
    add(crop.get("general_info"), "general_info")
    for section in ["harvest", "intercultural", "irrigation", "landPreparation"]:
        block = crop.get(section)
        if block:
            add(block.get("description"), section)

    # fertilizer — different shape from the others: text lives under key
    # "fertilizer" itself, not "description"
    fert = crop.get("fertilizer")
    if fert:
        add(fert.get("fertilizer"), "fertilizer")
        
    # 2. seed section — has structured fields, not just one description
    seed = crop.get("seed")
    if seed:
        parts = []
        if seed.get("seed_rate"):
            parts.append(f"Seed rate: {clean_html(str(seed['seed_rate']))}")
        if seed.get("treatment"):
            parts.append(f"Seed treatment: {clean_html(seed['treatment'])}")
        if seed.get("showing_method"):
            parts.append(f"Sowing method: {clean_html(seed['showing_method'])}")
        if seed.get("time_showing"):
            parts.append(f"Sowing time: {clean_html(seed['time_showing'])}")
        if seed.get("seedbed"):
            parts.append(f"Seedbed: {clean_html(seed['seedbed'])}")
        if parts:
            add(" | ".join(parts), "seed")

    # 3. each variety -> its OWN document (this is what your seed-rate-per-hectare
    #    example needs — precise numbers per named variety, not lumped together)
    for v in crop.get("variety") or []:
        vname = v.get("variety_name", "unnamed variety")
        parts = [f"Variety: {vname}"]
        if v.get("seed_rate"):
            unit = (v.get("seedRateUnit") or {}).get("unit_name", "")
            parts.append(f"Seed rate: {v['seed_rate']} {unit}")
        if v.get("avg_expected_yield"):
            parts.append(f"Average expected yield: {v['avg_expected_yield']}")
        if v.get("duration_start") and v.get("duration_end"):
            parts.append(f"Duration: {v['duration_start']}-{v['duration_end']} days")
        if v.get("special_character"):
            parts.append(clean_html(v["special_character"]))
        add(" | ".join(parts), "variety", extra={"variety_name": vname})

    # 4. each pesticide/herbicide entry -> its own document
    for p in crop.get("pesticide") or []:
        text = f"Problem: {p.get('disease_name', '')}. Control: {clean_html(p.get('control_measure'))}"
        add(text, "pesticide", extra={"disease_name": p.get("disease_name")})

    for h in crop.get("herbicide") or []:
        text = f"Target weeds: {h.get('pesticide_name', '')}. Guide: {clean_html(h.get('application_guide'))}"
        add(text, "herbicide")

    return docs


if __name__ == "__main__":
    with open("data/crops.json", encoding="utf-8") as f:
        raw = json.load(f)

    rows = raw["data"]["getAllCropsFullDetails"]["rows"]
    print(f"Loaded {len(rows)} crops")

    all_docs = []
    for crop in rows:
        all_docs.extend(crop_to_documents(crop))

    print(f"Produced {len(all_docs)} documents total")
    # print("\n--- Example documents from crop 0 ---")
    # for d in crop_to_documents(rows[0])[:3]:
    #     print(d["metadata"]["section"], "->", d["text"][:150])
    #     print()

    with open("data/documents.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

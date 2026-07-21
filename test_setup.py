from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")
response = llm.invoke("Say hello in one short sentence.")
print(response.content)



# from __future__ import annotations

# import json
# import re
# from pathlib import Path
# from typing import Any

# from bs4 import BeautifulSoup


# INPUT_FILE = Path("data/crops.json")
# OUTPUT_FILE = Path("data/documents.json")

# MAX_DOCUMENT_CHARS = 1800
# CHUNK_OVERLAP_CHARS = 150


# def has_value(value: Any) -> bool:
#     if value is None:
#         return False

#     if isinstance(value, str):
#         return bool(value.strip())

#     if isinstance(value, (list, dict, tuple, set)):
#         return bool(value)

#     return True


# def normalize_text(value: Any) -> str:
#     if not has_value(value):
#         return ""

#     text = str(value)
#     text = text.replace("\xa0", " ")
#     text = text.replace("\r\n", "\n")
#     text = text.replace("\r", "\n")

#     text = re.sub(r"[ \t\f\v]+", " ", text)
#     text = re.sub(r" *\n *", "\n", text)
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     return text.strip()


# def clean_html(raw_html: Any) -> str:
#     if not has_value(raw_html):
#         return ""

#     html = str(raw_html)
#     soup = BeautifulSoup(html, "html.parser")

#     for unwanted_tag in soup.find_all(["script", "style", "noscript"]):
#         unwanted_tag.decompose()

#     for image in soup.find_all("img"):
#         alt_text = normalize_text(image.get("alt"))
#         if alt_text:
#             image.replace_with(f" Image description: {alt_text} ")
#         else:
#             image.decompose()

#     for line_break in soup.find_all("br"):
#         line_break.replace_with("\n")

#     for list_item in soup.find_all("li"):
#         list_item.insert_before("\n- ")
#         list_item.append("\n")

#     block_tags = [
#         "p",
#         "div",
#         "section",
#         "article",
#         "header",
#         "footer",
#         "h1",
#         "h2",
#         "h3",
#         "h4",
#         "h5",
#         "h6",
#         "ul",
#         "ol",
#         "table",
#         "tr",
#     ]

#     for block_tag in soup.find_all(block_tags):
#         block_tag.insert_before("\n")
#         block_tag.append("\n")

#     for table_cell in soup.find_all(["td", "th"]):
#         table_cell.append(" | ")

#     text = soup.get_text(separator=" ")
#     return normalize_text(text)


# def clean_value(value: Any) -> str:
#     if not has_value(value):
#         return ""

#     if isinstance(value, str):
#         return clean_html(value)

#     return normalize_text(value)


# def join_parts(parts: list[str], separator: str = "\n") -> str:
#     cleaned_parts = [
#         normalize_text(part)
#         for part in parts
#         if has_value(part) and normalize_text(part)
#     ]

#     return separator.join(cleaned_parts)


# def humanize_key(key: str) -> str:
#     text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
#     text = text.replace("_", " ")
#     return text.strip().capitalize()


# def safe_identifier(value: Any) -> str:
#     text = normalize_text(value)

#     if not text:
#         return "root"

#     text = re.sub(r"[^0-9A-Za-z_-]+", "-", text)
#     text = re.sub(r"-+", "-", text)

#     return text.strip("-") or "root"


# def split_large_piece(
#     text: str,
#     max_chars: int,
#     overlap_chars: int,
# ) -> list[str]:
#     text = normalize_text(text)

#     if not text:
#         return []

#     if len(text) <= max_chars:
#         return [text]

#     pieces = []
#     start = 0
#     text_length = len(text)

#     while start < text_length:
#         end = min(start + max_chars, text_length)

#         if end < text_length:
#             minimum_break_position = start + int(max_chars * 0.6)
#             possible_breaks = [
#                 text.rfind("\n", minimum_break_position, end),
#                 text.rfind("।", minimum_break_position, end),
#                 text.rfind(".", minimum_break_position, end),
#                 text.rfind(" ", minimum_break_position, end),
#             ]

#             break_position = max(possible_breaks)

#             if break_position > start:
#                 end = break_position + 1

#         piece = text[start:end].strip()

#         if piece:
#             pieces.append(piece)

#         if end >= text_length:
#             break

#         new_start = max(end - overlap_chars, start + 1)

#         while new_start < text_length and not text[new_start].isspace():
#             new_start += 1

#         start = new_start

#     return pieces


# def split_text(
#     text: str,
#     max_chars: int = MAX_DOCUMENT_CHARS,
#     overlap_chars: int = CHUNK_OVERLAP_CHARS,
# ) -> list[str]:
#     text = normalize_text(text)

#     if not text:
#         return []

#     if len(text) <= max_chars:
#         return [text]

#     paragraphs = [
#         normalize_text(paragraph)
#         for paragraph in re.split(r"\n+", text)
#         if normalize_text(paragraph)
#     ]

#     units: list[str] = []

#     for paragraph in paragraphs:
#         if len(paragraph) <= max_chars:
#             units.append(paragraph)
#             continue

#         sentences = [
#             normalize_text(sentence)
#             for sentence in re.split(r"(?<=[।.!?])\s+", paragraph)
#             if normalize_text(sentence)
#         ]

#         if len(sentences) <= 1:
#             units.extend(
#                 split_large_piece(
#                     paragraph,
#                     max_chars=max_chars,
#                     overlap_chars=overlap_chars,
#                 )
#             )
#         else:
#             units.extend(sentences)

#     chunks: list[str] = []
#     current_chunk = ""

#     for unit in units:
#         if len(unit) > max_chars:
#             if current_chunk:
#                 chunks.append(current_chunk)
#                 current_chunk = ""

#             chunks.extend(
#                 split_large_piece(
#                     unit,
#                     max_chars=max_chars,
#                     overlap_chars=overlap_chars,
#                 )
#             )
#             continue

#         candidate = (
#             f"{current_chunk}\n{unit}".strip()
#             if current_chunk
#             else unit
#         )

#         if len(candidate) <= max_chars:
#             current_chunk = candidate
#         else:
#             if current_chunk:
#                 chunks.append(current_chunk)

#             current_chunk = unit

#     if current_chunk:
#         chunks.append(current_chunk)

#     return chunks


# def nested_unit_name(
#     record: dict,
#     nested_key: str,
#     fallback_key: str | None = None,
# ) -> str:
#     nested_value = record.get(nested_key)

#     if isinstance(nested_value, dict):
#         unit_name = clean_value(nested_value.get("unit_name"))

#         if unit_name:
#             return unit_name

#         unit_code = clean_value(nested_value.get("unit_code"))

#         if unit_code:
#             return unit_code

#     if fallback_key:
#         fallback_value = clean_value(record.get(fallback_key))

#         if fallback_value:
#             return fallback_value

#     return ""


# def format_quantity(
#     value: Any,
#     unit: str = "",
#     missing_unit_message: str = "",
# ) -> str:
#     cleaned_value = clean_value(value)

#     if not cleaned_value:
#         return ""

#     if unit:
#         return f"{cleaned_value} {unit}".strip()

#     if missing_unit_message:
#         return f"{cleaned_value} ({missing_unit_message})"

#     return cleaned_value


# def flatten_data(
#     value: Any,
#     prefix: str = "",
# ) -> list[str]:
#     lines: list[str] = []

#     if isinstance(value, dict):
#         for key, nested_value in value.items():
#             if not has_value(nested_value):
#                 continue

#             label = humanize_key(str(key))
#             nested_prefix = f"{prefix} {label}".strip()

#             if isinstance(nested_value, (dict, list)):
#                 lines.extend(flatten_data(nested_value, nested_prefix))
#             else:
#                 cleaned_value = clean_value(nested_value)

#                 if cleaned_value:
#                     lines.append(f"{nested_prefix}: {cleaned_value}")

#     elif isinstance(value, list):
#         for index, item in enumerate(value, start=1):
#             item_prefix = f"{prefix} item {index}".strip()
#             lines.extend(flatten_data(item, item_prefix))

#     elif has_value(value):
#         cleaned_value = clean_value(value)

#         if cleaned_value:
#             if prefix:
#                 lines.append(f"{prefix}: {cleaned_value}")
#             else:
#                 lines.append(cleaned_value)

#     return lines


# def extract_packet_information(packet_data: Any) -> str:
#     if not isinstance(packet_data, list):
#         return ""

#     packet_parts = []

#     for packet in packet_data:
#         if not isinstance(packet, dict):
#             continue

#         size = clean_value(packet.get("size"))
#         price = clean_value(packet.get("price"))

#         if size and price:
#             packet_parts.append(f"Size {size}, price {price}")
#         elif size:
#             packet_parts.append(f"Size {size}")
#         elif price:
#             packet_parts.append(f"Price {price}")

#     return "; ".join(packet_parts)


# def extract_season_names(seasons: Any) -> list[str]:
#     if not isinstance(seasons, list):
#         return []

#     season_names = []

#     for season_entry in seasons:
#         if not isinstance(season_entry, dict):
#             continue

#         season_value = season_entry.get("season")

#         if isinstance(season_value, dict):
#             possible_names = [
#                 season_value.get("season_name"),
#                 season_value.get("name"),
#                 season_value.get("title"),
#             ]

#             for possible_name in possible_names:
#                 cleaned_name = clean_value(possible_name)

#                 if cleaned_name:
#                     season_names.append(cleaned_name)
#                     break

#         elif has_value(season_value):
#             cleaned_name = clean_value(season_value)

#             if cleaned_name:
#                 season_names.append(cleaned_name)

#     return list(dict.fromkeys(season_names))


# def crop_to_documents(crop: dict) -> list[dict]:
#     crop_id = crop.get("id")
#     crop_name = clean_value(crop.get("crop_name"))
#     crop_bangla_name = clean_value(crop.get("crop_bangla_name"))

#     documents: list[dict] = []
#     seen_document_ids: set[str] = set()

#     base_metadata = {
#         "crop_id": crop_id,
#         "crop_name": crop_name,
#         "crop_bangla_name": crop_bangla_name,
#         "scientific_name": clean_value(crop.get("scientific_name")),
#         "crop_family": clean_value(crop.get("crop_family")),
#     }

#     def add_document(
#         content: Any,
#         section: str,
#         source_key: Any = "root",
#         extra_metadata: dict | None = None,
#         max_chars: int = MAX_DOCUMENT_CHARS,
#     ) -> None:
#         cleaned_content = clean_value(content)

#         if not cleaned_content:
#             return

#         source_identifier = safe_identifier(source_key)
#         parent_document_id = (
#             f"crop:{crop_id}:{safe_identifier(section)}:{source_identifier}"
#         )

#         chunks = split_text(
#             cleaned_content,
#             max_chars=max_chars,
#             overlap_chars=CHUNK_OVERLAP_CHARS,
#         )

#         if not chunks:
#             return

#         total_chunks = len(chunks)

#         for chunk_index, chunk in enumerate(chunks, start=1):
#             document_id = (
#                 f"{parent_document_id}:chunk:{chunk_index:03d}"
#             )

#             if document_id in seen_document_ids:
#                 continue

#             metadata = {
#                 **base_metadata,
#                 "section": section,
#                 "parent_document_id": parent_document_id,
#                 "document_id": document_id,
#                 "chunk_index": chunk_index,
#                 "chunk_count": total_chunks,
#             }

#             if extra_metadata:
#                 metadata.update(
#                     {
#                         key: value
#                         for key, value in extra_metadata.items()
#                         if has_value(value)
#                     }
#                 )

#             header_parts = [
#                 f"Crop: {crop_name}" if crop_name else "",
#                 (
#                     f"Crop Bangla name: {crop_bangla_name}"
#                     if crop_bangla_name
#                     else ""
#                 ),
#                 f"Section: {section}",
#             ]

#             text = join_parts(
#                 [
#                     join_parts(header_parts),
#                     chunk,
#                 ]
#             )

#             documents.append(
#                 {
#                     "text": text,
#                     "metadata": metadata,
#                 }
#             )

#             seen_document_ids.add(document_id)

#     general_information_parts = []

#     if has_value(crop.get("general_info")):
#         general_information_parts.append(
#             clean_html(crop.get("general_info"))
#         )

#     if has_value(crop.get("scientific_name")):
#         general_information_parts.append(
#             f"Scientific name: {clean_value(crop.get('scientific_name'))}"
#         )

#     if has_value(crop.get("crop_family")):
#         general_information_parts.append(
#             f"Crop family: {clean_value(crop.get('crop_family'))}"
#         )

#     if has_value(crop.get("average_production")):
#         general_information_parts.append(
#             "Average production value from source: "
#             f"{clean_value(crop.get('average_production'))}"
#         )

#     add_document(
#         join_parts(general_information_parts),
#         section="general_info",
#     )

#     simple_sections = {
#         "harvest": "harvest",
#         "intercultural": "intercultural",
#         "irrigation": "irrigation",
#         "landPreparation": "land_preparation",
#     }

#     for source_field, section_name in simple_sections.items():
#         section_data = crop.get(source_field)

#         if isinstance(section_data, dict):
#             add_document(
#                 section_data.get("description"),
#                 section=section_name,
#                 source_key=section_data.get("id", "root"),
#                 extra_metadata={
#                     "source_record_id": section_data.get("id"),
#                 },
#             )

#     fertilizer = crop.get("fertilizer")

#     if isinstance(fertilizer, dict):
#         fertilizer_id = fertilizer.get("id", "root")

#         add_document(
#             fertilizer.get("fertilizer"),
#             section="fertilizer",
#             source_key=fertilizer_id,
#             extra_metadata={
#                 "source_record_id": fertilizer.get("id"),
#             },
#         )

#     seed = crop.get("seed")

#     if isinstance(seed, dict):
#         seed_id = seed.get("id", "root")

#         if has_value(seed.get("seed_rate")):
#             add_document(
#                 (
#                     "Seed-rate value: "
#                     f"{clean_value(seed.get('seed_rate'))}. "
#                     "The source does not provide a reliable unit or area basis "
#                     "for this structured value."
#                 ),
#                 section="seed_rate",
#                 source_key=f"{seed_id}-rate",
#                 extra_metadata={
#                     "seed_id": seed.get("id"),
#                     "seed_rate_raw": seed.get("seed_rate"),
#                     "seed_rate_unit_status": "missing_or_ambiguous",
#                 },
#             )

#         seed_fields = {
#             "treatment": "seed_treatment",
#             "showing_method": "seed_sowing_method",
#             "time_showing": "seed_sowing_time",
#             "seedbed": "seedbed",
#         }

#         for source_field, section_name in seed_fields.items():
#             add_document(
#                 seed.get(source_field),
#                 section=section_name,
#                 source_key=f"{seed_id}-{source_field}",
#                 extra_metadata={
#                     "seed_id": seed.get("id"),
#                 },
#             )

#     climate = crop.get("climate")

#     if isinstance(climate, dict):
#         climate_id = climate.get("id", "root")

#         add_document(
#             climate.get("general_info"),
#             section="climate_guidance",
#             source_key=f"{climate_id}-guidance",
#             extra_metadata={
#                 "climate_id": climate.get("id"),
#             },
#         )

#         climate_requirements = []

#         temperature_start = clean_value(
#             climate.get("climate_temperature_start")
#         )
#         temperature_end = clean_value(
#             climate.get("climate_temperature_end")
#         )

#         if temperature_start or temperature_end:
#             climate_requirements.append(
#                 "Temperature source values: "
#                 f"{temperature_start or 'not provided'}"
#                 f" to {temperature_end or 'not provided'}"
#             )

#         rainfall_start = clean_value(
#             climate.get("climate_rainfall_start")
#         )
#         rainfall_end = clean_value(
#             climate.get("climate_rainfall_end")
#         )

#         if rainfall_start or rainfall_end:
#             climate_requirements.append(
#                 "Rainfall source values: "
#                 f"{rainfall_start or 'not provided'}"
#                 f" to {rainfall_end or 'not provided'}"
#             )

#         ph_start = clean_value(climate.get("climate_ph_start"))
#         ph_end = clean_value(climate.get("climate_ph_end"))

#         if ph_start or ph_end:
#             climate_requirements.append(
#                 f"Soil pH range: {ph_start or 'not provided'}"
#                 f" to {ph_end or 'not provided'}"
#             )

#         humidity_start = clean_value(climate.get("climate_humidity"))
#         humidity_end = clean_value(
#             climate.get("climate_humidity_end")
#         )

#         if humidity_start or humidity_end:
#             climate_requirements.append(
#                 "Humidity source values: "
#                 f"{humidity_start or 'not provided'}"
#                 f" to {humidity_end or 'not provided'}"
#             )

#         ec_start = clean_value(climate.get("climate_ec_start"))
#         ec_end = clean_value(climate.get("climate_ec_end"))

#         if ec_start or ec_end:
#             climate_requirements.append(
#                 f"EC source values: {ec_start or 'not provided'}"
#                 f" to {ec_end or 'not provided'}"
#             )

#         salinity_start = clean_value(climate.get("salinity_start"))
#         salinity_end = clean_value(climate.get("salinity_end"))

#         if salinity_start or salinity_end:
#             climate_requirements.append(
#                 "Salinity source values: "
#                 f"{salinity_start or 'not provided'}"
#                 f" to {salinity_end or 'not provided'}"
#             )

#         if has_value(climate.get("land_type")):
#             climate_requirements.append(
#                 f"Land type reference ID: {climate.get('land_type')}"
#             )

#         if has_value(climate.get("soil_texture")):
#             climate_requirements.append(
#                 "Soil texture reference ID: "
#                 f"{climate.get('soil_texture')}"
#             )

#         add_document(
#             join_parts(climate_requirements),
#             section="climate_requirements",
#             source_key=f"{climate_id}-requirements",
#             extra_metadata={
#                 "climate_id": climate.get("id"),
#                 "land_type_id": climate.get("land_type"),
#                 "soil_texture_id": climate.get("soil_texture"),
#             },
#         )

#     infestation_guidelines = crop.get(
#         "cropInfestationGuidelines"
#     )

#     if has_value(infestation_guidelines):
#         add_document(
#             join_parts(flatten_data(infestation_guidelines)),
#             section="infestation_guidelines",
#             source_key="infestation-guidelines",
#         )

#     for cost in crop.get("cropAdditionalCostInfo") or []:
#         if not isinstance(cost, dict):
#             continue

#         cost_id = cost.get("id", "root")
#         cost_type = clean_value(cost.get("cost_type"))
#         amount = clean_value(cost.get("amount"))

#         unit_info = cost.get("unitInfo")
#         unit_name = ""

#         if isinstance(unit_info, dict):
#             unit_name = (
#                 clean_value(unit_info.get("unit_name"))
#                 or clean_value(unit_info.get("unit_code"))
#             )

#         cost_parts = []

#         if cost_type:
#             cost_parts.append(f"Cost type: {cost_type}")

#         if amount:
#             cost_parts.append(
#                 f"Amount: {amount}{f' per {unit_name}' if unit_name else ''}"
#             )

#         if not unit_name and has_value(cost.get("unit")):
#             cost_parts.append(
#                 f"Unit reference ID: {cost.get('unit')}"
#             )

#         add_document(
#             join_parts(cost_parts),
#             section="crop_cost",
#             source_key=cost_id,
#             extra_metadata={
#                 "cost_id": cost.get("id"),
#                 "cost_type": cost_type,
#                 "unit_name": unit_name,
#                 "unit_id": cost.get("unit"),
#             },
#         )

#     for variety in crop.get("variety") or []:
#         if not isinstance(variety, dict):
#             continue

#         variety_id = variety.get("id", "root")
#         variety_name = (
#             clean_value(variety.get("variety_name"))
#             or "Unnamed variety"
#         )

#         variety_metadata = {
#             "variety_id": variety.get("id"),
#             "variety_name": variety_name,
#             "is_verified": variety.get("is_verified"),
#             "company_id": variety.get("company_id"),
#         }

#         overview_parts = [f"Variety name: {variety_name}"]

#         company_name = clean_value(variety.get("company_name"))

#         if company_name:
#             overview_parts.append(f"Company: {company_name}")

#         season_names = extract_season_names(
#             variety.get("seasons")
#         )

#         if season_names:
#             overview_parts.append(
#                 f"Declared seasons: {', '.join(season_names)}"
#             )
#             variety_metadata["declared_seasons"] = season_names

#         if has_value(variety.get("rating")):
#             overview_parts.append(
#                 f"Rating: {clean_value(variety.get('rating'))}"
#             )

#         if has_value(variety.get("price")):
#             overview_parts.append(
#                 "Price value from source: "
#                 f"{clean_value(variety.get('price'))}"
#             )

#         add_document(
#             join_parts(overview_parts),
#             section="variety_overview",
#             source_key=f"{variety_id}-overview",
#             extra_metadata=variety_metadata,
#         )

#         duration_start = clean_value(
#             variety.get("duration_start")
#         )
#         duration_end = clean_value(
#             variety.get("duration_end")
#         )

#         duration_parts = [f"Variety name: {variety_name}"]

#         if duration_start or duration_end:
#             duration_parts.append(
#                 f"Duration: {duration_start or 'not provided'}"
#                 f" to {duration_end or 'not provided'} days"
#             )

#         if has_value(variety.get("variety_duration")):
#             duration_parts.append(
#                 "Additional duration value: "
#                 f"{clean_value(variety.get('variety_duration'))}"
#             )

#         if len(duration_parts) > 1:
#             add_document(
#                 join_parts(duration_parts),
#                 section="variety_duration",
#                 source_key=f"{variety_id}-duration",
#                 extra_metadata=variety_metadata,
#             )

#         yield_parts = [f"Variety name: {variety_name}"]

#         if has_value(variety.get("avg_expected_yield")):
#             yield_parts.append(
#                 "Average expected yield value: "
#                 f"{clean_value(variety.get('avg_expected_yield'))}. "
#                 "The unit is not explicitly provided by this field."
#             )

#         if has_value(variety.get("variety_yield")):
#             yield_parts.append(
#                 "Variety yield value: "
#                 f"{clean_value(variety.get('variety_yield'))}"
#             )

#         if has_value(variety.get("yield_low")):
#             yield_parts.append(
#                 f"Lower yield value: {clean_value(variety.get('yield_low'))}"
#             )

#         if has_value(variety.get("yield_up")):
#             yield_parts.append(
#                 f"Upper yield value: {clean_value(variety.get('yield_up'))}"
#             )

#         production_unit = nested_unit_name(
#             variety,
#             nested_key="productionUnit",
#             fallback_key="production_unit",
#         )

#         if has_value(variety.get("production")):
#             yield_parts.append(
#                 "Production: "
#                 + format_quantity(
#                     variety.get("production"),
#                     production_unit,
#                     "unit not provided",
#                 )
#             )

#         if len(yield_parts) > 1:
#             add_document(
#                 join_parts(yield_parts),
#                 section="variety_yield",
#                 source_key=f"{variety_id}-yield",
#                 extra_metadata=variety_metadata,
#             )

#         if has_value(variety.get("seed_rate")):
#             seed_rate_unit = nested_unit_name(
#                 variety,
#                 nested_key="seedRateUnit",
#                 fallback_key="seed_rate_unit_name",
#             )

#             seed_rate_text = join_parts(
#                 [
#                     f"Variety name: {variety_name}",
#                     (
#                         "Seed-rate value: "
#                         + format_quantity(
#                             variety.get("seed_rate"),
#                             seed_rate_unit,
#                         )
#                     ),
#                     (
#                         "Area basis: not provided by the structured "
#                         "source field."
#                     ),
#                 ]
#             )

#             add_document(
#                 seed_rate_text,
#                 section="variety_seed_rate",
#                 source_key=f"{variety_id}-seed-rate",
#                 extra_metadata={
#                     **variety_metadata,
#                     "seed_rate_raw": variety.get("seed_rate"),
#                     "seed_rate_unit": seed_rate_unit,
#                     "seed_rate_area_basis": "unknown",
#                 },
#             )

#         add_document(
#             join_parts(
#                 [
#                     f"Variety name: {variety_name}",
#                     clean_html(variety.get("special_character")),
#                 ]
#             ),
#             section="variety_details",
#             source_key=f"{variety_id}-details",
#             extra_metadata=variety_metadata,
#         )

#         add_document(
#             join_parts(
#                 [
#                     f"Variety name: {variety_name}",
#                     clean_html(variety.get("time_showing")),
#                 ]
#             ),
#             section="variety_sowing_time",
#             source_key=f"{variety_id}-sowing-time",
#             extra_metadata=variety_metadata,
#         )

#         if has_value(variety.get("variety_seed")):
#             add_document(
#                 join_parts(
#                     [
#                         f"Variety name: {variety_name}",
#                         join_parts(
#                             flatten_data(variety.get("variety_seed"))
#                         ),
#                     ]
#                 ),
#                 section="variety_seed_information",
#                 source_key=f"{variety_id}-seed-information",
#                 extra_metadata=variety_metadata,
#             )

#     for pesticide in crop.get("pesticide") or []:
#         if not isinstance(pesticide, dict):
#             continue

#         pesticide_id = pesticide.get("id", "root")
#         disease_name = (
#             clean_value(pesticide.get("disease_name"))
#             or "Unnamed pest or disease"
#         )

#         pesticide_metadata = {
#             "pesticide_id": pesticide.get("id"),
#             "infestation_id": pesticide.get("infestation_id"),
#             "disease_name": disease_name,
#             "disease_type": clean_value(
#                 pesticide.get("disease_type")
#             ),
#             "priority": pesticide.get("priority"),
#             "is_verified": pesticide.get("is_verified"),
#         }

#         overview_parts = [
#             f"Problem name: {disease_name}",
#         ]

#         if has_value(pesticide.get("disease_type")):
#             overview_parts.append(
#                 "Problem type: "
#                 f"{clean_value(pesticide.get('disease_type'))}"
#             )

#         add_document(
#             join_parts(overview_parts),
#             section="pest_overview",
#             source_key=f"{pesticide_id}-overview",
#             extra_metadata=pesticide_metadata,
#         )

#         add_document(
#             join_parts(
#                 [
#                     f"Problem name: {disease_name}",
#                     clean_html(pesticide.get("damage_control")),
#                 ]
#             ),
#             section="pest_symptoms",
#             source_key=f"{pesticide_id}-symptoms",
#             extra_metadata=pesticide_metadata,
#         )

#         add_document(
#             join_parts(
#                 [
#                     f"Problem name: {disease_name}",
#                     clean_html(pesticide.get("control_measure")),
#                 ]
#             ),
#             section="pest_control",
#             source_key=f"{pesticide_id}-control",
#             extra_metadata=pesticide_metadata,
#         )

#         for chemical in pesticide.get("chemical") or []:
#             if not isinstance(chemical, dict):
#                 continue

#             chemical_id = chemical.get("id", "root")

#             trade_name = clean_value(
#                 chemical.get("trade_name")
#             )
#             generic_name = clean_value(
#                 chemical.get("generic_name")
#             )
#             company_name = clean_value(
#                 chemical.get("company_name")
#             )

#             application_dose_unit = nested_unit_name(
#                 chemical,
#                 nested_key="applicationDoseUnitInfo",
#                 fallback_key="application_dose_unit_name",
#             )

#             pesticide_amount_unit = nested_unit_name(
#                 chemical,
#                 nested_key="pesticideAmountUnitInfo",
#             )

#             chemical_parts = [
#                 f"Target problem: {disease_name}",
#             ]

#             if trade_name:
#                 chemical_parts.append(
#                     f"Trade name: {trade_name}"
#                 )

#             if generic_name:
#                 chemical_parts.append(
#                     f"Generic name: {generic_name}"
#                 )

#             if company_name:
#                 chemical_parts.append(
#                     f"Company: {company_name}"
#                 )

#             if has_value(chemical.get("application_dose")):
#                 chemical_parts.append(
#                     "Application dose: "
#                     + format_quantity(
#                         chemical.get("application_dose"),
#                         application_dose_unit,
#                         "unit not provided",
#                     )
#                 )

#             if has_value(chemical.get("pesticide_amount")):
#                 chemical_parts.append(
#                     "Product amount: "
#                     + format_quantity(
#                         chemical.get("pesticide_amount"),
#                         pesticide_amount_unit,
#                         "unit not provided",
#                     )
#                 )

#             if has_value(chemical.get("rating")):
#                 chemical_parts.append(
#                     f"Rating: {clean_value(chemical.get('rating'))}"
#                 )

#             if has_value(chemical.get("price")):
#                 chemical_parts.append(
#                     f"Price value: {clean_value(chemical.get('price'))}"
#                 )

#             packet_information = extract_packet_information(
#                 chemical.get("packetSizeAndPrice")
#             )

#             if packet_information:
#                 chemical_parts.append(
#                     f"Packet information: {packet_information}"
#                 )

#             application_guide = clean_html(
#                 chemical.get("application_guide")
#             )

#             if application_guide:
#                 chemical_parts.append(
#                     f"Application guide: {application_guide}"
#                 )

#             add_document(
#                 join_parts(chemical_parts),
#                 section="pesticide_chemical",
#                 source_key=chemical_id,
#                 extra_metadata={
#                     **pesticide_metadata,
#                     "chemical_id": chemical.get("id"),
#                     "trade_name": trade_name,
#                     "generic_name": generic_name,
#                     "company_id": chemical.get("company_id"),
#                     "application_dose_unit": application_dose_unit,
#                     "is_verified": chemical.get("is_verified"),
#                 },
#             )

#     for herbicide in crop.get("herbicide") or []:
#         if not isinstance(herbicide, dict):
#             continue

#         herbicide_id = herbicide.get("id", "root")
#         target_weeds = clean_value(
#             herbicide.get("pesticide_name")
#         )
#         trade_name = clean_value(herbicide.get("trade_name"))
#         generic_name = clean_value(
#             herbicide.get("generic_name")
#         )
#         company_name = clean_value(
#             herbicide.get("company_name")
#         )

#         application_dose_unit = nested_unit_name(
#             herbicide,
#             nested_key="applicationDoseUnitInfo",
#             fallback_key="application_dose_unit_name",
#         )

#         pesticide_amount_unit = nested_unit_name(
#             herbicide,
#             nested_key="pesticideAmountUnitInfo",
#         )

#         herbicide_parts = []

#         if target_weeds:
#             herbicide_parts.append(
#                 f"Target weeds: {target_weeds}"
#             )

#         if trade_name:
#             herbicide_parts.append(
#                 f"Trade name: {trade_name}"
#             )

#         if generic_name:
#             herbicide_parts.append(
#                 f"Generic name: {generic_name}"
#             )

#         if company_name:
#             herbicide_parts.append(
#                 f"Company: {company_name}"
#             )

#         if has_value(herbicide.get("application_dose")):
#             herbicide_parts.append(
#                 "Application dose: "
#                 + format_quantity(
#                     herbicide.get("application_dose"),
#                     application_dose_unit,
#                     "unit not provided",
#                 )
#             )

#         if has_value(herbicide.get("pesticide_amount")):
#             herbicide_parts.append(
#                 "Product amount: "
#                 + format_quantity(
#                     herbicide.get("pesticide_amount"),
#                     pesticide_amount_unit,
#                     "unit not provided",
#                 )
#             )

#         if has_value(herbicide.get("rating")):
#             herbicide_parts.append(
#                 f"Rating: {clean_value(herbicide.get('rating'))}"
#             )

#         if has_value(herbicide.get("price")):
#             herbicide_parts.append(
#                 f"Price value: {clean_value(herbicide.get('price'))}"
#             )

#         application_guide = clean_html(
#             herbicide.get("application_guide")
#         )

#         if application_guide:
#             herbicide_parts.append(
#                 f"Application guide: {application_guide}"
#             )

#         add_document(
#             join_parts(herbicide_parts),
#             section="herbicide_product",
#             source_key=herbicide_id,
#             extra_metadata={
#                 "herbicide_id": herbicide.get("id"),
#                 "target_weeds": target_weeds,
#                 "trade_name": trade_name,
#                 "generic_name": generic_name,
#                 "company_id": herbicide.get("company_id"),
#                 "priority": herbicide.get("priority"),
#                 "is_verified": herbicide.get("is_verified"),
#                 "application_dose_unit": application_dose_unit,
#             },
#         )

#     return documents


# def get_crop_rows(raw_data: dict) -> list[dict]:
#     try:
#         rows = raw_data["data"]["getAllCropsFullDetails"]["rows"]
#     except (KeyError, TypeError) as error:
#         raise ValueError(
#             "Could not find data.getAllCropsFullDetails.rows "
#             "inside the input JSON."
#         ) from error

#     if not isinstance(rows, list):
#         raise ValueError(
#             "data.getAllCropsFullDetails.rows must be a list."
#         )

#     return rows


# def validate_documents(documents: list[dict]) -> None:
#     seen_document_ids = set()

#     for index, document in enumerate(documents):
#         if not isinstance(document, dict):
#             raise ValueError(
#                 f"Document at index {index} is not a dictionary."
#             )

#         text = document.get("text")
#         metadata = document.get("metadata")

#         if not isinstance(text, str) or not text.strip():
#             raise ValueError(
#                 f"Document at index {index} has empty text."
#             )

#         if not isinstance(metadata, dict):
#             raise ValueError(
#                 f"Document at index {index} has invalid metadata."
#             )

#         document_id = metadata.get("document_id")

#         if not document_id:
#             raise ValueError(
#                 f"Document at index {index} has no document_id."
#             )

#         if document_id in seen_document_ids:
#             raise ValueError(
#                 f"Duplicate document_id found: {document_id}"
#             )

#         seen_document_ids.add(document_id)


# def main() -> None:
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"Input file was not found: {INPUT_FILE}"
#         )

#     with INPUT_FILE.open("r", encoding="utf-8") as file:
#         raw_data = json.load(file)

#     rows = get_crop_rows(raw_data)

#     print(f"Loaded {len(rows)} crops")

#     all_documents: list[dict] = []

#     for crop in rows:
#         if not isinstance(crop, dict):
#             continue

#         crop_documents = crop_to_documents(crop)
#         all_documents.extend(crop_documents)

#     validate_documents(all_documents)

#     OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

#     with OUTPUT_FILE.open("w", encoding="utf-8") as file:
#         json.dump(
#             all_documents,
#             file,
#             ensure_ascii=False,
#             indent=2,
#         )

#     section_counts: dict[str, int] = {}

#     for document in all_documents:
#         section = document["metadata"]["section"]
#         section_counts[section] = section_counts.get(section, 0) + 1

#     print(f"Produced {len(all_documents)} documents total")
#     print(f"Saved documents to: {OUTPUT_FILE}")
#     print("\nDocuments by section:")

#     for section, count in sorted(section_counts.items()):
#         print(f"  {section}: {count}")


# if __name__ == "__main__":
#     main()
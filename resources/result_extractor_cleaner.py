import json
import re


def extract_json_objects(response_text):
    try:
        json_objects = []
        matches = re.findall(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
        for match in matches:
            json_objects.append(json.loads(match))
        return json_objects
    except json.JSONDecodeError as e:
        return f"JSON decoding error: {e}"


def clean_text(text):
    text = str(text)
    for prefix, suffix in [("['", "']"), ('["', '"]'), ('[', ']'), ('[{', '}]')]:
        if text.startswith(prefix) and text.endswith(suffix):
            text = text[len(prefix):-len(suffix)]
    return text

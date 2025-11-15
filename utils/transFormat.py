import json
def format_json_for_prompt(data):
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
import json

def messageMerge(function_return, messages):
    tool_response_text = "\n\n"
    for item in function_return:
        tool_response_text += f"<FUNCTION>\n{item['name']}\n"
        tool_response_text += f"<ARGS>\n{json.dumps(item['args'], ensure_ascii=False, indent=2)}\n"
        tool_response_text += f"<RETURN>\n{json.dumps(item['return'], ensure_ascii=False, indent=2)}\n"
        tool_response_text += "<RESULT>"
    messages[-1]['content'] += tool_response_text

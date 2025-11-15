import regex as re
import json
import inspect

def extract_tool_calls(response):
    pattern = r"<FUNCTION>\s*(\w+)\s+<ARGS>\s*(\{.*?\})"

    matches = re.findall(pattern, response, re.DOTALL)
    tool_calls = []

    for match in matches:
        tool_name = match[0]
        args_str = match[1]

        try:
            # 确保是合法 JSON 字符串
            args = json.loads(args_str)
            tool_calls.append({
                "name": tool_name,
                "args": args
            })
        except Exception as e:
            print(f"Parameter parsing failed: {e}")
    
    return tool_calls

def has_config_parameter(func):
    return 'config' in inspect.signature(func).parameters


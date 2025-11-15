from typing import Callable, Dict, Any, Optional, List, get_type_hints
import inspect
import json

# Type mapping from Python types to JSON Schema types
TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null"
}

class FunctionRegistry:
    """
    A registry class for functions.
    Stores functions and their metadata, and can generate JSON Schema
    for use with large language models (LLMs).
    """
    def __init__(self):
        # Store registered function objects, keyed by function name
        self.functions: Dict[str, Callable] = {}
        # Store metadata for each function, including parameters, return type, description, etc.
        self.metadata: Dict[str, dict] = {}

    def register(self,
                 name: Optional[str] = None,          # Optional registration name for the function
                 description: str = "",               # Function description
                 parameters: Optional[List[Dict]] = None,  # List of parameter dicts
                 constraints: str = "",               # Parameter constraints or notes
                 returns: Optional[Dict] = None):    # Return value schema
        """
        Decorator method for registering a function and its metadata.
        """
        def decorator(func: Callable):
            nonlocal name
            if name is None:
                name = func.__name__  # Default to function's actual name

            # Build the parameters JSON Schema
            properties = {}  # Dict to store parameter type and description
            required = []    # List of required parameter names
            for p in parameters:
                param_name = p["name"]
                properties[param_name] = {
                    "type": p["type"],                # Parameter type
                    "description": p.get("description"),  # Parameter description
                }
                if p.get("required", True):  # Default is required
                    required.append(param_name)

            schema = {
                "type": "object",     # Parameters are represented as an object
                "properties": properties,
                "required": required
            }

            # Handle return value information
            if returns:
                return_info = returns  # Use user-provided return info if available
            else:
                # Infer return type from type hints
                return_type = TYPE_MAP.get(get_type_hints(func).get('return'), 'object')
                return_info = {
                    "type": return_type
                }

            # Register the function object
            self.functions[name] = func
            # Register the function metadata
            self.metadata[name] = {
                "name": name,              # Function name
                "description": description,  # Description
                "parameters": schema,        # Parameter JSON Schema
                "constraints": constraints,  # Constraints or additional notes
                "returns": return_info       # Return value schema
            }
            return func  # Return the original function
        return decorator

    def get_function(self, name: str) -> Callable:
        """Get the function object by name"""
        return self.functions.get(name)

    def get_metadata(self, name: str) -> dict:
        """Get the metadata for a function by name"""
        return self.metadata.get(name)

    def list_functions(self) -> list:
        """List all registered function names"""
        return list(self.functions.keys())

    def export_tool_schemas(self) -> list:
        """
        Export all registered functions in the "tool_call" format for LLMs.
        Returns a list of dicts like:
        [
            {
                "type": "function",
                "function": {
                    "name": ...,
                    "description": ...,
                    "constraints": ...,
                    "parameters": {...},
                    "returns": {...}
                }
            },
            ...
        ]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": meta["name"],
                    "description": meta["description"],
                    "constraints": meta["constraints"],
                    "parameters": meta["parameters"],
                    "returns": meta["returns"]
                }
            }
            for meta in self.metadata.values()
        ]

# Instantiate the function registry
function_register = FunctionRegistry()

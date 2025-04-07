import torch
import re
from pprint import pprint
from typing import Dict, Any, List, Tuple, Optional

dict_profile2torchdtype = {
    'c10::BFloat16': torch.bfloat16,
    'c10::Half': torch.half,
    'c10::Float': torch.float,
    'c10::Long': torch.long,
}

def parse_bool(s: str) -> bool:
    return s.lower() == 'true' if s else False

def parse_int(s: str) -> int:
    return int(s) if s else 0

def parse_float(s: str) -> float:
    return float(s) if s else 0.0

def parse_symint_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.strip('[]').split(',')] if s else []

def parse_bool_list(s: str) -> List[bool]:
    return [parse_bool(x.strip()) for x in s.strip('[]').split(',')] if s else []

def build_tensor(shape: Tuple[int, ...], stride: Tuple[int, ...], dtype_str: str, device: str = 'cpu') -> torch.Tensor:
    dtype = dict_profile2torchdtype.get(dtype_str, torch.float)
    return torch.randn(shape, dtype=dtype, device=device).as_strided(size=shape, stride=stride)

def summarize(val: Any) -> str:
    if isinstance(val, torch.Tensor):
        return f"Tensor(shape={tuple(val.shape)}, dtype={val.dtype}, stride={tuple(val.stride())}, device={val.device})"
    return str(val)

def get_schemas(op_name: str) -> List[torch._C.FunctionSchema]:
    schemas = torch._C._jit_get_all_schemas()
    return [s for s in schemas if s.name == op_name]

def parse_schema_string(schema_str: str) -> Tuple[str, List[Tuple[str, str, Optional[str], bool]], str]:
    match = re.match(r'^([^\(]+)\((.*)\)\s*->\s*(.*)$', schema_str.strip())
    if not match:
        raise ValueError(f"Cannot parse schema string: {schema_str}")
    op_name, args_str, return_type = match.groups()
    parts = args_str.split('*')
    pos_part = parts[0].rstrip(',').strip()
    kwarg_part = parts[1].lstrip(',').strip() if len(parts) > 1 else ""

    def _parse_args(raw_args: str, kwarg_only: bool) -> List[Tuple[str, str, Optional[str], bool]]:
        args = []
        for item in [x.strip() for x in raw_args.split(',') if x.strip()]:
            m = re.match(r'^(\S+)\s+(.*)$', item)
            if not m:
                raise ValueError(f"Invalid arg: {item}")
            arg_type, rest = m.groups()
            m2 = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$', rest)
            if not m2:
                raise ValueError(f"Invalid arg name/default: {rest}")
            arg_name, default = m2.group(1), m2.group(2).strip() if m2.group(2) else None
            args.append((arg_type, arg_name, default, kwarg_only))
        return args

    return op_name, _parse_args(pos_part, False) + _parse_args(kwarg_part, True), return_type.strip()

def _match_schema(event: Dict[str, Any], schemas: List[torch._C.FunctionSchema]) -> Optional[torch._C.FunctionSchema]:
    args_dict = event.get('args', {})
    input_types = args_dict.get('Input type', [])
    concrete_inpts = args_dict.get('Concrete Inputs', [])

    num_tensor_inputs = sum(1 for type_str in input_types if type_str in ['float', 'bfloat16', 'half', 'long'])
    num_concrete_inputs = len([val for val in concrete_inpts if val != ''])

    for schema in schemas:
        schema_str = str(schema)
        _, parsed_args, _ = parse_schema_string(schema_str)
        num_tensor_args = sum(1 for arg_type, _, _, _ in parsed_args if "Tensor" in arg_type)
        num_other_args = len(parsed_args) - num_tensor_args
        if num_tensor_args == num_tensor_inputs and num_other_args <= len(concrete_inpts):
            return schema
    return None

def _parse_and_build_arguments(event: Dict[str, Any], schema: torch._C.FunctionSchema, device: str = 'cpu', debug: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
    schema_str = str(schema)
    _, parsed_args, _ = parse_schema_string(schema_str)
    args_dict = event.get('args', {})
    input_dims = args_dict.get('Input Dims', [])
    input_strides = args_dict.get('Input Strides', [])
    concrete_inpts = args_dict.get('Concrete Inputs', [])
    input_types = args_dict.get('Input type', [])

    positional_values = []
    keyword_values = {}
    arg_idx = 0  # Single index for all arguments

    for arg_type, arg_name, default_str, kwarg_only in parsed_args:
        if debug:
            print(f"  Processing argument: {arg_type} {arg_name}, default={default_str}, kwarg_only={kwarg_only}")

        if arg_idx >= len(input_dims):
            val = None
            if debug:
                print(f"    No input data for argument {arg_name}")
        elif "Tensor" in arg_type:
            if input_dims[arg_idx]:
                shape = tuple(input_dims[arg_idx])
                stride = tuple(input_strides[arg_idx]) if arg_idx < len(input_strides) else None
                dtype_str = input_types[arg_idx] if arg_idx < len(input_types) else 'float'
                val = build_tensor(shape, stride, dtype_str, device)
                if debug:
                    print(f"    Built tensor: {summarize(val)}")
            else:
                val = None
                if debug:
                    print("    No tensor data found.")
        else:
            raw_val = concrete_inpts[arg_idx] if arg_idx < len(concrete_inpts) else None
            if raw_val and raw_val != '':
                try:
                    if arg_type.lower() == "bool":
                        val = parse_bool(raw_val)
                    elif arg_type.lower().startswith("bool["):
                        val = parse_bool_list(raw_val)
                    elif arg_type.lower() in ("int", "symint"):
                        val = parse_int(raw_val)
                    elif arg_type.lower() in ("float", "scalar"):
                        val = parse_float(raw_val)
                    elif "[]" in arg_type.lower():
                        val = parse_symint_list(raw_val)
                    else:
                        val = raw_val
                    if debug:
                        print(f"    Parsed value: {arg_name}={summarize(val)} from '{raw_val}'")
                except Exception as e:
                    if debug:
                        print(f"    Error parsing argument {arg_name}: {e}")
                    val = None
            elif default_str is not None:
                try:
                    val = float(default_str) if '.' in default_str else int(default_str)
                except ValueError:
                    val = default_str
                if debug:
                    print(f"    Using default value: {arg_name}={summarize(val)}")
            else:
                val = None
                if debug:
                    print(f"    No concrete input or default for {arg_name}")

        if not kwarg_only:
            positional_values.append(val)
        else:
            keyword_values[arg_name] = val
        arg_idx += 1

    return positional_values, keyword_values

def _call_op(op_name: str, positional_args: List[Any], keyword_args: Dict[str, Any]) -> Any:
    namespace, func_name = op_name.split("::", 1)
    op_callable = getattr(torch.ops.aten, func_name)
    return op_callable(*positional_args, **keyword_args)

def replay_op(event: Dict[str, Any], device: str = 'cpu', debug: bool = False) -> Optional[torch.Tensor]:
    op_name = event['name']
    namespace, func_name = op_name.split("::", 1)
    all_schemas = get_schemas(op_name)

    if debug:
        print(f"Attempting to replay operation: {op_name}")
        pprint(event)
        print(f"Available schemas: {[str(s) for s in all_schemas]}")

    matched_schema = _match_schema(event, all_schemas)
    if not matched_schema:
        print(f"Error: No matching schema found for {op_name} and event arguments.")
        return None

    if debug:
        print(f"Matched schema: {matched_schema}")

    try:
        positional_args, keyword_args = _parse_and_build_arguments(event, matched_schema, device, debug)
        if debug:
            print(f"\nCalling {op_name} with:")
            print(f"  Positional Args: {[summarize(arg) for arg in positional_args]}")
            print(f"  Keyword Args: {{ {', '.join(f'{k}: {summarize(v)}' for k, v in keyword_args.items())} }}")

        result = _call_op(op_name, positional_args, keyword_args)
        if debug:
            print("\nReplay Result:")
            if isinstance(result, torch.Tensor):
                print(f"Tensor(shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device})")
            else:
                print(summarize(result))
        return result
    except Exception as e:
        print(f"Error replaying operation {op_name}: {e}")
        return None

# Test with your event
event = {
    'UID': 8,
    'args': {
        'Concrete Inputs': ['', '', '', '1', '1', ''],
        'Ev Idx': 8,
        'External id': 9,
        'Fwd thread id': 0,
        'Input Dims': [[4096, 16384], [4096, 2048], [2048, 16384], [], [], [4096, 16384]],
        'Input Strides': [[16384, 1], [2048, 1], [16384, 1], [], [], [16384, 1]],
        'Input type': ['float', 'float', 'float', 'Scalar', 'Scalar', 'float'],
        'Record function id': 0,
        'Sequence number': 0
    },
    'cat': 'cpu_op',
    'children': [58, 60, 62],
    'cpu_op_root': True,
    'dur': 9.534,
    'gpu_events': [104, 106],
    'name': 'aten::addmm',
    'ph': 'X',
    'pid': 18819,
    't_end': 7631570493681.158,
    'tid': 18819,
    'tree': True,
    'ts': 7631570493671.624
}

result = replay_op(event, debug=True)
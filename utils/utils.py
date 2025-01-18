import importlib
#从filepath返回函数名为func_name的函数
import os

def import_function_from_path(filepath: str, func_name: str):
    module_name = os.path.basename(filepath).rstrip(".py")

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, func_name, None)
    if function is None:
        raise ImportError(f"Function {func_name} not found in {filepath}")

    return function
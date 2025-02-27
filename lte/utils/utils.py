import importlib
import torch
import os


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def import_function_from_path(filepath: str, func_name: str):
    module_name = os.path.basename(filepath).rstrip(".py")

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, func_name, None)
    if function is None:
        raise ImportError(f"Function {func_name} not found in {filepath}")

    return function
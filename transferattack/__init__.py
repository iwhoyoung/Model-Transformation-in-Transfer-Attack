import importlib


attack_zoo = {
    "mota": (".input_transformation.mota", "MoTA"),
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise ValueError("Unsupported attack algorithm {}".format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    return getattr(module, class_name)


__version__ = "1.0.0"

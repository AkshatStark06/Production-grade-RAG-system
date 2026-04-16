import yaml


def load_config(path):
    """
    Generic config loader for YAML files.
    Always requires explicit path for clarity.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config
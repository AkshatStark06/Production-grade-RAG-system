import yaml


def load_config(path=None):
    """
    Generic config loader for YAML files.
    Always requires explicit path for clarity.
    """
    if path is None:
        path = "config/settings.yaml"

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config
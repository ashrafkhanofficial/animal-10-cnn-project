import yaml

def load_config(config_path):
    """
    load a YAML configuration file.

    Args:
        config_path(str): Path to the YAML configuration file.

    returns:
        dict: Configurations as python dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
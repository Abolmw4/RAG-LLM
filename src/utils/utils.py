import yaml

def load_yaml_config_file(config_src: str):
    try:
        with open(config_src, 'r', encoding='utf-8') as file:
            configs = yaml.safe_load(file)
            return configs
    except FileNotFoundError:
        print(f"Error: The file at {config_src} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")

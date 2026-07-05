from datetime import datetime
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

class Logger:
    def __init__(self, file_path=None):
        self.file_path = file_path

    def _write(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] [{level}] {message}"

        print(text)

        if self.file_path:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def info(self, message):
        self._write("INFO", message)

    def warning(self, message):
        self._write("WARNING", message)

    def error(self, message):
        self._write("ERROR", message)

    def debug(self, message):
        self._write("DEBUG", message)



logger = Logger(file_path="logs/rag-llm.log")
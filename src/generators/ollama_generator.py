from src.generators.generator import Generator
from ollama import Client

class OllamaGenerator(Generator):
    def __init__(self, model_name: str="gemma3:4B", host: str="http://localhost:11434"):
        super().__init__()
        self.client = Client(host=host)
        self.model = model_name
        
    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError(f"prompt not be empty")

        response = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

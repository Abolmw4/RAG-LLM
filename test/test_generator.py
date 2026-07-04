import unittest
from src.generators.generator import Generator
from src.generators.ollama_generator import OllamaGenerator


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.generator: Generator = OllamaGenerator(model_name="gemma3:4B")
    
    def test_ollam_generator(self):
        result = self.generator.generate(prompt="سلام داداش خوبی؟")        
        self.assertFalse(not result.strip())
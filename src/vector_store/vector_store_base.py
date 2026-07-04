from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def add(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def search(self):
        pass

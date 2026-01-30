from abc import ABC, abstractmethod

class BaseGeneralWrapper(ABC):
    
    @abstractmethod
    def extract_entities(self):
        pass
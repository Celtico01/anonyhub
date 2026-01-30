from abc import ABC, abstractmethod

class BaseModelNERClass(ABC):
    @abstractmethod
    def extract_entities(self, text, labels):
        pass
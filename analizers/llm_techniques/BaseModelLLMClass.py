from abc import ABC, abstractmethod

class BaseModelLLMClass(ABC):
    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    @abstractmethod
    def extract_entities(self, text, labels=None):
        pass
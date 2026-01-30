from gliner import GLiNER
from BaseModelWrapper import BaseGeneralWrapper

class BaseGlinerWrapper(BaseGeneralWrapper):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = GLiNER.from_pretrained(self.model_name) 

    
    def extract_entities(self, text, labels : list):
        return super().extract_entities()
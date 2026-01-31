from gliner import GLiNER
from BaseModelWrapper import BaseGeneralWrapper

class BaseGlinerWrapper(BaseGeneralWrapper):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = GLiNER.from_pretrained(self.model_name) 

    def extract_entities(self, text, labels: list, threshold: float):
        """(text, label, start, end, score)"""
        entities = self.model.predict_entities(text=text, labels=labels, threshold=threshold)
        
        return [
            (ent['text'], ent['label'], ent['start'], ent['end'], ent['score']) 
            for ent in entities
        ]
    
class Gliner1MultiV21(BaseGeneralWrapper):

    def __init__(self):
        self.model_name = 'urchade/gliner_multi-v2.1'
        self.model = GLiNER.from_pretrained(self.model_name) 

    def extract_entities(self, text, labels: list, threshold: float):
        """(text, label, start, end, score)"""
        entities = self.model.predict_entities(text=text, labels=labels, threshold=threshold)
        
        return [
            (ent['text'], ent['label'], ent['start'], ent['end'], ent['score']) 
            for ent in entities
        ]
    
class Gliner2MultiV1(BaseGeneralWrapper):

    def __init__(self):
        self.model_name = 'fastino/gliner2-multi-v1'
        self.model = GLiNER.from_pretrained(self.model_name) 

    def extract_entities(self, text, labels: list, threshold: float):
        """(text, label, start, end, score)"""
        entities = self.model.predict_entities(text=text, labels=labels, threshold=threshold)
        
        return [
            (ent['text'], ent['label'], ent['start'], ent['end'], ent['score']) 
            for ent in entities
        ]
    
class Gliner1NvidiaPII(BaseGeneralWrapper):

    def __init__(self):
        self.model_name = 'nvidia/gliner-PII'
        self.model = GLiNER.from_pretrained(self.model_name) 

    def extract_entities(self, text, labels: list, threshold: float):
        """(text, label, start, end, score)"""
        entities = self.model.predict_entities(text=text, labels=labels, threshold=threshold)
        
        return [
            (ent['text'], ent['label'], ent['start'], ent['end'], ent['score']) 
            for ent in entities
        ]
    
class Gliner1MultiPII(BaseGeneralWrapper):

    def __init__(self):
        self.model_name = 'urchade/gliner_multi_pii-v1'
        self.model = GLiNER.from_pretrained(self.model_name) 

    def extract_entities(self, text, labels: list, threshold: float):
        """(text, label, start, end, score)"""
        entities = self.model.predict_entities(text=text, labels=labels, threshold=threshold)
        
        return [
            (ent['text'], ent['label'], ent['start'], ent['end'], ent['score']) 
            for ent in entities
        ]
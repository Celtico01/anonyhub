import torch
from transformers import pipeline
from ..wrapper.BaseModelWrapper import BaseGeneralWrapper

class BaseHuggingFaceWrapper(BaseGeneralWrapper):
    def __init__(self, model_name, device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
            
        self.model_name = model_name
        self.nlp = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple", 
            device=device
        )
        
        self.supported_labels = list(self.nlp.model.config.label2id.keys())
        self.max_length = self.nlp.model.config.max_position_embeddings

    def extract_entities(self, text, labels=None, stride=64):
        if labels:
            for label in labels:
                if label not in self.supported_labels:
                    raise ValueError(f"Label {label} não suportada. Labels do modelo: {self.supported_labels}")

        predictions = self.nlp(
            text, 
            stride=stride, 
            max_length=self.max_length
        )
        
        results = []
        for pred in predictions:
            label = pred['entity_group']
            if labels and label not in labels:
                continue
            try:
                results.append((pred['word'], label, pred['start'], pred['end'], round(float(pred['score']), 4)))
            except:
                results.append((pred['word'], label, pred['start'], pred['end']))
            
        return results
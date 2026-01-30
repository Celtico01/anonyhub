import spacy
from BaseModelWrapper import BaseGeneralWrapper

class BaseSpacyWrapper(BaseGeneralWrapper):
    def __init__(self, model_name):
        try:
            self.model_name = model_name
            self.model = spacy.load(self.model_name)
            self.supported_labels = self.model.get_pipe("ner").labels
        except OSError:
            spacy.cli.download(model_name)
            self.model = spacy.load(self.model_name)
            self.supported_labels = self.model.get_pipe("ner").labels
            

    def extract_entities(self, text, labels : list):
        for label in labels:
                if label not in self.supported_labels:
                    raise ValueError(f"Label not supported! label:{label}, supported labels for the model {self.model_name}: {self.supported_labels}")
        doc = self.model(text)
        
        return [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]
    
class SpacyPTNewsLG(BaseGeneralWrapper):
     def __init__(self):
        try:
            self.model_name = 'pt_core_news_lg'
            self.model = spacy.load(self.model_name)
            self.supported_labels = self.model.get_pipe("ner").labels
        except OSError:
            spacy.cli.download(self.model_name)
            self.model = spacy.load(self.model_name)
            self.supported_labels = self.model.get_pipe("ner").labels
     
     def extract_entities(self, text, labels):
        for label in labels:
            if label not in self.supported_labels:
                raise ValueError(f"Label not supported! label:{label}, supported labels for the model {self.model_name}: {self.supported_labels}")
        
        doc = self.model(text)

        return [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]


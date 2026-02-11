from BaseModelNERClass import BaseHuggingFaceWrapper

class LenerBRWrapper(BaseHuggingFaceWrapper):
    def __init__(self, device=None):
        super().__init__("pierreguillou/ner-bert-large-cased-pt-lenerbr", device)
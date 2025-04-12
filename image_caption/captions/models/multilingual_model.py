from transformers import pipeline

class MultilingualTranslator:
    def __init__(self):
        self.translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

    def translate(self, text, target_language):
        return self.translator(text, src_lang="en_XX", tgt_lang=target_language)[0]['translation_text']

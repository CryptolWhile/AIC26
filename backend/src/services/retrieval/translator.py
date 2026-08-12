from deep_translator import GoogleTranslator

class Translator():
    def __init__(self, from_lang='vi', to_lang='en'):
        self.__from_lang = from_lang
        self.__to_lang = to_lang
        
        # Initialize Google Translator from deep_translator
        self.model = GoogleTranslator(source=from_lang, target=to_lang)

    def preprocessing(self, text):
        return text.strip()

    def __call__(self, text):
        try:
            text = self.preprocessing(text)
            if not text:
                return text
            
            # Use deep_translator's GoogleTranslator
            translated = self.model.translate(text)
            return translated if translated else text
            
        except Exception as e:
            # Return original text if translation fails
            print(f"Translation failed: {e}")
            return text
# src/my_nlp_framework/core/tokenizer.py
class Tokenizer:
    def __init__(self, method='word'):
        self.method = method

    def tokenize(self, text):
        if self.method == 'word':
            return text.split()
        elif self.method == 'char':
            return list(text)
        # Add more methods if needed

    def detokenize(self, tokens):
        if self.method == 'word':
            return ' '.join(tokens)
        elif self.method == 'char':
            return ''.join(tokens)
        # Add more methods if needed

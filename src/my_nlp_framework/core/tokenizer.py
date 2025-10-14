class Tokenizer:
    def __init__(self, method='word'):
        self.method = method

    def tokenize(self, text):
        if self.method == 'word':
            return text.split()
        elif self.method == 'char':
            return list(text)

    def detokenize(self, tokens):
        if self.method == 'word':
            return ' '.join(tokens)
        elif self.method == 'char':
            return ''.join(tokens)
   

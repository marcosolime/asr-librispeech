from torchtext.data.utils import get_tokenizer
import itertools

"""
Word models must implement two methods

    - text_to_int(self, sentence : str) -> list[int]
        from a text string returns the sequence of ids
    
    - int_to_text(self, indices: list[int]) -> str
        from a list of ids returns the reconstructed text senteces

Conversion from string to id and viceversa is made according
to an internal vocabulary
"""

class WordModel():
    """
    Facade class. Gives access to methods of Unigram and Bigram models.
    """
    def __init__(self, word_model_type):
        if word_model_type == 'unigram':
            self.word_model = Unigram()
        elif word_model_type == 'bigram':
            self.word_model = Bigram()
        else:
            raise Exception("Error: you must provide a correct word model.")
    
    def get_name(self):
        return self.word_model.name

    def get_n_class(self):
        return self.word_model.n_class
    
    def get_blank_id(self):
        return self.word_model.blank_id

    def text_to_int(self, sentence: str):
        return self.word_model.text_to_int(sentence)

    def int_to_text(self, indices: list[int]):
        return self.word_model.int_to_text(indices)

class Unigram():
    def __init__(self):

        self.name = 'unigram'
        self.blank_id = 28
        self.n_class = 29

        self.SPACE = "[space]"
        self.characters =  "'" + self.SPACE + " ".join(" abcdefghijklmnopqrstuvwxyz")
        self.tokenizer = get_tokenizer("basic_english")
        self.tokens = self.tokenizer(self.characters)

        self.char_to_id = {char: idx for idx, char in enumerate(self.tokens)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.tokens)}

    def text_to_int(self, sentence: str):
        # Eg. "my name is elon" -> [14, 25, 1, 15, ...]
        idx_sequence = []
        for ch in sentence:
            idx = self.char_to_id[self.SPACE] if ch == " " else self.char_to_id[ch]
            idx_sequence.append(idx)
        return idx_sequence

    def int_to_text(self, indices):
        # Eg. [14, 25, 1, 15, ...] -> "my name is elon"
        sentence = []
        for i in indices:
            ch = self.id_to_char[i]
            sentence.append(ch)
        return "".join(sentence).replace(self.SPACE, " ")

class Bigram():
    def __init__(self):

        self.name = 'bigram'
        self.blank_id = 704
        self.n_class = 705
        
        self.SPACE = '[space]'
        self.APOSTROPHE = "'"

        characters = 'abcdefghijklmnopqrstuvwxyz'
        unigrams = list(characters)
        bi_grams = [''.join(bi_gram) for bi_gram in itertools.product(characters, repeat=2)]

        ngrams = set(unigrams + bi_grams + [self.SPACE, self.APOSTROPHE])
        ngrams = sorted(list(ngrams))
        ngrams = ' '.join(ngrams)

        tokenizer = get_tokenizer("basic_english")
        tokens = tokenizer(ngrams)

        self.ngram_to_id = {ngram: idx for idx, ngram in enumerate(tokens)}
        self.id_to_ngram = {idx: ngram for idx, ngram in enumerate(tokens)}
    
    def text_to_int(self, sentence: str):
        # Eg. "my name is elon" -> [351, 1, 354, 331, 1, 237, 1, 122, 394]
        idx_sequence = []
        i = 0
        while i < len(sentence):
            # is space
            if sentence[i] == " ":
                idx_sequence.append(self.ngram_to_id[self.SPACE])
                i += 1
                continue
            # is apostrophe
            elif sentence[i] == "'":
                idx_sequence.append(self.ngram_to_id[self.APOSTROPHE])
                i += 1
                continue
            # last char
            elif i == len(sentence)-1:
                idx_sequence.append(self.ngram_to_id[sentence[i]])
                break
            
            bigram = sentence[i:i+2]
            # second is space
            if bigram[-1] == " ":
                idx_sequence.append(self.ngram_to_id[sentence[i]])
                idx_sequence.append(self.ngram_to_id[self.SPACE])
            # second is apostrophe
            elif bigram[-1] == "'":
                idx_sequence.append(self.ngram_to_id[sentence[i]])
                idx_sequence.append(self.ngram_to_id[self.APOSTROPHE])
            # append bigram
            else:
                idx_sequence.append(self.ngram_to_id[bigram])
            i += 2
        return idx_sequence
        
    def int_to_text(self, indices):
        # Eg. [351, 1, 354, 331, 1, 237, 1, 122, 394] -> "my name is elon"
        text_sentence = []
        for i in indices:
            ngram = self.id_to_ngram[i]
            text_sentence.append(ngram)
        return "".join(text_sentence).replace(self.SPACE, " ")


# test
"""
ngram = WordModel('unigram')
original = "my name is elon"
encoded = ngram.text_to_int(original)
reconstructed = ngram.int_to_text(encoded)

print('original:', original)
print('encoded:', encoded)
print('reconstructed:', reconstructed)
"""


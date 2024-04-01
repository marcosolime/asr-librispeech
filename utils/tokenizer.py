from torchtext.data.utils import get_tokenizer

class Tokenizer():
    def __init__(self):
        self.SPACE = "[space]"
        self.characters =  "'" + self.SPACE + " ".join(" abcdefghijklmnopqrstuvwxyz")
        self.tokenizer = get_tokenizer("basic_english")
        self.tokens = self.tokenizer(self.characters)

        self.char_to_id = {char: idx for idx, char in enumerate(self.tokens)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.tokens)}

    def text_to_int(self, sentence: str):
        """
        Convert a lower case string into a list of ids.

        Eg. "my name is elon" -> [14, 25, 1, 15, ...]
        """
        idx_sequence = []
        for ch in sentence:
            idx = self.char_to_id[self.SPACE] if ch == " " else self.char_to_id[ch]
            idx_sequence.append(idx)
        return idx_sequence

    def int_to_text(self, indices):
        """
        Convert a list of ids into a string.

        Eg. [14, 25, 1, 15, ...] -> "my name is elon"
        """
        sentence = []
        for i in indices:
            ch = self.id_to_char[i]
            sentence.append(ch)
        return "".join(sentence).replace(self.SPACE, " ")
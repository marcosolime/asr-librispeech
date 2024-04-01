import torch

class DecoderBase():

    def __init__(self, blank_id=28):
        self.blank_id = blank_id
    
    def decode_prob(self, prob, tokenizer):
        """
        Decodes (a batch of) log-probabilities
        into characters.
        prob -> shape (e.g.) [32, 650, 29]
        """
        max_ids = torch.argmax(prob, dim=-1) # [32, 650]
        out = []
        for ids in max_ids:
            values = []
            for i, value in enumerate(ids):
                 # ignore blank id
                if value == self.blank_id:
                    continue
                # avoid repetitions
                if i != 0 and value == ids[i-1]:
                    continue
                values.append(value.item())
            out.append(tokenizer.int_to_text(values))
        return out

    
    def decode_labels(self, indices, len_indices, tokenizer):
        """
        Decodes (a batch of) ids into characters.
        indices -> shape: [32, 300]
        len_indices -> shape: [32]
        tokenizer -> tool to convert idx into chars
        """
        out = []
        for i, ids in enumerate(indices):
            len_ids = len_indices[i]
            unpad_ids = ids[:len_ids]
            out.append(tokenizer.int_to_text(unpad_ids.tolist()))
        return out
            


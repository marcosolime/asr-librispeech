import torch

class DecoderBase():

    def __init__(self, blank_id):
        self.blank_id = blank_id
    
    def decode_prob(self, prob, word_model):
        """
        Decodes (a batch of) log-probabilities
        into characters.
        prob -> shape (e.g.) [16, 650, 29]
        """
        prob = torch.transpose(prob, 0, 1)
        arg_maxes = torch.argmax(prob, dim=-1) # [16, 650]
        decodes = []

        for i, args in enumerate(arg_maxes):
            decode = []
            for j, index in enumerate(args):
                 # ignore blank id
                if index == self.blank_id:
                    continue
                # avoid repetitions
                if j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
            decodes.append(word_model.int_to_text(decode))
        return decodes

    
    def decode_labels(self, indices, len_indices, word_model):
        """
        Decodes (a batch of) ids into characters.
        indices -> shape: [32, 300]
        len_indices -> shape: [32]
        word_model -> tool to convert idx into chars
        """
        out = []
        for i, ids in enumerate(indices):
            len_ids = len_indices[i]
            unpad_ids = ids[:len_ids]
            out.append(word_model.int_to_text(unpad_ids.tolist()))
        return out
            


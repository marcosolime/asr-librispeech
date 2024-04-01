import numpy as np

def edit_distance(reference, hypothesis):
    """
    Edit distance: given two sequences, return
    the Levenshtein distance, ie. the minimum number
    of single character edits.

    It works for both strings and list of strings. 
    """
    
    # base case
    if len(reference) == 0: return len(hypothesis)
    if len(hypothesis) == 0: return len(reference)
    if reference == hypothesis: return 0

    # init cache
    cache = np.zeros(shape=(len(reference)+1, len(hypothesis)+1))
    cache[-1,:] = np.arange(start=len(hypothesis), stop=-1, step=-1)
    cache[:,-1] = np.arange(start=len(reference), stop=-1, step=-1)

    # bottom-up approach
    for i in range(len(reference)-1, -1, -1):
        for j in range(len(hypothesis)-1, -1, -1):
            if reference[i] == hypothesis[j]:
                cache[i,j] = cache[i+1,j+1]
            else:
                cache[i, j] = 1 + min(cache[i+1,j], cache[i,j+1], cache[i+1,j+1])
    
    #print(cache)
    return cache[0,0]

def word_errors(reference: str, hypothesis: str):
    """
    Find the word-level edit distance between reference and hypothesis.
    """
    reference = reference.lower()
    hypothesis = hypothesis.lower()

    ref_words = reference.split(' ')
    hyp_words = hypothesis.split(' ')

    distance = edit_distance(ref_words, hyp_words)
    return float(distance), len(ref_words)

def char_errors(reference, hypothesis):
    """
    Find the char-level edit distance between reference and hypothesis.
    """
    reference = reference.lower()
    hypothesis = hypothesis.lower()

    reference = " ".join(filter(None, reference.split(' ')))
    hypothesis = " ".join(filter(None, hypothesis.split(' ')))

    distance = edit_distance(reference, hypothesis)
    return float(distance), len(reference)

def wer(reference, hypothesis):
    """
    Returns a number between 0 and 1.
    Idea: how many reference words I got wrong in percentage.
    Eg. 0.1 -> GOOD, 0.7 -> BAD
    """

    distance, ref_len = word_errors(reference, hypothesis)
    if ref_len == 0:
        raise ValueError("Reference sentence appears to be empty.")
    return float(distance) / ref_len

def cer(reference, hypothesis):
    """
    Returns a number between 0 and 1.
    Idea: how many reference characters I got wrong in percentage.
    Eg. 0.1 -> GOOD, 0.7 -> BAD
    """
    distance, ref_len = char_errors(reference, hypothesis)
    if ref_len  == 0:
        raise ValueError("Reference sentence appears to be empty.")
    return float(distance) / ref_len

def batch_cer(batch_ref, batch_hyp, average=False):
    """
    Returns a list of CER for each pair-element in the batch
    """
    batch_size = len(batch_ref)
    out = []
    for i in range(batch_size):
        out.append(cer(batch_ref[i], batch_hyp[i]))
    
    if average:
        return sum(out) / batch_size
    return out

def batch_wer(batch_ref, batch_hyp, average=False):
    """
    Returns a list of WER for each pair-element in the batch
    """
    batch_size = len(batch_ref)
    out = []
    for i in range(batch_size):
        out.append(wer(batch_ref[i], batch_hyp[i]))
    
    if average:
        return sum(out) / batch_size
    return out
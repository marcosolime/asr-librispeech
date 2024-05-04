import numpy as np
import Levenshtein

def compute_wer(hypothesis, reference):
    """
    Returns a number between 0 and 1.
    Idea: how many reference words we got wrong in percentage.
    Eg. 0.1 -> GOOD, 0.7 -> BAD
    """
    hypothesis_words = hypothesis.split()
    reference_words = reference.split()
    
    wer = Levenshtein.distance(hypothesis_words, reference_words) / len(reference_words)
    return wer

def compute_cer(hypothesis, reference):
    """
    Returns a number between 0 and 1.
    Idea: how many reference characters we got wrong in percentage.
    Eg. 0.1 -> GOOD, 0.7 -> BAD
    """
    cer = Levenshtein.distance(hypothesis, reference) / len(reference)
    return cer

def avg_cer(batch_hyp, batch_ref):
    """
    Calculates CER for each hyp-ref pair, and returns the average
    """
    batch_size = len(batch_ref)
    out = []
    for i in range(batch_size):
        out.append(compute_cer(batch_hyp[i], batch_ref[i]))
    
    return sum(out) / batch_size

def avg_wer(batch_hyp, batch_ref):
    """
    Calculates WER for each hyp-ref pair, and returns the average
    """
    batch_size = len(batch_ref)
    out = []
    for i in range(batch_size):
        out.append(compute_wer(batch_hyp[i], batch_ref[i]))
    
    return sum(out) / batch_size

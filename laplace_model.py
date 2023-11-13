# implements laplace add-1 smoothing method for an n-gram language model

def laplace_model(ngrams):
    
    dictionary = {}
    for ngram in ngrams: # update frequencies
        ngram = tuple(ngram)
        if ngram in dictionary:
            dictionary[ngram] += 1
        else:
            dictionary[ngram] = 1
                
    unique_tuples = {tuple(inner_list) for inner_list in ngrams}
    V = len([list(t) for t in unique_tuples])  # unique n-grams
    
    for ngram in dictionary:
        count_ngram = dictionary[ngram]
        count_prefix = sum(1 for ng in ngrams if ng[:-1] == ngram[:-1])
        dictionary[ngram] = (count_ngram + 1) / (count_prefix + V)
    
    unseen = 1 / V #probability assigned to unseen n-grams
        
    return dictionary, unseen

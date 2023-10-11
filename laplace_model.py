def laplace_model(ngrams):
    
    dictionary = {}
    for ngram in ngrams: #update frequencies
        if ngram in dictionary:
            dictionary[ngram] += 1
        else:
            dictionary[ngram] = 1
    N = len(ngrams)
    V = len(dictionary)
    
    for ngram in dictionary:
        dictionary[ngram] = (dictionary[ngram] + 1) / (N+V) #calc cond ngram prob
    
    unseen = 1/(N + V)
        
    return dictionary, unseen
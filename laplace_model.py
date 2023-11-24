import math

def count_n_grams(ngrams):

    dictionary = {}
    for ngram in ngrams:
        ngram = tuple(ngram)
        if ngram in dictionary:
            dictionary[ngram] += 1
        else:
            dictionary[ngram] = 1
    
    return dictionary

def laplace_model(nminus1_grams_counts, 
                  n_grams_counts, 
                  test_n_gram, 
                  vocab_size):

    test_n_gram = tuple(test_n_gram)
    
    if test_n_gram in list(n_grams_counts.keys()):
        x = n_grams_counts.get(test_n_gram)
    else:
        x=0
    
    if test_n_gram[:-1] in list(nminus1_grams_counts.keys()):
        y = nminus1_grams_counts.get(test_n_gram[:-1])
    else:
        y=0
    
    prob = (x+1) / (y + vocab_size)
        
    return prob

def perplexity_ngram_model(nminus1_grams_counts, 
                           n_grams_counts, 
                           test_n_grams, 
                           vocab_size):

    log_prob_sum = 0
    
    for n_gram in test_n_grams:
        log_prob_sum += math.log(laplace_model(nminus1_grams_counts, 
                                               n_grams_counts, 
                                               n_gram, 
                                               vocab_size))
        
    unique_test_tokens = set()
    for lst in test_n_grams:
        unique_test_tokens.update(set(lst))
    N = len(unique_test_tokens)
    
    perplexity = math.exp(-log_prob_sum / N)

    return perplexity

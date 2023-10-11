def perplexity(model, ngrams, N):
    
    '''
    model : the learned language model
    ngrams : the data to test as a list of ngrams
    N : the length of the text to test
    '''
    
    perplexity = 1
    
    for i in range(len(ngrams)):
        try:
            perplexity *= 1/(model[0][ngrams[i]]) # denominator as learned by the model
        except:
            perplexity *= model[1] # model[1] = unseen probability
            
    perplexity = pow(perplexity, 1/N)
    
    return perplexity
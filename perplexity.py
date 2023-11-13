#perplexity implementation (formula containing exp and log to avoid numerical overflow

import math

def perplexity(learned_distribution, unseen_prob, ngrams, N):
    '''
    learned_distribution: the learned learned_distribution of the language model
    unseen_prob: the probability the model assigns to ngrams not included in the training set
    ngrams: the data to test as a list of ngrams
    N: the length of the text to test
    '''

    log_prob_sum = 0

    for i in range(len(ngrams)):
        try:
            log_prob_sum += math.log(1 / learned_distribution[tuple(ngrams[i])])
        except KeyError:
            log_prob_sum += math.log(unseen_prob)

    perplexity = math.exp(-log_prob_sum / N)

    return perplexity

import math
import torch
import torch.nn.functional as F

def perplexity_ngram_model(learned_distribution, unseen_prob, ngrams, N):
    '''
    learned_distribution: the learned learned_distribution of the language model
    unseen_prob: the probability the model assigns to ngrams not included in the training set
    ngrams: the data to test as a list of ngrams
    N: the length (words) of the test text
    '''

    log_prob_sum = 0

    for i in range(len(ngrams)):
        try:
            log_prob_sum += math.log(1 / learned_distribution[tuple(ngrams[i])])
        except KeyError:
            log_prob_sum += math.log(unseen_prob)
    
    perplexity = math.exp(-log_prob_sum / N)

    return perplexity

def perplexity_neural_model(model, test_batches, vocab_size, criterion):
    
    '''
    model: learned neural language model (ex. lstm)
    test_batches: list of batches to test (in indexed format)
    vocab_size: vocabulary size
    criterion: loss function pytorch instance    
    '''
    
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        
        for batch in test_batches:
            
            batch = batch.view(1, -1)
            
            output = model(batch[:,:-1])
            output = torch.reshape(output, (1, vocab_size))
            
            target = batch[:,-1]

            loss = criterion(output, target)
            
            total_loss += loss.item()

    average_loss = total_loss / len(test_batches)
    perplexity = math.exp(average_loss)

    return perplexity
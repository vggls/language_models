import math
import torch


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
        
    N = len(test_n_grams) #N is equal to the sum updates
    
    perplexity = math.exp(- log_prob_sum / N)

    return perplexity


def perplexity_network_model(test_sequence_of_integers,
                             sequence_length,
                             model,
                             model_type,                              
                             loss_fct,
                             vocab_size):
    
    assert model_type in ['lstm', 'transformer']

    sequences = [test_sequence_of_integers[i:i+sequence_length+1] for i in range(0, len(test_sequence_of_integers)-sequence_length)]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_loss = 0

    with torch.no_grad(): #since in validation phase there is no backprop and weight updates
    
        for seq in sequences:
    
            seq = torch.tensor(seq)
            seq = seq.to(device)
            seq = seq.view(1,-1)

            if model_type == 'lstm': #this is ok for now
                pred = model(seq[:,:-1])
                output = torch.reshape(pred, (1, vocab_size)) #reshape in order to add dim '1' to the first shape dimension
                target = seq[:,-1]
                loss = loss_fct(output, target)

            elif model_type == 'transformer':
                pred = model(seq[:,:-1])
                output = torch.reshape(pred[0], (1*sequence_length, vocab_size))
                target = seq[:,1:]
                target = torch.reshape(target, (-1,))
                loss = loss_fct(output, target)
            
            test_loss += loss.item()
    
    perplexity = math.exp( test_loss / len(sequences) )

    return perplexity

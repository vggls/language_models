import math
import torch

def perplexity_neural_model(test_sequence_of_integers,
                            sequence_length,
                            model,
                            loss_fct,
                            vocab_size):
    
    sequences = [test_sequence_of_integers[i:i+sequence_length+1] for i in range(0, len(test_sequence_of_integers)-sequence_length)]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    (hidden, cell) = (torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device),
                      torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device))

    test_loss = 0

    with torch.no_grad(): #since in validation phase there is no backprop and weight updates
    
        for seq in sequences:
    
            seq = torch.tensor(seq)
            seq = seq.to(device)
            seq = seq.view(1,-1)
            
            hidden.detach_(), cell.detach_()
    
            output, (hidden, cell) = model(seq[:,:-1], hidden, cell)
            output = torch.reshape(output, (1, vocab_size))

            target = seq[:,-1]
            
            loss = loss_fct(output, target)
            
            test_loss += loss.item()
    
    perplexity = math.exp( test_loss / len(sequences) )

    return perplexity
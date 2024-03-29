import os
import torch
import torch.optim as optim
from copy import deepcopy
import math

class Train():

    def __init__(self,
                 model,
                 model_type,
                 loss_fct,
                 optimizer,
                 scheduler,
                 train_sequence,
                 val_sequence,
                 sequence_length,
                 batch_size,
                 epochs,
                 patience,
                 name):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)

        # attributes
        self.model = model                               # nn model
        self.model_type = model_type                     # either 'lstm' or 'transformer'
        self.loss_fct = loss_fct                         # loss function
        self.optimizer = optimizer                       # optimizer
        self.scheduler = scheduler                       # to gradually adjust learning rate value
        self.train_sequence = train_sequence             # list of integers
        self.val_sequence = val_sequence                 # list of integers
        self.sequence_length = sequence_length           # integer - no. of tokens we use to predict the next token
        self.batch_size = batch_size                     # integer
        self.epochs = epochs                             # max number of epochs to train the model
        self.patience = patience                         # epochs to wait until EarlyStopping condition is satisfied
        self.name = name                                 # str

        assert self.model_type in ['lstm', 'transformer']
        
        self.model.to(self.device)

    #------------------------------------------------------------------------------------------------------
    def training(self):

        self.training_perplexity_history = []
        self.validation_perplexity_history = []

        # below block is used in the Early Stopping section of the loop
        self.threshold_val_loss    = 10e+5
        self.best_model            = deepcopy(self.model)
        self.unchanged_epochs      = 0
        self.early_stopping_checkpoints = []

        print('Starting training..')

        if not self.val_sequence:
            print('No validation data is used.')

        for e in range(0, self.epochs):

            self.model.train()
            self.train_epoch()

            if self.val_sequence:
                self.model.eval()
                self.validate_epoch()
                self.early_stopping_check()

                if (e+1!=1) and ((e+1) == self.early_stopping_checkpoints[-1]):
                    print('Epoch: {}/{} - Perplexity: training {:.3f}, validation {:.3f} - E.S. checkpoint'.
                        format(e+1, self.epochs, self.training_perplexity_history[-1], self.validation_perplexity_history[-1]))
                else:
                    print('Epoch: {}/{} - Perplexity: training {:.3f}, validation {:.3f}'.
                        format(e+1, self.epochs, self.training_perplexity_history[-1], self.validation_perplexity_history[-1]))

                if self.unchanged_epochs == self.patience:
                    break
            else:
                print('Epoch: {}/{} - Perplexity: training {:.3f}'.format(e+1, self.epochs, self.training_perplexity_history[-1]))
            
            if self.scheduler != None:
                self.scheduler.step()  #Update Scheduler for next epoch
            
        print('Training complete !')
        if self.val_sequence:
            return self.training_perplexity_history, self.validation_perplexity_history, self.early_stopping_checkpoints
        else:
            return self.training_perplexity_history

    #------------------------------------------------------------------------------------------------------
    def process_sequence(self, sequence):
        
        sequences = [sequence[i:i+self.sequence_length+1] for i in range(0, len(sequence)-self.sequence_length)]
        num_batches = len(sequence)//self.batch_size
        
        return sequences, num_batches

    #------------------------------------------------------------------------------------------------------
    def compute_batch_loss(self, batch):
        
        if self.model_type == 'lstm':
            output = self.model(batch[:,:-1])      #output.shape = (batch_size, vocab_size)
            target = batch[:,-1]                   #target.shape = (batch_size)
            loss = self.loss_fct(output, target)
                
        elif self.model_type == 'transformer':
            output = self.model(batch[:,:-1])      #output[0].shape = (batch_size, sequence_length, vocab_size)
            target = batch[:,1:]                   #target.shape = (batch_size, sequence_length)
            output_flat = torch.reshape(output[0], (output[0].shape[0]*output[0].shape[1], -1))
            target_flat = torch.reshape(target, (-1,))                                           
            loss = self.loss_fct(output_flat, target_flat)
        
        return loss

    #------------------------------------------------------------------------------------------------------
    def train_epoch(self):

        train_sequences, num_train_batches = self.process_sequence(self.train_sequence)

        train_loss = 0 #loss = -log_prob
        
        if self.scheduler != None:
            print('  lr value {}'.format(self.optimizer.param_groups[0]['lr']))
        
        for n in range(num_train_batches):

            batch = train_sequences[n*self.batch_size:(n+1)*self.batch_size]
            batch = torch.tensor(batch) # batch.shape is (batch_size, sequence_length+1)
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            loss = self.compute_batch_loss(batch)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            train_loss += loss.item()

        #train_perplexity = math.exp( train_loss / (self.batch_size * num_train_batches) )
        train_perplexity = math.exp(train_loss / num_train_batches)

        self.training_perplexity_history.append(train_perplexity)
        
    #------------------------------------------------------------------------------------------------------
    def validate_epoch(self):

        val_sequences, num_val_batches = self.process_sequence(self.val_sequence)

        val_loss = 0

        with torch.no_grad(): #since in validation phase there is no backprop and weight updates

            for n in range(num_val_batches):

                batch = val_sequences[n*self.batch_size:(n+1)*self.batch_size]
                batch = torch.tensor(batch)
                batch = batch.to(self.device)

                loss = self.compute_batch_loss(batch)

                val_loss += loss.item()

        val_perplexity = math.exp( val_loss / num_val_batches )

        self.validation_perplexity_history.append(val_perplexity)
        
    #------------------------------------------------------------------------------------------------------
    def early_stopping_check(self):

        epoch_val_perplexity = self.validation_perplexity_history[-1]

        if (epoch_val_perplexity <= self.threshold_val_loss):

            current_epoch = len(self.validation_perplexity_history)
            self.early_stopping_checkpoints.append(current_epoch)

            self.best_model = deepcopy(self.model)
            self.unchanged_epochs = 0
            self.threshold_val_loss  = epoch_val_perplexity

            if (current_epoch>=2):
                os.remove(f'model_epoch{self.early_stopping_checkpoints[-2]}_{self.name}.pt')
            torch.save(self.best_model, f'model_epoch{current_epoch}_{self.name}.pt')

        else:
            self.unchanged_epochs += 1

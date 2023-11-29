## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. In sections A-C we present the steps followed to construct for each model, while section D is dedicated to analysis of results, comparisons and discussion.

  A. 3-gram language model with Laplace smoothing <br>
  B. LSTM neural language model: I) with trainable embeddings, II) with pretrained embeddings **(currently working on improvements)** <br>
  C. Transformer model **(to do)** <br>
  D. Discussion **(to do)**

In our experiments, the Penn Treebank is downloaded from nltk and the sentences come in tokenized form.In our analysis, we consider all tokens in lower letter format, except for the '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-' tokens describing parentheses variations. The numbers and punctuation symbols were also preserved. In addition, a token is considered unknown, '< unk>' token, if it appears less than 3 times in the training tokens set. Based on this we construct the vocabulary V, which contains the set of words that the model sees during training. In turn is used for replacing with '< unk>' test tokens not included in it.

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences
- 2-grams (sequences of 2 words): for each tokenized sentence add one '< bos>' token at the beginning and one '< eos>' token at the end. Then extract the resulting 2-grams per sentence.
- 3-grams (sequences of 3 words): for each tokenized sentence add two '< bos>' tokens at the beginning and two '< eos>' tokens at the end. Then extract the resulting 3-grams per sentence.
- Calculate 3-gram model with add-1 smoothing:

  The model learns to calculate next word probabilities given the previous two words, as per following formula:
   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/bc95e121-3e6b-4d77-9992-64e4a3fb3359.png" height="200" width="600" />
   </p>
  The presence of 1 to the numerator and |V| to the denominator ensures that the model does not assign zero probability to any trigram, unseen (i.e. C(w_i-2, w_i-1, w_i)=0) or not.
- Test model performance by calculating perplexity over the test 3-grams.

   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/cb5e3128-1ee2-4582-968a-c279f4d52a62.png" height="65" width="520" />
   </p>

  In the above formula we note that 'log' refers to the natural logarithm (base e).

## B. LSTM neural language model
- Training-Validation-Test data: 3262-314-338 sentences
- Embedding layer: In order to feed words into a neural language model we must create their vector representations. This is achieved via an embedding layer which is put at the beginning of the neural architecture. This layer takes as input an integer representation of each word and maps it into a vector of desired length (embedding_dim hyperparameter). This layer could be either trainable (case I) or pre-trained (case II).
- LSTM language model general architecture: 
  
       Input (N,L+1) --> Embedding --> (N,L+1,E) matrix --> LSTM --> (N, L+1, H) vector --> Classification --> (N,|V|)   
                           layer                           layer(s)                            layer
  
         where N: num of batches
               L: sequence length
               E: embedding dim
               H: LSTM's hidden dimension
             |V|: vocabulary V size           

- discuss reduced vocabulary here and GloVe embeddings
- discuss the tie weights concept and why i did not apply it to the models. in case ii it is preferable if the output layer is learnable and not frozen. i could apply it to case i but
  i want the models to be as-comparable-as possible..
  
- The implemented models:
  
  The experiments currently presented in main.ipynb utilize the below **hyperparameters configuration**. The choice of values is 'handmade' based on case-by-case experimentation.
  
      embedding_dim = 256
      num_layers = 2 (the number of lstm layers)
      hidden_dim = 256
      dropout_rate = 0.3
      criterion = nn.CrossEntropyLoss()
      optimizer = Adam
      learning_rate = 0.001
      sequence_length = 50 (the number of words used to predict the next word)
      batch_size = 128 (we train the model in batches of 128 51-word sequences (sequence_length=50 words, target=1 word)
      epochs = 50
      patience = 10 (we monitor the validation perplexity and train as long as there is improvement within 10 epochs from the last improved value)
  
- Discuss perplexity here (and give the loss = logprob formula)

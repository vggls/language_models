## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. In sections A-C we present the steps followed for each model while section D is dedicated to comparisons and discussion.

  A. 3-gram language model with Laplace smoothing <br>
  B. LSTM neural language model: I) with trainable embeddings, II) with pretrained embeddings **(currently working on improvements)** <br>
  C. Transformer model **(to do)** <br>
  D. Discussion **(to do)**

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences, downloaded in tokenized form. We consider all tokens, including punctuation and numbers, except for '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-'. 
- For each token lower all letters.
- Unknown words: A token is considered unknown, '< unk>' token, if it appears less than 3 times in the training tokens set or if it contains the un-natural sequence ' \ /' (Penn Treebank comes with tokens of this form as well).
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

## B. LSTM neural language model (re-adapt based on A)
- Training-Validation-Test data: 3262-314-338 sentences, downloaded in tokenized form. We consider all tokens, including punctuation and numbers, except for '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-'.
- For each token lower all letters.
- Unknown words: Replace training tokens that appear less than 3 times with '< unk>' token and compute vocabulary V.
  Then replace test tokens not included in the vocabulary with '< unk>'.
- Embedding layer: In order to feed words into a neural language model we should create their vector representations. This is achieved via an embedding layer which is put at the beginning of the neural architecture. This layer takes as input an integer representation of each word and maps it into a vector of desired length (embedding_dim hyperparameter). This layer could be either trainable (case I) or pre-trained (case II).
- LSTM language model general architecture: <!-- shold put image here -->
  
        Embedding layer - LSTM layer(s) - Classification layer

- The implemented models:
  
  The experiments currently presented in main.ipynb utilize the below hyperparameters configuration. The choice of values is 'handmade' based on case-by-case experimentation.
  This section will be enriched by considering Bayesian optimization technique for choosing the final values **(currently working on this)**.
   <!-- discuss about tie weights as well -->
    - embedding_dim = 256
    - GloVe pretrained word embeddings (for case II)
    - num_layers = 2 (the number of lstm layers)
    - hidden_dim = 256
    - dropout_rate = 0.3
    - criterion = nn.CrossEntropyLoss()
    - optimizer = Adam
    - learning_rate = 0.001
    - sequence_length = 50 (the number of words used to predict the next word)
    - batch_size = 128 (we train the model in batches of 128 51-word sequences (sequence_length=50 words, target=1 word)
    - epochs = 50
    - patience = 10 (we monitor the validation perplexity and train as long as there is improvement within 10 epochs from the last improved value)
  

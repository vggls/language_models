## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. 

Sections A-C include the steps followed to construct each model and useful remarks to take into account while section D is dedicated to analysis of the results, comparisons and discussion.

  **A. 3-gram language model with Laplace smoothing** (*completed*) <br>
  **B. LSTM neural language model:**  (*to work on improvements - see end of section B*) <br>
      &nbsp; &nbsp; **- case I) with learnable embeddings** <br>
      &nbsp; &nbsp; **- case II) with pretrained embeddings** <br>
  **C. Transformer model** (*to do*) <br>
  **D. Discussion** (*to do*)

In our experiments, the Penn Treebank is downloaded from nltk and the sentences come in tokenized form.In our analysis, we consider all tokens in lower letter format, except for the '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-' tokens describing parentheses variations. The numbers and punctuation symbols were also preserved. In addition, a token is considered unknown, '< unk>' token, if it appears less than 3 times in the training tokens set. Based on this we construct the vocabulary V, which contains the set of words that the model sees during training. In turn is used for replacing with '< unk>' test tokens not included in it. We note that the vocabulary of the 3-gram model is slightly larger to the neural models ones, since for the neural models we had to define a small validation set as well. The test set is the same for all models.

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences
- 2-grams (sequences of 2 words): for each tokenized sentence add one '< bos>' token at the beginning and one '< eos>' token at the end. Then extract the resulting 2-grams per sentence.
- 3-grams (sequences of 3 words): for each tokenized sentence add two '< bos>' tokens at the beginning and two '< eos>' tokens at the end. Then extract the resulting 3-grams per sentence.
- Calculate 3-gram model with add-1 smoothing:

  The model learns to calculate next word probabilities given the previous two words, as per following formula:
   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/c1d237ca-d348-4658-a36f-3a31f5652314.png" height="170" width="600" />
   </p>
  The presence of 1 to the numerator and |V| (= vocabulary size) to the denominator ensures that the model does not assign zero probability to any trigram, unseen (i.e. C(w_i-2, w_i-1, w_i)=0) or not.
- Test model performance by calculating perplexity over the test 3-grams.

   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/3ffb26cf-2556-4bac-b139-9e0f1082e2d2.png" height="55" width="530" />
   </p>

  In the above formula we note that 'log' refers to the natural logarithm (base e).

## B. LSTM neural language model
- Training-Validation-Test data: 3262-314-338 sentences
- Embedding layer: In order to feed words into a neural language model we must create their vector representations. This is achieved via an embedding layer which is put at the beginning of the neural architecture. This layer takes as input an integer representation of each word and maps it into a vector of desired length (embedding_dim hyperparameter). This layer could be either trainable (case I) or pre-trained (case II).

  In regards to case II, we consider 300d pre-trained 6B-GloVe embeddings, which are kept frozen during training.
  We note that the embeddings do not contain representation for the '< eos>' and '< unk>' tokens. In our implementation, we assign the mean of all GloVe vectors to the '< eos>' token and a random vector, with values between GloVe min and max values, to the '< unk>' token.
  In addition, we note that there are 34 tokens included in the vocabulary of case I model (3259 size) which do not have a GloVe representation. To this purpose, in order to assign all vocabulary words to a GloVe embedding, we replaced these tokens with '< unk>' as well, resulting in a slightly smaller vocabulary (3225 size). This simple approach is one of many available to tackle this issue.
    
- LSTM language model general architecture:

  Due to the nature of language modelling task, in the LSTM layer we focus on the last time-step output only.
  
      (N,L) --> Embedding --> (N,L,E) --> LSTM --> (N,H) --> Classification --> (N,|V|)   
      input       layer        matrix    layer(s)  matrix        layer          matrix
  
         where N: batch size
               L: sequence length used to predict the next token
               E: embedding dim
               H: hidden dimension size (i.e. units) per LSTM layer
             |V|: vocabulary V size           

  <!--
  My remarks for each layer:
  a)Embedding layer: Per batch, we have N L-length sequences of tokens. Passing them throught the embedding layer we get an E=300-dim representation per token. Thus (N,L,E) is ok.
  b)LSTM layer: Fix a batch sequence and consider that the layer has H units. In parallel, we pass one-by-one the words (in vector form) to the units. Gradually, exhasuting all words (each word corresponds to a time-step) we get L predictions per unit. So, the LSTM output is normally (L,H) i.e. H predictions per time-step. So, it is valid to write (N,L,H) as well in the architecture. Since, for language modelling task, we are interested in predicting the sequence output only after the last time step we are particularly interested only in the last 'set' of predictions. So, only in the last H predictions produced by the LSTM layer. That is we write H instead of (L,H). 
  c)Classification layer: The job of a classification layer is to get as input the LSTM's vectorized view of the next word (the view is as large as we want, most specifically of size H) and map/assign it (with a probability) to a particular vocabulary word. Based on this, it is straightforward to say that an H-dim input results in a |V|-dim output and a (L,H) input to a (L,|V|) output.
   -->
  
- In order to train this kind of models we map token sequences to the next token in the text. The loss is determined by the probability the model assigns to the correct next word (which is known since we know the text). For a sequence of L tokens, the Cross-Entropy (CE) loss is given by:

 <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/a7d67ad2-c63c-4199-b84f-0cd0eb471880.png" height="45" width="500" />
   </p>
   
  &nbsp; &nbsp; &nbsp; &nbsp; In turn, the perplexity formula given in section A is adjusted accordingly.

- **Future improvements to include**:
    1. The choice of models hyperparameters values (see 'lstm_hyperparams.md' in 'learned models' folder) is currently based on case-by-case experimentation. This section will be enriched by implementing Bayesian optimization for hyperparameter tuning.
    2. Implement a more advanced technique to deal with vocabulary words that do not have a pre-trained representation (ex. subword embeddings or contextualized word embeddings)
 
 ## C. Transformer model
  to do

 ## D. Discussion
 to do


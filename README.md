This is an introductory repo to different architectures of Language Models (LM), trained and tested on the Penn Treebank. Language Modeling is the task of predicting the next word in a document. 

Contents: <br>

  **A. 3-gram model with Laplace smoothing** <br>
  **B. LSTM model:** <br>
      &nbsp; &nbsp; **- case I) with learnable embeddings** <br>
      &nbsp; &nbsp; **- case II) with pretrained embeddings** <br>
  **C. Pre-trained transformer model** <br>
  **D. Results** <br>
  **E. Text Generation & Discussion (to do)** <br>
  **F. Future improvements**

The Penn Treebank is downloaded from nltk and the sentences come in tokenized form.In our analysis, we consider all tokens in lower letter format, except for the '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-' tokens describing parentheses variations. The numbers and punctuation symbols were also preserved. In addition, a token is considered unknown, '< unk>' token, if it appears less than 3 times in the training tokens set. Based on this we construct the vocabulary V, which contains the set of words that the model sees during training. In turn is used for replacing with '< unk>' test tokens not included in it. We note that the vocabulary of the 3-gram model is slightly larger to the neural models ones, since for the neural models we had to define a small validation set as well. The test set is the same for all models.

## A. 3-gram model with Laplace smoothing
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

## B. LSTM model
- Training-Validation-Test data: 3262-314-338 sentences
- **Embedding layer**: In order to feed words into a neural language model we must create their vector representations. This is achieved via an embedding layer which is put at the beginning of the neural architecture. This layer takes as input an integer representation of each word and maps it into a vector of desired length (embedding_dim hyperparameter). This layer could be either trainable (case I) or pre-trained (case II).

  In regards to case II, we consider 300d pre-trained 6B-GloVe embeddings, which are kept frozen during training.
  We note that the embeddings do not contain representation for the '< eos>' and '< unk>' tokens. In our implementation, we assign the mean of all GloVe vectors to the '< eos>' token and a random vector, with values between GloVe min and max values, to the '< unk>' token.
  In addition, we note that there are 34 tokens included in the vocabulary of case I model (3259 size) which do not have a GloVe representation. To this purpose, in order to assign all vocabulary words to a GloVe embedding, we replaced these tokens with '< unk>' as well, resulting in a slightly smaller vocabulary (3225 size). This simple approach is one of many available to tackle this issue (see section F as well).

- In order to **train** this kind of models, we first put all the text tokens in a large input sequence, via their integer representation, and then process it **in a sequential manner**. To this purpose, we choose a hyperparameter called sequence_length and map sequences of length sequence_length to the next token. This procedure takes place iteratively, sliding over the token sequence and shifting -at each time step- the target token of interest by one position to the right.

  At time step t, the loss is determined by the probability the model assigns to the correct next word (which is known since we know the text). This learning approach is often called **teacher forcing**. For a sequence of L training tokens, the Cross-Entropy (CE) loss is given by the formula below. For any step t, due to the recurrence in the calculation of hidden states (i.e. h_(t+1) depends on h_t), the prediction y_(t+1) can be computed as long as y_t can be computed. This phenomenon results in a **sequential/serial loss calculation** over the time steps.
  
   <p align="center">
       <img src="https://github.com/vggls/language_models/assets/55101427/87fdb7f4-0c32-433e-a34a-e19a1cc769d1.png" height="65" width="550" />
     </p> 

- LSTM language model general **architecture**:

  Due to the nature of language modelling task, in the LSTM layer we focus on the last time-step output only.
  
      (N,L+1) --> Embedding --> (N,L,E) --> LSTM --> (N,H) --> Classification --> (N,|V|)   
       input        layer       matrix    layer(s)   matrix        layer          matrix
  
         where N: batch size
               L: integer-sequence length used to predict the next token (integer)
               E: embedding dim
               H: hidden dimension size (i.e. units) per LSTM layer
             |V|: vocabulary V size           

  <!--VERY IMPORTANT: No need to apply Softmax! The output is fed into an nn.CrossEntropyLoss, which applies softmax by default. See documentation.-->
  <!--
  My remarks for each layer:
  a)Embedding layer: Per batch, we have N L-length sequences of tokens. Passing them throught the embedding layer we get an E=300-dim representation per token. Thus (N,L,E) is ok.
  b)LSTM layer: Fix a batch sequence and consider that the layer has H units. In parallel, we pass one-by-one the words (in vector form) to the units. Gradually, exhasuting all words (each word corresponds to a time-step) we get L predictions per unit. So, the LSTM output is normally (L,H) i.e. H predictions per time-step. So, it is valid to write (N,L,H) as well in the architecture. Since, for language modelling task, we are interested in predicting the sequence output only after the last time step we are particularly interested only in the last 'set' of predictions. So, only in the last H predictions produced by the LSTM layer. That is we write H instead of (L,H). 
  c)Classification layer: The job of a classification layer is to get as input the LSTM's vectorized view of the next word (the view is as large as we want, most specifically of size H) and map/assign it (with a probability) to a particular vocabulary word. Based on this, it is straightforward to say that an H-dim input results in a |V|-dim output and a (L,H) input to a (L,|V|) output.
   -->
   
- For this kind of models, the **perplexity** formula, introduced in section A, can be adjusted accordingly as per above loss formula.
 
 ## C. Pre-trained transformer model
  - Training-Validation-Test data: 3262-314-338 sentences
  - We consider a pre-trained 'small' GPT2. During training we keep the embedding and transformer layers frozen and tune the linear 'head' to the needs of the training set.
  - Similar to the LSTM model, we create an integer representation of the training tokens, put them in a large input sequence and choose a sequence_length hyperparameter value. In order to **train** the model, we process the input sequence in a sequential manner, by mapping a sequence of sequence_length length to the sequence which is the initial one shifted by one time-step to the future.

    This kind of models, during training, process the input sequence w_1,..,w_L (L=sequence_length) **in parallel**; using the inputs w_1,..,w_k to calculate y_k, for k<=L. This results in L predictions y_1,..,y_L, whose losses are calculated **in parallel** as well.

    As per below formula, for a sequence w_1,..,w_k, the loss is determined by the probability the model assigns to the correct next word w_(k+1), mainting the **teacher forcing** approach of recurrent nets.
    <p align="center">
          <img src="https://github.com/vggls/language_models/assets/55101427/6f32296f-a5cf-48dc-ad44-d97856eddd71.png" height="60" width="490" />
        </p>

- small-GPT2 language model general **architecture**:

      (N,L+1) --> Embedding --> (N,L,E) --> 12 transformer --> (N,L,E) --> Linear Head --> (N,L,|V|)
       input       layers        matrix        layers           matrix        layer         matrix
  
         where N: batch size
               L: integer-sequence length
               E: embedding dim is 768 for the small version of GPT2
             |V|: vocabulary V size
    <!--VERY IMPORTANT: No need to apply Softmax! The output is fed into an nn.CrossEntropyLoss, which applies softmax by default. See documentation.-->
  
  More info about the nature of embedding and transformer layers for this architecture can be found at the original [publication](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- For this kind of models, the **perplexity** formula, introduced in section A, can be adjusted accordingly as per above loss formula.
  
 ## D. Results
 On the test set of 338 sentences:
 | Model  | Perplexity | Complexity |
 |  ---: | :---: | :---: | 
 | 3-gram with Laplace smoothin  | 1082.93 | - |
 | LSTM w/ learnable embeddings  | 248.95  | 2.9M |
 | LSTM w/ GloVe embeddings      | 195.72  | 1.9M |
 | GPT2 w/ trainable head        | 139.07  | 2.5M |

 ## E. Text Generation & Discussion
 (to do)
 <!-- comparison recurrent vs transformer ??  -->

 ## F. Future improvements
  1. As far as the LSTM with pre-trained embeddings is concerned, we will implement a more advanced approach to deal with vocabulary words that do not have a pre-trained representation (ex. subword embeddings or contextualized word embeddings).
  2. The choice of models hyperparameters values (see 'hyperparameters.txt' in the path 'notebooks/link_to_learned_models.md') is currently based on case-by-case experimentation. One may utilize Bayesian optimization techniques for more thorough tuning; ex. use of ray.tune. Due to limited hardware resources this task will be postponed for now.









  

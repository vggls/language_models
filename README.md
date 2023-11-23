## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. In sections A-C we present the steps followed for each model while section D is dedicated to comparisons and discussion.

  A. 3-gram language model with Laplace smoothing <br>
  B. LSTM neural language model: I) with trainable embeddings, II) with pretrained embeddings <br>
  C. Transformer model **(to do)** <br>
  D. Discussion **(to do)**

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences, downloaded in tokenized form    
- For each token lower the first letter
- Unknown words: Replace training tokens that appear less than 3 times with '< unk>' token and compute vocabulary V.
  Then replace test tokens not included in the vocabulary with '< unk>'.
- 2-grams (sequences of 2 words): for each tokenized sentence add one '< bos>' token at the beginning and one '< eos>' token at the end. Then extract the resulting 2-grams per sentence.
- 3-grams (sequences of 3 words): for each tokenized sentence add two '< bos>' tokens at the beginning and two '< eos>' tokens at the end. Then extract the resulting 3-grams per sentence.
- Calculate 3-gram model with add-1 smoothing:

  The model learns to calculate next word probabilities given the previous two words, as per following formula:
   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/a5c6495b-e898-44f9-bbbd-0d4d599f14f8.png" height="200" width="600" />
   </p>
  By adding 1 to the numerator and |V| to the denominator, we ensure that the model does not assign zero probability to any trigram, unseen (i.e. C(w_i-2, w_i-1, w_i)=0) or not.
- Test model performance by calculating perplexity over the test 3-grams.

   <p align="center">
     <img src="https://github.com/vggls/language_models/assets/55101427/cb5e3128-1ee2-4582-968a-c279f4d52a62.png" height="65" width="520" />
   </p>

## B. LSTM neural language model

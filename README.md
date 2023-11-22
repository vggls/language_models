## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. In sections A-D we present the steps followed for each model while section E is dedicated to comparisons and discussion.
- A. 3-gram language model with Laplace smoothing
- B. LSTM neural language model with trainable embeddings
- C. LSTM neural language model with pretrained embeddings
- D. Transformer model **(to do)**
- E. Discussion **(to do)**

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences, downloaded in tokenized form
- All words are transformed to lower letters
- Unknown words: replace training tokens that appear less than 3 times with '< unk>' token and compute vocabulary V.
  Then replace test tokens not included in the vocabulary with '< unk>'.
- For each training and test sentence add 2 '< bos>' tokens at the beginning and 2 '< eos>' tokens at the end
- Calculate training and test 3-grams
- Calculate 3-gram model with add-1 smoothing. The model learns to assign 1/|V| probability mass to any unseen test token, while the remaining (1-1/|V|) mass is allocated to the available training 3-grams based on their frequency in the training sentences. 
- Test model performance by calculating perplexity over the test 3-grams.

## B. LSTM neural language model with trainable embeddings

## C. LSTM neural language model with pretrained embeddings

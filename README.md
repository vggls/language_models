## Intro
This is an introductory repo to different architectures of language models, trained and tested on the Penn Treebank. We discuss:
- A. 3-gram language model with Laplace smoothing
- B. LSTM neural language model (build from scratch)
- C. LSTM neural language model with pretrained embeddings
- D. Transformer model **(to do)**

## A. 3-gram language model with Laplace smoothing
- Training-Test data: 3576-338 sentences, downloaded in tokenized form
- all words are transformed to lower letters
- unknown words: replace training tokens that appear less than 3 times with '<unk>' token and compute vocabulary
  Then replace test tokens not included in the vocabulary with '<unk>'.
- for each training and test sentence add 2 '<bos>' tokens at the beginning and 2 '<eos>' tokens at the end
- calculate training and test 3-grams
- calculate 3-gram model with add-1 smoothing
- test model performance by calculating perplexity over the test 3-grams

## B. LSTM neural language model (build from scratch)

## C. LSTM neural language model with pretrained embeddings

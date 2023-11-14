def add_unk_tokens (tokenized_sentences):
    
    # tokenized_sentences = list of tokens lists
    # replace all tokens that appear less than 3 times with <UNK>
    
    word_count = {}
    
    for sentence in tokenized_sentences:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    for i, sentence in enumerate(tokenized_sentences):
        for j, word in enumerate(sentence):
            if word_count[word] <= 2:
                tokenized_sentences[i][j] = '<UNK>'
    
    return tokenized_sentences

def training_ngrams(n, tokenized_sentences):
        
    ngrams = []
    for sent in tokenized_sentences:
        x = (n-1)*['<BOS>'] + sent + (n-1)*['<EOS>']
        for j in range(len(x) - (n - 1)):
            ngrams.append([x[j + i] for i in range(n)])
                         
    return ngrams

def test_ngrams(vocabulary, n, tokenized_sentences):

    # during testing, if a token is not in the vocabulary, replace it with <UNK>
    for sent in tokenized_sentences:
        for word in sent:
            if word not in vocabulary:
                sent[sent.index(word)] = '<UNK>'
            else:
                continue

    ngrams = []
    for sent in tokenized_sentences:
        for j in range(len(sent) - (n - 1)):
            ngrams.append([sent[j + i] for i in range(n)])

    return ngrams
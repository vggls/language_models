def lower(tokenized_treebank_sentences):
    
    lower = [[token.lower() for token in sentence] for sentence in tokenized_treebank_sentences]    

    return lower

def add_unk_tokens_for_training(tokenized_sentences):
    
    # for the purposes of vocabulary construction
    # replace all tokens that appear less than 3 times with <unk>
    
    word_count = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    tokenized_sentences_with_unk = tokenized_sentences
    for i, sentence in enumerate(tokenized_sentences_with_unk):
        for j, word in enumerate(sentence):
            if word_count[word] <= 2:
                tokenized_sentences_with_unk[i][j] = '<unk>'
    
    return tokenized_sentences_with_unk

def replace_with_unk_for_testing(vocabulary, tokenized_sentences):
    
    # during testing, if a token is not in the vocabulary, replace it with <unk>
    tokenized_sentences_with_unk = tokenized_sentences
    for sent in tokenized_sentences_with_unk:
        for word in sent:
            if word not in vocabulary:
                sent[sent.index(word)] = '<unk>'
    
    return tokenized_sentences_with_unk

def create_ngrams(n, tokenized_sentences):
        
    ngrams = []
    for sent in tokenized_sentences:
        x = (n-1)*['<bos>'] + sent + (n-1)*['<eos>']
        for j in range(len(x) - (n - 1)):
            ngrams.append([x[j + i] for i in range(n)])
                         
    return ngrams
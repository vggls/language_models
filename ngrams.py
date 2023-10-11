from nltk.corpus import treebank

# --------- Training data --------------------------
train_treebank = []
for j in range(150): # len(treebank.fileids()) = 199
    for i in treebank.sents(treebank.fileids()[j]):
        l = [j for j in i if '*' not in j] # remove tokens that contain '*'
        train_treebank.append(l)
        
# replace all tokens that appear less than 3 times with <UNK>
def unk (text):
    count = {}
    for sentence in text:
        for word in sentence:
            if word in count:
                count[word] += 1
            else:
                count[word] = 1

    for i, sentence in enumerate(text):
        for j, word in enumerate(sentence):
            if count[word] <= 2:
                text[i][j] = '<UNK>'
    
    return text

train_unk = unk(train_treebank)
    
# no. of elements replaced
n = 0
for sent in train_unk :
    n = n + len([i for i, j in enumerate(sent) if j == '<UNK>'])
print(f'No of elements replaced with <UNK>: {n}')
    
def training_ngrams(train_unk):
    
    # create n-grams, for n=1,2,3
    train_unigrams = []
    train_bigrams = []
    train_trigrams = []
    for sent in train_unk :
        sent.insert(0,'<BOS>')
        sent.append('<EOS>')
        for j in range(len(sent)):
            train_unigrams.append((sent[j]))
        for j in range(len(sent)-1):
            train_bigrams.append((sent[j], sent[j+1]))
        for j in range(len(sent)-2):
            train_trigrams.append((sent[j], sent[j+1], sent[j+2]))

    return train_unigrams, train_bigrams, train_trigrams

'''
train_unigrams, train_bigrams, train_trigrams = training_ngrams(train_unk)
print(len(train_unigrams), len(train_bigrams), len(train_trigrams))
'''

# ----------------- Test data ----------------------------------
test_treebank = []
for j in range(150, 199): # len(treebank.fileids()) = 199
    for i in treebank.sents(treebank.fileids()[j]):
        l = [j for j in i if '*' not in j]
        test_treebank.append(l)

def test_ngrams(test_treebank, train_unk):

    N = 0
    for sent in test_treebank:
        N += len(sent) #len(sent) counts the number of words in the sentence
    
    # during testing, if a token is not in L we do not replace it with <UNK>
    forL = []
    for sent in train_unk :
        forL.extend(sent)
    L = set(forL)
    
    for sent in test_treebank :
        for word in sent :
            if word not in L :
                sent[sent.index(word)] = '<UNK>'
            else :
                continue
            
    test_bigrams = []
    test_trigrams = []
    for sent in test_treebank :
        sent.insert(0,'<BOS>')
        sent.append('<EOS>')
        for j in range(len(sent)-1):
            test_bigrams.append((sent[j], sent[j+1]))
        for j in range(len(sent)-2):
            test_trigrams.append((sent[j], sent[j+1], sent[j+2]))
            
    return N, test_bigrams, test_trigrams

'''
N, test_bigrams, test_trigrams = test_ngrams(test_treebank, train_unk)
print(N, len(test_bigrams), len(test_trigrams))
'''
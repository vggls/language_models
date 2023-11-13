'''
from nltk.corpus import treebank

# ------------------------------ Training data -------------------------------
train_treebank = []
for j in range(150): # len(treebank.fileids()) = 199
    for i in treebank.sents(treebank.fileids()[j]):
        l = [j for j in i if '*' not in j] # remove tokens that contain '*'
        train_treebank.append(l)
'''

# replace all tokens that appear less than 3 times with <UNK>
def add_unk_tokens (list_of_sentences):
    
    word_count = {}
    
    for sentence in list_of_sentences:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    for i, sentence in enumerate(list_of_sentences):
        for j, word in enumerate(sentence):
            if word_count[word] <= 2:
                list_of_sentences[i][j] = '<UNK>'
    
    return list_of_sentences

#training_sentences = [sentence.copy() for sentence in train_treebank]

#add_unk_tokens(training_sentences)

'''
# no. of tokens replaced with <UNK>
n = 0
for sent in training_sentences :
    n = n + len([i for i, j in enumerate(sent) if j == '<UNK>'])
print(f'No of tokens replaced with <UNK>: {n}')
'''

#the vocabulary is useful for the testing phase
#vocabulary = set([item for sublist in training_sentences for item in sublist])

def training_ngrams(n, tokenized_sentences):
        
    ngrams = []
    for sent in tokenized_sentences:
        x = (n-1)*['<BOS>'] + sent + (n-1)*['<EOS>']
        for j in range(len(x) - (n - 1)):
            ngrams.append([x[j + i] for i in range(n)])
                         
    return ngrams

#train_trigrams = training_ngrams(3, training_sentences)

# ------------------------------ Test data ----------------------------------
'''
test_treebank = []
for j in range(150, 199): # len(treebank.fileids()) = 199
    for i in treebank.sents(treebank.fileids()[j]):
        l = [j for j in i if '*' not in j]
        test_treebank.append(l)
'''
#test_sentences = [sentence.copy() for sentence in test_treebank]

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
        for j in range(len(sent) - (n - 1)): #n=2: j in range(len(sent)-1)
            ngrams.append([sent[j + i] for i in range(n)]) #n=2: append sent[j], sent[j+1]

    return ngrams

#test_trigrams = test_ngrams(vocabulary, 3, test_sentences)
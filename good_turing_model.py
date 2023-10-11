import collections

def good_turing_model(ngrams):
    
    N = len(ngrams)
    
    c_dict = collections.Counter(ngrams) # says how many times each ngram occured
    n = collections.Counter(c_dict.values())
    Nc_dict = collections.OrderedDict(sorted(n.items())) # says how many times each frequency occured
    
    dictionary = {}
    for ngram in ngrams :
        c = c_dict[ngram]
        Nc = Nc_dict[c]
        for j in Nc_dict.keys():
            if j > c:
                break
        Nc_plus_1 = Nc_dict[j]
        dictionary[ngram] = (c+1) * Nc_plus_1 / (N * Nc)
    
    unseen = Nc_dict[1] / N
    
    return dictionary, unseen
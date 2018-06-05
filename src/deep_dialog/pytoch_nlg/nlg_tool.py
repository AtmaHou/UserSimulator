from nltk.tokenize.treebank import TreebankWordTokenizer


def treebank_tokenizer(sentence, max_length=0):
    """
    Tokenize and truncate sentence
    :param sentence: str, a sentence string
    :param max_length: int, max token included in the result, 0 for unlimited
    :return: list, a list of token
    """
    # split 's but also split <>, wait to use in further work
    t = TreebankWordTokenizer()
    word_lst = t.tokenize(sentence.lower().replace("$", "_B_"))
    # word_lst = t.tokenize(sentence.lower().replace("<", "LAB_").replace(">", "_RAB"))
    ret = []
    for w in word_lst:
         ret.append(w.replace("_B_", "$"))
         # ret.append(w.replace("LAB_", "<").replace("_RAB", ">"))
    if max_length > 0:
        return ret[: max_length]
    else:
        return ret

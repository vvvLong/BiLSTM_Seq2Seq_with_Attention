from collections import Counter


class Vocabulary(object):
    """a vocabulary for NLP corpus"""

    def __init__(self, max_vocab_size=50000, min_count=None, start_end_tokens=False, name="Vocabulary1"):
        self.name = name
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens  # whether include start and end tokens
        self.vocabulary_size = 2
        self.word2idx = {"<UNK>": 0, "<PAD>": 1}
        self.idx2word = ["<UNK>", "<PAD>"]
        self.idx2count = [0, 0]

    def buildVocabulary(self, data):
        """
        method to build a vocabulary
        :param list data: a list of strings, with each string being a sentence
        :return: None
        """
        # add start and end token
        if self.start_end_tokens:
            self.idx2word += ['<SOS>', '<EOS>']
            self.vocabulary_size += 2

        # count words
        counter = Counter(
            [word for sentence in data for word in sentence.split()])
        # filter words by their counts
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size - self.vocabulary_size)}
        # filter words with low frequency
        if self.min_count:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_count}

        # generate attributes
        self.idx2word += list(sorted(counter.keys()))
        self.idx2count = [counter.get(word, 0) for word in self.idx2word]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocabulary_size = len(self.idx2word)

    def indexer(self, word):
        """
        return word index
        :param str word: the word to query
        :return: word index
        """
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx['<UNK>']

from src.utils.config import Config
from src.utils.vocabulary import Vocabulary
import jieba
import re
import random
import torch
import logging


# map Chinese punctuations into English punctuations
ch2en = {
    '！': '!',
    '？': '?',
    '。': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def normalizeString(s):
    """
    lower case and trim punctuations and numbers
    :param str s: input string
    :return: normalised string
    """
    s = s.lower().strip()
    # check Chinese punctuations
    s = "".join(char if char not in ch2en.keys() else ch2en[char] for char in s)
    s = re.sub(r"([.!?])", r" \1", s)  # only keep . ! ?
    # keep alphabets, the three punctuations and Chinese characters
    s = re.sub(r"[^a-zA-Z.!?\u4e00-\u9fff]+", r" ", s)
    return s


def preprocess(data_path):
    """
    pre-process text data and return vocabularies
    :param str data_path: input data path
    :return: a list of lists of processed sentence pairs
    """
    # read data and split into pairs
    lines = open(data_path, encoding="utf-8").read().strip().split('\n')
    pairs = [[normalizeString(s) for s in line.split('\t')[:2]] for line in lines]

    # segment Chinese sentences
    for line in pairs:
        line[1] = " ".join(word for word in jieba.lcut(line[1]))

    return pairs


def getVocabulary(pairs, input_lang, output_lang, max_vocab_size, reverse=False, start_end_tokens=True):
    """
    generate vocabularies for the pairs
    :param list pairs: language sentence pairs
    :param str input_lang: input language name
    :param str output_lang: output language name
    :param int max_vocab_size: max vocabulary size
    :param bool reverse: whether to inverse the input and output sentences
    :param bool start_end_tokens: whether to use start and end tokens
    :return: two vocabularies
    """
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    # initialise vocabularies
    input_vocab = Vocabulary(name=input_lang, start_end_tokens=start_end_tokens, max_vocab_size=max_vocab_size)
    output_vocab = Vocabulary(name=output_lang, start_end_tokens=start_end_tokens, max_vocab_size=max_vocab_size)
    input_sentences = []
    output_sentences = []

    # build vocabularies
    for pair in pairs:
        input_sentences.append(pair[0])
        output_sentences.append(pair[1])
    input_vocab.buildVocabulary(input_sentences)
    output_vocab.buildVocabulary(output_sentences)

    return input_vocab, output_vocab


def sentence2index(vocabulary, sentence):
    """
    convert sentence into list of index
    :param Vocabulary vocabulary: the vocabulary of the language
    :param str sentence: input sentence
    :return: list of index
    """
    index = [vocabulary.indexer("<SOS>")]
    index += [vocabulary.indexer(word) for word in sentence.split()]
    index += [vocabulary.indexer("<EOS>")]
    return index


def train_test_split(data, test_split=0.3):
    """
    split data into train and text sets
    :param list data: text data list
    :param float test_split: text proportion
    :return: train data and test data
    """
    size = len(data)
    split = size - int(size * test_split)
    random.seed(66)
    random.shuffle(data)
    return data[:split], data[split:]


def translation(sentence, model, max_length, src_vocab, trg_vocab):
    """
    translate one source sentence
    :param str sentence: source sentence
    :param model: the Seq2Seq model
    :param int max_length: max sentence length
    :param src_vocab: source language vocabulary
    :param trg_vocab: target language vocabulary
    :return: one target sentence
    """
    sentence = normalizeString(sentence)
    index = sentence2index(src_vocab, sentence)
    source = torch.LongTensor(index).unsqueeze(1)  # (T, 1)
    model = model.cpu()
    target = model.translate(source, max_length)
    target = [trg_vocab.idx2word[idx] for idx in target]
    if target[-1] == "<EOS>":
        target = target[:-1]
    return "".join(word for word in target)


def create_logger(filename):
    """
    create a logger
    :param str filename: log file name include path
    :return: logger
    """
    logger = logging.getLogger(__name__)
    fmt = '%(message)s'
    format_str = logging.Formatter(fmt)  # formatting
    logger.setLevel(logging.INFO)  # level
    sh = logging.StreamHandler()  # output on screen
    sh.setFormatter(format_str)  
    th = logging.FileHandler(filename)
    th.setFormatter(format_str)  # output into file
    logger.addHandler(sh)  
    logger.addHandler(th)
    return logger

logger = create_logger(Config.log_path + "/training.log")


if __name__ == "__main__":
    data = preprocess(Config.data_path)
    English, Chinese = getVocabulary(data, "English", "Chinese")
    index = sentence2index(Chinese, data[2046][1])
    print(data[2046][1])
    print(index)
    print([Chinese.idx2word[idx] for idx in index])

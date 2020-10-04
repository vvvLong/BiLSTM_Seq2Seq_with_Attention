import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.process import sentence2index, preprocess, getVocabulary
from src.utils.config import Config


class TranslationData(Dataset):

    def __init__(self, data, src_vocab, trg_vocab):
        """
        Dataset for translation
        :param list data: processed data
        :param src_vocab: source language vocabulary
        :param trg_vocab: target language vocabulary
        """
        super(TranslationData, self).__init__()
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence = self.data[idx][0]
        trg_sentence = self.data[idx][1]
        src_index = sentence2index(self.src_vocab, src_sentence)
        trg_index = sentence2index(self.trg_vocab, trg_sentence)
        src = torch.LongTensor(src_index)
        trg = torch.LongTensor(trg_index)
        return src, trg


def collate_fn(batch):
    """
    customized collate function to dynamically padding mini-batches
    :param batch: the list of samples from Dataloader
    :return: padded mini-batch
    """
    pad_index = 1  # the <PAD> index in vocabulary
    src_list = [sample[0] for sample in batch]  # list of each language sentences
    trg_list = [sample[1] for sample in batch]

    def padding(sentence_list):
        """padding each sentence to the right"""
        max_len = max([sentence.size(0) for sentence in sentence_list])
        pad_sen = [sen.tolist() + [pad_index] * max(0, max_len - len(sen))
                   for sen in sentence_list]
        return torch.LongTensor(pad_sen).transpose(0, 1)  # shape of (T, B)

    return padding(src_list), padding(trg_list)


if __name__ == "__main__":
    data = preprocess(Config.data_path)
    src_vocab, trg_vocab = getVocabulary(data, "English", "Chinese")
    dataset = TranslationData(data, src_vocab, trg_vocab)
    source, target = dataset[66]
    print(source.size())
    print(target.size())
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for i, batch in enumerate(loader):
        source, target = batch
        print(source.size())
        print(target.size())
        break

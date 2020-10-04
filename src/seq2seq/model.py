import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.5):
        """
        Bidirectional LSTM encoder; aggregate the outputs of two directions by summation
        :param int vocab_size: size of the source language vocabulary
        :param int embedding_size: size of embedded
        :param int hidden_size: size of hidden states
        :param int n_layers: number of layers
        :param float dropout: dropout rate
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_sentence, hidden=None):
        """
        forward propagation
        :param torch.Tensor input_sentence: source sentence; shape of (T, B)
        :param tuple hidden: previous hidden states, (h, c)
        :return: outputs sequence, the last hidden states include h and c
        """
        embedded = self.embedding(input_sentence)  # (T, B, E)
        outputs, hidden = self.lstm(embedded, hidden)
        # aggregate hidden states of two directions by sum
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])  # (T, B, H)
        return outputs, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size):
        """
        attention weights; two-layer MLP as alignment model
        :param int hidden_size: the hidden state size of encoder
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn1 = nn.Linear(2*hidden_size, hidden_size)  # first fc layer
        self.attn2 = nn.Linear(hidden_size, 1)  # second fc layer

    def forward(self, encoder_outputs, hidden):
        """
        calculate attention score
        :param torch.Tensor encoder_outputs: sequence of outputs from encoder
        :param torch.Tensor hidden: the previous hidden h of decoder, shape of (1, B, H)
        :return: attention score
        """
        length = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, T, H)
        hidden = hidden.repeat(length, 1, 1).transpose(0, 1)  # (B, T, H)
        # calculate energies
        energies = self.attn1(torch.cat([encoder_outputs, hidden], dim=2))
        energies = F.relu(energies)  # (B, T, H)
        energies = self.attn2(energies)  # (B, T, 1)
        # calculate attention weights
        attention = F.softmax(energies.squeeze(2), dim=1).unsqueeze(1)  # (B, 1, T)
        return attention


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.5):
        """
        :param int vocab_size: size of the target language vocabulary
        :param int embedding_size: size of embedded
        :param int hidden_size: size of hidden states
        :param int n_layers: number of layers
        :param float dropout: dropout rate
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        self.attention = Attention(hidden_size)  # encoder and decoder will use same hidden size for simplicity
        self.out = nn.Linear(2*hidden_size, vocab_size)

    def forward(self, input_word, hidden, encoder_outputs):
        """
        forward propagation
        :param torch.Tensor input_word: the input word tensor, shape of (B, )
        :param tuple hidden: previous hidden states
        :param torch.Tensor encoder_outputs: output from encoder
        :return: log likelihood of output words; final hidden states
        """
        # get input word embeddings
        embedded = self.embedding(input_word)  # (B, E)
        embedded = embedded.unsqueeze(0)  # (1, B, E)
        # calculate context
        attention = self.attention(encoder_outputs, hidden[0][-1:])  # hidden h uses the last layer
        context = attention.bmm(encoder_outputs.transpose(0, 1))  # (B, 1, H)
        context = context.transpose(0, 1)  # (1, B, H)
        # concatenate LSTM input
        lstm_input = torch.cat([embedded, context], dim=2)  # (1, B, E+H)
        output, hidden = self.lstm(lstm_input, hidden)
        # concatenate final layer input
        context = context.squeeze(0)  # (B, H)
        output = output.squeeze(0)  # (B, H)
        final_input = torch.cat([output, context], dim=1)  # (B, 2H)
        # final output layer
        output = self.out(final_input)  # (B, N)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        """
        the final Seq2Seq model with attention
        :param Encoder encoder: the Encoder class
        :param Decoder decoder: the Decoder class
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing=0.5):
        """
        forward propagation
        :param torch.Tensor source: the source sentences, shape of (T', B)
        :param torch.Tensor target: the target sentences, shape of (T, B)
        :param float teacher_forcing: the teacher forcing ratio
        :return: translated sentences
        """
        # initialize outputs
        target_len = target.size(0)
        batch_size = source.size(1)
        vocab_size = self.decoder.vocab_size
        outputs = Variable(torch.zeros(target_len, batch_size, vocab_size))  # (T, B, N)

        # encoding source sentences
        encoder_outputs, hidden = self.encoder(source)
        # initialize decoder as the final hidden states of encoder
        hidden = (hidden[0][:self.decoder.n_layers], hidden[1][:self.decoder.n_layers])

        # decoding word by word
        output = Variable(target.data[0, :])  # first output <SOS>, shape of (B,)
        for t in range(1, target_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output  # fill in outputs
            top1 = output.max(1)[1]  # the word index with the largest log likelihood
            # teacher forcing
            is_teacher = random.random() < teacher_forcing
            output = Variable(target.data[t] if is_teacher else top1)
        return outputs

    def translate(self, source, max_length):
        """
        translate one sentence
        :param torch.Tensor source: the source sentence, shape of (T', 1)
        :param int max_length: max translation length
        :return: translated sentence
        """
        # encoding source sentences
        encoder_outputs, hidden = self.encoder(source)
        # initialize decoder as the final hidden states of encoder
        hidden = (hidden[0][:self.decoder.n_layers], hidden[1][:self.decoder.n_layers])
        outputs = []
        # decoding word by word
        output = torch.LongTensor([2])  # first output <SOS>, shape of (1,)
        for t in range(1, max_length+1):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            top1 = output.max(1)[1]  # the word index with the largest log likelihood
            outputs.append(top1.item())  # fill in outputs
            # break if <EOS>
            if top1 == 3:
                break
            output = top1
        return outputs


if __name__ == "__main__":
    test_encoder = Encoder(20, 10, 5)
    test_decoder = Decoder(20, 10, 5)
    model = Seq2Seq(test_encoder, test_decoder)

    src = torch.randint(high=20, size=(20, 8))
    trg = torch.randint(high=20, size=(10, 8))

    res = model(src, trg)
    print(res.shape)

    test_sen = torch.randint(high=20, size=(20, 1))
    res = model.translate(test_sen, 10)
    print(res)

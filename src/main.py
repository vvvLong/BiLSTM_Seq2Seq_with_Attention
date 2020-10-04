from src.utils.process import train_test_split, preprocess, getVocabulary, translation, logger
from src.utils.config import Config
from src.utils.dataset import TranslationData, collate_fn
from src.seq2seq.model import Encoder, Decoder, Seq2Seq
from src.seq2seq.train_eval import train, evaluate

import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch import optim

from datetime import datetime
import pickle


# import and process data
data = preprocess(Config.data_path)
# data = data[:5000]  # test a small set
en_vocab, cn_vocab = getVocabulary(data, "English", "Chinese", Config.max_vocab_size)
train_data, eval_data = train_test_split(data, test_split=0.3)

# get dataloader
train_dataset = TranslationData(train_data, en_vocab, cn_vocab)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, collate_fn=collate_fn)

eval_dataset = TranslationData(eval_data, en_vocab, cn_vocab)
eval_loader = DataLoader(eval_dataset, batch_size=Config.batch_size, collate_fn=collate_fn)

# prepare training
encoder = Encoder(en_vocab.vocabulary_size, Config.embedding_size, Config.hidden_size, Config.n_layers, Config.dropout)
decoder = Decoder(cn_vocab.vocabulary_size, Config.embedding_size, Config.hidden_size, Config.n_layers, Config.dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(Config.device)

optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
criterion = NLLLoss(ignore_index=1)  # ignore padding index

# training epochs
training_losses = []
best_eval_loss = 100000
for epoch in range(Config.epochs):
    logger.info(f"\n{datetime.now().strftime('%H:%M:%S')} | Training epoch {epoch+1} ...")
    logger.info("-" * 30)
    training_losses += train(model, train_loader, optimizer, criterion, cn_vocab.vocabulary_size,
                             Config.grad_clip, Config.teacher_forcing)
    logger.info(f"\n{datetime.now().strftime('%H:%M:%S')} | Evaluating epoch {epoch + 1} ...")
    logger.info("-" * 30)
    eval_loss = evaluate(model, eval_loader, criterion, cn_vocab.vocabulary_size)
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        logger.info(f"\n{datetime.now().strftime('%H:%M:%S')} | Saving model ...")
        torch.save(model.state_dict(), Config.model_path + "/seq2seq.bin")
        with open(Config.model_path + '/training_losses.bin', 'wb') as f:
            pickle.dump(training_losses, f)
        with open(Config.model_path + '/eval_loss.bin', 'wb') as f:
            pickle.dump(eval_loss, f)

# translate one sentence
sentence = "I want to go out tonight!"
logger.info("\n[In] " + sentence)                  
logger.info("[Out] " + translation(sentence, model, Config.max_sentence_length, en_vocab, cn_vocab))

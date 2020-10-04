import torch
from torch.nn.utils import clip_grad_norm_
from datetime import datetime
from src.utils.config import Config
from src.utils.process import logger


def train(model, dataloader, optimizer, criterion, vocab_size, grad_clip, teacher_forcing):
    """
    training over one epoch
    :param model: the Seq2Seq model
    :param dataloader: training dataloader
    :param optimizer: training optimiser
    :param criterion: loss function
    :param vocab_size: target vocabulary size
    :param grad_clip: max gradient
    :param teacher_forcing: teacher forcing ratio for training
    :return: list of losses per 100 mini-batches
    """
    model.train()
    batch_losses = []
    batch_loss = 0
    n_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        source, target = batch
        source = source.to(Config.device)
        target = target.to(Config.device)
        target_len = target.size(0)
        optimizer.zero_grad()
        output = model(source, target, teacher_forcing)  # forward propagation
        output = output.to(Config.device)
        loss = criterion(output[1:].view(-1, vocab_size),
                         target[1:].contiguous().view(-1))  # calculate NLL loss, ignore first token <SOS>
        loss.backward()  # backward propagation
        clip_grad_norm_(model.parameters(), grad_clip)  # clip gradients
        optimizer.step()  # update parameters
        batch_loss += loss.data.item() / target_len

        # print results every 100 mini-batches
        if i % 100 == 0 and i != 0:
            batch_loss = batch_loss / 100  # average loss
            batch_losses.append(batch_loss)
            logger.info("%s | Finished %.1f%% | Mini-batch %d | Avg Loss: %5.2f" %
                        (datetime.now().strftime('%H:%M:%S'), (i+1) / n_batches * 100, i+1, batch_loss))
            batch_loss = 0
    return batch_losses


def evaluate(model, dataloader, criterion, vocab_size):
    """
    evaluation over one epoch
    :param model: the Seq2Seq model
    :param dataloader: training dataloader
    :param criterion: loss function
    :param vocab_size: target vocabulary size
    :return: average evaluation loss
    """
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        batch_loss = 0
        n_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
            source, target = batch
            source = source.to(Config.device)
            target = target.to(Config.device)
            target_len = target.size(0)
            output = model(source, target, teacher_forcing=0.0)  # forward propagation
            output = output.to(Config.device)
            loss = criterion(output[1:].view(-1, vocab_size),
                             target[1:].contiguous().view(-1))  # calculate NLL loss
            eval_loss += loss.data.item() / target_len
            batch_loss += loss.data.item() / target_len

            # print results every 100 mini-batches
            if i % 100 == 0 and i != 0:
                batch_loss = batch_loss / 100  # average loss
                logger.info("%s | Finished %.1f%% | Mini-batch %d | Avg Loss: %5.2f" %
                            (datetime.now().strftime('%H:%M:%S'), (i + 1) / n_batches * 100, i + 1, batch_loss))
                batch_loss = 0
    return eval_loss / len(dataloader)

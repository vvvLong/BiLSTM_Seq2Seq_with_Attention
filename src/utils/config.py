import os
import torch


class Config(object):
    """some model configurations"""

    # paths
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(os.path.split(cur_path)[0])[0]
    data_path = root_path + "/data/cmn.txt"
    model_path = root_path + "/model"
    log_path = root_path + "/logs"

    # data processing
    max_vocab_size = 10000

    # model setting
    embedding_size = 300
    hidden_size = 256
    n_layers = 2
    dropout = 0.5
    teacher_forcing = 0.5

    # model training
    batch_size = 32
    learning_rate = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = 10.0
    epochs = 100

    # translation
    max_sentence_length = 20

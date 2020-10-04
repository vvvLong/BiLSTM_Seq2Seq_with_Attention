# BiLSTM_Seq2Seq_with_Attention
A bidirectional LSTM with attention on English to Chinese translation dataset

This small practice includes:
* an implementation of classic Seq2Seq model
* a customised vocabulary, torch Dataset and Dataloader with dynamic padding
* usage of GPU if available
* only requirements of PyTorch and standard Python 3 libraries

# Model structure
* Encoder: Bidirectional 2-layer LSTM
* Decoder: Unidirectional 2-layer LSTM
* Attention: 2-layer MLP

# File structure
* src: all Python scripts; run main.py to train; change any model configs in config.py
* data: path to translation dataset
* logs: log file containing the details of training with 100 epochs
* model: path to saved models and results

# References
* [English to Chinese data][1]
* [Original paper of Attention model][2]
* [mini seq2seq by keon][3]
* [PyTorch official tutorial][4]

[1]: https://www.manythings.org/anki/
[2]: https://arxiv.org/pdf/1409.0473.pdf
[3]: https://github.com/keon/seq2seq
[4]: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

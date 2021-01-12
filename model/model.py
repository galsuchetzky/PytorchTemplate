import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EncoderRNN(BaseModel):
    def __init__(self, input_size, hidden_size, vocab):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.embedding = nn.Embedding(self.vocab.n_tokens, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, h_0):
        ### YOUR CODE HERE
        # input dim: (seq_len,1)
        # input_embed dim: (seq_len,1, hidden_size)
        input_embed = self.embedding(input)
        # h_0 shape: (1,1,hidden_size)
        # output shape: (seq_len,1, hidden_size)
        output, h_0 = self.gru(input_embed, h_0)
        ### --------------
        return output, h_0

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderSimple(BaseModel):
    def __init__(self, input_size, hidden_size, vocab, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.output_size = self.vocab.n_tokens

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # for projecting the last hidden state of the encoder to the decoder space,
        # as the first decoder hidden state, in case the two dimensions don't match
        self.W_p = nn.Linear(enc_hidden_size, hidden_size)

        self.gru_cell = nn.GRUCell(self.input_size, self.hidden_size)

        # for output
        self.W_s = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, targets, h, evaluation_mode=False, **kwargs):
        ### YOUR CODE HERE
        outputs = []  # For keeping the outputs
        # h dim before (1,1,hidden_size)
        # h dim after (1,hidden_size)
        h = self.W_p(h).squeeze(0)  # Project the last hidden state of the encoder to the input size of the decoder.
        start_token = torch.tensor(self.vocab.t2i[SOS_STR], device=device).view(1, -1)
        # targets dim (target_seq_len, 1)
        targets = torch.cat((start_token,
                             targets))  # For code simplicity, the first input that is fed to the rnn is the start token, so by setting in to be the first target i can just put it in as the first input.

        for i, target in enumerate(targets[:-1]):
            # target dim (1)
            # input dim  (1, hidden_size)
            input = s
            elf.embedding(target)
            if evaluation_mode and i > 0:  # use the output of the model rather then the target as input only for evaluation
                input = self.embedding(torch.tensor(torch.argmax(output, dim=1)))

            h = self.gru_cell(input, h)
            # h dim (1,hidden_size)
            output = self.W_s(h)
            # output dim (1,output_size)
            outputs.append(output)
        # outputs dim (seq_len, output_size)
        outputs = torch.vstack(outputs)
        ### --------------
        return outputs


class EncoderDecoder(BaseModel):
    def __init__(self, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, vocab):
        super().__init__()

        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.vocab = vocab
        self.output_size = self.vocab.n_tokens
        self.encoder = EncoderRNN(self.enc_input_size, self.enc_hidden_size, self.vocab)
        self.decoder = DecoderSimple(self.enc_input_size, self.enc_hidden_size, self.vocab)


    def forward(self, targets, h, evaluation_mode=False, **kwargs):
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel

# TODO add documentation and shapes everywhere.

class MnistModel(BaseModel):
    def __init__(self, device, num_classes=10):
        super().__init__(device)
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
    """
    Simple encoder for seq2seq.
    """

    def __init__(self, input_size, hidden_size, vocab, device):
        """
        :param input_size: The size of the input tensor.
        :param hidden_size: The size of the hidden state on the RNN.
        :param vocab: The input vocabulary.
        """
        super().__init__(device)
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, h_0):
        """
        Runs the forward pass of the encoder.
        :param input: The input sequence for the encoder.
        :param h_0: The final hidden state.
        :return: the output of the sequence and the initial hidden state.
        """
        ### YOUR CODE HERE
        # input dim: (seq_len,1)
        # input_embed dim: (seq_len,1, hidden_size)
        input_embeds = self.embedding(input)
        # h_0 shape: (1,1,hidden_size)
        # output shape: (seq_len,1, hidden_size)
        # print('input device:', input_embeds.get_device())
        # print('hidden device:', h_0.get_device())
        output, h_0 = self.gru(input_embeds, h_0)
        ### --------------
        return output, h_0

    def init_hidden(self):
        """
        :return: The initial hidden state of the encoder.
        """
        return torch.zeros(1, 1, self.hidden_size)


class DecoderSimple(BaseModel):
    """
    simple decoder for seq2seq.
    """

    def __init__(self, input_size, enc_hidden_size, hidden_size, vocab, sos_str, eos_str, device, **kwargs):
        """
        :param input_size: The size of the input tensor.
        :param hidden_size: The size of the hidden states of the decoder.
        :param vocab: The output vocabulary.
        :param enc_hidden_size: The size of the encoder hidden state.
        :param kwargs: Additional arguments.
        """
        super().__init__(device)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.vocab = vocab
        self.output_size = len(self.vocab)
        self.SOS_STR = sos_str
        self.EOS_STR = eos_str

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
        start_token = torch.tensor(self.vocab[self.SOS_STR], device=self.device).view(1, -1)

        # print("device of start:", start_token.get_device())
        # print("device of targets:", targets.get_device())
        # targets dim (target_seq_len, 1)
        targets = torch.cat((start_token, targets))
        for i, target in enumerate(targets[:-1]):
            # target dim (1)
            # input dim  (1, hidden_size)
            input = self.embedding(target)
            # use the output of the model rather then the target as input only for evaluation
            if evaluation_mode and i > 0:
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
    """
    Seq2Seq basic.
    """

    def __init__(self, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, vocab, device):
        """
        :param enc_input_size: The dimension of the input embeddings for the encoder.
        :param dec_input_size: The dimension of the input embeddings for the decoder.
        :param enc_hidden_size: The size of the encoder hidden state.
        :param dec_hidden_size: The size of the decoder hidden state.
        :param vocab: The vocabulary of the input and output.
        """
        super().__init__(device)

        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.vocab = vocab
        self.device = device
        self.output_size = len(self.vocab)
        self.SOS_STR = '<sos>'
        self.EOS_STR = '<eos>'
        self.encoder = EncoderRNN(self.enc_input_size, self.enc_hidden_size, self.vocab, self.device)
        self.decoder = DecoderSimple(self.enc_input_size, self.enc_hidden_size, self.dec_hidden_size, self.vocab,
                                     self.SOS_STR, self.EOS_STR, self.device)

    def forward(self, input_tensor, target_tensor, evaluation_mode=False, **kwargs):
        encoder_hidden_first = self.encoder.init_hidden().to(self.device)
        encoder_outputs, encoder_h_m = self.encoder(input_tensor, encoder_hidden_first)
        decoder_hidden = encoder_h_m
        decoder_outputs = self.decoder(target_tensor, decoder_hidden,
                                       enc_input=input_tensor, enc_outputs=encoder_outputs)

        return decoder_outputs

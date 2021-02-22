import torch
import torch.nn as nn

from .base_BREAK_model import BaseBREAKModel
from .base_model import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
# TODO add documentation and shapes everywhere.
# TODO remove the "your code here" from the models. give credits in a docstring above to the course.


class EncoderRNN(BaseModel):
    """
    Simple encoder for seq2seq.
    """

    def __init__(self, batch_size, input_size, hidden_size, vocab, device):
        """
        :param input_size: The size of the input tensor.
        :param hidden_size: The size of the hidden state on the RNN.
        :param vocab: The input vocabulary.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab = vocab

        # TODO use padding_idx in embedding
        self.embedding = nn.Embedding(len(self.vocab), hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, h_0):
        """
        Runs the forward pass of the encoder.
        :param input: The input sequence for the encoder.
        :param h_0: The final hidden state.
        :return: the output of the sequence and the initial hidden state.
        """
        ### YOUR CODE HERE
        # input dim: (batch_size, seq_len)
        # input_embed dim: (batch_size, seq_len, hidden_size)
        input_embeds = self.embedding(input)
        # h_0 shape: (1,batch_size,hidden_size)
        # output shape: (batch_size, seq_len, hidden_size)
        output, h_0 = self.gru(input_embeds, h_0)
        ### --------------
        return output, h_0

    def init_hidden(self):
        """
        :return: The initial hidden state of the encoder.
        """
        return torch.zeros(1, self.batch_size, self.hidden_size)


class DecoderSimple(BaseModel):
    """
    simple decoder for seq2seq.
    """

    def __init__(self, batch_size, input_size, enc_hidden_size, hidden_size, vocab, sos_str, eos_str, device, **kwargs):
        """
        :param input_size: The size of the input tensor.
        :param hidden_size: The size of the hidden states of the decoder.
        :param vocab: The output vocabulary.
        :param enc_hidden_size: The size of the encoder hidden state.
        :param kwargs: Additional arguments.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.vocab = vocab
        self.output_size = len(self.vocab)
        self.SOS_STR = sos_str
        self.EOS_STR = eos_str

        # TODO use padding_idx in embedding
        self.embedding = nn.Embedding(self.output_size, embedding_dim=self.hidden_size)

        # for projecting the last hidden state of the encoder to the decoder space,
        # as the first decoder hidden state, in case the two dimensions don't match
        self.W_p = nn.Linear(enc_hidden_size, hidden_size)

        self.gru_cell = nn.GRUCell(self.input_size, self.hidden_size)

        # for output
        self.W_s = nn.Linear(self.hidden_size, self.output_size)
        # TODO maybe replace linear layer with embedding matrix - usually tied to encoder embedding matrix)
        # TODO add dropout layers

    def forward(self, targets, h, evaluation_mode=False, **kwargs):
        ### YOUR CODE HERE
        # if evaluation_mode:
        #     generation_length = 256
        # else:
        #     generation_length = len(targets)
        outputs = []  # For keeping the outputs
        # h dim before (1,batch_size, enc_hidden_size)
        # h dim after (batch_size, hidden_size)
        h = self.W_p(h).squeeze(0)  # Project the last hidden state of the encoder to the input size of the decoder.
        # start_token = torch.tensor(self.vocab[self.SOS_STR], device=self.device).view(1, -1)

        start_token = torch.full(size=(self.batch_size, 1), fill_value=self.vocab[self.SOS_STR], device=self.device)
        # end_token = torch.full(size=(self.batch_size, 1), fill_value=self.vocab[self.EOS_STR], device=self.device)

        # targets dim before (batch_size, target_seq_len)
        # targets dim after (batch_size, target_seq_len + 1)
        targets = torch.cat((start_token, targets), dim=1)
        # loop through each index in the targets (all the batch targets together), except the last one
        # TODO in eval mode need to loop until EOS token reached
        # for i in range(generation_length):
        for i, target in enumerate(torch.transpose(targets, 0, 1)[:-1]):
            if evaluation_mode and i > 0:  # use the output of the model rather then the target as input only for evaluation
                argmax_idx = torch.tensor(torch.argmax(output, dim=1))
                input = self.embedding(argmax_idx)
            else:
                input = self.embedding(target)
            # target dim (batch_size)
            # input dim  (batch_size, hidden_size)
            # input = self.embedding(target)
            # use the output of the model rather then the target as input only for evaluation
            # if evaluation_mode and i > 0:
            #     input = self.embedding(torch.tensor(torch.argmax(output, dim=1)))

            h = self.gru_cell(input, h)
            # h dim (batch_size, hidden_size))
            output = self.W_s(h)
            # output dim (batch_size,output_size)
            outputs.append(output)
        # outputs dim (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)
        ### --------------
        return outputs


class EncoderDecoder(BaseBREAKModel):
    """
    Seq2Seq basic.
    """

    def __init__(self, batch_size, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, vocab, device):
        """
        :param enc_input_size: The dimension of the input embeddings for the encoder.
        :param dec_input_size: The dimension of the input embeddings for the decoder.
        :param enc_hidden_size: The size of the encoder hidden state.
        :param dec_hidden_size: The size of the decoder hidden state.
        :param vocab: The vocabulary of the input and output.
        :param device: The device to use for the model.
        """
        super().__init__(device)

        self.batch_size = batch_size
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.vocab = vocab
        self.device = device
        self.output_size = len(self.vocab)
        self.SOS_STR = '<sos>'
        self.EOS_STR = '<eos>'
        self.encoder = EncoderRNN(self.batch_size, self.enc_input_size, self.enc_hidden_size, self.vocab, self.device)
        self.decoder = DecoderSimple(self.batch_size, self.enc_input_size, self.enc_hidden_size, self.dec_hidden_size,
                                     self.vocab,
                                     self.SOS_STR, self.EOS_STR, self.device)

    def forward(self, input_tensor, target_tensor, evaluation_mode=False, **kwargs):
        # TODO stop predicting when reaching EOS (dont evaluate the rest predicted), but keep predicting for the rest of the batch_length
        # TODO when in eval_mode do not use target at all
        encoder_hidden_first = self.encoder.init_hidden().to(self.device)
        encoder_outputs, encoder_h_m = self.encoder(input_tensor, encoder_hidden_first)
        decoder_hidden = encoder_h_m
        decoder_outputs = self.decoder(target_tensor, decoder_hidden,
                                       enc_input=input_tensor, enc_outputs=encoder_outputs)

        return decoder_outputs


class BartBREAK(BaseBREAKModel):
    def __init__(self, batch_size, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, vocab, device):
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_BREAK_model import BaseBREAKModel
from .base_model import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
from utils.util import masked_softmax
from .loss import masked_sequence_cross_entropy_with_logits


# TODO add documentation and shapes everywhere.
# TODO remove the "your code here" from the models. give credits in a docstring above to the course.


class EncoderRNN(BaseModel):
    """
    Simple encoder for seq2seq.
    """

    def __init__(self, batch_size, input_size, hidden_size, is_optimal_encoder, dropout_rate, vocab,
                 encoder_embedding_weight, device):
        """
        :param input_size: The size of the input tensor.
        :param hidden_size: The size of the hidden state on the RNN.
        :param vocab: The input vocabulary.
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.is_optimal_encoder = is_optimal_encoder
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), hidden_size)
        self.embedding.weight = encoder_embedding_weight
        if self.is_optimal_encoder:
            self.num_layers = 2
            self.dropout_rate = dropout_rate
            self.bidirectional = True
            self.num_directions = 2
            self.linear_hidden = nn.Linear(self.num_directions * self.num_layers, 1)
            self.linear_output = nn.Linear(self.num_directions, 1)
        else:
            self.num_layers = 1
            self.dropout_rate = 0
            self.bidirectional = False
            self.num_directions = 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, batch_first=True,
                          dropout=self.dropout_rate, bidirectional=self.bidirectional)
        self.a = 1

    def forward(self, input, h_0):
        """
        Runs the forward pass of the encoder.
        :param input: The input sequence for the encoder.
        :param h_0: The final hidden state.
        :return: the output of the sequence and the initial hidden state.
        """
        # input dim: (batch_size, seq_len)
        # input_embed dim: (batch_size, seq_len, hidden_size)
        input_embeds = self.embedding(input)
        # h_0 shape: (1,batch_size,hidden_size)
        # output shape: (batch_size, seq_len, hidden_size)
        output, h_0 = self.gru(input_embeds, h_0)
        if self.is_optimal_encoder:
            h_0 = h_0.transpose(0, 2)
            h_0 = self.linear_hidden(h_0)
            h_0 = h_0.transpose(0, 2)

            output = self.linear_output(output)
        return output, h_0

    def init_hidden(self):
        """
        :return: The initial hidden state of the encoder.
        """
        init_h = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        torch.nn.init.xavier_uniform(init_h)
        return init_h


class DecoderRNN(BaseModel):
    """
    simple decoder for seq2seq.
    """

    def __init__(self, batch_size, input_size, enc_hidden_size, hidden_size, is_dynamic,
                 is_attention, is_tied_weights, is_dropout, is_xavier, is_multilayer, dropout_rate, vocab, sos_str,
                 eos_str,
                 tied_weight, device, **kwargs):
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
        self.is_dynamic = is_dynamic
        self.is_attention = is_attention
        self.is_tied_weights = is_tied_weights
        self.is_dropout = is_dropout
        self.is_xavier = is_xavier
        self.is_multilayer = is_multilayer

        self.vocab = vocab
        self.output_size = len(self.vocab)
        self.SOS_STR = sos_str
        self.EOS_STR = eos_str

        self.embedding = nn.Embedding(self.output_size, embedding_dim=self.input_size)

        # for projecting the last hidden state of the encoder to the decoder space,
        # as the first decoder hidden state, in case the two dimensions don't match
        self.W_project_hidden = nn.Linear(enc_hidden_size, hidden_size)

        self.gru_cell0 = nn.GRUCell(self.input_size, self.hidden_size)
        self.gru_cell1 = nn.GRUCell(self.hidden_size, self.hidden_size)

        # for attention
        self.W_project_outputs = nn.Linear(self.enc_hidden_size, self.hidden_size)
        self.W_attn_combine = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # For Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        if is_xavier:
            torch.nn.init.xavier_uniform_(self.W_project_hidden.weight)
            torch.nn.init.xavier_uniform_(self.W_project_outputs.weight)
            torch.nn.init.xavier_uniform_(self.W_attn_combine.weight)

        # for output
        self.W_out = nn.Linear(self.hidden_size, self.output_size)
        if self.is_tied_weights:
            self.embedding.weight = tied_weight
            self.W_out.weight = tied_weight

    def CalculateAttention(self, hidden, encoder_hiddens_proj):
        """ Calculates the attention vector. """
        # TODO explanation https://stackoverflow.com/a/50829107/15281056
        # encoder_hiddens_proj dim (batch_size, question_length, hidden_size)
        # scores dim (batch_size, source_seq_len, 1)
        scores = torch.bmm(encoder_hiddens_proj, hidden.unsqueeze(2))

        # reshape to match encoder. dim (batch_size, 1, source_seq_len)
        reshaped_scores = scores.unsqueeze(1).squeeze(-1)
        # softmax over tokens of the question
        # TODO maybe change to dim 1
        reshaped_scores = F.softmax(reshaped_scores, dim=2)

        # result dim (batch_size, hidden_size)
        result = torch.bmm(reshaped_scores, encoder_hiddens_proj)
        result = result.squeeze(1)
        return result

    def forward(self, targets, h, lexicon_ids, enc_input, enc_outputs, evaluation_mode=False, **kwargs):

        outputs = []
        masks = []
        # h dim before (1,batch_size, enc_hidden_size)
        # h dim after (batch_size, hidden_size)
        # Project the last hidden state of the encoder to the input size of the decoder.
        h = self.W_project_hidden(h).squeeze(0)
        if self.is_dropout:
            h = self.dropout(h)

        # start_token = torch.tensor(self.vocab[self.SOS_STR], device=self.device).view(1, -1)
        start_token = torch.full(size=(self.batch_size, 1), fill_value=self.vocab[self.SOS_STR], device=self.device)
        # end_token = torch.full(size=(self.batch_size, 1), fill_value=self.vocab[self.EOS_STR], device=self.device)

        # targets dim before (batch_size, target_seq_len)
        # targets dim after (batch_size, target_seq_len + 1)
        targets = torch.cat((start_token, targets), dim=1)

        # enc_outputs dim (batch_size, question_length, enc_hidden_size)
        # encoder_hiddens_proj dim (batch_size, question_length, hidden_size)
        encoder_hiddens_proj = self.W_project_outputs(enc_outputs)
        h0 = h
        h1 = h
        # loop through each index in the targets (all the batch targets together), except the last one
        for i, target in enumerate(torch.transpose(targets, 0, 1)[:-1]):
            if evaluation_mode and i > 0:  # use the output of the model rather then the target as input only for evaluation
                argmax_idx = torch.tensor(torch.argmax(output, dim=1))
                input = self.embedding(argmax_idx)
            else:
                input = self.embedding(target)
            # target dim (batch_size)
            # input dim  (batch_size, hidden_size)

            if self.is_dropout:
                input = self.dropout(input)

            # h dim (batch_size, hidden_size)
            h0 = self.gru_cell0(input, h0)
            h1 = self.gru_cell1(h0, h1)
            if self.is_multilayer:
                h = h1
            else:
                h = h0
            if self.is_dropout:
                h = self.dropout(h)

            if self.is_attention:
                attention_vec = self.CalculateAttention(h, encoder_hiddens_proj)
                last_hidden = self.W_attn_combine(torch.cat((h, attention_vec), dim=1))
                last_hidden = F.relu(last_hidden)
            else:
                last_hidden = h
            # TODO it must be the last computation in each loop
            # output dim (batch_size,output_size)
            output = self.W_out(last_hidden)
            if self.is_dynamic:
                fill_value = -1e32
                # the returned Tensor has the same torch.dtype and torch.device as this tensor
                lexicon_adjustment = output.new_full((self.batch_size, self.output_size), fill_value)
                lexicon_adjustment[torch.arange(self.batch_size).unsqueeze(1), lexicon_ids] = 0
                output += lexicon_adjustment
                # we take masked softmax in order to drop thousands of irrelevant classes in loss calc
                mask = (lexicon_adjustment != fill_value).float()
                # perform softmax only in evaluation mode. during training it is part of the loss
                if evaluation_mode:
                    # dim =-1 for softmax over each row in the batch
                    output = masked_softmax(output, mask, dim=-1)
            else:
                mask = output.new_full((self.batch_size, self.output_size), 1).float()
            outputs.append(output)
            masks.append(mask)
        # outputs dim (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)
        masks = torch.stack(masks, dim=1)
        return outputs, masks


class EncoderDecoder(BaseBREAKModel):
    """
    Seq2Seq basic.
    """

    def __init__(self, batch_size, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size, is_dynamic,
                 is_attention, is_tied_weights, is_dropout, is_xavier, is_multilayer, is_optimal_encoder, dropout_rate,
                 vocab, device):
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
        self.is_dynamic = is_dynamic
        self.is_attention = is_attention
        self.is_tied_weights = is_tied_weights
        self.is_dropout = is_dropout
        self.is_xavier = is_xavier
        self.is_multilayer = is_multilayer
        self.is_optimal_encoder = is_optimal_encoder
        self.encoder_embedding = nn.Embedding(self.output_size, embedding_dim=self.enc_hidden_size)
        torch.nn.init.xavier_uniform_(self.encoder_embedding.weight)
        if self.is_tied_weights:
            if self.enc_hidden_size != self.dec_hidden_size:
                raise ValueError('When using the tied flag, enc_hidden_size must be equal to dec_hidden_size')

        self.encoder = EncoderRNN(self.batch_size, self.enc_input_size, self.enc_hidden_size, self.is_optimal_encoder,
                                  dropout_rate,
                                  self.vocab, self.encoder_embedding.weight, self.device)
        self.decoder = DecoderRNN(self.batch_size, self.enc_input_size, self.enc_hidden_size, self.dec_hidden_size,
                                  self.is_dynamic, self.is_attention, self.is_tied_weights, self.is_dropout,
                                  self.is_xavier, self.is_multilayer, dropout_rate, self.vocab, self.SOS_STR,
                                  self.EOS_STR,
                                  self.encoder_embedding.weight, self.device)

    def forward(self, input_tensor, target_tensor, lexicon_ids, evaluation_mode=False, **kwargs):
        encoder_hidden_first = self.encoder.init_hidden().to(self.device)
        encoder_outputs, encoder_h_m = self.encoder(input_tensor, encoder_hidden_first)
        decoder_hidden = encoder_h_m
        decoder_outputs = self.decoder(target_tensor, decoder_hidden, lexicon_ids,
                                       enc_input=input_tensor, enc_outputs=encoder_outputs)

        return decoder_outputs

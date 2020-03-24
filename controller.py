import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from torch.distributions import Categorical
from operations import *

class Controller(nn.Module):
    def __init__(self, args, hidden_size=100, steps=4, device='cpu'):
        super(Controller, self).__init__()
        self.embedding_size = args.embedding_size
        self.len_nodes = steps + 1
        self.len_OPS = len(OP_NAME)
        self.len_combs = len(COMB_NAME)

        self.hidden_size = hidden_size
        self.steps = steps
        self.device = device

        len_action = self.len_nodes + self.len_OPS + self.len_combs
        self.embedding = nn.Embedding(len_action, self.embedding_size)

        self.node_decoders = nn.ModuleList()
        for step in range(steps):
            self.node_decoders.append(nn.Linear(hidden_size, step+2))

        #operations: identity, 3x3 conv, 3x3 maxpool
        self.op_decoder = nn.Linear(hidden_size, self.len_OPS)

        #combine: add, concat
        self.comb_decoder = nn.Linear(hidden_size, self.len_combs)


        self.rnn = nn.LSTMCell(self.embedding_size, hidden_size)

        self.hidden = self.init_hidden()
        self.init_parameters()

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input).squeeze(1)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def sample(self):
        input = torch.LongTensor([[self.len_nodes + self.len_OPS]]).to(self.device)
        h_t, c_t = self.hidden
        ps = []
        log_ps = []
        actions_index = []

        for type in range(2):
            for node in range(self.steps):
                #node1
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0,action_index]
                log_p =F.log_softmax(logits, dim=-1)[0,action_index]
                ps.append(p)
                log_ps.append(log_p)
                actions_index.append(action_index)

                #node2
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                ps.append(p)
                log_ps.append(log_p)
                actions_index.append(action_index)

                #op1
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                ps.append(p)
                log_ps.append(log_p)
                actions_index.append(action_index)

                #op2
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                ps.append(p)
                log_ps.append(log_p)
                actions_index.append(action_index)

                #comb
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.comb_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                ps.append(p)
                log_ps.append(log_p)
                actions_index.append(action_index)

                input = action_index + self.len_nodes + self.len_OPS

        ps = torch.cat(ps)
        log_ps = torch.cat(log_ps)
        actions_index = torch.cat(actions_index)

        return ps, log_ps, actions_index

    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        return (h_t, c_t)

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.node_decoders:
            decoder.bias.data.fill_(0)
        self.op_decoder.bias.data.fill_(0)
        self.comb_decoder.bias.data.fill_(0)
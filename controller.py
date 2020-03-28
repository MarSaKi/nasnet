import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from torch.distributions import Categorical
from operations import *
'''import argparse

parser = argparse.ArgumentParser('minst')
parser.add_argument('--data', type=str, default='./mnist')
parser.add_argument('--train_portion', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=1024)

parser.add_argument('--model_epochs', type=int, default=2)
parser.add_argument('--model_lr', type=float, default=0.025)
parser.add_argument('--model_lr_min', type=float, default=0.001)
parser.add_argument('--model_weight_decay', type=float, default=3e-4)
parser.add_argument('--model_momentum', type=float, default=0.9)
parser.add_argument('--init_channel', type=int, default=4)

parser.add_argument('--arch_epochs', type=int, default=500)
parser.add_argument('--arch_lr', type=float, default=3.5e-4)
parser.add_argument('--episodes', type=int, default=8)
parser.add_argument('--entropy_weight', type=float, default=1e-5)
parser.add_argument('--baseline_weight_decay', type=float, default=0.95)
parser.add_argument('--embedding_size', type=int, default=32)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()'''

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

        self.init_parameters()

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def sample(self):
        input = torch.LongTensor([self.len_nodes + self.len_OPS]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []

        for type in range(2):
            for node in range(self.steps):
                #node1
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0,action_index]
                log_p =F.log_softmax(logits, dim=-1)[0,action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                #node2
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                #op1
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                #op2
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                #comb
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.comb_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                input = action_index + self.len_nodes + self.len_OPS

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)

        return actions_p, actions_log_p, actions_index

    def get_p(self, actions_index):
        input = torch.LongTensor([self.len_nodes + self.len_OPS]).to(self.device)
        h_t, c_t = self.init_hidden()
        t = 0
        actions_p = []
        actions_log_p = []

        for type in range(2):
            for node in range(self.steps):
                # node1
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                # node2
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.node_decoders[node])
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                # op1
                input = action_index
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                # op2
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                # comb
                input = action_index + self.len_nodes
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.comb_decoder)
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                input = action_index + self.len_nodes + self.len_OPS

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)

        return actions_p, actions_log_p

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

'''controller = Controller(args)
actions_p, actions_log_p, actions_index = controller.sample()
actions_p_new, actions_log_p_new = controller.get_p(actions_index)
actions_importance = actions_p_new / actions_p
print(actions_importance[1].requires_grad, actions_importance[5].requires_grad)
print(actions_importance)'''

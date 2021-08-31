import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class SynthesizerAttention(nn.Module):
    """
    A synthesizer multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MLP Params
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):

        ### TODO:
        ### [part g]: Write your SynthesizerAttention below.
        ###   Do not modify __init__().
        ### Hints:
        ###   - Paste over the CausalSelfAttention above and modify it minimally.
        ###   - Consider especially the parameters self.w1, self.w2 and self.b2.
        ###       How do these map to the matrices in the handout?

        ### START CODE HERE
        #def forward(self, x, layer_past=None):
            # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
            # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head (so that is d/h)

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        w1 = self.w1(x).view(B, T, self.config.n_embed, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        w2 = self.w2(x).view(B, T, self.config.block_size-1, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        b2 = self.b2(x).view(B, T, self.config.block_size-1).transpose(1, 2)
        v = self.value(x).view(B, T, self.config.n_embed,  C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        ##
        # Hi all, I am a little confused by the "init code" in SynthesizerAttention class. There, self.w2 and self.b2 are defined. But I don't know 
        # why the shape has one element with config.block_size-1? Shouldn't we compute a matrix with shape block_size x block_size?
        #
        # Barthold (CF): Interesting question! I believe this is due to the key contribution of the synthesizer, ie replacing the dot product in 
        # self-attention. As you can see in the handout and in the code, for CausalSelfAttention we can instantiate all linear layers with the hyper parameter d 
        # (config.n_embd) due to the nice dot products making sequence lengths, l, "matching". However, for the synthesizer we can do this only for 
        # the first "layer" (A in in the handout, w1 in the code). The second "layer" (B or w2) is of dimension d/h x l and, hence, dependent on the 
        # sequence length which may vary. And this is, where I think nn.Parameters() comes in handy. It is more or less the same as nn.Linear() but 
        # you can slice it. This allows you (hint for implementation) to slice w2 and b2, which have been initialized with config.block_size-1 
        # (maximum possible length) to an fitting size depending on actual input length.
        # BTW the authors stress your point explicitly in their paper: "On Parameters Depending on Sequence Length Random and dense Synthesizers both 
        # rely on parameters that depend on length l. In general, we define a maximum length and dynamically truncate to the actual length of each 
        # batch. We note that this is in similar spirit to trainable positional encodings which have been common practice in Transformer models. Hence, 
        # we do not forsee any issue here. In the case that this is really a problem, one potential solution is to project to a smaller value b and 
        # tile b to the maximum sequence length. We leave this exploration to future work." HTH (and is a correct interpretation)!
        ##
        # self.w1 (is A in the hand-out), self.w2 (is B in the hand-out, is of dimension d/h x l) and self.b2
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.ReLU(w1) @ w2[:,:T].transpose(-2, -1) + b2[:T].transpose(-2, -1)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y    
        ### END CODE HERE

        raise NotImplementedError

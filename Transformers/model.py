class SelfAttention(nn.Module):
      def __init__(self, dim):
          super().__init__()

          self.dim = dim
          self.Q = nn.Linear(self.dim, self.dim)
          self.K = nn.Linear(self.dim, self.dim)
          self.V = nn.Linear(self.dim, self.dim)

      def forward(self, x):

          Q = self.Q(x)
          K = self.K(x)
          V = self.V(x)

          att = (Q@K.T)/math.sqrt(self.dim) 

          # causal mask 
          T = att.shape[0]
          mask = torch.tril(torch.ones(T, T))

          att = att.masked_fill(mask == 0, float('-inf'))

          weights = F.softmax(att, dim=-1)

          out = weights @ V  # (T, dim)

          return out


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = self.MaskedSelfAttention(d_model, n_heads)
        self.ffn = self.FeedForward(d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffn(self.ln2(x))    # residual + ff
        return x

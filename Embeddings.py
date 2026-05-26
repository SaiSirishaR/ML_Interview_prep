# Step 1: Input text --> Integers
# Step 2: Integers --> Tensor
# Step 3: Tensor --> Embeddings


###################################### Step 1: character tokeniser #####################################

corpus = 'Hello world!'

chars = sorted(list(set(corpus)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encoder(text):
    return [stoi[f] for f in text]

def decoder(dec_inp):
    return [itos[f] for f in dec_inp]

inp_text = 'Hello!'
encode = encoder(inp_text)
print("encoder", encode)

#######################################################################################################

#################################### Step 3: Embeddings ###############################################

class TokenEmbedding(nn.Module):
      def __init__(self, vocab_size, dim):
          super().__init__()
          self.vocab_size = vocab_size
          self.dim = dim
          self.embed = nn.Embedding(self.vocab_size, self.dim)
          
      def forward(self, inp):
          return self.embed(inp)

#######################################################################################################

#################################### Step 2: IDs to Tensors ###########################################

x = torch.tensor(encode, dtype = torch.long)

model = TokenEmbedding(len(stoi), 8)

embeds = model(x)

print("Embeddinds are:",embeds)

#######################################################################################################


decode = decoder(encode)
print("decoder", decode)

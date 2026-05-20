corpus = 'helloworld'

corpus = sorted(list(set(corpus)))
stoi = {ch: i for i, ch in enumerate(corpus)}
itos = {i: ch for ch, i in stoi.items()}

print("string to int", stoi)
print("int to string", itos)

def encoder(text):
    encode = [stoi[v] for v in text]
    return encode

def decoder(enc):
    decode = [itos[vv] for vv in enc]
    return decode    

encoded = encoder('hello')    
print( "Encoded output is", encoded)
decoded = decoder(encoded)    
print("Decoded text is", decoded)

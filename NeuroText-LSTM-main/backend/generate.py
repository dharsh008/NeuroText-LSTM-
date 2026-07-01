import torch
import torch.nn.functional as F
from model import CharLSTM
import numpy as np
import os

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    
    net = CharLSTM(len(checkpoint['tokens']), 
                   n_hidden=checkpoint['n_hidden'], 
                   n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    net.chars = checkpoint['tokens']
    net.int2char = dict(enumerate(net.chars))
    net.char2int = {ch: ii for ii, ch in net.int2char.items()}
    
    return net

def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        
        # one-hot encode
        x = np.zeros((1, 1, len(net.chars)), dtype=np.float32)
        x[0, 0, net.char2int[char]] = 1.
        
        inputs = torch.from_numpy(x)
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
            
        # select the next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h

def sample(net, size, prime='The', top_k=None):
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

if __name__ == "__main__":
    if os.path.exists("model.pth"):
        net = load_checkpoint("model.pth")
        print(sample(net, 1000, prime='The', top_k=5))
    else:
        print("Model not found. Run train.py first.")

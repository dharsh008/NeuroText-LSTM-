import torch
import torch.nn as nn
from model import CharLSTM
import numpy as np
import pickle
import os

def one_hot_encode(arr, n_labels):
    # Initialize the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def get_batches(arr, batch_size, seq_length):
    # Total number of batches we can make
    n_batches = len(arr) // (batch_size * seq_length)
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size * seq_length]
    
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # Iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def train(net, data, epochs=10, batch_size=8, seq_length=50, lr=0.001, clip=5):
    net.train()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create training and validation data
    val_idx = int(len(data) * 0.9)
    train_data, val_data = data[:val_idx], data[val_idx:]
    
    if os.path.exists("model.pth"):
        print("Model already trained, skipping...")
        return

    print("Training the LSTM model...")
    for e in range(epochs):
        # Initializing hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(train_data, batch_size, seq_length):
            # One-hot encode our data and make them Tensors
            x = one_hot_encode(x, len(net.chars))
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # Zero accumulated gradients
            net.zero_grad()
            
            # Get the output from the model
            output, h = net(inputs, h)
            
            # Calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            
        print(f"Epoch: {e+1}/{epochs}... Loss: {loss.item():.4f}")

    # Save the model
    checkpoint = {
        'n_hidden': net.n_hidden,
        'n_layers': net.n_layers,
        'state_dict': net.state_dict(),
        'tokens': net.chars
    }
    with open('model.pth', 'wb') as f:
        torch.save(checkpoint, f)

if __name__ == "__main__":
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])
    
    net = CharLSTM(len(chars))
    net.chars = chars
    net.int2char = int2char
    net.char2int = char2int
    
    # Train for a few epochs for demonstration
    train(net, encoded, epochs=100, batch_size=2, seq_length=15)

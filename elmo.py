import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import nltk
from collections import Counter

def load_brown_corpus(corpus_path):
    """
    Load Brown Corpus from the specified directory
    """
    corpus = []
    
    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)
        
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    words = nltk.word_tokenize(line.strip().lower())
                    
                    if words:
                        corpus.append(words)
    
    return corpus

def create_vocabulary(corpus, max_vocab_size=10000):
    """
    Create vocabulary from corpus
    
    Args:
    - corpus: List of tokenized sentences
    - max_vocab_size: Maximum number of words in vocabulary
    
    Returns:
    - vocab: Dictionary mapping words to indices
    - reverse_vocab: Dictionary mapping indices to words
    """
    word_counts = Counter(word for sentence in corpus for word in sentence)
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '< SOS >': 2, '<EOS>': 3}
    
    for word, count in word_counts.most_common(max_vocab_size - len(vocab)):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    
    return vocab, reverse_vocab

class BrownCorpusDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=5, max_context_length=20):
        self.data = []
        self.vocab = vocab
        self.window_size = window_size
        self.max_context_length = max_context_length
        
        for sentence in corpus:
            for i in range(len(sentence)):
                # Create context window
                context = sentence[max(0, i-window_size):i] + sentence[i+1:min(len(sentence), i+window_size+1)]
                target = sentence[i]
                
                # Pad or truncate context to fixed length
                if len(context) > max_context_length:
                    context = context[:max_context_length]
                else:
                    context = context + ['<PAD>'] * (max_context_length - len(context))
                
                # Convert words to indices
                context_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in context]
                target_index = self.vocab.get(target, self.vocab['<UNK>'])
                
                self.data.append((context_indices, target_index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    """
    contexts, targets = zip(*batch)
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    return contexts, targets

class ELMoModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(ELMoModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layers
        self.bilstm_forward = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                                      bidirectional=False, batch_first=True)
        self.bilstm_backward = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                                       bidirectional=False, batch_first=True)
        
        # Output layers for prediction
        self.output_forward = nn.Linear(hidden_dim, vocab_size)
        self.output_backward = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, direction='forward'):
        # Embed the input
        embedded = self.embedding(x)
        
        # Forward and backward passes
        if direction == 'forward':
            output, _ = self.bilstm_forward(embedded)
            prediction = self.output_forward(output)
        else:
            # Reverse input for backward direction
            x_reversed = torch.flip(x, [1])
            embedded_reversed = self.embedding(x_reversed)
            output, _ = self.bilstm_backward(embedded_reversed)
            prediction = self.output_backward(output)
        
        return prediction

def train_elmo(model, train_loader, criterion, optimizer, device, direction='forward'):
    model.train()
    total_loss = 0
    
    for batch_context, batch_target in train_loader:
        batch_context = batch_context.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(batch_context, direction)
        
        loss = criterion(predictions[:, -1, :], batch_target)  

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Load Brown Corpus
    corpus_path = 'brown/brown/brown'
    brown_corpus = load_brown_corpus(corpus_path)
    
    vocab, reverse_vocab = create_vocabulary(brown_corpus)
    
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 64
    MAX_CONTEXT_LENGTH = 20
    
    dataset = BrownCorpusDataset(brown_corpus, vocab, max_context_length=MAX_CONTEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ELMoModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        # Train forward direction
        forward_loss = train_elmo(model, dataloader, criterion, optimizer, device, 'forward')
        
        # Train backward direction
        backward_loss = train_elmo(model, dataloader, criterion, optimizer, device, 'backward')
        
        print(f'Epoch {epoch+1}: Forward Loss = {forward_loss:.4f}, Backward Loss = {backward_loss:.4f}')
    
    torch.save(model.state_dict(), 'bilstm.pt')
    
    import json
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    print(f"Total vocabulary size: {len(vocab)}")
    print("Model and vocabulary saved successfully!")

if __name__ == '__main__':
    main()

    
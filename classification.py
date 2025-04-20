import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

from elmo import ELMoModel

class AGNewsDataset(Dataset):
    def __init__(self, descriptions, labels, elmo_model, vocab, max_length=50):
        self.descriptions = descriptions
        self.labels = labels
        self.elmo_model = elmo_model
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in self.descriptions[idx].split()]
        
        description_indices = description_indices[:self.max_length]
        description_indices += [self.vocab['<PAD>']] * (self.max_length - len(description_indices))
        
        description_tensor = torch.tensor(description_indices)
        
        with torch.no_grad():
            elmo_embeddings = []
            for layer in range(2):  
                layer_embedding = self.elmo_model.embedding(description_tensor)
                elmo_embeddings.append(layer_embedding)
            
            combined_embedding = self.combine_embeddings(elmo_embeddings)
        
        return combined_embedding, torch.tensor(self.labels[idx], dtype=torch.long)

    
    def combine_embeddings(self, embeddings, method='concat'):
        if method == 'mean':
            combined = torch.mean(torch.stack(embeddings), dim=0)
        elif method == 'concat':
            combined = torch.cat(embeddings, dim=-1)
        
        return combined

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        output, (hidden, _) = self.rnn(x) 
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        output = self.fc(hidden)
        return output

def train_classifier(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def create_vocab(descriptions):
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for desc in descriptions:
        for word in desc.split():
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    
    return word_to_idx

def load_pretrained_elmo(vocab_size, embedding_dim=300, hidden_dim=512):
    elmo_model = ELMoModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    
    pretrained_state = torch.load('bilstm.pt')
    
    new_state_dict = elmo_model.state_dict()
    
    for name, param in pretrained_state.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            new_state_dict[name].copy_(param)
    
    elmo_model.load_state_dict(new_state_dict)
    
    return elmo_model

def main():
    os.makedirs('news', exist_ok=True)
    
    df = pd.read_csv('news/train.csv')
    
    descriptions = df['Description'].tolist()
    labels = (df['Class Index'] - 1).tolist()
    vocab = create_vocab(descriptions)
    

    X_train, X_test, y_train, y_test = train_test_split(
        descriptions, labels, test_size=0.2, random_state=42
    )
    
    elmo_model = load_pretrained_elmo(vocab_size=len(vocab))
    elmo_model.eval() 
    train_dataset = AGNewsDataset(X_train, torch.tensor(y_train), elmo_model, vocab)
    test_dataset = AGNewsDataset(X_test, torch.tensor(y_test), elmo_model, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = 600  
    num_classes = len(set(labels))
    
    classifier = ClassificationModel(
        input_dim=input_dim, 
        hidden_dim=256, 
        output_dim=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print("Starting Training...")
    for epoch in range(5):
        train_loss = train_classifier(classifier, train_loader, criterion, optimizer, device)
        train_acc = evaluate_model(classifier, train_loader, device)
        test_acc = evaluate_model(classifier, test_loader, device)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, '
              f'Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}')
    
    torch.save(classifier.state_dict(), 'ag_news_classifier.pt')
    print("Training completed and model saved.")

if __name__ == '__main__':
    main()
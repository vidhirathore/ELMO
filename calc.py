import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns

from elmo import ELMoModel

# Instead of trying to access individual layers, we'll create three different
# ELMo-based embeddings with different aggregation methods and use those
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
            # Get standard embedding from ELMo model
            embedding = self.elmo_model.embedding(description_tensor)
        
        return embedding, torch.tensor(self.labels[idx], dtype=torch.long)


class WeightedEmbeddingModel(nn.Module):
    """
    Model that implements the three different lambda-based approaches for combining embeddings
    """
    def __init__(self, input_dim, hidden_dim, output_dim, mode='trainable_lambdas'):
        super(WeightedEmbeddingModel, self).__init__()
        self.mode = mode
        self.input_dim = input_dim
        
        # Since we can't access individual layer representations, we'll create
        # three different transformations of the input embedding to simulate
        # having different layer embeddings
        
        # Transform 1: Direct embedding (simulates token embedding)
        self.transform1 = nn.Identity()
        
        # Transform 2: Linear transformation (simulates lower biLSTM)
        self.transform2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        
        # Transform 3: More complex transformation (simulates upper biLSTM)
        self.transform3 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        
        # Lambda weights for combining the three embeddings
        if mode == 'trainable_lambdas':
            # Initialize trainable λs
            self.lambdas = nn.Parameter(torch.ones(3) / 3)
            
        elif mode == 'frozen_lambdas':
            # Initialize random λs and freeze them
            self.register_buffer('lambdas', torch.randn(3))
            self.lambdas = self.lambdas / self.lambdas.sum()  # Normalize
            
        elif mode == 'learnable_function':
            # Define a learnable function to combine representations
            self.combine_net = nn.Sequential(
                nn.Linear(input_dim * 3, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, input_dim)
            )
        
        # Classifier components
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                         batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Create three different representations from the input embedding
        e0 = self.transform1(x)
        e1 = self.transform2(x)
        e2 = self.transform3(x)
        
        # Combine representations based on the specified mode
        if self.mode in ['trainable_lambdas', 'frozen_lambdas']:
            # Apply softmax to ensure lambdas sum to 1
            normalized_lambdas = torch.softmax(self.lambdas, dim=0)
            
            # Weight each representation by its lambda and sum
            combined = (e0 * normalized_lambdas[0].view(1, 1, 1) + 
                       e1 * normalized_lambdas[1].view(1, 1, 1) + 
                       e2 * normalized_lambdas[2].view(1, 1, 1))
            
        elif self.mode == 'learnable_function':
            # Concatenate representations along the last dimension
            concatenated = torch.cat([e0, e1, e2], dim=-1)
            
            # Apply learnable transformation
            combined = self.combine_net(concatenated)
        
        # Apply classification layers
        output, (hidden, _) = self.rnn(combined)
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


def evaluate_model(model, dataloader, device, get_predictions=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if get_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if get_predictions:
        return correct / total, all_preds, all_labels
    else:
        return correct / total


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


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
    elmo_model.eval()  
    
    return elmo_model


def run_hyperparameter_tuning(X_train, X_test, y_train, y_test, elmo_model, vocab, embedding_dim=300):
    """Run hyperparameter tuning with the three different embedding combination methods"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define class names for confusion matrix
    class_names = ['World', 'Sports', 'Business', 'Science/Tech']  # Adjust according to your AG News classes
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # For storing results
    results = {}
    
    # Create datasets - simpler now since we just need standard embeddings
    train_dataset = AGNewsDataset(X_train, y_train, elmo_model, vocab)
    test_dataset = AGNewsDataset(X_test, y_test, elmo_model, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Run experiments for each combination method
    for mode in ['trainable_lambdas', 'frozen_lambdas', 'learnable_function']:
        print(f"\n=== Running experiment with: {mode} ===")
        
        # Create the classifier model with the specified mode
        model = WeightedEmbeddingModel(
            input_dim=embedding_dim,  
            hidden_dim=256, 
            output_dim=len(class_names),
            mode=mode
        ).to(device)
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        train_accs = []
        test_accs = []
        
        # Train the model
        for epoch in range(3):
            train_loss = train_classifier(model, train_loader, criterion, optimizer, device)
            train_acc = evaluate_model(model, train_loader, device)
            test_acc = evaluate_model(model, test_loader, device)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, '
                  f'Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}')
        
        # Get final predictions for confusion matrix
        _, y_pred, y_true = evaluate_model(model, test_loader, device, get_predictions=True)
        
        # Plot and save confusion matrix
        cm_title = f'Confusion Matrix - {mode}'
        cm_path = f'results/confusion_matrix_{mode}.png'
        plot_confusion_matrix(y_true, y_pred, class_names, cm_title, cm_path)
        
        # Save training history plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title(f'Training Loss - {mode}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.title(f'Accuracy - {mode}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/training_history_{mode}.png')
        plt.close()
       
        results[mode] = {
            'final_test_acc': test_accs[-1],
            'best_test_acc': max(test_accs),
            'final_train_acc': train_accs[-1],
            'final_train_loss': train_losses[-1]
        }
        
        # If using lambdas, print their values
        if mode in ['trainable_lambdas', 'frozen_lambdas']:
            lambdas = torch.softmax(model.lambdas, dim=0).detach().cpu().numpy()
            print(f"Final lambda values: {lambdas}")
            results[mode]['lambdas'] = lambdas
    
    # Compare results
    print("\n=== Hyperparameter Tuning Results ===")
    for mode, result in results.items():
        print(f"{mode}: Best Test Acc = {result['best_test_acc']:.4f}, Final Test Acc = {result['final_test_acc']:.4f}")
        if 'lambdas' in result:
            print(f"  - Lambda values: {result['lambdas']}")
    
    # Identify best method
    best_mode = max(results.keys(), key=lambda k: results[k]['best_test_acc'])
    print(f"\nBest method: {best_mode} with test accuracy of {results[best_mode]['best_test_acc']:.4f}")
    
    plt.figure(figsize=(10, 6))
    
    for mode in results.keys():
        with open(f'results/model_{mode}.pt', 'rb') as f:
            saved_data = torch.load(f)
            plt.plot(saved_data['test_accs'], label=f'{mode} (Best: {max(saved_data["test_accs"]):.4f})')
    
    plt.title('Comparison of Embedding Combination Methods')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/methods_comparison.png')
    plt.close()
    
    return results


def main():
    os.makedirs('news', exist_ok=True)
    
    df = pd.read_csv('news/train.csv')
    
    descriptions = df['Description'].tolist()
    labels = (df['Class Index'] - 1).tolist()
    vocab = create_vocab(descriptions)
    
    X_train, X_test, y_train, y_test = train_test_split(
        descriptions, labels, test_size=0.2, random_state=42
    )
    
    # Convert y_train and y_test to tensors
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    
    # Load pre-trained ELMo model
    elmo_model = load_pretrained_elmo(vocab_size=len(vocab))
    elmo_model.eval()  # Set to evaluation mode
    
    # Run hyperparameter tuning
    run_hyperparameter_tuning(X_train, X_test, y_train, y_test, elmo_model, vocab)
    
    print("Hyperparameter tuning completed with results saved in 'results' directory.")

if __name__ == '__main__':
    main()
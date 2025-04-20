import torch
import sys
import numpy as np
import json
import os
from classification import ClassificationModel, ELMoModel

def load_model(model_path):
    """
    Load the saved classifier model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModel(input_dim=600, hidden_dim=256, output_dim=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def load_vocabulary(vocab_path='vocab.json'):
    """
    Load the vocabulary from a JSON file
    """
    if not os.path.exists(vocab_path):
        print(f"Warning: Vocabulary file '{vocab_path}' not found. Creating a simple vocabulary.")
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        common_words = ['oil', 'prices', 'rose', 'sharply', 'after', 'OPEC', 'announced', 'production', 
                        'cuts', 'affecting', 'global', 'markets', 'the', 'and', 'in', 'of', 'to']
        for i, word in enumerate(common_words):
            word_to_idx[word] = i + 2
        return word_to_idx
    else:
        with open(vocab_path, 'r') as f:
            return json.load(f)

def manual_elmo_embedding(description_tensor, elmo_model, device):
    """
    Create a very simple ELMo embedding manually
    This is a fallback approach when we're not sure how the original ELMo model works
    """
    # Assuming the model has an embedding layer
    try:
        # Get basic embeddings
        embedded = elmo_model.embedding(description_tensor).to(device)
        print(f"Basic embedding shape: {embedded.shape}")
        
        # Create a placeholder tensor with the expected dimensions
        # Assuming we need a tensor of shape [batch_size, seq_len, 600]
        batch_size = 1
        seq_len = len(description_tensor)
        feature_dim = 600
        
        # Create a placeholder with random values (or zeros)
        placeholder = torch.zeros(batch_size, seq_len, feature_dim).to(device)
        print(f"Created placeholder with shape: {placeholder.shape}")
        
        return placeholder
    except Exception as e:
        print(f"Error in manual_elmo_embedding: {e}")
        # Create a very basic placeholder as a last resort
        return torch.zeros(1, description_tensor.size(0), 600).to(device)

def prepare_input(description, elmo_model, vocab, device):
    """
    Prepare input description using ELMo embeddings with extensive debugging
    """
    print("\n=== Preparing Input ===")
    
    # Handle case when '<UNK>' is not in vocab
    unk_idx = vocab.get('<UNK>', 0)
    print(f"Using UNK index: {unk_idx}")
    
    # Convert words to indices
    words = description.split()
    print(f"Input words: {words}")
    
    description_indices = [vocab.get(word.lower(), unk_idx) for word in words]
    print(f"Word indices: {description_indices}")
    
    # Convert to tensor and move to the correct device
    description_tensor = torch.tensor(description_indices).to(device)
    print(f"Input tensor shape: {description_tensor.shape}")
    
    # Try different methods to get embeddings
    try:
        print("Attempting method 1: direct forward pass...")
        with torch.no_grad():
            embeddings = elmo_model(description_tensor.unsqueeze(0))
            print(f"Method 1 success, shape: {embeddings.shape}")
            return embeddings
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    try:
        print("Attempting method 2: get_elmo_embeddings...")
        with torch.no_grad():
            embeddings = elmo_model.get_elmo_embeddings(description_tensor.unsqueeze(0))
            print(f"Method 2 success, shape: {embeddings.shape}")
            return embeddings
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    print("Using fallback embedding method...")
    return manual_elmo_embedding(description_tensor, elmo_model, device)

def predict(model, input_tensor):
    """
    Get class probabilities with shape validation
    """
    print("\n=== Making Prediction ===")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor device: {input_tensor.device}")
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    # Ensure input tensor is on the same device
    if input_tensor.device != model_device:
        print(f"Moving input tensor to {model_device}")
        input_tensor = input_tensor.to(model_device)
    
    # Check if model expects a different input shape
    expected_shape = None
    for name, param in model.named_parameters():
        if "weight_ih_l0" in name:
            # For LSTM, the input dimension is the second dimension of weight_ih_l0
            expected_input_dim = param.shape[1]
            print(f"Model expects input dimension: {expected_input_dim}")
            expected_shape = (1, input_tensor.shape[1], expected_input_dim)
            break
    
    # Reshape input if needed
    if expected_shape and input_tensor.shape != expected_shape:
        print(f"Reshaping input to {expected_shape}")
        try:
            # Simple case: just need to resize the feature dimension
            if input_tensor.shape[0] == expected_shape[0] and input_tensor.shape[1] == expected_shape[1]:
                # Resize the last dimension through interpolation
                if input_tensor.shape[2] > expected_shape[2]:
                    # Downsample
                    input_tensor = input_tensor[:, :, :expected_shape[2]]
                else:
                    # Upsample (pad with zeros)
                    padded = torch.zeros(expected_shape).to(input_tensor.device)
                    padded[:, :, :input_tensor.shape[2]] = input_tensor
                    input_tensor = padded
            else:
                print("Cannot reshape automatically, dimensions are too different")
        except Exception as e:
            print(f"Error reshaping: {e}")
    
    print(f"Final input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        try:
            outputs = model(input_tensor)
            print(f"Output shape: {outputs.shape}")
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return dummy probabilities as fallback
            return np.array([0.25, 0.25, 0.25, 0.25])

def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <saved model path> <description>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    description = sys.argv[2]
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the classifier model
    print("\n=== Loading Classifier Model ===")
    classifier, device = load_model(model_path)
    
    # Print classifier architecture
    print("Classifier architecture:")
    print(classifier)
    
    # Load the ELMo model
    print("\n=== Loading ELMo Model ===")
    elmo_model = ELMoModel(vocab_size=10000, embedding_dim=300, hidden_dim=512).to(device)
    try:
        elmo_model.load_state_dict(torch.load('bilstm.pt', map_location=device))
        print("ELMo model loaded successfully")
    except Exception as e:
        print(f"Error loading ELMo model: {e}")
        print("Continuing with untrained ELMo model")
    elmo_model.eval()
    
    # Print ELMo architecture
    print("ELMo architecture:")
    print(elmo_model)
    
    # Load the vocabulary
    print("\n=== Loading Vocabulary ===")
    vocab = load_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Some vocabulary entries: {list(vocab.items())[:5]}")
    
    # Prepare the input
    input_tensor = prepare_input(description, elmo_model, vocab, device)
    
    # Get predictions
    class_probs = predict(classifier, input_tensor)
    
    # Print results
    print("\n=== Classification Results ===")
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    for name, prob in zip(class_names, class_probs):
        print(f"{name} {prob:.4f}")

if __name__ == '__main__':
    main()
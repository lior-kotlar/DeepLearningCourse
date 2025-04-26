import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

THRESHOLD = 0.05  # Lower threshold
LR = 0.01      # Smaller learning rate
N_EPOCS = 30     # More epochs
BATCH_SIZE = 16  # Larger batch size

def load_train_data(data_directory):
    """Load antigens from files with the specific naming convention from ex1_data"""
    if not os.path.isdir(data_directory):
        print(f"Data directory '{data_directory}' does not exist")
        exit(-1)
    
    # Look for the specific allele files and the negative file
    alleles = {}
    negative_antigens = []
    
    # Expected file pattern: A0101_pos, A0201_pos, etc., and neg
    file_names = os.listdir(data_directory)
    
    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                antigens = [line.strip() for line in f if line.strip()]
                
                if file_name == "negs.txt":
                    negative_antigens = antigens
                    print(f"Loaded {len(antigens)} negative examples from {file_name}")
                else:
                    # Remove .txt extension
                    allele_name = os.path.splitext(file_name)[0]
                    alleles[allele_name] = antigens
                    print(f"Loaded {len(antigens)} antigens for allele {allele_name}")
    
    print(f"Loaded data for {len(alleles)} alleles: {list(alleles.keys())}")
    return alleles, negative_antigens

class MLP_B(nn.Module):
    def __init__(self):
        super(MLP_B, self).__init__()
        self.fc1 = nn.Linear(180, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 6)  # Dynamic number of alleles

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def decide(network_output):
    # network_output: (batch_size, num_alleles)
    probabilities = torch.sigmoid(network_output)
    # Max probability must exceed threshold (0.2)
    detection = torch.max(probabilities, dim=1)[0] > THRESHOLD
    return detection

def preprocess_antigen(antigen):
    """Convert a 9-letter protein sequence to a 180-dimensional feature vector"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    feature_vector = np.zeros(180)
    for i, aa in enumerate(antigen):
        aa_idx = amino_acids.find(aa)
        if aa_idx != -1:
            feature_vector[i * 20 + aa_idx] = 1
    return feature_vector

class AntigenDataset(Dataset):
    def __init__(self, positive_antigens, negative_antigens):
        self.antigens = []
        self.labels = []
        self.original_antigens = []  # Store original antigen sequences
        self.source_indices = []  # Track which source each sample comes from
        
        num_alleles = len(positive_antigens)
        
        # Process positive examples (antigens recognized by at least one allele)
        for allele_idx, (allele_name, antigens) in enumerate(positive_antigens.items()):
            for antigen in antigens:
                feature_vector = preprocess_antigen(antigen)
                self.antigens.append(feature_vector)
                # Using one-hot encoding for the alleles
                label = torch.zeros(num_alleles)
                label[allele_idx] = 1
                self.labels.append(label)
                self.original_antigens.append((antigen, allele_name))
                self.source_indices.append(allele_idx)  # Track allele index
        
        # Process negative examples (antigens not recognized by any allele)
        for antigen in negative_antigens:
            feature_vector = preprocess_antigen(antigen)
            self.antigens.append(feature_vector)
            # For negative examples, all alleles have equal low probability
            self.labels.append(torch.zeros(num_alleles))
            self.original_antigens.append((antigen, "negative"))
            self.source_indices.append(num_alleles)  # Use num_alleles as index for negative samples
    
    def __len__(self):
        return len(self.antigens)
    
    def __getitem__(self, idx):
        return torch.tensor(self.antigens[idx], dtype=torch.float32), self.labels[idx]
    
    def get_stratified_split(self, test_ratio=0.1):
        """
        Create stratified train/test split - each source file contributes
        proportionally to both training and test sets
        """
        train_indices = []
        test_indices = []
        
        # Get unique source indices
        unique_sources = set(self.source_indices)
        
        # For each source, split its samples
        for source in unique_sources:
            # Get indices for this source
            source_indices = [i for i, s in enumerate(self.source_indices) if s == source]
            
            # Shuffle indices
            np.random.shuffle(source_indices)
            
            # Calculate split
            test_size = int(len(source_indices) * test_ratio)
            
            # Add to test and train sets
            test_indices.extend(source_indices[:test_size])
            train_indices.extend(source_indices[test_size:])
        
        return train_indices, test_indices

def run(model, dataloader, num_epochs):
    """Train the model using the provided dataloader"""
    criterion = nn.BCEWithLogitsLoss()  # Soft labels supported
    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)  # logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    
    return model

def run_with_validation(model, train_loader, val_loader, num_epochs):
    """Train the model and check validation accuracy"""
    criterion = nn.BCEWithLogitsLoss()  # Soft labels supported
    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    best_val_accuracy = 0
    epochs_without_improvement = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_accuracy = test_model(model, val_loader)
        
        # Check for improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")
            break
        
        # Print loss for the current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    return model


def get_train_test(alleles_dict, negative_antigens, test_ratio=0.1):
    """Split data into training and testing sets using stratified sampling"""
    dataset = AntigenDataset(alleles_dict, negative_antigens)
    
    # Get stratified split
    train_indices, test_indices = dataset.get_stratified_split(test_ratio)
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Print split statistics
    print(f"Dataset split: {len(train_dataset)} training samples, {len(test_dataset)} testing samples")
    
    return train_loader, test_loader

def get_train_val_test(alleles_dict, negative_antigens, val_ratio=0.1, test_ratio=0.1):
    """Split data into training, validation, and testing sets using stratified sampling"""
    dataset = AntigenDataset(alleles_dict, negative_antigens)
    
    # Get stratified split
    train_indices, test_indices = dataset.get_stratified_split(test_ratio + val_ratio)
    
    # Calculate the size of the validation set
    val_size = int(len(train_indices) * val_ratio / (1 - val_ratio - test_ratio))
    train_indices, val_indices = train_indices[val_size:], train_indices[:val_size]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Print split statistics
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} testing samples")
    
    return train_loader, val_loader, test_loader

def test_model(model, test_loader):
    """Test the model and calculate accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    # Track separate metrics for positive and negative examples
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            detections = decide(outputs) 

            is_positive = (labels.sum(dim=1) == 1)
            
            true_pos += ((detections == True) & (is_positive == True)).sum().item()
            false_pos += ((detections == True) & (is_positive == False)).sum().item()
            true_neg += ((detections == False) & (is_positive == False)).sum().item()
            false_neg += ((detections == False) & (is_positive == True)).sum().item()

            total += labels.size(0)
            correct += (detections == is_positive).sum().item()
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    return accuracy

def predict_single_antigen(model, antigen, allele_names):
    """Make a prediction for a single antigen"""
    model.eval()
    feature_vector = preprocess_antigen(antigen)
    input_tensor = torch.tensor([feature_vector], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)  # raw logits
        probabilities = torch.sigmoid(output)[0]
    
    detection = (torch.max(probabilities) > THRESHOLD).item()

    result = {
        "detection": "Detect" if detection else "Not Detect",
        "probabilities": {allele_name: prob.item() for allele_name, prob in zip(allele_names, probabilities)}
    }
    
    return result


def main():
    # Default data directory
    data_directory = "exercise1/data/ex1_data/clean_data"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    
    # Load allele data and negative examples
    alleles_dict, negative_antigens = load_train_data(data_directory)
    
    # Check if we have the required data
    if not alleles_dict:
        print("No allele data found. Please check the data directory.")
        exit(-1)
    
    if not negative_antigens:
        print("No negative examples found. Please make sure the 'negs.txt' file exists.")
        exit(-1)
    
    # Print statistics
    total_positive = sum(len(antigens) for antigens in alleles_dict.values())
    total_negative = len(negative_antigens)
    print(f"Total positive examples: {total_positive}")
    print(f"Total negative examples: {total_negative}")
    print(f"Positive to negative ratio: {total_positive/total_negative:.2f}")
    
    # Get the allele names for reporting
    allele_names = list(alleles_dict.keys())
    
    # Split data into training and testing sets with stratified sampling

    train_loader, test_loader = get_train_test(alleles_dict, negative_antigens)
    #train_loader, val_loader, test_loader = get_train_val_test(alleles_dict, negative_antigens)

    # Initialize and train model
    model = MLP_B()
    
    model = run(model, train_loader, N_EPOCS)
    #model = run_with_validation(model, train_loader, val_loader, N_EPOCS)
  
    # Evaluate model
    accuracy = test_model(model, test_loader)

    print(accuracy)
    # Example prediction
    print("\nExample predictions:")
    # Example from positive set
    for allele_name, antigens in alleles_dict.items():
        if antigens:
            positive_example = antigens[0]
            print(f"Positive example ({positive_example}) from {allele_name}:")
            result = predict_single_antigen(model, positive_example, allele_names)
            print(f"  Result: {result['detection']}")
            print(f"  Probabilities: {result['probabilities']}")
            break
    
    # Example from negative set
    if negative_antigens:
        negative_example = negative_antigens[0]
        print(f"\nNegative example ({negative_example}):")
        result = predict_single_antigen(model, negative_example, allele_names)
        print(f"  Result: {result['detection']}")
        print(f"  Probabilities: {result['probabilities']}")
    
    # Save the model
    torch.save(model.state_dict(), "antigen_detector_model.pth")
    print("\nModel saved as 'antigen_detector_model.pth'")

if __name__ == '__main__':
    main()
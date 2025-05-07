from matplotlib import pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from prepreprocess import load_train_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import os

#HYPERS
MODEL = 'B'
THRESHOLD = 0.3
FALSE_NEG_WEIGHT = 3.0 
LR = 0.01     
N_EPOCS = 30
BATCH_SIZE = 16

class MLP_B(nn.Module):
    def __init__(self):
        super(MLP_B, self).__init__()
        self.fc1 = nn.Linear(180, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def __str__(self):
        return "B"

    
class MLP_C(nn.Module):
    def __init__(self):
        super(MLP_C, self).__init__()
        self.fc1 = nn.Linear(180, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def __str__(self):
        return "C"
    
class MLP_D(nn.Module):
    def __init__(self):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(180, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def __str__(self):
        return "D"

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
        self.original_antigens = []
        self.source_indices = []
        
        num_alleles = len(positive_antigens)
        
        for allele_idx, (allele_name, antigens) in enumerate(positive_antigens.items()):
            for antigen in antigens:
                feature_vector = preprocess_antigen(antigen)
                self.antigens.append(feature_vector)
                label = torch.zeros(num_alleles)
                label[allele_idx] = 1
                self.labels.append(label)
                self.original_antigens.append((antigen, allele_name))
                self.source_indices.append(allele_idx)
        
        for antigen in negative_antigens:
            feature_vector = preprocess_antigen(antigen)
            self.antigens.append(feature_vector)
            self.labels.append(torch.zeros(num_alleles))
            self.original_antigens.append((antigen, "negative"))
            self.source_indices.append(num_alleles)

    def get_stratified_split(self, test_ratio=0.1):
        """
        Create stratified train/test split - each source file contributes
        proportionally to both training and test sets
        """
        train_indices = []
        test_indices = []
        unique_sources = set(self.source_indices)
        for source in unique_sources:
            source_indices = [i for i, s in enumerate(self.source_indices) if s == source]
            np.random.shuffle(source_indices)
            test_size = int(len(source_indices) * test_ratio)
            
            test_indices.extend(source_indices[:test_size])
            train_indices.extend(source_indices[test_size:])
        
        return train_indices, test_indices
    
    def __len__(self):
        return len(self.antigens)
    
    def __getitem__(self, idx):
        return torch.tensor(self.antigens[idx], dtype=torch.float32), self.labels[idx]
    
    def is_positive(self, idx):
        """Helper method to check if a sample is positive (for any allele)"""
        return self.labels[idx].sum() > 0

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, false_neg_weight=3.0):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.false_neg_weight = false_neg_weight
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        loss = bce_loss(inputs, targets)
        weights = torch.ones_like(targets)
        weights[targets > 0] = self.false_neg_weight
        weighted_loss = loss * weights
        return weighted_loss.mean()

def get_train_test(alleles_dict, negative_antigens, test_ratio=0.1):
    """Split data into training and testing sets using stratified sampling"""
    dataset = AntigenDataset(alleles_dict, negative_antigens)
    train_indices, test_indices = dataset.get_stratified_split(test_ratio)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Dataset split: {len(train_dataset)} training samples, {len(test_dataset)} testing samples")
    
    return train_loader, test_loader

def decide(network_output, threshold):
    probabilities = torch.sigmoid(network_output)
    detection = torch.max(probabilities, dim=1)[0] > threshold
    return detection

def evaluate_model(model, dataloader, criterion, threshold):
    """Evaluate the model and return loss and accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            detections = decide(outputs, threshold)
            is_positive = (labels.sum(dim=1) == 1)
            correct += (detections == is_positive).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, num_epochs, threshold=THRESHOLD, false_neg_weight=FALSE_NEG_WEIGHT):
    """Train the model and track both training and testing error"""
    criterion = WeightedBCEWithLogitsLoss(false_neg_weight=false_neg_weight)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Eval training
        model.eval()
        _, train_accuracy = evaluate_model(model, train_loader, criterion, threshold)
        train_accuracies.append(train_accuracy)
        
        # Eval test
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, threshold)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs)
    
    return model

def plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epochs):
    """Plot training and testing metrics over epochs"""
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, test_losses, 'r-', label='Testing Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, test_accuracies, 'r-', label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


def test_model(model, test_loader, threshold):
    """Test the model and calculate accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            detections = decide(outputs, threshold) 

            is_positive = (labels.sum(dim=1) == 1)
            
            true_pos += ((detections == True) & (is_positive == True)).sum().item()
            false_pos += ((detections == True) & (is_positive == False)).sum().item()
            true_neg += ((detections == False) & (is_positive == False)).sum().item()
            false_neg += ((detections == False) & (is_positive == True)).sum().item()

            total += labels.size(0)
            correct += (detections == is_positive).sum().item()

    
    accuracy = 100 * correct / total if total > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return accuracy

def extract_9mers(protein_sequence):
    """extract 9mers out of the protein sequence"""
    return [protein_sequence[i:i+9] for i in range(len(protein_sequence) - 8)]

def predict_sars(model, peptides_9mer, allele_names):
    """Checks the top 3 peptides in the peptides list and by which allele"""
    model.eval()
    results = []

    for peptide in peptides_9mer:
        x = preprocess_antigen(peptide)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(x_tensor).squeeze(0) 
        for i, score in enumerate(output):
            results.append({
                'peptide': peptide,
                'allele': allele_names[i],
                'score': float(score)
            })
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results


def main(): 
    data_directory = "exercise1/data/ex1_data/clean_data"
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]

    alleles_dict, negative_antigens = load_train_data(data_directory)

    train_loader, test_loader = get_train_test(alleles_dict, negative_antigens)

    if MODEL == 'B':
        model = MLP_B()
    elif MODEL == 'C':
        model = MLP_C()
    else:
        model = MLP_D()
    
    print(f"Training model {model} with false negative weight: {FALSE_NEG_WEIGHT}")
    model = train_model(model,train_loader,test_loader,N_EPOCS,THRESHOLD)

    test_model(model, test_loader, THRESHOLD)

    spike_sequence = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
    "NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV"
    "NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE"
    "GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT"
    "LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK"
    "CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN"
    "CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD"
    "YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC"
    "NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN"
    "FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP"
    "GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSY"
    "ECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI"
    "SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQE"
    "VFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC"
    "LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAM"
    "QMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN"
    "TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRA"
    "SANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA"
    "ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDP"
    "LQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL"
    "QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD"
    "SEPVLKGVKLHYT")

    results = predict_sars(model, extract_9mers(spike_sequence), list(alleles_dict.keys()))
    top_3 = results[:3]

    # Print as table
    print(f"{'Rank':<5} {'Peptide':<12} {'Allele':<20} {'Score':>8}")
    print("-" * 50)
    for i, res in enumerate(top_3, 1):
        print(f"{i:<5} {res['peptide']:<12} {res['allele']:<20} {res['score']:>8.4f}")

if __name__ == '__main__':
    main()
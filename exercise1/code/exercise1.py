import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
from prepreprocess import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
THRESHOLD = 0.3
LR = 0.01
CRITERIA = [nn.CrossEntropyLoss]
N_EPOCS = 30


class peptide_dataset(Dataset):

    def __init__(self, allels_antigens):
        self.amino_dictionary = None
        self.antigens = []
        self.allel_labels = []
        self.amino_dictionary = self.make_amino_dictionary()
        self.num_classes = len(allels_antigens)

        for class_idx, antigen_list in enumerate(allels_antigens):
            self.antigens.extend(antigen_list)
            self.allel_labels.extend([class_idx] * len(antigen_list))


    def __len__(self):
        return len(self.antigens)


    def __getitem__(self, idx):
        antigen_string = self.antigens[idx]
        return self.encode_antigen(antigen_string), self.allel_labels[idx]


    def encode_antigen(self, antigen):
        encoding_vector = []
        for char in antigen:
            encoding_vector.append(self.amino_dictionary[char])
        as_tensor = torch.tensor(encoding_vector, dtype=torch.long)
        output = fun.one_hot(as_tensor, num_classes=len(amino_acids)).float()
        return output


    def make_amino_dictionary(self):
        amino_dictionary = {}
        for i, one_acid in enumerate(amino_acids):
            amino_dictionary[one_acid] = i
        return amino_dictionary


class trainer:
    def __init__(self, data_directory):
        self.peptide_dataset, self.idx_to_class_name = self.load_train_data(data_directory)


    def load_train_data(self, data_directory):# assumes there are no shared words between any files
        print(f'data_directory {data_directory}')
        if not os.path.isdir(data_directory):
            print("Data directory does not exist")
            exit(-1)
        file_names = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
        file_names_no_extension = [os.path.splitext(filename)[0] for filename in file_names]
        antigen_groups_by_allel = []
        idx_to_class_name = {}
        for class_idx, (no_ext, file_name) in enumerate(zip(file_names_no_extension, file_names)):
            idx_to_class_name[class_idx] = no_ext
            file_full_path = os.path.join(data_directory, file_name)
            with open(file_full_path, "r") as f:
                allel_antigens = [line.strip() for line in f if line.strip()]
                print(len(allel_antigens))
                antigen_groups_by_allel.append(allel_antigens)
        dataset = peptide_dataset(antigen_groups_by_allel)
        return dataset, idx_to_class_name




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

def decide(network_output):
    probabilities = torch.softmax(network_output, dim=1)
    max_probability = torch.max(probabilities, dim=1).values
    min_probability = torch.min(probabilities, dim=1).values
    if max_probability - min_probability > THRESHOLD:
        return True
    return False


def run(mod, crit, input, labels, num_epochs):
    model = mod
    criterion = crit()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


def preprocess_antigen(antigen):
    """
    Convert a 9-letter protein sequence to a 180-dimensional feature vector
    using one-hot encoding for each amino acid.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    feature_vector = np.zeros(180)
    for i, aa in enumerate(antigen):
        aa_idx = amino_acids.find(aa)
        if aa_idx != -1:
            feature_vector[i * 20 + aa_idx] = 1
    return feature_vector


def process_data(alleles_dict):
    """
    Process the loaded data to create training features and labels
    """
    all_antigens = []
    all_labels = []

    # Assuming each key in alleles_dict corresponds to one of the 6 alleles
    for idx, (allele_name, antigens) in enumerate(alleles_dict.items()):
        for antigen in antigens:
            # Convert antigen to feature vector - assumes preprocess_antigen is defined in prepreprocess.py
            feature_vector = preprocess_antigen(antigen)
            all_antigens.append(feature_vector)
            all_labels.append(idx)  # Use the index as the label

    # Convert to tensors
    X = torch.tensor(all_antigens, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    return X, y


# def train_model():
#     # Load data
#     alleles_dict = load_train_data("data_directory")
#
#     # Process data
#     X, y = process_data(alleles_dict)
#
#     # Initialize model
#     model = MLP_B()
#
#     # Train model
#     model = run(model, nn.CrossEntropyLoss, X, y, N_EPOCS)
#
#     return model

def test_model(model, test_antigens):
    """
    Test the model on new antigens
    """
    model.eval()
    results = []

    for antigen in test_antigens:
        # Convert antigen to feature vector
        feature_vector = preprocess_antigen(antigen)
        input_tensor = torch.tensor([feature_vector], dtype=torch.float32)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)

        # Make decision using your threshold-based approach
        detection = decide(output)
        results.append("Detect" if detection else "Not Detect")

    return results


def get_train_test():
    pass


def main():
    # train, test = get_train_test()
    # trained_model = train_model(train)
    # results = test_model(trained_model, test)
    #
    # print("Results:")
    # for antigen, result in zip(test, results):
    #     print(f"{antigen}: {result}")
    peptide_trainer = trainer(sys.argv[1])


if __name__ == '__main__':
    main()
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
THRESHOLD = 0.3
LR = 0.0001
CRITERIA = [nn.CrossEntropyLoss]
N_EPOCS = 20
FIXED_NUMBER_CLASSES = 6
REJECTION_METHOD = False
MAX_PROBABILITY_THRESHOLD = 0.5
WEIGHTED = True
BATCH_SIZE = 64
LINEAR = False


class Peptide_Data:

    def __init__(self, allels_antigens):
        self.amino_dictionary = None
        self.antigens_as_strings = []
        self.encoded_antigens = []
        self.encoded_allel_labels = []
        self.amino_dictionary = self.make_amino_dictionary()
        self.num_classes = len(allels_antigens)

        class_sizes = []
        for class_idx, antigen_list in enumerate(allels_antigens):
            class_sizes.append(len(antigen_list))
            self.antigens_as_strings.extend(antigen_list)
            onehot_tensor_antigen_list = [self.encode_antigen(antigen) for antigen in antigen_list]
            self.encoded_antigens.extend(onehot_tensor_antigen_list)
            tensored_label_list = [class_idx] * len(antigen_list)
            encoded_labels_list = [self.encode_label_class(label) for label in tensored_label_list]
            self.encoded_allel_labels.extend(encoded_labels_list)

        self.peptide_length = len(self.encoded_antigens[0])
        self.class_proportions = self.get_class_proportion(class_sizes)


    def getitem_as_string(self, idx):
        return self.antigens_as_strings[idx], self.encoded_allel_labels[idx]


    def encode_antigen(self, antigen):
        encoding_vector = []
        for char in antigen:
            encoding_vector.append(self.amino_dictionary[char])
        as_tensor = torch.tensor(encoding_vector, dtype=torch.long)
        output = fun.one_hot(as_tensor, num_classes=len(amino_acids)).float()
        return output


    def encode_label_onehot(self, label):
        if REJECTION_METHOD:
            EQUAL_DIST = torch.tensor([0.0] * FIXED_NUMBER_CLASSES)
            onehot_label = EQUAL_DIST if label == FIXED_NUMBER_CLASSES \
                else fun.one_hot(torch.tensor(label),
                                 num_classes=self.num_classes).float()
        else:
            onehot_label = fun.one_hot(torch.tensor(label),
                                 num_classes=self.num_classes).float()
        return onehot_label

    def encode_label_class(self, label):
        return label


    def make_amino_dictionary(self):
        amino_dictionary = {}
        for i, one_acid in enumerate(amino_acids):
            amino_dictionary[one_acid] = i
        return amino_dictionary


    def get_class_proportion(self, class_sizes):
        max = np.max(class_sizes)
        min = np.min(class_sizes)
        proportions = [np.sqrt(max/class_size) for class_size in class_sizes] if max/min > 3 else [1] * len(class_sizes)
        proportions = [2.5] * self.num_classes
        proportions[-1] = 1
        print(proportions)
        return proportions



class peptide_dataset(Dataset):

    def __init__(self, peptides, labels):
        assert len(peptides) == len(labels)
        self.peptides = peptides
        self.labels = labels


    def __len__(self):
        return len(self.peptides)


    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]



class trainer:
    def __init__(self, data_directory):
        self.num_epochs = N_EPOCS
        self.batch_size = BATCH_SIZE
        self.peptide_data, self.idx_to_class_name = self.load_peptide_data(data_directory)
        self.train_loader, self.test_loader = self.get_loaders()
        # self.model = MLP_B(self.peptide_data.peptide_length,
        #                    FIXED_NUMBER_CLASSES,
        #                    linear) if REJECTION_METHOD else (MLP_B(self.peptide_data.peptide_length,
        #                                                   self.peptide_data.num_classes,
        #                                                   linear))

        self.model = MLP_B(self.peptide_data.peptide_length,
                           self.peptide_data.num_classes)
        self.loss_function = nn.CrossEntropyLoss(weight=torch.tensor(self.peptide_data.class_proportions).float()) \
            if WEIGHTED else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)


    def load_peptide_data(self, data_directory):# assumes there are no shared words between any files
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
        peptide_data = Peptide_Data(antigen_groups_by_allel)
        return peptide_data, idx_to_class_name


    def get_loaders(self):

        x_train, x_test, y_train, y_test = train_test_split(
            self.peptide_data.encoded_antigens,
            self.peptide_data.encoded_allel_labels,
            test_size=0.1,
            # stratify=self.peptide_dataset.encoded_allel_labels,
            random_state=42
        )

        train_dataset = peptide_dataset(x_train, y_train)
        test_dataset = peptide_dataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader


    def prediction_from_probabilities(self, probabilities):
        max = np.max(probabilities)
        min = np.min(probabilities)
        if max < MAX_PROBABILITY_THRESHOLD:
            pass


    def train(self):
        train_losses = []
        test_losses = []
        test_accuracies = []
        confusion_matrices = []
        running_train_loss = 0.

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch_inputs, batch_labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                # print(f'batch inputs {batch_inputs}\nbatch labels {batch_labels}')
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_function(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)


            self.model.eval()
            running_test_loss = 0.0
            correct = 0
            total = 0

            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch_inputs, batch_labels in self.test_loader:
                    outputs = self.model(batch_inputs)
                    loss = self.loss_function(outputs, batch_labels)
                    running_test_loss += loss.item()
                    probabilities = torch.softmax(outputs, dim=1)
                    # batch_labels = batch_labels.argmax(dim=1)
                    preds = outputs.argmax(dim=1)
                    # print(f'preds: {preds}\nprobabilities: {probabilities}\nlabels: {batch_labels_as_class_idx}\noutputs: {outputs}')
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

            avg_test_loss = running_test_loss / len(self.test_loader)
            test_losses.append(avg_test_loss)

            accuracy = correct / total
            test_accuracies.append(accuracy)

            cm = confusion_matrix(all_labels, all_predictions, labels=list(range(self.peptide_data.num_classes)))
            confusion_matrices.append(cm)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {accuracy * 100:.2f}%")

        return train_losses, test_losses, test_accuracies, confusion_matrices

    def plot_loss_curves(self, train_losses, test_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


    def analyze_confusion_matrix(self, cm):
        print("Per-class accuracy:")
        for i in range(self.peptide_data.num_classes):
            class_name = self.idx_to_class_name[i]
            true_positives = cm[i, i]
            total = cm[i].sum()
            acc = true_positives / total if total > 0 else 0.0
            print(f"  Class '{class_name}': {acc * 100:.2f}% accuracy")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.idx_to_class_name.values())
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.show()



class MLP_B(nn.Module):
    def __init__(self, peptide_length, num_classes):
        super().__init__()
        input_dim = peptide_length * len(amino_acids)
        bigger_layer = 1024
        print(f'input_dim {input_dim}')
        self.linear = LINEAR
        self.flatten = nn.Flatten()
        self.first = nn.Linear(input_dim, bigger_layer)
        self.second = nn.Linear(bigger_layer, bigger_layer)
        self.third = nn.Linear(bigger_layer, num_classes)


    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        x = self.flatten(x)
        if self.linear:
            x = self.first(x)
            x = self.second(x)
            x = self.third(x)
        else:
            x = torch.relu(self.first(x))
            x = torch.relu(self.second(x))
            x = torch.relu(self.third(x))
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

# def test_model(model, test_antigens):
#     """
#     Test the model on new antigens
#     """
#     model.eval()
#     results = []
#
#     for antigen in test_antigens:
#         # Convert antigen to feature vector
#         feature_vector = preprocess_antigen(antigen)
#         input_tensor = torch.tensor([feature_vector], dtype=torch.float32)
#
#         # Forward pass
#         with torch.no_grad():
#             output = model(input_tensor)
#
#         # Make decision using your threshold-based approach
#         detection = decide(output)
#         results.append("Detect" if detection else "Not Detect")
#
#     return results


def get_train_test():
    pass


def main():
    peptide_trainer = trainer(sys.argv[1])
    train_losses, test_losses, test_accuracies, confusion_matrices = peptide_trainer.train()
    peptide_trainer.plot_loss_curves(train_losses, test_losses)
    final_cm = confusion_matrices[-1]
    peptide_trainer. analyze_confusion_matrix(final_cm)



if __name__ == '__main__':
    main()
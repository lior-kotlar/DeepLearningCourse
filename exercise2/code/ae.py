import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns



NUM_DIGITS = 10
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
LARGE_ENCODER_1_LAYER = 16
LARGE_ENCODER_2_LAYER = 16
LARGE_ENCODER_3_LAYER = 16
SMALL_ENCODER_1_LAYER = 4
SMALL_ENCODER_2_LAYER = 5
SMALL_ENCODER_3_LAYER = 6
CLASSIFIER = 0
DECODER = 1
SMALL = 0
LARGE = 1
LARGE_LATENT_DIMENSION = 16
SMALL_LATENT_DIMENSION = 4
MNIST_IMAGE_SIZE = 28
TRAINING_SUBSET_SIZE = 100
sizes_dict = {0:'small', 1:'large'}
model_type_dict = {CLASSIFIER: 'classifier', DECODER: 'decoder'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Conv_AE_custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Conv_AE_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer_size = 4
        self.second_layer_size = 5
        self.third_layer_size = 6
        self.encoder = Encoder(self.first_layer_size, self.second_layer_size, self.third_layer_size)
        self.decoder = Decoder(self.first_layer_size, self.second_layer_size, self.third_layer_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Conv_AE_large(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size):
        super().__init__()
        self.encoder = Encoder(first_layer_size, second_layer_size, third_layer_size)
        self.decoder = Decoder(first_layer_size, second_layer_size, third_layer_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# size = ((input_size-kernel_size+2*padding)/stride) + 1

class Encoder(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size, latent_dimension):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, first_layer_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(first_layer_size, second_layer_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second_layer_size, third_layer_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(third_layer_size, latent_dimension)
        )

    def forward(self, x):
        output = self.encoder(x)
        return output

class Decoder(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size, latent_dimension):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, third_layer_size),
            nn.ReLU(),
            nn.Unflatten(1, (third_layer_size, 1, 1)),
            nn.ConvTranspose2d(third_layer_size, second_layer_size, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(second_layer_size, first_layer_size, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(first_layer_size, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.decoder(x)
        return output



class MLPClassifier(nn.Module):
    def __init__(self, latent_dimension):
        super().__init__()
        self.ff = nn.Linear(latent_dimension, NUM_DIGITS)

    def forward(self, x):
        return self.ff(x)

class EncoderDecoder(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size, latent_dimension_d,
                 model_type, pretrained_encoder_path = None):
        super().__init__()
        self.pretrained = False
        self.encoder = Encoder(first_layer_size, second_layer_size, third_layer_size, latent_dimension_d)
        if pretrained_encoder_path:
            self.encoder.load_state_dict(torch.load(pretrained_encoder_path))
            self.pretrained = True
        if model_type == CLASSIFIER:
            self.second_component = MLPClassifier(latent_dimension=latent_dimension_d)
            self.flatten_after_encoder = True
        else:
            self.second_component = Decoder(first_layer_size, second_layer_size, third_layer_size, latent_dimension_d)
            self.flatten_after_encoder = False

    def forward(self, x):
        encoded = self.encoder(x)
        # if self.flatten_after_encoder:
        #     encoded = encoded.view(encoded.size(0), -1)
        output = self.second_component(encoded)
        return output

def plot_losses_and_reconstruction(outputs, losses, plot_save_directory, num_epochs):
    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9,2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i+1)
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9+i+1)
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])

        save_path = os.path.join(plot_save_directory, f"reconstruction_epoch_{k}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


    plt.figure(figsize=(8,5))
    plt.plot(range(num_epochs), losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plot_save_directory, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def get_mnist_training_sets(train_subset = False):
    transform = transforms.ToTensor()
    mnist_training_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    mnist_test_data = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )

    if train_subset:
        subset_indices = torch.arange(TRAINING_SUBSET_SIZE)
        train_subset_dataset = torch.utils.data.Subset(mnist_training_data, subset_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

    else:
        train_loader = torch.utils.data.DataLoader(
        mnist_training_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        mnist_test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, test_loader

def plot_loss_and_accuracy(train_losses, test_losses,
                           train_accuracies, test_accuracies,
                           plot_save_directory,
                           learning_rate_updates_epochs = None,
                           subset = False):
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    if learning_rate_updates_epochs:
        for epoch_idx in learning_rate_updates_epochs:
            plt.axvline(x=epoch_idx + 1, color='red', linestyle='--', alpha=0.6)
            max_loss = max(train_losses + test_losses)
            plt.text(epoch_idx + 1, max_loss, 'LR ↓', color='red', fontsize=8, ha='center', va='bottom')

    file_name = f'loss and accuracy {TRAINING_SUBSET_SIZE} examples.png' if subset else 'loss and accuracy full dataset.png'

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plot_save_directory, file_name)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def get_training_tools(model_type, latent_dimension, model_size,
                       pretrained_encoder_path=None, train_data_subset = False):

    if model_size == LARGE:
        first_layer_size = LARGE_ENCODER_1_LAYER
        second_layer_size = LARGE_ENCODER_2_LAYER
        third_layer_size = LARGE_ENCODER_3_LAYER
    elif model_size == SMALL:
        first_layer_size = SMALL_ENCODER_1_LAYER
        second_layer_size = SMALL_ENCODER_2_LAYER
        third_layer_size = SMALL_ENCODER_3_LAYER
    else:
        print(f'model_size {model_size} not supported')
        exit(1)
    model = EncoderDecoder(first_layer_size=first_layer_size, second_layer_size=second_layer_size,
                           third_layer_size=third_layer_size, model_type=model_type,
                           pretrained_encoder_path= pretrained_encoder_path,
                           latent_dimension_d=latent_dimension)
    if model_type == CLASSIFIER:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    model.to(device)
    if model.pretrained:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.second_component.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        print('Loaded pretrained encoder, and froze its weights')
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1)
    train_loader, test_loader = get_mnist_training_sets(train_data_subset)

    return model, criterion, optimizer, scheduler, train_loader, test_loader


def show_reconstructions_by_class(encoder_decoder, dataloader, plot_save_directory, samples_per_class=5):
    encoder_decoder.eval()

    digit_to_images = {i: [] for i in range(10)}

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            for img, label in zip(imgs, labels):
                if len(digit_to_images[label.item()]) < samples_per_class:
                    digit_to_images[label.item()].append(img.unsqueeze(0))
            if all(len(v) == samples_per_class for v in digit_to_images.values()):
                break

    fig, axs = plt.subplots(10, samples_per_class * 2, figsize=(samples_per_class * 2, 10))
    fig.suptitle("Original (left) vs Reconstruction (right) - per digit", fontsize=14)

    for digit in range(10):
        imgs = torch.cat(digit_to_images[digit], dim=0)
        with torch.no_grad():
            recons = encoder_decoder(imgs)

        for i in range(samples_per_class):
            axs[digit, 2*i].imshow(imgs[i].squeeze().cpu().numpy(), cmap='gray')
            axs[digit, 2*i].axis('off')
            axs[digit, 2*i + 1].imshow(recons[i].squeeze().cpu().numpy(), cmap='gray')
            axs[digit, 2*i + 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    save_path = os.path.join(plot_save_directory, f"reconstructions_by_class.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def intra_class_distance(data_loader, encoder, plot_save_directory):
    all_latents = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            latents = encoder(imgs)
            all_latents.append(latents.cpu())
            all_labels.append(labels.cpu())

    all_latents = torch.cat(all_latents, dim=0)  # [N, latent_dim]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    class_means = {}
    for digit in range(10):
        class_latents = all_latents[all_labels == digit]
        class_means[digit] = class_latents.mean(dim=0)

    distance_matrix = torch.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            distance_matrix[i, j] = nn.functional.pairwise_distance(
                class_means[i].unsqueeze(0),
                class_means[j].unsqueeze(0),
                p=2
            )

    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix.numpy(), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Inter-Class Distance Matrix in Latent Space")
    plt.xlabel("Digit Class")
    plt.ylabel("Digit Class")
    save_path = os.path.join(plot_save_directory, f"distance_matrix.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    latents_2d = tsne.fit_transform(all_latents.numpy())

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        idx = all_labels == digit
        plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], label=str(digit), alpha=0.6)

    plt.legend()
    plt.title("t-SNE of Latent Space (by Digit Class)")
    save_path = os.path.join(plot_save_directory, f"tSNE latent space.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


def compose_status_message(second_component_type, model_size, pretrained_encoder_path, train_data_subset):
    status_message = (f'starting to train {sizes_dict[model_size]} '
                      f'encoder to {model_type_dict[second_component_type]} model, ')
    if pretrained_encoder_path:
        status_message += f'with a pretrained encoder {pretrained_encoder_path}.'
    else:
        status_message += f'training both encoder and {model_type_dict[second_component_type]}.'

    if train_data_subset:
        status_message += f' using only {TRAINING_SUBSET_SIZE} examples.'
    else:
        status_message += f' using full training set.'
    return status_message


def encoder_to_decoder_training(plot_save_directory,
                                latent_dimension,
                                model_size, pretrained_encoder_path = None,
                                save_encoder_path=None,
                                train_data_subset = False):
    (model, criterion,optimizer,
     scheduler, train_loader, test_loader) = get_training_tools(model_type=DECODER,
                                                                latent_dimension=latent_dimension,
                                                                model_size=model_size,
                                                                pretrained_encoder_path=pretrained_encoder_path,
                                                                train_data_subset=train_data_subset)

    status_message = compose_status_message(DECODER, model_size, pretrained_encoder_path, train_data_subset)
    print(status_message)

    outputs = []
    train_losses, test_losses = [], []
    learning_rate_updates_epochs = []
    prev_lr = LEARNING_RATE
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for img, _ in train_loader:
            img = img.to(device)
            reconstruction = model(img)
            loss = criterion(reconstruction, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        outputs.append((epoch, img, reconstruction))

        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for img, labels in test_loader:
                img = img.to(device)
                reconstruction = model(img)
                loss = criterion(reconstruction, img)
                test_running_loss += loss.item()
        test_loss = test_running_loss / len(test_loader)
        test_losses.append(test_loss)

        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.8f} to {current_lr:.8f}")
            learning_rate_updates_epochs.append(epoch)
            prev_lr = current_lr

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

    plot_save_directory = plot_save_directory + ' d4' if latent_dimension == SMALL_LATENT_DIMENSION else (
            plot_save_directory + ' d16')

    show_reconstructions_by_class(model, test_loader, plot_save_directory=plot_save_directory)

    intra_class_distance(test_loader, model.encoder, plot_save_directory=plot_save_directory)

    plot_losses_and_reconstruction(outputs, train_losses, plot_save_directory, NUM_EPOCHS)

    if save_encoder_path:
        torch.save(model.encoder.state_dict(), save_encoder_path)
        print(f'saved trained encoder model to {save_encoder_path}')

def encoder_to_classifier_training(plot_save_directory,
                                   latent_dimension = LARGE_LATENT_DIMENSION,
                                   pretrained_encoder_path = None,
                                   save_classifier_encoder_path = None,
                                   train_data_subset = False):
    if pretrained_encoder_path and save_classifier_encoder_path:
        print('ERROR: choose whether to use pretrained reconstruction encoder or to save classifying encoder')
        exit(1)
    message = compose_status_message(CLASSIFIER, LARGE, pretrained_encoder_path, train_data_subset)
    print(message)
    # if pretrained_encoder_path:
    #     print(f'starting to train encoder to classifier model, with a pretrained encoder {pretrained_encoder_path}')
    # else:
    #     print('starting to train encoder to classifier model. training both encoder and classifier.')
    (model, criterion, optimizer,
     scheduler, train_loader, test_loader) = get_training_tools(model_type=CLASSIFIER,
                                                                model_size=LARGE,
                                                                latent_dimension=latent_dimension,
                                                                pretrained_encoder_path=pretrained_encoder_path,
                                                                train_data_subset=train_data_subset)
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    learning_rate_updates_epochs = []
    prev_lr = LEARNING_RATE
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss/total
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        #=== testing
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()*labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        test_loss = test_loss/total
        test_accuracy = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.8f} to {current_lr:.8f}")
            learning_rate_updates_epochs.append(epoch)
            prev_lr = current_lr

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%")

    plot_loss_and_accuracy(train_losses, test_losses,
                           train_accuracies, test_accuracies,
                           plot_save_directory,
                           learning_rate_updates_epochs,
                           subset=train_data_subset)

    if save_classifier_encoder_path and not train_data_subset:
        torch.save(model.encoder.state_dict(), save_classifier_encoder_path)
        print(f'saved trained encoder model to {save_classifier_encoder_path}')

def classifier_to_reconstruction_training(plot_save_directory,
                                          pretrained_reconstruction_encoder_path):
    if pretrained_reconstruction_encoder_path is None:
        print('pretrained reconstruction encoder path not provided')
        exit(1)



def q1(small_model = True, large_model = True,
       save_encoder_path = None, both_d_sizes = True,
       directory_to_save_plots ='./plots/q1'):
    if small_model:
        small_ed_save_directory = f'{directory_to_save_plots}/small autoencoder'
        if both_d_sizes:
            encoder_to_decoder_training(plot_save_directory=small_ed_save_directory, model_size=SMALL,
                                        latent_dimension=LARGE_LATENT_DIMENSION,
                                        pretrained_encoder_path=None, save_encoder_path=save_encoder_path)

        encoder_to_decoder_training(plot_save_directory=small_ed_save_directory, model_size=SMALL,
                                    latent_dimension=SMALL_LATENT_DIMENSION,
                                    pretrained_encoder_path=None, save_encoder_path=save_encoder_path)

    if large_model:
        large_ed_save_directory = f'{directory_to_save_plots}/large autoencoder'

        if both_d_sizes:
            encoder_to_decoder_training(plot_save_directory=large_ed_save_directory, model_size = LARGE,
                                    latent_dimension=SMALL_LATENT_DIMENSION,
                                    pretrained_encoder_path = None, save_encoder_path=save_encoder_path)

        encoder_to_decoder_training(plot_save_directory=large_ed_save_directory, model_size=LARGE,
                                    latent_dimension=LARGE_LATENT_DIMENSION,
                                    pretrained_encoder_path=None, save_encoder_path=save_encoder_path)


def q2(encoder_classifier_plot_save_directory = './plots/q2',
       save_classifier_encoder_path = './saved models/classifier_encoder_pretrained.pth'):

    encoder_to_classifier_training(encoder_classifier_plot_save_directory,
                                   pretrained_encoder_path= None,
                                   save_classifier_encoder_path= save_classifier_encoder_path,
                                   train_data_subset=False)

    encoder_to_classifier_training(encoder_classifier_plot_save_directory,
                                   pretrained_encoder_path=None,
                                   save_classifier_encoder_path=save_classifier_encoder_path,
                                   train_data_subset=True)

def q3():
    path_to_save_trained_reconstruction_encoder = "./saved models/reconstruction_encoder_pretrained.pth"
    if not os.path.isfile(path_to_save_trained_reconstruction_encoder):
        directory_to_save_encoder_decoder_plots ='./plots/q3'
        q1(small_model= False, large_model=True, both_d_sizes=False,
           save_encoder_path=path_to_save_trained_reconstruction_encoder,
           directory_to_save_plots=directory_to_save_encoder_decoder_plots)
    directory_to_save_classifier_plot = "./plots/q3/pretrained"
    encoder_to_classifier_training(plot_save_directory=directory_to_save_classifier_plot,
                                   pretrained_encoder_path = path_to_save_trained_reconstruction_encoder,
                                   save_classifier_encoder_path = None)
    encoder_to_classifier_training(plot_save_directory=directory_to_save_classifier_plot,
                                   pretrained_encoder_path=path_to_save_trained_reconstruction_encoder,
                                   save_classifier_encoder_path=None,
                                   train_data_subset=True)

def q4():
    path_to_save_trained_classifier_encoder = "./saved models/classifier_encoder_pretrained.pth"
    if not os.path.isfile(path_to_save_trained_classifier_encoder):
        directory_to_save_classifier_encoder_plot = './plots/q4'
        q2(encoder_classifier_plot_save_directory = directory_to_save_classifier_encoder_plot,
           save_classifier_encoder_path = path_to_save_trained_classifier_encoder)
    directory_to_save_plot = "./plots/q4/pretrained"
    encoder_to_decoder_training(plot_save_directory=directory_to_save_plot,
                                model_size=LARGE,
                                latent_dimension=LARGE_LATENT_DIMENSION,
                                pretrained_encoder_path=path_to_save_trained_classifier_encoder,
                                save_encoder_path=None)


def main():
    q1()
    q2()
    q3()
    q4()


if __name__ == '__main__':
    main()

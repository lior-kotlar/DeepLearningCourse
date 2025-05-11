import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


# size = ((input_size-kernel_size+2*padding)/stride) + 1
NUM_DIGITS = 10
BATCH_SIZE = 64
NUM_EPOCHS = 20
ENCODER_1_LAYER = 16
ENCODER_2_LAYER = 32
ENCODER_3_LAYER = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Encoder(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, first_layer_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(first_layer_size, second_layer_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second_layer_size, third_layer_size, 7),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, first_layer_size, second_layer_size, third_layer_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(third_layer_size, second_layer_size, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(second_layer_size, first_layer_size, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(first_layer_size, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

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

class MLP_classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.ff = nn.Linear(input_size, NUM_DIGITS)


    def forward(self, x):
        return self.ff(x)

class q2_encoder_classifier(nn.Module):
    def __init__(self, encoder_1_layer, encoder_2_layer, encoder_3_layer):
        super().__init__()
        self.encoder = Encoder(encoder_1_layer, encoder_2_layer, encoder_3_layer)
        self.classifer = MLP_classifier(input_size=encoder_3_layer)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        classified = self.classifer(encoded)
        return classified

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
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plot_save_directory, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def get_mnist_training_sets(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_training_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    mninst_test_data = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_training_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        mninst_test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def q1_encoder_decoder_training(plot_save_directory, model_type):

    data_loader, _ = get_mnist_training_sets(BATCH_SIZE)
    model = model_type.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model_type.parameters(), lr=1e-3, weight_decay=1e-5)


    outputs = []
    losses = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for img, _ in data_loader:
            img = img.to(device)
            reconstruction = model(img)
            loss = criterion(reconstruction, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        outputs.append((epoch, img, reconstruction))

    plot_losses_and_reconstruction(outputs, losses, plot_save_directory, NUM_EPOCHS)


def q1():
    small_ae_save_directory = './plots/small autoencoder'
    large_ae_save_directory = './plots/large autoencoder'
    q1_encoder_decoder_training(small_ae_save_directory, Conv_AE_small())
    q1_encoder_decoder_training(large_ae_save_directory, Conv_AE_large(16, 32, 64))


def q2_training(plot_save_directory):
    model = q2_encoder_classifier(ENCODER_1_LAYER, ENCODER_2_LAYER, ENCODER_3_LAYER)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader, test_loader = get_mnist_training_sets(BATCH_SIZE)
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
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

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%")


    epochs = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

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
    plot_path = os.path.join(plot_save_directory, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def q2():
    small_ae_save_directory = './plots/q2 classifier'
    q2_training(small_ae_save_directory)


def main():
    q2()


if __name__ == '__main__':
    main()

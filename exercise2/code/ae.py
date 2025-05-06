import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class autoencoder_from_youtube(nn.Module):
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


def experiment_training_function():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data,
                                              batch_size=64,
                                              shuffle=True)


    model = autoencoder_from_youtube()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 10
    outputs = []
    for epoch in range(num_epochs):
        for img, _ in data_loader:
            img = img.reshape(-1, 28*28)
            reconstruction = model(img)
            loss = criterion(reconstruction, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        outputs.append((epoch, img, reconstruction))


def main():
    experiment_training_function()


if __name__ == '__main__':
    main()

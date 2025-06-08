########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt


BATCH_SIZE = 64
OUTPUT_SIZE = 2
HIDDEN_SIZE = 128        # to experiment with

run_recurrent = True    # else run Token-wise MLP
use_RNN = True          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 2
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset
train_dataset, test_dataset, num_words, input_size = ld.get_data_set(BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # what else?
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.out_activation = nn.Softmax(dim=1)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):

        concatenated = torch.cat((x, hidden_state), dim=1)
        hidden = self.tanh(self.in2hidden(concatenated))
        output = self.hidden2out(hidden)
        output = self.out_activation(output)
        
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class PY_RNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, sequence_length, num_classes):
        super(PY_RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        #         print(out.shape)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out

    def name(self):
        return "PYRNN"

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

# Implements GRU Unit
class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        # self.something =
        # etc ...

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        pass
        # Implementation of GRU cell

        # missing implementation

        # return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size,hidden_size)
        # additional layer(s)


    def name(self):
        return "MLP"

    def forward(self, x):

        # Token-wise MLP network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        # rest

        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size,hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        # rest ...


    def name(self):
        return "MLP_atten"

    def forward(self, x):
        pass
        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size
        # with zero padding at the ends

        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        # query = ...
        # keys = ...
        # vals = ...


        # return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):

    # implement
    pass

# select model to use
def select_model():
    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, OUTPUT_SIZE, HIDDEN_SIZE)
            model = PY_RNN(input_size, 2, HIDDEN_SIZE, sequence_length=100, num_classes=2)
        else:
            model = ExGRU(input_size, OUTPUT_SIZE, HIDDEN_SIZE)
    else:
        if atten_size > 0:
            model = ExLRestSelfAtten(input_size, OUTPUT_SIZE, HIDDEN_SIZE)
        else:
            model = ExMLP(input_size, OUTPUT_SIZE, HIDDEN_SIZE)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))

    return model


def save_train_test_loss_plot(train_losses, test_losses, test_steps,
                              title="Training and Test Loss Over Time",
                              filename='exercise3/plots and outputs/loss_plot.jpeg'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.6)
    plt.plot(test_steps, test_losses, label="Test Loss", marker='o', linestyle='--')
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

def train():
    model = select_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = 1.0
    test_loss = 1.0

    train_loss_history = []
    test_loss_history = []
    test_steps = []
    # training steps in which a test step is executed every test_interval

    for epoch in range(num_epochs):

        itr = 0 # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:   # getting training batches

            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

            else:
            # Token-wise networks (MLP / MLP + Atten.)
                sub_score = []
                if atten_size > 0:
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

            # cross-entropy loss

            loss = criterion(output, labels)

            # optimize in training iterations

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                test_loss_history.append(test_loss)
                test_steps.append(epoch + itr / len(train_dataset))  # fractional epoch
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
                train_loss_history.append(train_loss)

            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )

                if not run_recurrent:
                    nump_subs = sub_score.detach().numpy()
                    labels = labels.detach().numpy()
                    print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

                # saving the model
                torch.save(model, model.name() + ".pth")

    save_train_test_loss_plot(train_loss_history, test_loss_history, test_steps)
def main():
    train()

if __name__ == "__main__":
    main()
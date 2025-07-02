########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################
import pandas as pd
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt
import os
import shutil

BATCH_SIZE = 64
OUTPUT_SIZE = 2
HIDDEN_SIZE_LARGE = 128        # to experiment with
HIDDEN_SIZE_SMALL = 64
MODEL_SAVE_DIRECTORY = 'exercise3/models'
PLOT_SAVE_DIRECTORY = 'exercise3/plots'

run_recurrent = False    # else run Token-wise MLP
use_RNN = True         # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten
SHOW_CUSTOM_REVIEWS = True

reload_model = False
EPOCHS = 10
learning_rate = 0.001
test_interval = 100

# Loading sataset, use toy = True for obtaining a smaller dataset
train_dataset, test_dataset, custom_dataset, num_words, input_size = ld.get_data_set(BATCH_SIZE, toy=SHOW_CUSTOM_REVIEWS)

number_to_label = ('positive', 'negative')

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

    @staticmethod
    def name():
        return "RNN"

    def is_rnn(self):
        return True

    def forward(self, x, hidden_state):

        # print(f'input shape is {x.shape}')
        concatenated = torch.cat((x, hidden_state), dim=1)
        hidden = self.tanh(self.in2hidden(concatenated))
        output = self.hidden2out(hidden)
        output = self.out_activation(output)
        
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.reset_gate = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        self.update_gate = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    @staticmethod
    def name():
        return "GRU"

    def is_rnn(self):
        return True

    def forward(self, x, hidden_state):
        x = x.to(hidden_state.device)
        concatenated = torch.cat([hidden_state, x], dim=1)
        z_t = self.sigmoid(self.update_gate(concatenated))
        r_t = self.sigmoid(self.reset_gate(concatenated))
        u = hidden_state * r_t
        cat_mid = torch.cat([u, x], dim=1)
        h_t_tag = self.tanh(self.fc(cat_mid))
        hidden = (z_t * hidden_state) + ((1 - z_t) * h_t_tag)
        output = self.output_fc(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

class ExMLP1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP1, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.layer1 = MatMul(input_size,hidden_size)
        self.layer2 = MatMul(hidden_size,hidden_size)
        self.layer3 = MatMul(hidden_size,output_size)

    @staticmethod
    def name():
        return "MLP_no_norm"

    def is_rnn(self):
        return False

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)
        return x

class ExMLP2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP2, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.layer1 = MatMul(input_size,hidden_size)
        self.b_norm = nn.BatchNorm1d(num_words)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    @staticmethod
    def name():
        return "MLP_with_batchnorm"

    def is_rnn(self):
        return False

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.b_norm(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.b_norm(x)
        x = self.layer3(x)
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

hidden_sizes = [HIDDEN_SIZE_SMALL, HIDDEN_SIZE_LARGE]
architecture_names = ['RNN', 'GRU', 'MLP_no_norm', 'MLP_with_batchnorm', 'MLP_atten']
name2arch = {
    'RNN': ExRNN,
    'GRU': ExGRU,
    'MLP_no_norm': ExMLP1,
    'MLP_with_batchnorm': ExMLP2,
    'MLP_atten': ExLRestSelfAtten
}

# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    # implement
    pass

def select_model(name = None, hidden_dimension=HIDDEN_SIZE_LARGE):
    if name is None:
        if run_recurrent:
            if use_RNN:
                model = ExRNN(input_size, OUTPUT_SIZE, hidden_dimension)
            else:
                model = ExGRU(input_size, OUTPUT_SIZE, hidden_dimension)
        else:
            if atten_size > 0:
                model = ExLRestSelfAtten(input_size, OUTPUT_SIZE, hidden_dimension)
            else:
                model = ExMLP1(input_size, OUTPUT_SIZE, hidden_dimension)

    else: model = name2arch[name](input_size, OUTPUT_SIZE, hidden_dimension)

    print(f'Using model: {model.name()}, with hidden size: {hidden_dimension}')

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))

    return model

def create_save_directories(model_name, hidden_dimension):
    folder_name = f'{model_name}_{hidden_dimension}'
    plot_save_directory_path = os.path.join(PLOT_SAVE_DIRECTORY, folder_name)

    if os.path.exists(plot_save_directory_path):
        shutil.rmtree(plot_save_directory_path)
    os.makedirs(plot_save_directory_path)

    if not os.path.exists(MODEL_SAVE_DIRECTORY):
        os.makedirs(MODEL_SAVE_DIRECTORY)

    return plot_save_directory_path

def save_accuracy_plot(test_steps, train_accuracies, test_accuracies,
               model_name, hidden_dimension, save_directory_path,
                       best_model_iteration):
    title = f'Training and Test Accuracies - hidden size {hidden_dimension}'
    filename = f'accuracy_plot_{model_name}_{hidden_dimension}.jpeg'
    plot_save_path = os.path.join(save_directory_path, filename)
    print(f'now saving accuracies to {plot_save_path}')
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot([int(s * len(train_accuracies) / EPOCHS) for s in test_steps],
             test_accuracies, label='Test Accuracy')
    best_model_iteration_index = int(best_model_iteration * len(train_accuracies) / EPOCHS)

    plt.axvline(best_model_iteration_index, color='red', linestyle=':', linewidth=1.2,
                label='best model saved')
    if best_model_iteration_index in test_steps:
        best_loss_value = test_accuracies[test_steps.index(best_model_iteration_index)]
        plt.scatter(best_model_iteration_index, best_loss_value,
                    color='red', marker='*', s=120, zorder=5)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()

def save_train_test_loss_plot(train_losses, test_losses, test_steps,
                              model_name, hidden_dimension, save_directory_path,
                              best_model_iteration):

    title = f'Training and Test Loss - {model_name} - {hidden_dimension}'
    filename = f'loss_plot_{model_name}_{hidden_dimension}.jpeg'
    plot_save_path = os.path.join(save_directory_path, filename)
    test_step_indices = [int(e * len(train_losses) / EPOCHS) for e in test_steps]
    best_model_iteration_index = int(best_model_iteration * len(train_losses) / EPOCHS)
    print(f'now saving loss plots: {plot_save_path}')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.6)
    plt.plot(test_step_indices, test_losses, label="Test Loss", marker='o', linestyle='--')

    plt.axvline(best_model_iteration_index, color='red', linestyle=':', linewidth=1.2,
                label='best model saved')
    if best_model_iteration_index in test_step_indices:
        best_loss_value = test_losses[test_step_indices.index(best_model_iteration_index)]
        plt.scatter(best_model_iteration_index, best_loss_value,
                    color='red', marker='*', s=120, zorder=5)

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path, bbox_inches='tight')

def save_plots(train_losses, test_losses, test_steps,
               train_accuracies, test_accuracies,
               model_name, hidden_dimension, save_directory_path,
               best_model_iteration):

    save_train_test_loss_plot(train_losses, test_losses, test_steps,
               model_name, hidden_dimension, save_directory_path,
                              best_model_iteration)

    save_accuracy_plot(test_steps, train_accuracies, test_accuracies,
               model_name, hidden_dimension, save_directory_path,
                       best_model_iteration)

def compute_accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum().item()
    total = true_labels.size(0)
    return correct / total

def tensor_numbers_to_labels(numbers):
    as_list = numbers.tolist()
    output = [number_to_label[int(number)] for number in as_list]
    return output

def trim_subscores(subscores, reviews_text):
    for sample_idx in range(len(reviews_text)):
        review_length_words = len(reviews_text[sample_idx])
        if review_length_words < 100:
            subscores[sample_idx][review_length_words:] = torch.tensor([0.0, 0.0])
    return subscores

def average_logits(subscores, reviews_text):
    #subscores here have to be trimmed before being passed to this function
    sum_logits = torch.sum(subscores, dim=1)
    averages = []
    for sample_idx in range(len(reviews_text)):
        review_length_words = len(reviews_text[sample_idx])
        review_logits_sum = sum_logits[sample_idx]
        average_logit = review_logits_sum / review_length_words
        print(f'review length: {review_length_words}, review sum: {review_logits_sum}, average: {average_logit}')
        averages.append(average_logit.tolist())
    return np.array(averages, dtype=np.float32)

def average_logits_tensor(subscores, reviews_text):
    lengths = torch.tensor([len(r) for r in reviews_text],
                           dtype=subscores.dtype,  # float for division
                           device=subscores.device)  # same device as logits
    sum_logits = subscores.sum(dim=1)
    lengths = lengths.to(subscores.device).unsqueeze(1)
    avg_logits = sum_logits / lengths
    return avg_logits

def make_confustion_matrix_table(
        reviews_text, labels, subscores, save_dir):

    print(subscores.shape)
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    labels = labels.view(-1).cpu()

    average_review_logits = average_logits(subscores, reviews_text)

    preds = np.argmax(average_review_logits, axis=1)
    quad = []
    for t, p in zip(labels, preds):
        if   t == 1 and p == 1: quad.append('TN')
        elif t == 0 and p == 1: quad.append('FN')
        elif t == 0 and p == 0: quad.append('TP')
        else:                   quad.append('FP')

    subscores_cpu = subscores.cpu()

    for idx, (text, q) in enumerate(zip(reviews_text, quad)):
        words = text[:subscores_cpu.size(1)]
        logits = subscores_cpu[idx].detach().numpy()       # (100, 2)

        rows = [[w,
                 f"{log[0]:+.2f}",
                 f"{log[1]:+.2f}"]
                for w, log in zip(words, logits)]

        fig_h = max(2, 0.25 * len(rows))
        fig, ax = plt.subplots(figsize=(6, fig_h))
        ax.axis('off')

        table = ax.table(
            cellText=rows,
            colLabels=['word',
                       f'{number_to_label[0]}_logit',
                       f'{number_to_label[1]}_logit'],
            loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        title = (f"Review {idx+1} – {q} "
                 f"(true={number_to_label[labels[idx]]}, "
                 f"pred={number_to_label[preds[idx]]})")
        fig.suptitle(title, fontsize=10, y=1.02)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"review_{idx}_{q}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved table for review {idx}: {filepath}")

def make_custom_prediction(best_model, hidden_dimension, save_directory_path):
    model_name = best_model.name()
    title = "Model Predictions vs. True Labels"
    filename = f'Predictions_Custom_Data_{model_name}_{hidden_dimension}.jpeg'
    plot_save_path = os.path.join(save_directory_path, filename)
    custom_accuracy_history = []
    for labels, reviews, reviews_text in custom_dataset:
        if best_model.is_rnn():  # Recurrent nets (RNN/GRU)
            hidden_state = best_model.init_hidden(int(labels.shape[0])).to(device)
            output = None
            for i in range(num_words):
                output, hidden_state = best_model(reviews[:, i, :], hidden_state)  # HIDE
            if output is None:
                print('error with predicting custom prediction or something')
                exit(1)
            predictions = torch.argmax(output, axis=1)

        else:
            if atten_size > 0:
                sub_score, atten_weights = best_model(reviews)
            else:
                sub_score = best_model(reviews)
                sub_score = trim_subscores(sub_score, reviews_text)
            output = average_logits(sub_score, reviews_text)
            predictions = torch.tensor(np.argmax(output, axis=1))

        if not best_model.is_rnn():
            make_confustion_matrix_table(reviews_text, labels, sub_score, save_directory_path)


        _, true_labels = torch.max(labels, 1)
        predicted_as_strings = tensor_numbers_to_labels(predictions)
        true_labels_as_strings = tensor_numbers_to_labels(true_labels)
        correct = (predictions == true_labels)
        # print(f'predicted labels: {predictions}\n'
        #       f'true labels: {true_labels}\n'
        #       f'correct labels: {correct}\n'
        #       f'predicted labels as strings: {predicted_as_strings}\n'
        #       f'true labels as strings: {true_labels_as_strings}\n')
        accuracy = compute_accuracy(predictions, true_labels)
        custom_accuracy_history.append(accuracy)
        df = pd.DataFrame({
            'Reviews': list(range(1, len(reviews_text) + 1)),
            'True Labels': true_labels_as_strings,
            'Predicted Labels': predicted_as_strings
        })
        df['Correct'] = df['True Labels'] == df['Predicted Labels']
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(df)))
        ax.axis("off")
        ax.set_title(title, fontsize=16, weight='bold', pad=20)

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
            if row > 0 and not df.iloc[row - 1]["Correct"]:
                cell.set_facecolor('#ffdddd')  # light red

        plt.tight_layout()

        plt.savefig(plot_save_path, dpi=300)
        plt.close()

def experiment_with_custom_review():
    pass

def train(model = None, hidden_dimension=HIDDEN_SIZE_LARGE):

    if model is None:
        model = select_model(hidden_dimension=hidden_dimension)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, threshold=0.0001)

    train_loss = 1.0
    test_loss = 1.0

    train_loss_history = []
    test_loss_history = []
    test_steps = []

    train_accuracy_history = []
    test_accuracy_history = []

    plot_save_directory_path = create_save_directories(model.name(), hidden_dimension)

    best_model = None
    best_test_loss = float('inf')

    prev_lr = learning_rate
    best_model_iteration = 0

    for epoch in range(EPOCHS):

        itr = 0 # iteration counter within each epoch
        test_iterations = 0

        for labels, reviews, reviews_text in train_dataset:   # getting training batches

            labels = labels.to(device)
            reviews = reviews.to(device)

            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                test_iterations += 1
                labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch
            else:
                test_iter = False

            if model.is_rnn(): # Recurrent nets (RNN/GRU)
                hidden_state = model.init_hidden(int(labels.shape[0])).to(device)
                output = None
                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
                if output is None:
                    print('error with predicting custom prediction or something')
                    exit(1)
                predictions = torch.argmax(output, axis=1)

            else:
            # Token-wise networks (MLP / MLP + Atten.)
                sub_score = []
                if atten_size > 0:
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)
                    sub_score = trim_subscores(sub_score, reviews_text)

                output = average_logits_tensor(sub_score, reviews_text)
                predictions = torch.argmax(output, axis=1)

            _, true_labels = torch.max(labels, 1)
            loss = criterion(output, labels)
            accuracy = compute_accuracy(predictions, true_labels)

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                scheduler.step(test_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != prev_lr:
                    print(f"Learning rate reduced from {prev_lr:.8f} to {new_lr:.8f}")
                    prev_lr = new_lr
                test_loss_history.append(test_loss)
                test_accuracy_history.append(accuracy)
                current_test_step = epoch + itr / len(train_dataset)
                test_steps.append(current_test_step)  # fractional epoch

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )

                if not model.is_rnn():
                    nump_subs = sub_score.detach().numpy()
                    labels = labels.detach().numpy()
                    print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])

                # saving the model
                if test_loss < best_test_loss:
                    print(f'better model found new loss {test_loss:.4f}')
                    best_model_iteration = current_test_step
                    best_test_loss = test_loss
                    best_model = model

            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
                train_loss_history.append(train_loss)
                train_accuracy_history.append(accuracy)


    make_custom_prediction(best_model, hidden_dimension, plot_save_directory_path)

    model_save_path = os.path.join(MODEL_SAVE_DIRECTORY, f'{best_model.name()}_{hidden_dimension}.pth')
    print(f'saving model to {model_save_path}')
    torch.save(best_model, model_save_path)

    save_plots(train_loss_history, test_loss_history, test_steps,
               train_accuracy_history, test_accuracy_history,
               best_model.name(), hidden_dimension, plot_save_directory_path,
               best_model_iteration)

def q1():
    model = select_model(ExRNN.name())
    train(model=model, hidden_dimension=HIDDEN_SIZE_LARGE)
    train(model=model, hidden_dimension=HIDDEN_SIZE_SMALL)
    model = select_model(ExGRU.name())
    train(model=model, hidden_dimension=HIDDEN_SIZE_LARGE)
    train(model=model, hidden_dimension=HIDDEN_SIZE_SMALL)

def q2():
    for hidden_size in hidden_sizes:
        model = select_model(ExMLP1.name(), hidden_size)
        train(model, hidden_dimension=hidden_size)
        model = select_model(ExMLP2.name(), hidden_size)
        train(model, hidden_dimension=hidden_size)

def main():
    q1()
    q2()

if __name__ == "__main__":
    main()
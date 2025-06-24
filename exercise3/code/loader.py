import torch

# import torchtext as tx
from torchtext.vocab import GloVe
# from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
# from torchtext.data.functional import to_map_style_dataset
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
embedding_size = 100
Train_size=30000



def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]


def load_data_set(load_my_reviews=False):
    custom_iter = None
    data=pd.read_csv("exercise3/data/IMDB Dataset.csv")
    train_data=data[:Train_size]
    train_iter=ReviewDataset(train_data["review"],train_data["sentiment"])
    test_data=data[Train_size:]
    if load_my_reviews:
        custom_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
        custom_iter = ReviewDataset(custom_data["review"],custom_data["sentiment"])
    test_data=test_data.reset_index(drop=True)
    test_iter=ReviewDataset(test_data["review"],test_data["sentiment"])
    return train_iter, test_iter, custom_iter


embedding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokenize)


def preprocess_review(s):
    cleaned = tokenize(s)
    embedded = embedding.get_vecs_by_tokens(cleaned)
    if embedded.shape[0] != 100 or embedded.shape[1] != 100:
        embedded = torch.nn.functional.pad(embedded, (0, 0, 0, MAX_LENGTH - embedded.shape[0]))
    return torch.unsqueeze(embedded, 0)


def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]


def collate_batch(batch):
    label_list = []
    review_list = []
    embedding_list=[]
    for review,label in batch:
        label_list.append(preprocess_label(label))### label
        review_list.append(tokenize(review))### the  actuall review
        processed_review = preprocess_review(review).detach()
        embedding_list.append(processed_review) ### the embedding vectors
    label_list = torch.tensor(label_list, dtype=torch.float32).reshape((-1, 2))
    embedding_tensor= torch.cat(embedding_list)
    return label_list.to(device), embedding_tensor.to(device) ,review_list


##########################
# ADD YOUR OWN TEST TEXT #
##########################

my_test_texts = []
my_test_texts.append("this movie is very very bad ,the worst movie")
my_test_texts.append("this movie is so great")
my_test_texts.append("I really liked the fish and animations the anther casting was not so good")
my_test_texts.append("i am happy that we left halfway through the movie. i recommend everyone to do the same")
my_test_labels = ["negative", "positive", "positive", "negative"]

##########################
##########################


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, labels):
        'Initialization'
        self.labels = labels
        self.reviews = review_list

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y



def get_data_set(batch_size, toy=False):
        train_data, test_data, custom_data = load_data_set(load_my_reviews=toy)

        train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                     shuffle=True, collate_fn=collate_batch)
        custom_dataloader = DataLoader(custom_data, batch_size=batch_size,
                                       shuffle=False, collate_fn=collate_batch) if toy else None
        return train_dataloader, test_dataloader, custom_dataloader, MAX_LENGTH, embedding_size

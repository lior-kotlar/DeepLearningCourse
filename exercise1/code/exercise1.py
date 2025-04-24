import numpy as np
import os
import sys


def load_train_data(data_directory):
    if not os.path.isdir(data_directory):
        print("Data directory does not exist")
        exit(-1)
    file_names = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
    file_names_no_extension = [os.path.splitext(filename)[0] for filename in file_names]
    allels = {}
    for no_ext, file_name in zip(file_names_no_extension, file_names):
        file_full_path = os.path.join(data_directory, file_name)
        with open(file_full_path, "r") as f:
            allel_antigens = [line.strip() for line in f if line.strip()]
            allels[no_ext] = allel_antigens
    print(allels.keys())




def main():
    dataset_directory = sys.argv[1]
    load_train_data(dataset_directory)


if __name__ == '__main__':
    main()
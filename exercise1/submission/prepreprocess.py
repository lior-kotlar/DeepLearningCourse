import os
import sys
from collections import defaultdict


clean_directory_name = 'clean data'


def remove_shared_words(data_directory):
    """removes shared words from negative and different alleles of the dataset"""
    word_to_files = defaultdict(set)
    file_to_words = {}
    if not os.path.isdir(data_directory):
        print("Data directory does not exist")
        exit(-1)
    file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".txt")]
    for file_path in file_paths:
        with open(file_path, "r", encoding='utf-8') as file:
            words = set(line.strip() for line in file if line.strip())
            file_to_words[file_path] = words
            for word in words:
                word_to_files[word].add(file_path)

    shared_words = {word for word, files in word_to_files.items() if len(files) > 1}

    clean_directory_path = os.path.join(data_directory, clean_directory_name)

    os.makedirs(clean_directory_path, exist_ok=True)

    for file_path, words in file_to_words.items():
        unique_words = words - shared_words
        newfile_name = os.path.basename(file_path)
        output_file_path = os.path.join(clean_directory_path, newfile_name)
        with open(output_file_path, "w", encoding='utf-8') as out_file:
            for word in unique_words:
                out_file.write(f'{word}\n')

def load_train_data(data_directory):
    """Load antigens from files with the specific naming convention from ex1_data"""
    if not os.path.isdir(data_directory):
        print(f"Data directory '{data_directory}' does not exist")
        exit(-1)
    
    alleles = {}
    negative_antigens = []

    file_names = os.listdir(data_directory)
    
    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                antigens = [line.strip() for line in f if line.strip()]
                
                if file_name == "negs.txt":
                    negative_antigens = antigens
                    print(f"Loaded {len(antigens)} negative examples from {file_name}")
                else:
                    allele_name = os.path.splitext(file_name)[0]
                    alleles[allele_name] = antigens
                    print(f"Loaded {len(antigens)} antigens for allele {allele_name}")
    
    print(f"Loaded data for {len(alleles)} alleles: {list(alleles.keys())}")
    return alleles, negative_antigens
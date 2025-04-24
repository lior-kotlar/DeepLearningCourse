import os
import sys
from collections import defaultdict


clean_directory_name = 'clean data'


def remove_shared_words(data_directory):
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


def main():
    remove_shared_words(sys.argv[1])


if __name__ == '__main__':
    main()
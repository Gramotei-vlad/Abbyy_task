from typing import Dict, Tuple, List
from datasets import Dataset


def get_dataset(dataset: Dataset, split_name: str) -> Tuple[List[List[str]], List[List[int]]]:
    sentences, labels = [], []

    for elem in dataset[split_name]:
        tokens = elem['tokens']
        tags = elem['pos_tags']

        assert len(tokens) == len(tags)

        sentences.append(tokens)
        labels.append(tags)

    return sentences, labels


def get_class_map():
    class_map = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11,
                 'DT': 12, 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21,
                 'NNP': 22, 'NNPS': 23, 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30,
                 'RBR': 31, 'RBS': 32, 'RP': 33, 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39,
                 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46}

    return class_map

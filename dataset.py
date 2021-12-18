from typing import Dict, Tuple, List
from datasets import Dataset


def get_class_map() -> Dict[str, int]:
    class_map = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7,
                 'I-INTJ': 8, 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15,
                 'I-PRT': 16, 'B-SBAR': 17, 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}
    return class_map


def get_dataset(dataset: Dataset, split_name: str) -> Tuple[List[List[str]], List[List[int]]]:
    sentences, labels = [], []

    for elem in dataset[split_name]:
        tokens = elem['tokens']
        tags = elem['chunk_tags']

        assert len(tokens) == len(tags)

        sentences.append(tokens)
        labels.append(tags)

    return sentences, labels

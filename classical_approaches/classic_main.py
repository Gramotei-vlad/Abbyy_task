import argparse

from datasets import load_dataset
from dataset import get_dataset
from viterbi_algorithm import ViterbiAlgorithm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Запуск классического подхода для задачи POS')
    parser.add_argument('--input', type=str, default='She saw a cat.',
                        help='Строка, которую нужно разметить')
    args = parser.parse_args()

    dataset = load_dataset('conll2003')
    train_texts, train_labels = get_dataset(dataset, 'train')
    test_texts, test_labels = get_dataset(dataset, 'test')

    algorithm = ViterbiAlgorithm()
    algorithm.tag(train_texts, train_labels)

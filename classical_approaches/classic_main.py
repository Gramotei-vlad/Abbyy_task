import argparse

from datasets import load_dataset
from dataset import get_dataset, get_class_map
from viterbi_algorithm import ViterbiAlgorithm
from seqeval.metrics import classification_report


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Запуск классического подхода для задачи POS')
    parser.add_argument('--input', type=str, default='She saw a cat',
                        help='Строка, которую нужно разметить')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Если истина, то выдаст результат для строки в args.input')
    args = parser.parse_args()

    dataset = load_dataset('conll2003')
    train_texts, train_labels = get_dataset(dataset, 'train')
    test_texts, test_labels = get_dataset(dataset, 'test')

    label2idx = get_class_map()
    idx2label = {idx: label for label, idx in label2idx.items()}

    algorithm = ViterbiAlgorithm()
    all_tags = algorithm.tag(train_texts, train_labels)

    result_predictions = []
    result_labels = []
    if args.predict:
        split_sentence = args.input.split()
        prediction = algorithm.run(split_sentence, all_tags)

        for word, pred_class in zip(split_sentence, prediction):
            tag = idx2label[pred_class]
            print(f'Word: {word}, prediction_class: {tag}')
    else:
        for text, labels in zip(test_texts, test_labels):
            prediction = algorithm.run(text, all_tags)
            tags_prediction = [idx2label[pred] for pred in prediction]
            tags_labels = [idx2label[label] for label in labels]
            result_predictions.append(tags_prediction)
            result_labels.append(tags_labels)

        print(classification_report(result_labels, result_predictions))



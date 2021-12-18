import argparse
import wandb
import fasttext


from dataset import get_dataset, get_class_map
from utils import *
from model import NerModel
from loss import SparseCategoricalFocalLossWithLogits
from training import TrainLoop
from typing import Generator
from datasets import load_dataset
from fasttext.util import download_model


BEST_MODEL_DIR = f'best_model'
LAST_MODEL_DIR = f'last_model'

ANOTHER_CLASS_NAME = 'O'
EMBEDDING_DIM = 300
PAD_TOKEN = 'PAD_TOKEN'

# ИСПРАВЬ WANDB


if __name__ == '__main__':

    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    parser = argparse.ArgumentParser('Скрипт для обучения модели по распознаванию полей из платежек')
    parser.add_argument('--dirty_text', action='store_true', default=False,
                        help='Если истина, то текст будет вместе с символами, иначе потрет их')
    parser.add_argument('--embedding_path', type=str, default='FastText/wiki.ru.bin',
                        help='Путь до файла с FastText эмбеддингами.')
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='Размерность LSTM блока')
    parser.add_argument('--embedding_output_dim', type=int, default=100,
                        help='Размерность выхода эмбеддинг слоя.')
    parser.add_argument('--use_bioes', action='store_true', default=False,
                        help='Использовать ли bioes формат меток')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча.')
    parser.add_argument('--main_metric', type=str, default='f1 score', choices=['f1 score', 'precision', 'recall'],
                        help='Название основной метрики')
    parser.add_argument('--epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--lr', type=float, default=0.001, help='Скорость обучения у оптимизатора.')
    parser.add_argument('--logs_dir', type=str, default='LogsMLC',
                        help='Директория с логами для tensorboard, лучшим и последним состоянием моделей')
    parser.add_argument('--use_focal', action='store_true', default=False, help='Использовать focal loss?')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='Обучать ли модель с весами классов?')
    parser.add_argument('--input_signature_name', type=str, default='input_tokens',
                        help='Название ключа сигнатуры для передачи тензоров в модель')
    parser.add_argument('--output_signature_name', type=str, default='predictions',
                        help='Название ключа для получения предсказаний модели')
    parser.add_argument('--inference', action='store_true', default=False,
                        help='Если истина, то не обучает модель, а прогоняет на тестовых данных')
    parser.add_argument('--inference_model_path', type=str, default='Model',
                        help='Путь до модели, которую нужно прогнать на тестовом сете'
                             'Параметр имеет смысл, если --inference истина')

    args = parser.parse_args()

    wandb.init(project="Abbyy_task", entity="v-kosukhin",
               config={
                   'use_class_weights': args.use_class_weights,
                   'learning rate': args.lr,
                   'batch size': args.batch_size,
                   'use focal loss': args.use_focal,
                   'epochs': args.epochs,
                   'lstm hidden size': args.lstm_hidden_size,
               })

    if args.use_focal and args.use_class_weights:
        raise ValueError(f"Focal loss doesn't use class weights")

    dataset = load_dataset('conll2003')

    train_texts, train_labels = get_dataset(dataset, 'train')
    val_texts, val_labels = get_dataset(dataset, 'validation')
    test_texts, test_labels = get_dataset(dataset, 'test')

    download_model('en', if_exists='ignore')
    fasttext_model = fasttext.load_model('cc.en.300.bin')

    full_texts = train_texts + val_texts + test_texts
    full_labels = train_labels + val_labels + test_labels

    label2idx = get_class_map()
    num_classes = len(label2idx)

    idx2label = {idx: label for label, idx in label2idx.items()}

    result_sentences = [train_texts, val_texts, test_texts]
    result_labels = [train_labels, val_labels, test_labels]

    assert len(result_sentences[0]) == len(result_labels[0])
    assert len(result_sentences[1]) == len(result_labels[1])
    assert len(result_sentences[2]) == len(result_labels[2])

    def train_generator() -> Generator[Tuple[List[str], List[int]], None, None]:
        for sentence, label in zip(result_sentences[0], result_labels[0]):
            yield sentence, label


    def val_generator() -> Generator[Tuple[List[str], List[int]], None, None]:
        for sentence, label in zip(result_sentences[1], result_labels[1]):
            yield sentence, label


    def test_generator() -> Generator[Tuple[List[str], List[int]], None, None]:
        for sentence, label in zip(result_sentences[2], result_labels[2]):
            yield sentence, label

    train_bucket_bounds = [get_max_length_sentence(result_sentences[0])]
    val_bucket_bounds = [get_max_length_sentence(result_sentences[1])]
    test_bucket_bounds = [get_max_length_sentence(result_sentences[2])]

    train_data = tf.data.Dataset.from_generator(train_generator, (tf.string, tf.int32),
                                                (tf.TensorShape([None]), tf.TensorShape([None])))
    val_data = tf.data.Dataset.from_generator(val_generator, (tf.string, tf.int32),
                                              (tf.TensorShape([None]), tf.TensorShape([None])))
    test_data = tf.data.Dataset.from_generator(test_generator, (tf.string, tf.int32),
                                               (tf.TensorShape([None]), tf.TensorShape([None])))

    train_data = train_data.shuffle(len(result_sentences[0]), seed=SEED, reshuffle_each_iteration=True)

    train_data = train_data.bucket_by_sequence_length(lambda x, _: tf.shape(x)[0], train_bucket_bounds,
                                                      [args.batch_size] * (len(train_bucket_bounds) + 1),
                                                      padding_values=(PAD_TOKEN, label2idx[ANOTHER_CLASS_NAME]))
    val_data = val_data.bucket_by_sequence_length(lambda x, _: tf.shape(x)[0], val_bucket_bounds,
                                                  [args.batch_size] * (len(val_bucket_bounds) + 1),
                                                  padding_values=(PAD_TOKEN, label2idx[ANOTHER_CLASS_NAME]))
    test_data = test_data.bucket_by_sequence_length(lambda x, _: tf.shape(x)[0], test_bucket_bounds,
                                                    [args.batch_size] * (len(test_bucket_bounds) + 1),
                                                    padding_values=(PAD_TOKEN, label2idx[ANOTHER_CLASS_NAME]))

    if args.inference:
        model = tf.saved_model.load(args.inference_model_path)
    else:
        model = NerModel(args.lstm_hidden_size, num_classes, args.output_signature_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    class_weights = None
    if args.use_focal:
        criterion = SparseCategoricalFocalLossWithLogits()
    else:
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if args.use_class_weights:
            class_weights = get_class_weights(result_labels[0])

    train_loop = TrainLoop(model, criterion, optimizer, fasttext_model, args.epochs, class_weights,
                           idx2label, args.logs_dir, args.main_metric,
                           args.input_signature_name, args.output_signature_name,
                           mlc_write_path='')

    if args.inference:
        train_loop.test_model(test_data)
    else:
        train_loop.run(train_data, val_data)
        train_loop.test_model(test_data)

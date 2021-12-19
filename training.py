import pickle
import wandb
import tensorflow as tf

from typing import Tuple, Dict, List
from utils import entity_to_string, bytes_to_words, get_fasttext_embeddings
from metrics import get_metrics
from tqdm import tqdm
from seqeval.metrics import classification_report
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

BEST_MODEL_DIR = f'best_model'
LAST_MODEL_DIR = f'last_model'


class TrainLoop:
    def __init__(self, model: Model, criterion: callable, optimizer: Optimizer, fasttext_model,
                 epochs: int, class_weights: Dict[int, float], idx2label: Dict[int, str],
                 logs_dir: str, main_metric: str, input_signature_name: str, output_signature_name: str, **kwargs):
        """

        :param model: модель типа tf.keras.Model
        :param criterion: функция потерь;
        :param optimizer: оптимизатор;
        :param fasttext_model: модель для получения FastText эмбеддингов;
        :param epochs: кол-во эпох обучения;
        :param class_weights: словарь вида {индекс класса: вес класса};
        :param idx2label: словарь вида {индекс класса: название класса};
        :param embedding2word: словарь вида {эмбеддинг слова: слово}
        :param logs_dir: директория, в которую будут писаться логи и в которую будут сохраняться модели;
        :param main_metric: название метрики, по которой будет сохраняться лучшая модель;
        :param input_signature_name: название ключа сигнатуры для передачи тензоров в модель;
        :param output_signature_name: название ключа, чтобы получить предсказания от графа;
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.fasttext_model = fasttext_model
        self.epochs = epochs
        self.class_weights = class_weights
        self.main_metric = main_metric
        self.idx2label = idx2label
        self.logs_dir = logs_dir
        self.input_signature_name = input_signature_name
        self.output_signature_name = output_signature_name
        self.mlc_write_path = kwargs.get('mlc_write_path', '')

        self.is_training = True
        self.best_metric_value = 0.

    @staticmethod
    def _get_predictions(logits: Tensor) -> Tensor:
        """
        Возвращает предсказанные классы из логитов
        :param logits: [batch_size, sentence_length, amount_classes] выходы модели;
        :return: [batch_size, amount_classes] предсказанные классы;
        """
        probs = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(probs, axis=-1)
        return predictions

    def _get_tensor_weights(self, logits: Tensor):
        """
        Генерит веса для каждого элемента в тензоре
        :return: [batch_size, sentence_len] тензор c весами
        """

        def to_weights(predicted_classes):
            weights = [self.class_weights[int(class_.numpy())] for class_ in predicted_classes]
            weights = tf.convert_to_tensor(weights, tf.float32)
            return weights

        predictions = self._get_predictions(logits)
        predictions = tf.cast(predictions, tf.float32)
        tensor_weights = tf.map_fn(to_weights, predictions)
        return tensor_weights

    def _train_step(self, embeddings_batch: Tensor, labels_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Один шаг обучения модели;
            :param embeddings_batch: тензор из эмбеддингов слов для входа модели;
            :param labels_batch: тензор из реальных классов каждого токена;
            :return: кортеж из лосса и выходных логитов;
            """
        with tf.GradientTape() as tape:
            logits = self.model(embeddings_batch)[self.output_signature_name]

            if self.class_weights is not None:
                tensor_weights = self._get_tensor_weights(logits)
                loss = self.criterion(labels_batch, logits, sample_weight=tensor_weights)
            else:
                loss = self.criterion(labels_batch, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))

        return loss, logits

    def _val_step(self, embeddings_batch: Tensor, labels_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Один шаг для тестирования модели на валидации
        :param embeddings_batch: [batch_size, sentence_length] тензор из эмбеддингов слов для входа модели;
        :param labels_batch: [batch_size, sentence_length] тензор из реальных классов каждого токена;
        :return: кортеж из лосса и выходных логитов
        """

        logits = self.model(embeddings_batch)[self.output_signature_name]

        if self.class_weights is not None:
            tensor_weights = self._get_tensor_weights(logits)
            loss = self.criterion(labels_batch, logits, sample_weight=tensor_weights)
        else:
            loss = self.criterion(labels_batch, logits)

        return loss, logits

    def _handle_batch(self, words_batch: Tensor, labels_batch: Tensor, loss_metric=None) \
            -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """
        Выполняет один шаг обучения
        :param words_batch: [batch_size, sentence_len, embeddings_dim] батч со словами
        :param labels_batch: [batch_size, sentence_len] батч меток;
        :return: word_sentences - [batch_size, sentence_len] - список слов;
                 real_classes - [batch_size, sentence_len] - список реальных классов;
                 predicted_classes - [batch_size, sentence_len] - список предсказанных классов
        """
        embeddings_batch = get_fasttext_embeddings(self.fasttext_model, words_batch)

        if self.is_training:
            loss, logits = self._train_step(embeddings_batch, labels_batch)
        else:
            loss, logits = self._val_step(embeddings_batch, labels_batch)

        curr_predicted = self._get_predictions(logits)
        curr_predicted = curr_predicted.numpy()
        labels_batch = labels_batch.numpy()
        words_batch = words_batch.numpy()

        if loss_metric is not None:
            loss_metric(loss)

        word_sentences = bytes_to_words(words_batch)
        real_classes = entity_to_string(labels_batch, self.idx2label)
        predicted_classes = entity_to_string(curr_predicted, self.idx2label)

        return word_sentences, real_classes, predicted_classes

    def _handle_epoch(self, data: tf.data.Dataset, step: int, loss_metric, summary_writer):
        """
        Выполняет одну эпоху обучения/валидации. На этапе валидации сохраняет лучшую модель,
        согласно метрики в self.main_metric
        :param data: набор данных;
        :param step: текущая эпоха;
        :param loss_metric: метрика подсчета лосса;
        :param summary_writer: записывает результаты для tensorboard
        :return:
        """
        result_sentences = []
        result_labels = []
        result_predictions = []
        prefix = 'train' if self.is_training else 'val'

        with summary_writer.as_default():
            for words_batch, labels_batch in tqdm(data):
                word_sentences, real_classes, predicted_classes = self._handle_batch(words_batch, labels_batch,
                                                                                     loss_metric)

                result_sentences.extend(word_sentences)
                result_labels.extend(real_classes)
                result_predictions.extend(predicted_classes)

            class_metrics = get_metrics(result_labels, result_predictions)

            precision = class_metrics['precision']
            recall = class_metrics['recall']
            f1 = class_metrics['f1 score']

            wandb.log({f'{prefix} loss': loss_metric.result(),
                       f'{prefix} precision score': precision,
                       f'{prefix} recall score': recall,
                       f'{prefix} f1 score': f1},
                      step=step)

            loss_metric.reset_states()

        if not self.is_training:
            curr_metric_value = class_metrics[self.main_metric]
            if curr_metric_value > self.best_metric_value:
                self.best_metric_value = curr_metric_value
                tf.saved_model.save(self.model, f'{self.mlc_write_path}{self.logs_dir}/{BEST_MODEL_DIR}/',
                                    signatures={self.input_signature_name: self.model.__call__})

    def run(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset):
        """
        Запуск обучения модели;
        В конце сохраняет последнюю версию модели, а также сохраняет словарь вида {индекс: название класса}
        :param train_data: набор данных для тренировки;
        :param val_data: набор данных для валидации;
        :return:
        """

        train_summary_writer = tf.summary.create_file_writer(f'{self.mlc_write_path}{self.logs_dir}/logs/train')
        val_summary_writer = tf.summary.create_file_writer(f'{self.mlc_write_path}{self.logs_dir}/logs/val')

        train_loss_metric = tf.keras.metrics.Mean('train loss', dtype=tf.float32)
        val_loss_metric = tf.keras.metrics.Mean('val loss', dtype=tf.float32)

        for epoch in range(self.epochs):
            self.is_training = True
            self._handle_epoch(train_data, epoch, train_loss_metric, train_summary_writer)
            self.is_training = False
            self._handle_epoch(val_data, epoch, val_loss_metric, val_summary_writer)

        with open(f'{self.mlc_write_path}{self.logs_dir}/idx2label.pkl', 'wb') as f:
            pickle.dump(self.idx2label, f, pickle.HIGHEST_PROTOCOL)

        tf.saved_model.save(self.model, f'{self.mlc_write_path}{self.logs_dir}/{LAST_MODEL_DIR}/',
                            signatures={self.input_signature_name: self.model.__call__})

    def test_model(self, test_data: tf.data.Dataset):
        """
        Прогоняет модель на тестовых данных;
        :param test_data: набор тестовых данных;
        :return:
        """
        test_sentences = []
        test_predictions = []
        test_labels = []
        self.is_training = False

        for embeddings_batch, labels_batch in tqdm(test_data):
            word_sentences, real_classes, predicted_classes = self._handle_batch(embeddings_batch, labels_batch)

            test_sentences.extend(word_sentences)
            test_predictions.extend(real_classes)
            test_labels.extend(predicted_classes)

        print(classification_report(test_labels, test_predictions))

    def predict(self, predict_string: str):
        """
        Для данной строки выдает предсказания
        :param predict_string: Исходная строка
        :return:
        """
        predict_list = predict_string.split()

        tensor_string = tf.convert_to_tensor(predict_list)
        tensor_string = tf.expand_dims(tensor_string, axis=0)
        embedding_string = get_fasttext_embeddings(self.fasttext_model, tensor_string)
        logits = self.model(embedding_string)[self.output_signature_name]

        curr_predicted = self._get_predictions(logits)
        curr_predicted = curr_predicted.numpy()
        predicted_classes = entity_to_string(curr_predicted, self.idx2label)

        for word, pred_class in zip(predict_list, predicted_classes):
            print(f'Word: {word}. Class: {pred_class}')

import re
import numpy as np
import tensorflow as tf

from collections import defaultdict
from typing import Dict, List, Union, Tuple
from flair.data import Sentence
from tensorflow.python.framework.ops import Tensor

UNKNOWN_INDEX = 0
PADDING_INDEX = 1
EMBEDDING_DIM = 200
EMBEDDING_VECTOR = [0 for _ in range(EMBEDDING_DIM)]
ANOTHER_CLASS_NAME = 'PAD_TOKEN'
EMBEDDING_PAD_TOKEN = tuple([0. for _ in range(EMBEDDING_DIM)])
PAD_TOKEN = 'PAD_TOKEN'
START_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
END_PREFIX = 'E-'
ALONE_PREFIX = 'S-'

PATTERN = re.compile(f'[BIES]-')


def get_set_of_words(sentences: List[List[int]]) -> List[int]:
    """
    Находит все уникальные токены из списков
    :param sentences: список из предложений, разбитых на токены;
    :return: множество из уникальных слов
    """
    unique_words = set()
    for sentence in sentences:
        for word in sentence:
            unique_words.add(word)
    return sorted(unique_words)


def get_word_2_index_dict(unique_words: List[int]) -> Dict[int, int]:
    """
    Для каждого токена генерит его уникальный индекс
    :param unique_words: множество из уникальных слов;
    :return: словарь вида {слово: индекс слова}
    """
    word2idx = dict()

    for idx, word in enumerate(unique_words):
        word2idx[word] = idx
        idx += 1

    return word2idx


def get_text_idxs(sentences: List[List[str]], word2idx: Dict[str, int]) -> np.array:
    """
    Переводит слова в их соответствующие индексы;
    :param sentences: список список из токенов;
    :param word2idx: словарь вида {слово: индекс слова};
    :return: numpy массив из индексов слов;
    """
    idx_sentences = []
    for sentence in sentences:
        curr_sentence = []
        for word in sentence:
            idx = word2idx[word]
            curr_sentence.append(idx)

        idx_sentences.append(np.asarray(curr_sentence))
    return np.asarray(idx_sentences)


def get_text_embeddings(sentence: List[str], word2embedding: Dict[str, Tuple[float]]) -> List[Tuple[float]]:
    """
    Получает эембеддинги каждого слова.
    :param sentence: список токенов;
    :param word2embedding: словарь вида {слово: эмбеддинг}
    :return: список из эмбеддингов
    """
    embeddings = []
    for word in sentence:
        curr_embedding = word2embedding[word]
        embeddings.append(curr_embedding)
    return embeddings


def get_embeddings(embedding_model, words_batch: Tensor) -> Tensor:
    """
    Получает ембеддинги для слов в тензоре
    :param embedding_model: модель из модуля fasttext
    :param words_batch: тензор тензоров из слов;
    :return: тензор вида [batch_dim, sentence_len, embedding_dim]
    """
    words_batch = words_batch.numpy()
    embeddings_batch = []
    for sentence in words_batch:
        sentence_embeddings = []
        for word in sentence:
            real_word = tf.compat.as_str_any(word)
            if real_word == PAD_TOKEN:
                embedding = EMBEDDING_VECTOR
            else:
                embedding = embedding_model.get_word_vector(real_word)

            sentence_embeddings.append(embedding)
        embeddings_batch.append(sentence_embeddings)
    embeddings_batch = tf.convert_to_tensor(embeddings_batch, dtype=tf.float32)
    return embeddings_batch


def get_bpe_embeddings(embedding_model, words_batch: Tensor):
    words_batch = words_batch.numpy()
    embeddings_batch = []
    for sentence in words_batch:
        sentence_embeddings = []
        real_sentence = []
        for word in sentence:
            real_word = tf.compat.as_str_any(word)
            real_sentence.append(real_word)

        sentence = Sentence(real_sentence)

        embedding_model.embed(sentence)

        for token in sentence:
            if token.text == PAD_TOKEN:
                sentence_embeddings.append(EMBEDDING_VECTOR)
            else:
                list_embedding = token.embedding.numpy().tolist()
                sentence_embeddings.append(list_embedding)

        embeddings_batch.append(sentence_embeddings)

    embeddings_batch = tf.convert_to_tensor(embeddings_batch, dtype=tf.float32)
    return embeddings_batch


def bytes_to_words(bytes_batch: Tensor) -> List[List[str]]:
    """
    Переводит слова-байты в чистые слова;
    :param bytes_batch: [batch_size, sentence_len] батч слов-байтов
    :return: список из предложений;
    """

    sentences = []
    for words in bytes_batch:
        sentence = []
        for word in words:
            word = tf.compat.as_str_any(word)
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def entity_to_string(batch_idxs: np.array, idx2label: Dict[int, str]) \
        -> Union[List[str], List[List[str]]]:
    """
    Переводит индексы в соответствующие слова;
    :param batch_idxs: numpy массив массивов из токентов
    :param idx2label: словарь вида {индекс: слово}
    :return: список, где индексы переделаны в их слова;
    """
    predicted_classes = []
    for sentence in batch_idxs:
        predicted_classes.append([idx2label[max_idx] for max_idx in sentence])
    return predicted_classes


def get_max_length_sentence(sentences: Union[List[List[int]], List[List[str]], np.array]) -> int:
    """
    Находит длину максимального предложения
    :param sentences: список списков из слов
    :return: длина максимального предложения
    """
    max_length = 0
    for sentence in sentences:
        max_length = max(max_length, len(sentence))
    return max_length


def get_class_weights(train_labels: List[List[int]]) -> Dict[int, float]:
    """
    Функция для получения весов классов,
    где вес класса равен (1 - кол-во этого класса / кол-во представителей всех классов);
    :param train_labels: список списков из категорий каждого токена;
    :return: словарь вида {индекс класса: вес класса}
    """
    amount_classes = defaultdict(int)
    for doc_labels in train_labels:
        for label in doc_labels:
            amount_classes[label] += 1
    amount_labels = sum(list(amount_classes.values()))
    weights_classes = {class_label: (1 - cls_amount / amount_labels)
                       for class_label, cls_amount in sorted(amount_classes.items())}

    # В данных есть метки, которые не используются.
    not_used_classes = []
    used_classes = list(weights_classes.keys())
    idx = 0
    for class_ in used_classes:
        if class_ != idx:
            not_used_classes.append(idx)
            idx += 2
        else:
            idx += 1

    for class_ in not_used_classes:
        weights_classes[class_] = 1.

    # Padding token
    weights_classes[47] = 1
    return weights_classes


def remove_spaces(field_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Удаляет все пробелы в строках значений словаря, т.е. 'слово1 слово2' -> 'слово1слово2';
    :param field_dict: словарь вида {название поля: значение поля}
    :return: словарь вида {название поля: значение поля без пробела}
    """
    for key, value in field_dict.items():
        field_dict[key] = value.replace(' ', '')
    return field_dict


def get_zero_field(field_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Для всех полей, кроме суммы, берется его первое вхождение.
    Для суммы последнее, ибо в теории и на практике это работает лучше.
    Если поле не найдено или его нет в разметке, то для него вернет просто ''
    :param field_dict: словарь вида {название поля: несколько предсказанных значений поля}
    :return: словарь вида {название поля: первое предсказанное значение поля}
    """
    result_dict = dict()
    for field, field_values in field_dict.items():
        if field == 'sum':
            result_dict[field] = field_values[-1] if len(field_values) > 0 else ''
        else:
            result_dict[field] = field_values[0] if len(field_values) > 0 else ''
    return result_dict


def get_bioes_labels(labels: List[str]) -> List[str]:
    """
    Функция для получения bioes меток из обычные меток
    :param labels: список с обычными метками
    :return: список с bioes метками
    """
    result_labels = []
    prev_label = ANOTHER_CLASS_NAME

    def change_last_label(bioes_labels: List[str]):
        """
        Меняет последнюю метку, если выполняются условия
        :param bioes_labels: текущий список с bioes метками
        :return: None
        """
        if len(bioes_labels) > 0:
            if bioes_labels[-1].startswith(START_PREFIX):
                bioes_labels[-1] = bioes_labels[-1].replace(START_PREFIX, ALONE_PREFIX)
            elif bioes_labels[-1].startswith(INSIDE_PREFIX):
                bioes_labels[-1] = bioes_labels[-1].replace(INSIDE_PREFIX, END_PREFIX)

    for curr_label in labels:
        if prev_label != curr_label:
            change_last_label(result_labels)
            bioes_label = curr_label if curr_label == ANOTHER_CLASS_NAME else f'{START_PREFIX}{curr_label}'
        else:
            bioes_label = curr_label if curr_label == ANOTHER_CLASS_NAME else f'{INSIDE_PREFIX}{curr_label}'

        prev_label = curr_label
        result_labels.append(bioes_label)

    change_last_label(result_labels)
    return result_labels


def to_usual_labels(labels: List[str]) -> List[str]:
    """
    Функция для преобразования bioes меток в обычные.
    Если bioes меток не было, то вернется исходный список
    :param labels: список с возможно bioes метками
    :return: список с обычными метками
    """
    usual_labels = [PATTERN.sub('', label) for label in labels]
    return usual_labels

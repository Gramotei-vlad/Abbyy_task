from collections import defaultdict, Counter
from typing import List, Tuple, Union


START_PROB = 1e-10


class ViterbiAlgorithm:
    def __init__(self):
        self.unigram_cnt = defaultdict(lambda: 0)
        self.bigram_cnt = defaultdict(lambda: 0)
        self.tag_cnt = defaultdict(lambda: 0)
        self.tag_word_cnt = Counter()
        self.transition_probs = defaultdict(lambda: START_PROB)
        self.emmission_probs = defaultdict(lambda: START_PROB)

    @staticmethod
    def get_ngrams(text: List[str], n: int):
        return [tuple(text[i: i + n]) for i in range(len(text))]

    def get_bigram_cnt(self, tags: List[Union[str, int]]):
        ngrams = self.get_ngrams(tags, 2)
        for bigram in ngrams:
            self.bigram_cnt[bigram] += 1
        return self.bigram_cnt

    def get_unigram_cnt(self, tags: List[Union[str, int]]):
        for tag in tags:
            self.unigram_cnt[tag] += 1
        return self.unigram_cnt

    def get_tag_and_words_cnt(self, tags_and_words: List[Tuple[int, str]]):
        for tag, word in tags_and_words:
            self.tag_cnt[tag] += 1
            self.tag_word_cnt[(tag, word)] += 1
        return self.tag_word_cnt

    def get_transition_probs(self, tags: List[Union[str, int]]):
        bigrams = self.get_ngrams(tags, 2)
        for bigram in bigrams:
            self.transition_probs[bigram] = self.bigram_cnt[bigram] / self.unigram_cnt[bigram[0]]
        return self.transition_probs

    def get_emmission_probs(self, tags_and_words: List[Tuple[int, str]]):
        for tag, word in tags_and_words:
            self.emmission_probs[(tag, word)] = self.tag_word_cnt[(tag, word)] / self.tag_cnt[tag]
        return self.emmission_probs

    def initial_probabilities(self, tag: int):
        return self.transition_probs["START", tag]

    def run(self, input_text: List[str], in_states: List[Union[int, str]]):
        states = set(in_states)
        states.remove("START")
        states.remove("END")
        trails = {}
        for s in states:
            trails[s, 0] = self.initial_probabilities(s) * self.emmission_probs[s, input_text[0]]

        for o in range(1, len(input_text)):
            obs = input_text[o]
            for s in states:
                v1 = [(trails[k, o - 1] * self.transition_probs[k, s] * self.emmission_probs[s, obs], k)
                      for k in states]
                k = sorted(v1)[-1][1]
                trails[s, o] = trails[k, o - 1] * self.transition_probs[k, s] * self.emmission_probs[s, obs]

        result_answer = []
        for o in range(len(input_text) - 1, -1, -1):
            k = sorted([(trails[k, o], k) for k in states])[-1][1]
            result_answer.append(k)
        result_answer.reverse()
        return result_answer

    def tag(self, sentences: List[List[str]], tags: List[List[int]]):
        tagged_words = []
        all_tags = []
        for sentence, tag in zip(sentences, tags):
            all_tags.append("START")

            assert len(sentence) == len(tag)
            for (word, label) in zip(sentence, tag):
                all_tags.append(label)
                tagged_words.append((label, word))
            all_tags.append("END")

        self.get_tag_and_words_cnt(tagged_words)

        self.bigram_cnt = self.get_bigram_cnt(all_tags)
        self.unigram_cnt = self.get_unigram_cnt(all_tags)

        self.get_transition_probs(all_tags)
        self.get_emmission_probs(tagged_words)

        return all_tags


import numpy as np
from nltk.corpus import dependency_treebank

sents = dependency_treebank.parsed_sents()
words = dependency_treebank.sents()
POS = dependency_treebank.tagged_sents()


def create_vocab_arr():
    return np.array(sorted(set([word for sentence in dependency_treebank.sents() for word in sentence])))


def create_POS_arr():
    return np.array(sorted(set([word[1] for tagged_sentence in dependency_treebank.tagged_sents() for word in tagged_sentence])))


class MST_Parser:
    def __init__(self, learning_rate: int, num_of_features: int):
        self.weights = np.zeros(num_of_features)
        self.learning_rate = learning_rate
        self.vocab = create_vocab_arr()
        self.POS = create_POS_arr()
        self.bigram_dict = {word1: {word2: 0 for word2 in self.vocab} for word1 in self.vocab}
        i = 0
        for word1 in self.vocab:
            for word2 in self.vocab:
                self.bigram_dict[word1][word2] = i
                i += 1
        self.POS_dict = {pos1: {pos2: 0 for pos2 in self.POS} for pos1 in self.POS}
        j = 0
        for pos1 in self.POS:
            for pos2 in self.POS:
                self.POS_dict[pos1][pos2] = j
                j += 1


    def _feature_function_word_bigram(self, word1, word2):
        arr = np.zeros(len(self.vocab)**2)
        arr[self.bigram_dict[word1[0]][word2[0]]] = 1
        arr[self.bigram_dict[word2[0]][word1[0]]] = 1
        return arr

    def _feature_function_POS_bigram(self, word1, word2):
        arr = np.zeros(len(self.POS) ** 2)
        arr[self.POS_dict[word1[1]][word2[1]]] = 1
        arr[self.POS_dict[word2[1]][word1[1]]] = 1
        return arr

    def concat_features(self, word1, word2):
        return np.concatenate((self._feature_function_word_bigram(word1, word2), self._feature_function_POS_bigram(word1, word2)), axis=0)

    def score_two_words(self, v1, v2, l):
        feature_repr = self.concat_features(v1, v2)


    def train(self, n_iterations: int, batch_size: int, sentences):
        for r in reange(n_iterations):
            for i in range(batch_size):
                pass
        return


    def test(self):
        return
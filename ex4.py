import numpy as np
from nltk.corpus import dependency_treebank

sents = dependency_treebank.parsed_sents()
words = dependency_treebank.sents()


def create_vocab_arr():
    return np.array(sorted(set([word for sentence in dependency_treebank.sents() for word in sentence])))


class MST_Parser:
    def __init__(self, learning_rate: int, num_of_features: int):
        self.weights = np.zeros(num_of_features)
        self.learning_rate = learning_rate
        self.vocab = create_vocab_arr()
        self.bigram_dict = {word1: {word2: 0 for word2 in self.vocab} for word1 in self.vocab}
        i = 0
        for word1 in self.vocab:
            for word2 in self.vocab:
                self.bigram_dict[word1][word2] = i
                i += 1

    def _feature_function_word_bigram(self, word1, word2):
        arr = np.zeros(len(self.vocab)**2)
        arr[self.bigram_dict[word1][word2]] = 1
        return arr

    def _feature_function_POS_bigram(self, word1, word2):
        return

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
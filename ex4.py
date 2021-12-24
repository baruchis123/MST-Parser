import random

import numpy as np
from copy import deepcopy
from collections import namedtuple
from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

# sent = dependency_treebank.parsed_sents()[0]
# for node in sents.nodes:
#     print(node)
# words = dependency_treebank.sents()
# print(words[0])
# POS = dependency_treebank.tagged_sents()
# for node in sents[0].nodes:
#     print(sents[0].nodes[node])
# print(sents[0])
# print(words[0])
# print(POS[0: 2])


def create_vocab_arr():
    return np.array(sorted(set([word for sentence in dependency_treebank.sents() for word in sentence] + ["ROOT"])))


def create_POS_arr():
    return np.array(sorted(set([word[1] for tagged_sentence in dependency_treebank.tagged_sents() for word in tagged_sentence] + ["TOP"])))


class MST_Parser:
    def __init__(self, learning_rate: int):
        self.ARC = namedtuple('ARC', ['head', 'tail', 'weight'])
        self.Ferature_repr = namedtuple('Feature_repr', ['bigram_ind', 'POS_ind'])
        self.__learning_rate = learning_rate
        self.__vocab = create_vocab_arr()
        self.__POS = create_POS_arr()
        self.__dimension_size = len(self.__vocab) ** 2 + len(self.__POS) ** 2
        self.__weights = dict()
        self.bigram_word_index_dict = {}
        i = 0
        for word in self.__vocab:
            self.bigram_word_index_dict[word] = i
            i += 1
        # self.__bigram_dict = {word1: {word2: 0 for word2 in self.__vocab} for word1 in self.__vocab}
        # i = 0
        # for word1 in self.__vocab: #TODO verify dim size of vocab
        #     for word2 in self.__vocab:
        #         self.__bigram_dict[word1][word2] = i
        #         i += 1
        self.__POS_dict = {pos1: {pos2: 0 for pos2 in self.__POS} for pos1 in self.__POS}
        j = 0
        for pos1 in self.__POS: #TODO verify dim size of POS dict
            for pos2 in self.__POS:
                self.__POS_dict[pos1][pos2] = j
                j += 1

    def find_pair_index(self, word1, word2):
        return self.bigram_word_index_dict[word1]*len(self.__vocab) + self.bigram_word_index_dict[word2]

    def __feature_function_word_bigram(self, word1, word2):
        first_word = word1["word"] if word1["word"] is not None else "ROOT"
        second_word = word2["word"] if word2["word"] is not None else "ROOT"
        return self.find_pair_index(first_word, second_word)

    def __feature_function_POS_bigram(self, word1, word2):
        return self.__POS_dict[word1["tag"]][word2["tag"]] + len(self.__vocab)

    def __feature_function(self, word1, word2):
        return self.Ferature_repr(self.__feature_function_word_bigram(word1, word2), self.__feature_function_POS_bigram(word1, word2))

    def __score_two_words(self, v1, v2):
        # index_words = self.__bigram_dict[v1[0]][v2[0]]
        index_words = self.find_pair_index(v1[0], v2[0])
        index_POS = self.__POS_dict[v1[1]][v2[1]]
        bigram_weight = self.__weights[index_words] if index_words in self.__weights.keys() else 0
        pos_weight = self.__weights[index_POS] if index_POS in self.__weights.keys() else 0
        return bigram_weight + pos_weight

    def __create_arc(self, v1, v2, v1_ind, v2_ind):
        return self.ARC(v1_ind, v2_ind, -self.__score_two_words(v1, v2))

    def __inference(self, sent):
        arcs_list = [self.__create_arc(v1, v2, v1_ind, v2_ind) for v1_ind, v1 in enumerate(sent) for v2_ind, v2 in
                     enumerate(sent) if v1_ind != v2_ind]
        maximum_spanning_tree = min_spanning_arborescence_nx(arcs_list, 0).values()
        maximum_spanning_tree = [self.ARC(arc.head, arc.tail, -arc.weight) for arc in maximum_spanning_tree]
        return maximum_spanning_tree

    def __create_gold_standard_tuple(self, sent):
        words = list(sent.nodes.values())
        list1 = [self.__create_arc((sent.nodes[word["head"]]["word"] if sent.nodes[word["head"]]["word"] is not None else "ROOT", sent.nodes[word["head"]]["tag"]), (word["word"], word["tag"]), word["head"], word["address"]) for word in words if word["head"] is not None]
        return list1

    def __sum_of_edges(self, tree, sentence):
        res = dict()
        for arc in tree:
            feature = self.__feature_function(sentence.nodes[arc.head], sentence.nodes[arc.tail])
            if feature.bigram_ind not in res.keys():
                res[feature.bigram_ind] = 0
            if feature.POS_ind not in res.keys():
                res[feature.POS_ind] = 0
            res[feature.bigram_ind] += 1
            res[feature.POS_ind] += 1
        return res

    def __update_weights(self, cur_weight, gold_standard_tree, cur_tree, sentence):
        gold_minus_cur = self.__minus_weights(self.__sum_of_edges(gold_standard_tree, sentence),
                                              self.__sum_of_edges(cur_tree, sentence))
        leraning_rate_mult_gold_minus_cur = self.__mult_scalar_weight(self.__learning_rate, gold_minus_cur)
        return self.__plus_weights(cur_weight, leraning_rate_mult_gold_minus_cur)
        # return cur_weight + self.__learning_rate * \
        #        (self.__sum_of_edges(gold_standard_tree, sentence) -
        #                  self.__sum_of_edges(cur_tree, sentence))

    def __minus_weights(self, weight1, weight2):
        new_weight = deepcopy(weight1)
        for ind in weight2:
            if ind not in new_weight.keys():
                new_weight[ind] = 0
            new_weight[ind] -= weight2[ind]
        return new_weight

    def __plus_weights(self, weight1, weight2):
        new_weight = deepcopy(weight1)
        for ind in weight2:
            if ind not in new_weight.keys():
                new_weight[ind] = 0
            new_weight[ind] += weight2[ind]
        return new_weight

    def __mult_scalar_weight(self, scalar, weight):
        new_weight = deepcopy(weight)
        new_weight.update((x, scalar * y) for x, y in new_weight.items())
        return new_weight


    def train(self, n_iterations: int, batch_size: int, training_set):
        print('started training')
        self.__weights = dict()
        start = random.randint(0, len(training_set) - batch_size)
        for r in range(n_iterations):
            print(f"r = {r}")
            for i, sent in enumerate(training_set[start: start + batch_size]):
                print(f"i = {i}")
                tagged_sent = self.__create_tagged_sent(sent)
                maximum_spanning_tree = self.__inference(tagged_sent)
                actual_spanning_tree = self.__create_gold_standard_tuple(sent)
                cur_iter_weight = self.__update_weights(self.__weights, actual_spanning_tree, maximum_spanning_tree, sent)
                self.__weights = self.__plus_weights(self.__weights, cur_iter_weight)
                start = random.randint(0, len(training_set) - batch_size)
        avarage_divide = n_iterations * batch_size
        self.__weights.update((x, y / avarage_divide) for x, y in self.__weights.items())
        # self.__weights /= (n_iterations * batch_size)

    def predict(self, sent):
        return self.__inference(sent)

    def __create_tagged_sent(self, sent):
        return [(word["word"] if word["word"] is not None else "ROOT", word["tag"]) for word in sent.nodes.values()]


    def eval(self, sent):
        tagged_sent = self.__create_tagged_sent(sent)
        maximum_spanning_tree = self.__inference(tagged_sent)
        actual_spanning_tree = self.__create_gold_standard_tuple(sent)
        max_set = set(maximum_spanning_tree)
        intersection = max_set.intersection(set(actual_spanning_tree))
        return len(intersection) / len(tagged_sent)

    def test(self,test_set):
        test_set_size = len(test_set)
        res = 0
        for sent in test_set:
            res += self.eval(sent)
        return res/test_set_size

if __name__ == '__main__':
    parser = MST_Parser(1)
    break_off = int(0.9*len(dependency_treebank.parsed_sents()))
    train_set = dependency_treebank.parsed_sents()[:break_off]
    test_set = dependency_treebank.parsed_sents()[break_off:]
    parser.train(2, 2, train_set)
    print(parser.test(test_set))


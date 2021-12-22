import numpy as np
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
        self.__learning_rate = learning_rate
        self.__vocab = create_vocab_arr()
        self.__POS = create_POS_arr()
        self.__dimension_size = len(self.__vocab) ** 2 + len(self.__POS) ** 2
        self.__weights = np.zeros(self.__dimension_size)
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
        arr = np.zeros(len(self.__vocab) ** 2)
        first_word = word1["word"] if word1["word"] is not None else "ROOT"
        second_word = word2["word"] if word2["word"] is not None else "ROOT"
        arr[self.find_pair_index(first_word, second_word)] = 1
        return arr

    def __feature_function_POS_bigram(self, word1, word2):
        arr = np.zeros(len(self.__POS) ** 2)
        arr[self.__POS_dict[word1["tag"]][word2["tag"]]] = 1
        return arr

    def __feature_function(self, word1, word2):
        return np.concatenate((self.__feature_function_word_bigram(word1, word2), self.__feature_function_POS_bigram(word1, word2)), axis=0)

    def __score_two_words(self, v1, v2):
        # index_words = self.__bigram_dict[v1[0]][v2[0]]
        index_words = self.find_pair_index(v1[0], v2[0])
        index_POS = self.__POS_dict[v1[1]][v2[1]]
        return self.__weights[index_words] + self.__weights[index_POS]

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
        res = np.zeros(self.__dimension_size)
        for arc in tree:
            res += self.__feature_function(sentence.nodes[arc.head], sentence.nodes[arc.tail])
        return res

    def __update_weights(self, cur_weight, gold_standard_tree, cur_tree, sentence):
        return cur_weight + self.__learning_rate * \
               (self.__sum_of_edges(gold_standard_tree, sentence) -
                         self.__sum_of_edges(cur_tree, sentence))

    def train(self, n_iterations: int, batch_size: int, training_set):
        weights = np.zeros(self.__dimension_size)
        start = 0
        for r in range(n_iterations):
            for sent in training_set[start: start + batch_size]:
                tagged_sent = self.__create_tagged_sent(sent)
                maximum_spanning_tree = self.__inference(tagged_sent)
                actual_spanning_tree = self.__create_gold_standard_tuple(sent)
                updated_weight = self.__update_weights(weights, actual_spanning_tree, maximum_spanning_tree, sent)
                self.__weights += updated_weight
                weights = updated_weight
                start += batch_size
        self.__weights / (n_iterations * batch_size)

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
    parser.train(2, 64, train_set)
    print(parser.test(test_set))


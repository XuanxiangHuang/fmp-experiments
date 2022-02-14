#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Decision Trees
import collections
from Orange.classification import TreeLearner
from Orange.tree import DiscreteNode, MappedDiscreteNode, NumericNode
from Orange.data import Table
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


class DecisionTree:
    """
        Using Orange, learning DT classifier from data.
    """
    def __init__(self, dataset, max_depth=7, learning_rate=0.2):
        datatable = Table(dataset)
        self.features = [att.name for att in datatable.domain.attributes]
        self.classes = datatable.domain.class_var.values

        X_train, X_test, y_train, y_test = \
            train_test_split(datatable.X, datatable.Y, test_size=learning_rate)

        learner = TreeLearner(max_depth=max_depth, binarize=True)

        train_datatable = Table.from_numpy(datatable.domain, X_train, y_train)
        self.classifier = learner(train_datatable)

        self.train_acc = accuracy_score(y_train, self.classifier(X_train))
        self.test_acc = accuracy_score(y_test, self.classifier(X_test))

    def train(self):
        """
            Training decision tree with given datasets.

            :return: none.
        """
        return round(self.train_acc, 3), round(self.test_acc, 3)

    def __str__(self):
        res = ''
        res += ('#Total Node: {0}\n'.format(self.classifier.node_count()))
        res += ('#Leaf Node: {0}\n'.format(self.classifier.leaf_count()))
        res += ('Depth: {0}\n'.format(self.classifier.depth()))
        res += self.classifier.print_tree()
        return res

    @staticmethod
    def save_model(dt, filename):
        """
            Save DT to pickle model.

            :param dt: decision tree classifier.
            :param filename: filename storing dt classifier.
            :return: none.
        """
        with open(filename, "wb") as f:
            pickle.dump(dt, f)

    @classmethod
    def from_file(cls, filename):
        """
            Load DT classifier from file.

            :param filename: decision tree classifier in pickle.
            :return: decision tree.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def predict_one(self, in_x):
        """
            Get prediction of given one instance or sample.

            :param in_x: given instance.
            :return: prediction.
        """
        return self.classifier(in_x)

    def support_var(self, root):
        """
            Get dependency set of DT, i.e. exclude all don't-cared features.

            :param root: root of DT or sub-DT.
            :return: a set of feature index.
        """
        vars_ = set()
        for nd in self.bfs(root):
            if not len(nd.children):
                continue
            if nd.attr_idx not in vars_:
                vars_.add(nd.attr_idx)
        return vars_

    def save_xpg_format(self, inst, filename):
        """
            Save DT and given instance to a .xpg format file.

            :param inst: given instance.
            :param filename: .xpg file.
            :return: none.
        """
        n_idx = 1
        new_nid = dict()
        vars_ = set()
        internal = []
        terminal = []
        prediction = self.classifier(inst)
        str_t = "T: "
        str_i = "NT: "
        for nd in self.bfs(self.classifier.root):
            new_nid.update({nd: n_idx})
            if not len(nd.children):
                terminal.append(nd)
                str_t += f"{n_idx} "
            else:
                internal.append(nd)
                if nd.attr_idx not in vars_:
                    vars_.add(nd.attr_idx)
                str_i += f"{n_idx} "
            n_idx += 1
        str_t = str_t[:-1] + "\n"
        str_i = str_i[:-1] + "\n"

        output = f"NN: {n_idx - 1}\n"
        output += "Root: 1\n"

        str_t += "TDef:\n"
        for nd in terminal:
            probs = nd.value / np.sum(nd.value)
            target = np.argmax(probs, axis=-1)
            if target == prediction:
                str_t += f"{new_nid[nd]} 1\n"
            else:
                str_t += f"{new_nid[nd]} 0\n"
        output += str_t

        str_n2v = ""
        str_p2c = ""
        for nd in internal:
            str_n2v += f"{new_nid[nd]} {nd.attr.name}\n"
            for c_id, chd in enumerate(nd.children):
                if type(nd) == DiscreteNode:
                    val = nd.attr.values[c_id]
                    if inst[nd.attr_idx] == int(val):
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 1\n"
                    else:
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 0\n"
                elif type(nd) == MappedDiscreteNode:
                    val = [nd.attr.values[j] for j in sorted(chd.condition)]
                    if str(inst[nd.attr_idx]) in val:
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 1\n"
                    else:
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 0\n"
                elif type(nd) == NumericNode:
                    assert (type(inst[nd.attr_idx]) is float or np.float)
                    if inst[nd.attr_idx] <= nd.threshold and (c_id == 0):
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 1\n"
                    elif inst[nd.attr_idx] > nd.threshold and (c_id == 1):
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 1\n"
                    else:
                        str_p2c += f"{new_nid[nd]} {new_nid[chd]} 0\n"
                else:
                    assert False, 'Unreachable node tree'

        output += str_i
        output += "NTDef:\n"
        output += str_p2c

        output += f"NV: {len(vars_)}\n"
        output += "VarDef:\n"
        output += str_n2v

        savefile = open(filename, "w")
        savefile.write(output)
        savefile.close()

    def bfs(self, root):
        """
            Iterate through nodes in breadth first search (BFS) order.

            :param root: root node of decision tree.
            :return: a set of all tree nodes in BFS order.
        """
        yield from self._bfs(root, set())

    def _bfs(self, root, visited):
        queue = collections.deque()
        queue.appendleft(root)
        while queue:
            node = queue.pop()
            if node and node not in visited:
                if node.children:
                    for child in node.children:
                        queue.appendleft(child)
                visited.add(node)
                yield node

    def size(self, root):
        """
            Return size of DT
            :param root: root node.
            :return: size
        """
        visited = set()
        for nd in self.bfs(root):
            visited.add(nd)
        return len(visited)
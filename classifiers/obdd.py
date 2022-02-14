#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Ordered Binary Decision Diagrams (OBDDs)
import collections
from dd.cudd import BDD, Function
from sklearn.metrics import accuracy_score
import pickle


def decimalToBinary(n,m):
    """
        convert an integer n to a binary and with specified length m
    :param n: integer
    :param m: length of binary
    :return: a list (size m) of 0,1 representing this integer
    """
    bits = f'{n:0{m}b}'
    bits_list = list(bits)
    return [int(i) for i in bits_list]


def parse_cnf_header(filename):
    """
        Parse CNF file header

        :param filename: file in .cnf format.
        :return: set of features and clauses.
    """
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    lines = list(filter(lambda l: (not (l.startswith('c') or l.strip() == '')), lines))
    assert (lines[0].strip().startswith('p cnf '))
    nv = (lines[0].strip().split())[2]
    n_claus = (lines[0].strip().split())[3]
    assert len(lines[1:]) == int(n_claus)

    features = [f'f_{i}' for i in range(int(nv))]

    claus = []
    for l in lines[1:]:
        tmp_clau = l.strip().split()
        clau = [int(_) for _ in tmp_clau]
        assert clau[-1] == 0
        claus.append(clau[:-1])

    return features, claus


class DdNode:
    """
        OBDD nodes structure.
    """

    def __init__(self, v_id=None, low=None, high=None, n_id=None):
        self.v_id = v_id
        self.low = low
        self.high = high
        self.n_id = n_id


class OBDD:
    """
        Reduced Ordered Binary Decision Diagrams (OBDD),
        without complemented edges (CEs).
        Maintaining only structure of OBDD classifier,
        without Boolean operations, without Universal/Existential quantifier, etc..
    """

    def __init__(self):
        self.root = None
        self.nv = None
        self.nn = None
        self.idx2nd = None
        self.T0 = None
        self.T1 = None
        self.features = None
        self.targets = None

        self.var2lvl = None
        self.lvl2var = None
        self.tup2node = None
        self.nd_idx = None
        self.comp_tab = None

    def compile(self, in_model, in_type, features, targets=None):
        """
            Compiling models to OBDD classifier.
            Currently support BuDDy package, one can also use CUDD package,
            but need to transform complemented edges to regular edges.

            :param in_model: a set of rules or a Boolean network
            :param in_type: 'DL' indicates a set of Ordered rules;
                            'DS' indicates a set of Unordered rules;
                            'BNet' indicates a Boolean network,
                            where each element is a If-Then-Else gate (var, low, high);
                            'CNF' indicates a set of clauses
            :param features: a set of features.
            :param targets: a set of target classes.
            :return: an OBDD classifier.
        """
        if in_type != 'DL' and in_type != 'DS' and in_type != 'BNet' and in_type != 'CNF':
            print('unknown input model')
            assert False

        cudd = BDD()
        cudd.configure(reordering=True, garbage_collection=True)
        self.nv = len(features)
        self.idx2nd = dict()
        self.T0 = DdNode(n_id=0)
        self.T1 = DdNode(n_id=1)
        self.features = features
        self.targets = targets
        for i in range(self.nv):
            cudd.declare(f'x_{i}')
        if in_type == 'CNF':
            self.compile_cnf_cudd(in_model, cudd)

    def obdd_from_cudd(self, cudd: BDD, root):
        """
            Extracting OBDD classifier from CUDD.

            :param cudd: CUDD manager.
            :param root: root node of OBDD in CUDD.
            :return: OBDD classifier object, each node is a DdNode object.
        """
        print(f"#### start standardization, size of cudd: {root.dag_size} ####")
        self.var2lvl = dict()
        self.lvl2var = dict()
        self.tup2node = dict()
        self.nd_idx = 2
        self.comp_tab = dict()

        for var in cudd.vars:
            lvl = cudd.level_of_var(var)
            self.var2lvl.update({var: lvl})
            self.lvl2var.update({lvl: var})

        cudd2obdd = dict()
        node = self.transform(root, cudd.false, cudd.true, cudd2obdd)
        if root.negated:
            node = self.negation(node)
        self.root = node
        self.nn = self.bdd_size(self.root)

        all_nds = set()
        for ele in cudd2obdd:
            assert type(ele) == Function
            all_nds.add(ele)

        assert len(all_nds) == len(cudd2obdd)

        print(f"#### standardization finish, size of obdd: {self.bdd_size(self.root)} ####")
        print("#### check standardization ####")
        power = min(15, self.nv)
        d_len = 2 ** power
        for j in range(d_len):
            # extract value of cared features
            tmp_sample1 = decimalToBinary(j,self.nv)
            tmp_sample2 = dict()
            for k, val in enumerate(tmp_sample1):
                if val:
                    tmp_sample2.update({f'x_{k}':True})
                else:
                    tmp_sample2.update({f'x_{k}':False})
            pred1 = self.predict_one(tmp_sample1)
            ret = cudd.let(tmp_sample2,root)
            assert ret == cudd.true or ret == cudd.false
            pred2 = (ret == cudd.true)
            if pred1 != pred2:
                print(f"standardization failed: inconsistent on instance {tmp_sample1}",
                      f"OBDD predicts {pred1} while CUDD predicts {pred2}")
        for j in range(d_len*2, d_len*4, 4):
            tmp_sample1 = decimalToBinary(j,self.nv)
            tmp_sample2 = dict()
            for k, val in enumerate(tmp_sample1):
                if val:
                    tmp_sample2.update({f'x_{k}':True})
                else:
                    tmp_sample2.update({f'x_{k}':False})
            pred1 = self.predict_one(tmp_sample1)
            ret = cudd.let(tmp_sample2,root)
            assert ret == cudd.true or ret == cudd.false
            pred2 = (ret == cudd.true)
            if pred1 != pred2:
                print(f"standardization failed: inconsistent on instance {tmp_sample1}",
                      f"OBDD predicts {pred1} while CUDD predicts {pred2}")

        print(f"#### standardization succeed ####")

    ######################### CUDD to standard OBDD #########################
    def find_or_add_node(self, var_idx, low, high):
        """
            find or add a new node
        :param var_idx: variable
        :param low: low
        :param high: high
        :return: node (var,low,high)
        """
        assert not self.identical_node(low, high)
        key = (var_idx, low, high)
        node = self.tup2node.get(key)
        if node is None:
            node = DdNode(var_idx, low, high, self.nd_idx)
            self.idx2nd.update({self.nd_idx: node})
            self.nd_idx += 1
            self.tup2node.update({key: node})
        return node

    def negation(self, nd):
        """
            negating nd
        :param nd: node
        :return: negation of node
        """
        if nd == self.T1:
            return self.T0
        if nd == self.T0:
            return self.T1
        if nd in self.comp_tab:
            return self.comp_tab[nd]

        low = self.negation(nd.low)
        high = self.negation(nd.high)
        neg_nd = self.find_or_add_node(nd.v_id, low, high)

        assert nd not in self.comp_tab
        self.comp_tab.update({nd: neg_nd})

        if neg_nd not in self.comp_tab:
            self.comp_tab.update({neg_nd: nd})
        else:
            assert self.identical_node(nd, self.comp_tab[neg_nd])
        return neg_nd

    def transform(self, nd, zero, one, cudd_to_obdd):
        """
            transform OBDD (with CE) to OBDD (without CE)

        :param nd: node
        :param zero: terminal 0
        :param one:  terminal 1
        :param cudd_to_obdd: mapping cudd nodes to obdd nodes
        :return: transformed OBDD
        """
        if nd in cudd_to_obdd:
            return cudd_to_obdd[nd]

        assert nd not in cudd_to_obdd

        if nd == zero:
            return self.T0
        if nd == one:
            return self.T1
        low = self.transform(nd.low, zero, one, cudd_to_obdd)
        high = self.transform(nd.high, zero, one, cudd_to_obdd)
        assert not nd.high.negated
        if nd.low != zero and nd.low.negated:
            low = self.negation(low)
        var_str = nd.var.split('_')

        ret_nd = self.find_or_add_node(int(var_str[1]), low, high)
        cudd_to_obdd.update({nd: ret_nd})
        return ret_nd

    def identical_node(self, nd1, nd2):
        """
            Check if two node identical, compare (var, low, high)
        :param nd1: node 1
        :param nd2: node 2
        :return: true if two nodes are identical.
        """
        if nd1.n_id == nd2.n_id:
            return True
        if nd1 == self.T1 and nd2 == self.T1:
            return True
        if nd1 == self.T0 and nd2 == self.T0:
            return True
        if not self.is_terminal(nd1) and not self.is_terminal(nd2) \
                and nd1.v_id == nd2.v_id \
                and nd1.low == nd2.low and nd1.high == nd2.high:
            return True
        return False

    ######################### CUDD to standard OBDD #########################

    def compile_cnf_cudd(self, clauses, cudd: BDD):
        pd = cudd.true
        for i, clau in enumerate(clauses):
            sm = cudd.false
            for lit in clau:
                var = cudd.var(f'x_{abs(lit) - 1}')
                if lit < 0:
                    var = ~var
                tmp = var | sm
                sm = tmp
            out = pd & sm
            pd = out
        root = pd
        self.obdd_from_cudd(cudd, root)

    def is_terminal(self, nd):
        """
            Check if given node is a terminal node.

            :param nd: given node.
            :return: true if it is a terminal node else false.
        """
        return nd == self.T0 or nd == self.T1

    def total_assignment(self, assignment):
        """
            Get terminal node given a total assignment.

            :param assignment: a total assignment.
            :return: true if assignment evaluated to terminal one else false.
        """
        assert (len(assignment) == self.nv)
        nd = self.root
        while not self.is_terminal(nd):
            if assignment[nd.v_id]:
                nd = nd.high
            else:
                nd = nd.low
        assert self.is_terminal(nd)
        return nd == self.T1

    def predict_one(self, in_x):
        """
            Return prediction of one given instance.

            :param in_x: total instance (not partial instance).
            :return: prediction of this instance.
        """
        assignment = [int(pt) for pt in in_x]
        return self.total_assignment(assignment)

    def predict_all(self, in_x):
        """
            Return a list of prediction given a list of instances.

            :param in_x: a list of total instances.
            :return: predictions of all instances.
        """
        y_pred = []
        for ins in in_x:
            assignment = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(assignment))
        return y_pred

    def accuracy(self, in_x, y_true):
        """
            Compare the output of bdd and desired prediction
        :param in_x: a list of total instances.
        :param y_true: desired prediction
        :return: accuracy in float.
        """
        y_pred = []
        for ins in in_x:
            assignment = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(assignment))
        acc = accuracy_score(y_true, y_pred)
        return acc

    def bdd_size(self, nd):
        """
            Return size of OBDD rooted at given node.

            :param nd: root node.
            :return: size of OBDD.
        """
        counter = set()
        self._bdd_size(nd, counter)
        return len(counter) + 2

    def _bdd_size(self, nd, counter):
        if nd.n_id in counter:
            return
        counter.add(nd.n_id)
        if not self.is_terminal(nd.low):
            self._bdd_size(nd.low, counter)
        if not self.is_terminal(nd.high):
            self._bdd_size(nd.high, counter)

    def bfs(self, root):
        """
            Iterate through nodes in breadth first search (BFS) order.

            :param root: root node of OBDD.
            :return: a set of nodes in BFS-order.
        """
        yield from self._bfs(root, set())

    def _bfs(self, root, visited):
        queue = collections.deque()
        queue.appendleft(root)
        while queue:
            nd = queue.pop()
            if nd not in visited:
                low = None
                high = None
                if nd != self.T0 and nd != self.T1:
                    low = nd.low
                    high = nd.high
                if low is not None:
                    queue.appendleft(low)
                if high is not None:
                    queue.appendleft(high)
                visited.add(nd)
                yield nd

    @staticmethod
    def save_model(obdd, filename):
        """
            Save OBDD to pickle model.

            :param obdd: given OBDD classifier.
            :param filename: file storing OBDD.
            :return: none.
        """
        with open(filename, "wb") as f:
            pickle.dump(obdd, f)

    @classmethod
    def from_file(cls, filename):
        """
            Load OBDD classifier from file.

            :param filename: OBDD in pickle.
            :return: OBDD classifier.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def support_var(self, root):
        """
            Get dependency set of OBDD, i.e. exclude all don't-cared features.

            :param root: root node of OBDD or sub-OBDD.
            :return: a set of variable index.
        """
        vars_ = set()
        for nd in self.bfs(root):
            if self.is_terminal(nd):
                continue
            if nd.v_id not in vars_:
                vars_.add(nd.v_id)
        return vars_

    def save_xpg_format(self, inst, filename):
        """
            Save OBDD and given instance to a .xpg format file.

            :param inst: given instance.
            :param filename: .xpg file.
            :return: none.
        """
        prediction = self.predict_one(inst)
        n_idx = 1
        new_nid = dict()
        vars_ = set()
        internal = []

        str_t = "T: "
        str_i = "NT: "
        for nd in self.bfs(self.root):
            new_nid.update({nd.n_id: n_idx})
            if self.is_terminal(nd):
                str_t += f"{str(n_idx)} "
            else:
                internal.append(nd)
                if nd.v_id not in vars_:
                    vars_.add(nd.v_id)
                str_i += f"{str(n_idx)} "
            n_idx += 1
        str_t = str_t[:-1] + "\n"
        str_i = str_i[:-1] + "\n"

        if self.nn is None:
            self.nn = self.bdd_size(self.root)

        output = f"NN: {self.nn}\n"
        output += "Root: 1\n"

        output += str_t
        output += "TDef:\n"
        if prediction:
            output += f"{new_nid[self.T0.n_id]} 0\n{new_nid[self.T1.n_id]} 1\n"
        else:
            output += f"{new_nid[self.T0.n_id]} 1\n{new_nid[self.T1.n_id]} 0\n"

        str_p2c = ""
        str_n2v = ""
        for nd in internal:
            low = nd.low
            high = nd.high
            str_n2v += f"{new_nid[nd.n_id]} {self.features[nd.v_id]}\n"
            if int(inst[nd.v_id]):
                str_p2c += f"{new_nid[nd.n_id]} {new_nid[low.n_id]} 0\n"
                str_p2c += f"{new_nid[nd.n_id]} {new_nid[high.n_id]} 1\n"
            else:
                str_p2c += f"{new_nid[nd.n_id]} {new_nid[low.n_id]} 1\n"
                str_p2c += f"{new_nid[nd.n_id]} {new_nid[high.n_id]} 0\n"

        output += str_i
        output += "NTDef:\n"
        output += str_p2c
        output += f"NV: {len(vars_)}\n"
        output += "VarDef:\n"
        output += str_n2v
        savefile = open(filename, "w")
        savefile.write(output)
        savefile.close()

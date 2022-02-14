#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Feature Membership
import time
from copy import deepcopy
from queue import Queue
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysdd.sdd import SddNode
from threading import Timer


class FeatureMembership(object):
    """
        Feature membership query. Compute an AXp containing given feature.
    """

    def __init__(self, root: SddNode, nv, features, sup_vars, verb=0):
        self.root = root
        self.nv = nv
        self.features = features
        self.sup_vars = sup_vars
        self.verbose = verb
        self.leaves = set()
        self.nd2id = dict()
        self.lit2idx = dict()
        self.vpool = IDPool()
        self.f_timeout = False
        self.preprocess()

    def new_var(self, name):
        """
            Find or new a PySAT variable.
            See PySat.

            :param name: name of variable
            :return: index of variable
        """
        return self.vpool.id(f'{name}')

    def preprocess(self):
        """
            Preprocessing, initializing some attributes.
            :return: None
        """
        pos_lits = []
        for ii in range(self.nv):
            pos_lits.append(self.sup_vars[ii].literal)
            self.lit2idx.update({self.sup_vars[ii].literal: ii})

        visited = set()
        q = Queue()
        q.put(self.root)
        while not q.empty():
            nd = q.get()
            if nd in visited:
                continue
            visited.add(nd)
            self.nd2id.update({nd: len(visited)})
            if nd.is_decision():
                for prime, sub in nd.elements():
                    q.put(prime)
                    q.put(sub)
                    if (prime, sub) not in visited:
                        visited.add((prime, sub))
                        self.nd2id.update({(prime, sub): len(visited)})
            else:
                assert nd.is_true() or nd.is_false() or nd.is_literal()
        assert q.empty()
        q.put(self.root)
        expand = set()
        while not q.empty():
            nd = q.get()
            if nd in expand:
                continue
            expand.add(nd)
            if nd.is_decision():
                for prime, sub in nd.elements():
                    assert not prime.is_true() and not prime.is_false()
                    if sub.is_true():
                        q.put(prime)
                    elif sub.is_false():
                        continue
                    else:
                        q.put(prime)
                        q.put(sub)
            else:
                assert nd not in self.leaves
                if nd.is_literal():
                    self.leaves.add(nd)

    def get_replica0(self, lits, feat_id):
        """
            Encoding 0-th replica of DAG.
            :param lits: given instance.
            :param feat_id: desired feature.
            :return: clauses
        """
        cls = CNF()
        appear = set()
        q = Queue()
        q.put(self.root)
        expand = set()
        while not q.empty():
            nd = q.get()
            if nd in expand:
                continue
            expand.add(nd)
            if nd.is_decision():
                # OR-node
                var_n = self.new_var(f'n_0_{self.nd2id[nd]}')
                tmp = []
                for prime, sub in nd.elements():
                    # AND-node and its successors
                    # true and false nodes are excluded in the queue
                    assert not prime.is_true() and not prime.is_false()
                    if sub.is_true():
                        # only prime has effect
                        var_c = self.new_var(f'n_0_{self.nd2id[(prime, sub)]}')
                        var_p = self.new_var(f'n_0_{self.nd2id[prime]}')
                        cls.append([-var_p, var_c])
                        cls.append([var_p, -var_c])
                        assert var_c not in tmp
                        tmp.append(var_c)
                        q.put(prime)
                    elif sub.is_false():
                        # skip this branch
                        continue
                    else:
                        var_c = self.new_var(f'n_0_{self.nd2id[(prime, sub)]}')
                        var_p = self.new_var(f'n_0_{self.nd2id[prime]}')
                        var_s = self.new_var(f'n_0_{self.nd2id[sub]}')
                        cls.append([-var_p, -var_s, var_c])
                        cls.append([var_p, -var_c])
                        cls.append([var_s, -var_c])
                        assert var_c not in tmp
                        tmp.append(var_c)
                        q.put(prime)
                        q.put(sub)
                for item in tmp:
                    cls.append([-item, var_n])
                cls.append([-var_n] + tmp)
        for nd in self.leaves:
            assert nd.is_literal()
            lit = nd.literal
            fid = self.lit2idx[abs(lit)]
            assert self.sup_vars[fid].literal == abs(lit)
            if lits[fid] == lit:
                cls.append([self.new_var(f'n_0_{self.nd2id[nd]}')])
            else:
                assert lits[fid] == -lit
                cls.append([-self.new_var(f's_{fid}'), -self.new_var(f'n_0_{self.nd2id[nd]}')])
                cls.append([self.new_var(f's_{fid}'), self.new_var(f'n_0_{self.nd2id[nd]}')])
                appear.add(fid)
        cls.append([-self.new_var(f'n_0_{self.nd2id[self.root]}')])
        cls.append([self.new_var(f's_{feat_id}')])
        noappear = []
        for i in range(self.nv):
            if i not in appear:
                noappear.append(i)
                cls.append([-self.new_var(f's_{i}')])
        print(f'There are #{len(noappear)} vars set to free, out of {self.nv}.')
        if feat_id in noappear:
            print(f'{feat_id} is free before solving')
        return cls

    def get_replica(self, lits, k):
        """
            Encode k-th replica.
            :param lits: given literal
            :param k: k > 0, feature index == k-1
            :return: hard and soft clauses
        """
        assert k > 0
        clsk = CNF()
        expand = set()
        q = Queue()
        q.put(self.root)
        while not q.empty():
            nd = q.get()
            if nd in expand:
                continue
            expand.add(nd)
            if nd.is_decision():
                var_n = self.new_var(f'n_{k}_{self.nd2id[nd]}')
                tmp = []
                for prime, sub in nd.elements():
                    assert not prime.is_true() and not prime.is_false()
                    if sub.is_true():
                        var_c = self.new_var(f'n_{k}_{self.nd2id[(prime, sub)]}')
                        var_p = self.new_var(f'n_{k}_{self.nd2id[prime]}')
                        clsk.append([-var_p, var_c])
                        clsk.append([var_p, -var_c])
                        assert var_c not in tmp
                        tmp.append(var_c)
                        q.put(prime)
                    elif sub.is_false():
                        continue
                    else:
                        var_c = self.new_var(f'n_{k}_{self.nd2id[(prime, sub)]}')
                        var_p = self.new_var(f'n_{k}_{self.nd2id[prime]}')
                        var_s = self.new_var(f'n_{k}_{self.nd2id[sub]}')
                        clsk.append([-var_p, -var_s, var_c])
                        clsk.append([var_p, -var_c])
                        clsk.append([var_s, -var_c])
                        assert var_c not in tmp
                        tmp.append(var_c)
                        q.put(prime)
                        q.put(sub)
                for item in tmp:
                    clsk.append([-item, var_n])
                clsk.append([-var_n] + tmp)
        clsk.append([-self.new_var(f's_{k-1}'), self.new_var(f'n_{k}_{self.nd2id[self.root]}')])
        clsk.append([self.new_var(f's_{k-1}'), -self.new_var(f'n_{k}_{self.nd2id[self.root]}')])
        for nd in self.leaves:
            lit = nd.literal
            fid = self.lit2idx[abs(lit)]
            assert self.sup_vars[fid].literal == abs(lit)
            if lits[fid] == lit:
                clsk.append([self.new_var(f'n_{k}_{self.nd2id[nd]}')])
            else:
                assert lits[fid] == -lit
                if k == fid+1:
                    clsk.append([self.new_var(f'n_{k}_{self.nd2id[nd]}')])
                else:
                    clsk.append([-self.new_var(f's_{fid}'), -self.new_var(f'n_{k}_{self.nd2id[nd]}')])
                    clsk.append([self.new_var(f's_{fid}'), self.new_var(f'n_{k}_{self.nd2id[nd]}')])
        return clsk

    def dfs_postorder(self, root):
        """
            Iterate through nodes in depth first search (DFS) post-order.

            :param root: a node of SDD.
            :return: a set of nodes in DFS-post-order.
        """

        #####################################################
        def _dfs_postorder(nd, visited):
            if nd in visited:
                return
            if nd.is_decision():
                for prime, sub in nd.elements():
                    yield from _dfs_postorder(prime, visited)
                    yield from _dfs_postorder(sub, visited)
            if nd not in visited:
                visited.add(nd)
                yield nd

        #####################################################
        yield from _dfs_postorder(root, set())

    def consistency(self, lits, fix):
        """
            Given instance and weak AXp
            check if the SDD is consistent or not.
            :param lits: given instance.
            :param fix: fix array (i.e. a weak axp)
            :return: true if consistent otherwise not.
        """
        assign = dict()
        for nd in self.leaves:
            assert nd.is_literal()
            lit = nd.literal
            fid = self.lit2idx[abs(lit)]
            assert self.sup_vars[fid].literal == abs(lit)
            if lits[fid] == lit:
                assign.update({nd: 1})
            else:
                assert lits[fid] == -lit
                if fix[fid]:
                    assign.update({nd: 0})
                else:
                    assign.update({nd: 1})
        for nd in self.dfs_postorder(self.root):
            if nd.is_decision():
                tmp = []
                for prime, sub in nd.elements():
                    assert not prime.is_true() and not prime.is_false()
                    if sub.is_true():
                        tmp.append(assign[prime])
                    elif sub.is_false():
                        continue
                    else:
                        tmp.append(assign[prime] * assign[sub])
                    if 1 in tmp:
                        break
                if 1 in tmp:
                    assign.update({nd: 1})
                else:
                    assign.update({nd: 0})
        assert assign[self.root] == 1 or assign[self.root] == 0
        return assign[self.root] == 1

    def answer(self, lits, feat_id, time_limit=1800):
        """
            Answer FM query "if there is an AXp containing feat_id":
            :param lits: given instance
            :param feat_id: desired feature
            :param time_limit: timeout
            :return: weakAXp/None, #CNF-vars, #CNF-clauses, timeout flag
        """
        def interrupt(slver1):
            """
                Interrupting the SAT solver if timeout.
                And set flag f_timeout to True.
                :param slver: given solver.
                :return: None
            """
            self.f_timeout = True
            slver1.interrupt()

        assert (self.nv == len(lits))
        for ii in range(self.nv):
            assert self.sup_vars[ii].literal == abs(lits[ii])

        if self.verbose:
            print('(Answer) Feature Membership of SDD into CNF formulas ...')

        cls = self.get_replica0(lits, feat_id)
        slv_guess = Solver(name="Glucose4", bootstrap_with=cls.clauses)
        timer = Timer(time_limit, interrupt, [slv_guess])
        timer.start()

        if self.verbose:
            print('Start solving...')
        time_solving_start = time.process_time()

        failed = False
        weak_axp = []
        clst = self.get_replica(lits, feat_id+1)
        slv_guess.append_formula(clst.clauses)
        nvars = slv_guess.nof_vars()
        nclaus = slv_guess.nof_clauses()
        if slv_guess.solve_limited(expect_interrupt=True):
            model = slv_guess.get_model()
            assert model
            for lit in model:
                name = self.vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 's':
                    if lit > 0:
                        weak_axp.append(int(name[1]))
            assert feat_id in weak_axp
        if self.f_timeout:
            print(f'Time out ({time_limit:.1f} secs)')
            failed = True
        timer.cancel()
        self.f_timeout = False

        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Solving time: {time_solving_end:.1f} secs")
        slv_guess.delete()
        if weak_axp:
            assert not failed
        if failed:
            assert len(weak_axp) == 0
        return weak_axp, nvars, nclaus, failed

    def extract(self, lits, feat_id, weak_axp, method='kc'):
        """
            Given a weak AXp as a seed, extract one AXp.

            :param lits: given instance
            :param feat_id: desired feature
            :param method:
                    1) 'enc', appending |X|+1 replica and call SAT solver.
                    2) 'kc', use operation supported by the PySDD package.
                    3) 'gt', graph traversal.
                    Note that method 2) is exponential time in worst case but faster in the experiment,
                    and the memory may increase dramatically.
                    3) is polytime in theory and practice but slower in experiment,
                    possibly because the algorithm is not optimized.
            :param weak_axp: given weak AXp.
            :return: one AXp
        """
        if method not in ('enc', 'kc', 'gt'):
            print(f'invalid parameter {method}')
            return None

        nvars = 0
        nclaus = 0

        assert feat_id in weak_axp
        if self.verbose:
            print(f'({method}) Start extracting...')
        time_solving_start = time.process_time()

        fix = [False] * self.nv
        for k in weak_axp:
            fix[k] = True

        if method == 'enc':
            cls = self.get_replica0(lits, feat_id)
            clst = self.get_replica(lits, feat_id+1)
            with Solver(name="Glucose4", bootstrap_with=cls.clauses) as slv_check:
                slv_check.append_formula(clst.clauses)
                for i in range(self.nv):
                    if not fix[i]:
                        slv_check.add_clause([-self.new_var(f's_{i}')])
                    elif i != feat_id:
                        clsi = self.get_replica(lits, i+1)
                        slv_check.append_formula(clsi.clauses)
                nvars = slv_check.nof_vars()
                nclaus = slv_check.nof_clauses()
                assert slv_check.solve()
                model = slv_check.get_model()
                assert model
                sat_axp = []
                for lit in model:
                    name = self.vpool.obj(abs(lit)).split(sep='_')
                    if name[0] == 's':
                        if lit > 0:
                            sat_axp.append(int(name[1]))
                for k in sat_axp:
                    fix[k] = True
        elif method == 'kc':
            # we disable dynamic minimization,
            # hence the size of SDD may increase dramatically.
            lits_ = deepcopy(lits)
            for i in range(self.nv):
                if not fix[i]:
                    lits_[i] = None
            for k in weak_axp:
                fix[k] = not fix[k]
                lits_[k] = None
                out = self.root
                for lit_ in lits_:
                    if lit_:
                        out = out.condition(lit_)
                if not out.is_false():
                    lits_[k] = lits[k]
                    fix[k] = not fix[k]
        else:
            for k in weak_axp:
                fix[k] = not fix[k]
                if self.consistency(lits, fix):
                    fix[k] = not fix[k]

        assert fix[feat_id]
        axp = [j for j in range(self.nv) if fix[j]]
        if self.verbose:
            feats_output = [self.features[i] for i in axp]
            if self.verbose == 1:
                print(f"AXp: {axp}")
            else:
                print(f"AXp: {axp} ({feats_output})")
        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Extracting time: {time_solving_end:.1f} secs")
        return axp, nvars, nclaus

    def check_one_axp(self, lits, axp):
        """
            Check if given axp is a subset-minimal weak axp
            of instance.

            :param axp: given axp.
            :return: true if it is subset-minimal weak axp, else false.
        """
        fix = [False] * self.nv
        for i in axp:
            fix[i] = True
        lits_ = deepcopy(lits)
        for i in range(self.nv):
            if not fix[i]:
                lits_[i] = None
        tmp = self.root
        for lit_ in lits_:
            if lit_:
                tmp = tmp.condition(lit_)
        if not tmp.is_false():
            print(f'given axp {axp} is not a weak AXp')
            return False
        for i in range(self.nv):
            if fix[i]:
                fix[i] = not fix[i]
                lits_[i] = None
                tmp = self.root
                for lit_ in lits_:
                    if lit_:
                        tmp = tmp.condition(lit_)
                if not tmp.is_false():
                    lits_[i] = lits[i]
                    fix[i] = not fix[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def GnC(self, lits, feat_id, time_limit=1800):
        """
            Generate and Check:
            1) m+-1 copies;
            2) 1 call to SAT solver.
            :param lits: given instance.
            :param feat_id: desired feature.
            :param time_limit: timeout.
            :return: AXp/None, #CNF-vars, #CNF-clauses, timeout flag
        """
        def interrupt(slver1):
            """
                Interrupting the SAT solver if timeout.
                And set flag f_timeout to True.
                :param slver: given solver.
                :return: None
            """
            self.f_timeout = True
            slver1.interrupt()

        assert (self.nv == len(lits))
        for ii in range(self.nv):
            assert self.sup_vars[ii].literal == abs(lits[ii])

        if self.verbose:
            print('(GnC) Feature Membership of SDD into CNF formulas ...')

        appear = set()
        for nd in self.leaves:
            lit = nd.literal
            fid = self.lit2idx[abs(lit)]
            if lits[fid] == lit:
                continue
            else:
                appear.add(fid)

        cls = self.get_replica0(lits, feat_id)
        slv_guess = Solver(name="Glucose4", bootstrap_with=cls.clauses)
        timer = Timer(time_limit, interrupt, [slv_guess])
        timer.start()

        if self.verbose:
            print('Start solving...')
        time_solving_start = time.process_time()

        failed = False
        axp = []
        for i in range(self.nv):
            if i in appear:
                clsi = self.get_replica(lits, i+1)
                slv_guess.append_formula(clsi.clauses)
        nvars = slv_guess.nof_vars()
        nclaus = slv_guess.nof_clauses()
        if slv_guess.solve_limited(expect_interrupt=True):
            model = slv_guess.get_model()
            assert model
            for lit in model:
                name = self.vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 's':
                    if lit > 0:
                        axp.append(int(name[1]))
            assert feat_id in axp
            if self.verbose:
                feats_output = [self.features[i] for i in axp]
                if self.verbose == 1:
                    print(f"AXp: {axp}")
                else:
                    print(f"AXp: {axp} ({feats_output})")
        if self.f_timeout:
            print(f'Time out ({time_limit:.1f} secs)')
            failed = True
        timer.cancel()
        self.f_timeout = False

        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Solving time: {time_solving_end:.1f} secs")
        slv_guess.delete()
        if axp:
            assert not failed
        if failed:
            assert len(axp) == 0
        return axp, nvars, nclaus, failed
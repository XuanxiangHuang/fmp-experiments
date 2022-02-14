#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Feature Membership
import time
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from xpg import XpGraph, MarcoXpG
from threading import Timer


class FeatureMembership(object):
    """
        Feature membership query. Compute an AXp containing given feature
        or enumerating all AXps containing given feature.
    """

    def __init__(self, xpg: XpGraph, verb=0):
        self.xpg = xpg
        self.verbose = verb
        self.vpool = IDPool()
        self.f_timeout = False

    def new_var(self, name):
        """
            Inner function,
            Find or new a PySAT variable.
            See PySat.

            :param name: name of variable
            :return: index of variable
        """
        return self.vpool.id(f'{name}')

    def get_replica0(self, feat_id):
        """
            Encoding 0-th replica of DAG.
            :param feat_id: desired feature.
            :return: clauses
        """
        cls = CNF()
        appear = set()
        G = self.xpg.graph
        neg_leaf0 = []
        for chd in G.nodes:
            if chd is self.xpg.root:
                continue
            if not G.out_degree(chd) and G.nodes[chd]['target']:
                continue
            if not G.out_degree(chd) and not G.nodes[chd]['target']:
                neg_leaf0.append(self.new_var(f'n_0_{chd}'))
            var_c = self.new_var(f'n_0_{chd}')
            parent = []
            for nd in G.predecessors(chd):
                var_n = self.new_var(f'n_0_{nd}')
                u = self.new_var('u_{0}'.format(G.nodes[nd]['var']))
                r = self.new_var(f'r_0_{nd}_{chd}')
                parent.append(r)
                if G.edges[nd, chd]['label']:
                    cls.append([-r, var_n])
                    cls.append([-var_n, r])
                else:
                    cls.append([-var_n, -u, r])
                    cls.append([-r, var_n])
                    cls.append([-r, u])
                    if G.nodes[nd]['var'] not in appear:
                        appear.add(G.nodes[nd]['var'])
            cls.append([-var_c] + parent)
            for item in parent:
                cls.append([-item, var_c])
        cls.append([self.new_var(f'n_0_{self.xpg.root}')])
        cls.append([self.new_var('ev_0')])
        cls.append([-self.new_var(f'u_{feat_id}')])
        cls.append(neg_leaf0 + [self.new_var('ev_0')])
        for item in neg_leaf0:
            cls.append([-self.new_var('ev_0'), -item])
        noappear = []
        for i in range(self.xpg.nv):
            if i not in appear:
                noappear.append(i)
                cls.append([self.new_var(f'u_{i}')])
        print(f'There are #{len(noappear)} vars set to free, out of {self.xpg.nv}.')
        if feat_id in noappear:
            print(f'{feat_id} is free before solving')
        return cls

    def get_replica(self, k):
        """
            Encode k-th replica.
            :param k: k > 0, feature index == k-1
            :return: hard and soft clauses
        """
        assert k > 0
        G = self.xpg.graph
        clsk = CNF()
        neg_leaf = []
        for chd in G.nodes:
            if chd is self.xpg.root:
                continue
            if not G.out_degree(chd) and G.nodes[chd]['target']:
                continue
            if not G.out_degree(chd) and not G.nodes[chd]['target']:
                neg_leaf.append(self.new_var(f'n_{k}_{chd}'))
            var_c = self.new_var(f'n_{k}_{chd}')
            parent = []
            for nd in G.predecessors(chd):
                var_n = self.new_var(f'n_{k}_{nd}')
                u = self.new_var('u_{0}'.format(G.nodes[nd]['var']))
                r = self.new_var(f'r_{k}_{nd}_{chd}')
                parent.append(r)
                if G.edges[nd, chd]['label']:
                    clsk.append([-r, var_n])
                    clsk.append([-var_n, r])
                else:
                    if k-1 == G.nodes[nd]['var']:
                        clsk.append([-r, var_n])
                        clsk.append([-var_n, r])
                    else:
                        clsk.append([-var_n, -u, r])
                        clsk.append([-r, var_n])
                        clsk.append([-r, u])
            clsk.append([-var_c] + parent)
            for item in parent:
                clsk.append([-item, var_c])
        clsk.append([self.new_var(f'n_{k}_{self.xpg.root}')])
        clsk.append(neg_leaf + [self.new_var(f'ev_{k}')])
        for item in neg_leaf:
            clsk.append([-self.new_var(f'ev_{k}'), -item])
        clsk.append([self.new_var(f'u_{k-1}'), -self.new_var(f'ev_{k}')])
        clsk.append([self.new_var(f'ev_{k}'), -self.new_var(f'u_{k-1}')])
        return clsk

    def answer(self, feat_id, time_limit=1800):
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

        if self.verbose:
            print('(Answer) Feature Membership of XpG into CNF formulas ...')

        cls = self.get_replica0(feat_id)
        slv_guess = Solver(name="Glucose4", bootstrap_with=cls.clauses)
        timer = Timer(time_limit, interrupt, [slv_guess])
        timer.start()

        if self.verbose:
            print('Start solving...')
        time_solving_start = time.process_time()

        failed = False
        weak_axp = []
        clst = self.get_replica(feat_id+1)
        slv_guess.append_formula(clst.clauses)
        nvars = slv_guess.nof_vars()
        nclaus = slv_guess.nof_clauses()
        if slv_guess.solve_limited(expect_interrupt=True):
            model = slv_guess.get_model()
            assert model
            for lit in model:
                name = self.vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 'u':
                    if lit < 0:
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

    def extract(self, feat_id, weak_axp, method='horn'):
        """
            Given a weak AXp as a seed, extract one AXp.

            :param feat_id: desired feature
            :param method:
                    1) 'enc', appending |X|+1 replica and call SAT solver.
                    2) 'horn', use Horn encoding supported by the XpG package.
                    3) 'gt', graph traversal algorithm supported by the XpG package.
            :param weak_axp: given weak AXp.
            :return: one AXp
        """
        if method not in ('enc', 'horn', 'gt'):
            print(f'invalid parameter {method}')
            return None

        nvars = 0
        nclaus = 0

        assert feat_id in weak_axp
        if self.verbose:
            print(f'({method}) Start extracting...')
        time_solving_start = time.process_time()

        fix = [False] * self.xpg.nv
        for k in weak_axp:
            fix[k] = True

        if method == 'enc':
            cls = self.get_replica0(feat_id)
            clst = self.get_replica(feat_id+1)
            with Solver(name="Glucose4", bootstrap_with=cls.clauses) as slv_check:
                slv_check.append_formula(clst.clauses)
                for i in range(self.xpg.nv):
                    if not fix[i]:
                        slv_check.add_clause([self.new_var(f'u_{i}')])
                    elif i != feat_id:
                        clsi = self.get_replica(i+1)
                        slv_check.append_formula(clsi.clauses)
                nvars = slv_check.nof_vars()
                nclaus = slv_check.nof_clauses()
                assert slv_check.solve()
                model = slv_check.get_model()
                assert model
                sat_axp = []
                for lit in model:
                    name = self.vpool.obj(abs(lit)).split(sep='_')
                    if name[0] == 'u':
                        if lit < 0:
                            sat_axp.append(int(name[1]))
                for k in sat_axp:
                    fix[k] = True
                if self.verbose:
                    feats_output = [self.xpg.features[i] for i in sat_axp]
                    if self.verbose == 1:
                        print(f"AXp: {sat_axp}")
                    else:
                        print(f"AXp: {sat_axp} ({feats_output})")
        elif method == 'horn':
            marco = MarcoXpG(self.xpg, verb=1, Horn=True)
            xpg_axp = marco.find_axp(fix)
            for k in xpg_axp:
                fix[k] = True
        else:
            marco = MarcoXpG(self.xpg, verb=1, Horn=False)
            xpg_axp = marco.find_axp(fix)
            for k in xpg_axp:
                fix[k] = True

        assert fix[feat_id]
        axp = [j for j in range(self.xpg.nv) if fix[j]]
        time_solving_end = time.process_time() - time_solving_start
        if self.verbose:
            print(f"Extracting time: {time_solving_end:.1f} secs")
        return axp, nvars, nclaus

    def check_one_axp(self, axp):
        """
            Check if given axp is an AXp.

            :param axp: potential abductive explanation.
            :return: true if given axp is an AXp
                        else false.
        """
        univ = [True] * self.xpg.nv
        for i in axp:
            univ[i] = not univ[i]

        if self.xpg.path_to_zero(univ):
            print(f'given axp {axp} is not a weak AXp')
            return False

        for i in range(len(univ)):
            if not univ[i]:
                univ[i] = not univ[i]
                if self.xpg.path_to_zero(univ):
                    univ[i] = not univ[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def GnC(self, feat_id, time_limit=1800):
        """
            Generate and Check:
            1) m+1 copies;
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

        if self.verbose:
            print('(GnC) Feature Membership of XpG into CNF formulas ...')

        cls = self.get_replica0(feat_id)
        slv_guess = Solver(name="Glucose4", bootstrap_with=cls.clauses)
        timer = Timer(time_limit, interrupt, [slv_guess])
        timer.start()

        if self.verbose:
            print('Start solving...')
        time_solving_start = time.process_time()

        failed = False
        axp = []
        for i in range(self.xpg.nv):
            clsi = self.get_replica(i+1)
            slv_guess.append_formula(clsi.clauses)
        nvars = slv_guess.nof_vars()
        nclaus = slv_guess.nof_clauses()
        if slv_guess.solve_limited(expect_interrupt=True):
            model = slv_guess.get_model()
            assert model
            for lit in model:
                name = self.vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 'u':
                    if lit < 0:
                        axp.append(int(name[1]))
            assert feat_id in axp
            if self.verbose:
                feats_output = [self.xpg.features[i] for i in axp]
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


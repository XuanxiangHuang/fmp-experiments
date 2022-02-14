#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments (answer fmp and find an AXp)
import csv
import resource
import sys
from Orange.data import Table
from math import ceil
from classifiers.decision_tree import DecisionTree
from classifiers.obdd import OBDD
from xpg import XpGraph
from pysdd.sdd import Vtree, SddManager, SddNode
from fmp.xpg_membership import FeatureMembership as FMXPG
from fmp.sdd_membership import FeatureMembership as FMSDD


def dt_get_one_axp(data_name, dataset, model, insts_seed_file, feats_file, tmp_dir, method='horn'):
    axps = []
    cnf_nv = []
    cnf_claus = []
    T_time = 0
    M_time = 0
    m_time = 99999
    nv = 0
    nn = 0
    seeds = []
    ########### read instance seed file ###########
    with open(insts_seed_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            seeds.append(int(line[0]))
    ########### read instance seed file ###########
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    assert len(seeds) == len(feat_lines)
    d_len = len(seeds)
    failed_count = 0
    ########### generate instance ###########
    datatable = Table(dataset)
    inst_data = Table.from_table_rows(datatable, seeds)
    ########### generate instance ###########
    for idx, line in enumerate(inst_data.X):
        ########### for each instance, generate a xpg file ###########
        tmp_inst = line
        assert len(tmp_inst) == len(model.features)
        tmp_xpg_file = tmp_dir + f"/{data_name}_xpg.txt"
        model.save_xpg_format(inst=tmp_inst, filename=tmp_xpg_file)
        ########### for each instance, generate a xpg file ###########
        ########### load this xpg file and explain ###########
        xpG = XpGraph.from_file(tmp_xpg_file)
        nv = xpG.nv
        nn = len(xpG.graph.nodes)
        feat_mem = FMXPG(xpG, verb=1)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        f_id = int(feat_lines[idx])
        assert 0 <= f_id <= xpG.nv-1

        print(f"{data_name}, {idx}-th inst file out of {d_len}")
        print(f"SAT encoding: query on feature {f_id} out of {xpG.nv} features:")
        axp = []
        weakaxp, nv_cnf, claus_cnf, failed = feat_mem.answer(f_id)
        if failed:
            failed_count += 1
        elif weakaxp:
            print('Answer Yes')
            axp, nv_cnf_, claus_cnf_ = feat_mem.extract(f_id, weakaxp, method)
            if method == 'enc':
                nv_cnf += nv_cnf_
                claus_cnf += claus_cnf_
            axps.append(axp)
        else:
            print('=============== no AXp exists ===============')
        cnf_nv.append(nv_cnf)
        cnf_claus.append(claus_cnf)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                  resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start

        if axp:
            assert f_id in axp
            print('=== check AXp ===')
            assert feat_mem.check_one_axp(axp) is True
            print('=== check succeed ===')

        T_time += time_i
        if time_i > M_time:
            M_time = time_i
        if time_i < m_time:
            m_time = time_i

    exp_results = f"{data_name} & {d_len} & "
    exp_results += f"{nv} & {nn} & "
    exp_results += f"{ceil(len(axps) / d_len * 100):.0f} ({ceil(failed_count / d_len * 100):.0f}) & "
    exp_results += f"{max([len(x) for x in axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in axps]) / len(axps)):.0f} & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n"\
        .format(T_time, M_time, m_time, T_time / d_len)

    print(exp_results)

    with open('results/kr22/xpg_fm_axp.txt', 'a') as f:
        f.write(exp_results)


def obdd_get_one_axp(data_name, model, insts_file, feats_file, tmp_dir, method='horn'):
    axps = []
    cnf_nv = []
    cnf_claus = []
    T_time = 0
    M_time = 0
    m_time = 99999
    nv = 0
    nn = 0
    tested = set()
    ########### read instance file ###########
    with open(insts_file, 'r') as fp:
        inst_lines = fp.readlines()
    ########### read instance file ###########
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    assert len(inst_lines) == len(feat_lines)
    d_len = len(inst_lines)
    failed_count = 0
    for idx, line in enumerate(inst_lines):
        ########### for each instance, generate a xpg file ###########
        tmp_inst = [int(v.strip()) for v in line.split(',')]
        assert tuple(tmp_inst) not in tested
        tested.add(tuple(tmp_inst))
        assert len(tmp_inst) == len(model.features)
        tmp_xpg_file = tmp_dir + f"/{data_name}_xpg.txt"
        model.save_xpg_format(inst=tmp_inst, filename=tmp_xpg_file)
        ########### for each instance, generate a xpg file ###########
        ########### load this xpg file and explain ###########
        xpG = XpGraph.from_file(tmp_xpg_file)
        nv = xpG.nv
        nn = len(xpG.graph.nodes)
        feat_mem = FMXPG(xpG, verb=1)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        f_id = int(feat_lines[idx])
        assert 0 <= f_id <= xpG.nv-1

        print(f"{data_name}, {idx}-th inst file out of {d_len}")
        print(f"SAT encoding: query on feature {f_id} out of {xpG.nv} features:")
        axp = []
        weakaxp, nv_cnf, claus_cnf, failed = feat_mem.answer(f_id)
        if failed:
            failed_count += 1
        elif weakaxp:
            print('Answer Yes')
            axp, nv_cnf_, claus_cnf_ = feat_mem.extract(f_id, weakaxp, method)
            if method == 'enc':
                nv_cnf += nv_cnf_
                claus_cnf += claus_cnf_
            axps.append(axp)
        else:
            print('=============== no AXp exists ===============')
        cnf_nv.append(nv_cnf)
        cnf_claus.append(claus_cnf)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                  resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start

        if axp:
            assert f_id in axp
            print('=== check AXp ===')
            assert feat_mem.check_one_axp(axp) is True
            print('=== check succeed ===')

        T_time += time_i
        if time_i > M_time:
            M_time = time_i
        if time_i < m_time:
            m_time = time_i

    exp_results = f"{data_name} & {d_len} & "
    exp_results += f"{nv} & {nn} & "
    exp_results += f"{ceil(len(axps) / d_len * 100):.0f} ({ceil(failed_count / d_len * 100):.0f}) & "
    exp_results += f"{max([len(x) for x in axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in axps]) / len(axps)):.0f} & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n"\
        .format(T_time, M_time, m_time, T_time / d_len)

    print(exp_results)

    with open('results/kr22/xpg_fm_axp.txt', 'a') as f:
        f.write(exp_results)


def support_vars(sdd: SddManager):
    """
        Given a SDD manager, return support variables,
        i.e. variables that used/referenced by SDD node.
        :param sdd: SDD manager
        :return:
    """
    all_vars = [_ for _ in sdd.vars]
    nv = len(all_vars)
    sup_vars = [None] * nv

    for i in range(nv):
        lit = all_vars[i].literal
        assert (lit == i + 1)
        neglit = -all_vars[i].literal
        if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
            sup_vars[i] = all_vars[i]
    return sup_vars


def to_lits(sup_vars, inst):
    lits = [None] * len(inst)

    for j in range(len(inst)):
        if sup_vars[j]:
            if int(inst[j]):
                lits[j] = sup_vars[j].literal
            else:
                lits[j] = -sup_vars[j].literal
    return lits


def prediction(root: SddNode, lits):
    out = root
    for item in lits:
        if item:
            out = out.condition(item)
    assert out.is_true() or out.is_false()
    return True if out.is_true() else False


def sdd_get_one_axp(sdd_file, vtree_file, circuit, insts_file, feats_file, method='kc'):
    name = circuit
    ######################  Pre-processing: original #####################
    # string to bytes
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    # Disable gc and minimization
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    root.ref()
    sdd.garbage_collect()
    assert not root.garbage_collected()
    # obtain all variables (don't cared variables are None)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    # get all features
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    # extract cared features and variables
    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing: original #####################
    T_time = 0
    M_time = 0
    m_time = 99999
    tested = set()
    sat_axps = []
    cnf_nv = []
    cnf_claus = []
    ###################### read instance file ######################
    with open(insts_file, 'r') as fp:
        inst_lines = fp.readlines()
    ###################### read instance file ######################
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    assert len(inst_lines) == len(feat_lines)
    d_len = len(inst_lines)
    failed_count = 0

    for i, s in enumerate(inst_lines):
        tmp_inst = [int(v.strip()) for v in s.split(',')]
        # extract value of cared features
        assert tuple(tmp_inst) not in tested
        tested.add(tuple(tmp_inst))

        assert len(tmp_inst) == tmp_nv

        inst = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                inst.append(tmp_inst[ii])

        lits = to_lits(sup_vars, inst)
        pred = prediction(root, lits)

        assert pred is False

        feat_mem = FMSDD(root, nv, features, sup_vars, verb=1)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        f_id = int(feat_lines[i])
        assert 0 <= f_id <= nv - 1
        print(f"{name}, {i}-th inst file out of {d_len}")
        print(f"SAT encoding: query on feature {f_id} out of {nv} features:")
        sat_axp = []
        sat_weakaxp, nv_cnf, claus_cnf, failed = feat_mem.answer(lits, f_id)
        if failed:
            failed_count += 1
        elif sat_weakaxp:
            print("Answer Yes")
            sat_axp, nv_cnf_, claus_cnf_ = feat_mem.extract(lits, f_id, sat_weakaxp, method)
            if method == 'enc':
                nv_cnf += nv_cnf_
                claus_cnf += claus_cnf_
            sat_axps.append(sat_axp)
        else:
            print('=============== no AXp exists ===============')
        cnf_nv.append(nv_cnf)
        cnf_claus.append(claus_cnf)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start

        if sat_axp:
            assert f_id in sat_axp
            print('=== check AXp ===')
            assert feat_mem.check_one_axp(lits, sat_axp) is True
            print('=== check succeed ===')

        if method == 'kc':
            print('=== SDD garbage collection ===')
            sdd.garbage_collect()
            assert not root.garbage_collected()

        T_time += time_i
        if time_i > M_time:
            M_time = time_i
        if time_i < m_time:
            m_time = time_i

    exp_results = f"{name} & {d_len} & "
    exp_results += f"{nv} & nn & "
    exp_results += f"{ceil(len(sat_axps) / d_len * 100):.0f} ({ceil(failed_count / d_len * 100):.0f}) & "
    exp_results += f"{max([len(x) for x in sat_axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in sat_axps]) / len(sat_axps)):.0f} & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n" \
        .format(T_time, M_time, m_time, T_time / d_len)

    print(exp_results)

    with open('results/kr22/sdd_fm_axp.txt', 'a') as f:
        f.write(exp_results)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}_list.txt", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"############ {name} ############")

            if args[2] == '-dt':
                dataset = f"datasets/{name}.csv"
                test_insts_seed = f"samples/test_insts/{bench_name}/{name}_seed.csv"
                test_feats = f"samples/test_feats/{bench_name}/{name}.csv"
                save_xpg = f"models/tmp/{bench_name}"
                md_file = f"models/dts/{bench_name}/{name}.pkl"
                in_model = DecisionTree.from_file(md_file)
                nn = in_model.size(in_model.classifier.root)
                dt_get_one_axp(name, dataset, in_model, test_insts_seed, test_feats, save_xpg)
            elif args[2] == '-obdd':
                test_insts = f"samples/test_insts/{bench_name}/{name}.csv"
                test_feats = f"samples/test_feats/{bench_name}/{name}.csv"
                save_xpg = f"models/tmp/{bench_name}"
                md_file = f"models/obdds/{bench_name}/{name}.pkl"
                in_model = OBDD.from_file(md_file)
                if in_model.nn > 7500:
                    obdd_get_one_axp(name, in_model, test_insts, test_feats, save_xpg)
            elif args[2] == '-sdd':
                # circuit
                circuit_sdd = f"models/sdds/{bench_name}/{name}.txt"
                circuit_sdd_vtree = f"models/sdds/{bench_name}/{name}_vtree.txt"
                test_insts = f"samples/test_insts/{bench_name}/{name}.csv"
                test_feats = f"samples/test_feats/{bench_name}/{name}.csv"
                sdd_get_one_axp(circuit_sdd, circuit_sdd_vtree, name, test_insts, test_feats)
    exit(0)

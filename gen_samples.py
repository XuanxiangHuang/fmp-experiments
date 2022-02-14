#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate tested instances/features
import sys
from Orange.data import Table
import random
import csv
import numpy as np
from classifiers.decision_tree import DecisionTree
from classifiers.obdd import OBDD
from pysdd.sdd import Vtree, SddManager, SddNode


def dt_gen_tested_insts(data_name, dataset, num_test, save_dir):
    data = Table(dataset)
    print("Dataset instances:", len(data))
    sample_seed_row = np.array(random.sample(list(range(len(data))), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_insts/{save_dir}/{data_name}_seed.csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


def dt_gen_tested_feats(data_name, dt_model: DecisionTree, num_test, save_dir):
    nv = len(dt_model.support_var(dt_model.classifier.root))
    print("Features:", nv)
    if num_test < nv:
        sample_seed_row = np.array(random.sample(list(range(nv)), num_test))
    else:
        sample_seed_row = np.array(np.random.choice(list(range(nv)), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{save_dir}/{data_name}.csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


def obdd_gen_tested_insts(circuit, obdd_model: OBDD, num_test, save_dir):
    name = circuit
    tested = set()
    d_len = num_test

    round_i = 0
    while round_i < d_len:
        tmp_sample = []
        for ii in range(obdd_model.nv):
            tmp_sample.append(random.randint(0, 1))
        while tuple(tmp_sample) in tested:
            tmp_sample = []
            for ii in range(obdd_model.nv):
                tmp_sample.append(random.randint(0, 1))

        assert tuple(tmp_sample) not in tested
        tested.add(tuple(tmp_sample))

        pred = obdd_model.predict_one(tmp_sample)
        if pred:
            print(f"instance: {tmp_sample}, prediction: {pred}")

        round_i += 1

    assert len(tested) == num_test
    data = []
    for item in tested:
        csv_item = list(item)
        assert len(csv_item) == obdd_model.nv
        data.append(csv_item)

    with open(f"samples/test_insts/{save_dir}/{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return


def obdd_gen_tested_feats(circuit, obdd_model: OBDD, num_test, save_dir):
    name = circuit
    nv = len(obdd_model.support_var(obdd_model.root))
    if num_test < nv:
        sample_seed_row = np.array(random.sample(list(range(nv)), num_test))
    else:
        sample_seed_row = np.array(np.random.choice(list(range(nv)), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{save_dir}/{name}.csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


def support_vars(sdd: SddManager):
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


def sdd_gen_tested_insts(sdd_file, vtree_file, circuit, num_test, save_dir, features_list=None):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing:  #####################

    basic_feats = []
    gen_feats = []
    gen_feats_lines = None
    if features_list:
        with open(features_list, 'r') as flp:
            lines = flp.readlines()
        index = 0
        while index < len(lines):
            lit_description = lines[index].strip().split(sep=':')
            if lit_description[1].startswith(" logic") and lit_description[0] != '0':
                basic_feats.append(int(lit_description[0]))
            elif lit_description[1].startswith(" conj"):
                if len(gen_feats) == 0:
                    gen_feats_lines = lines[index:]
                gen_feats.append(int(lit_description[0]))
            index += 1

    tested = set()
    d_len = num_test

    round_i = 0
    if features_list is None:
        while round_i < d_len:
            tmp_sample = []
            for ii in range(tmp_nv):
                tmp_sample.append(random.randint(0, 1))
            while tuple(tmp_sample) in tested:
                tmp_sample = []
                for ii in range(tmp_nv):
                    tmp_sample.append(random.randint(0, 1))

            assert tuple(tmp_sample) not in tested

            sample = []
            for ii in range(tmp_nv):
                if tmp_sup_vars[ii]:
                    sample.append(tmp_sample[ii])

            lits = to_lits(sup_vars, sample)
            pred = prediction(root, lits)

            if pred:
                continue

            tested.add(tuple(tmp_sample))
            round_i += 1

    assert len(tested) == num_test
    data = []
    for item in tested:
        csv_item = list(item)
        assert len(csv_item) == tmp_nv
        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(int(csv_item[ii]))

        lits = to_lits(sup_vars, sample)
        pred = prediction(root, lits)
        assert pred is False
        data.append(csv_item)

    with open(f"samples/test_insts/{save_dir}/{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return


def sdd_gen_tested_feats(sdd_file, vtree_file, circuit, num_test, save_dir, features_list=None):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []

    if features_list:
        all_lits = set()
        with open(features_list, 'r') as flp:
            lines = flp.readlines()
        index = 0
        while index < len(lines):
            lit_description = lines[index].strip().split(sep=':')
            if lit_description[1].startswith(" logic") and lit_description[0] != '0':
                all_lits.add(int(lit_description[0]))
            index += 1

        for jj in range(tmp_nv):
            if tmp_sup_vars[jj] and (jj+1) in all_lits:
                sup_vars.append(tmp_sup_vars[jj])
                features.append(tmp_features[jj])
        nv = len(sup_vars)
        assert (len(features) == nv)
    else:
        for jj in range(tmp_nv):
            if tmp_sup_vars[jj]:
                sup_vars.append(tmp_sup_vars[jj])
                features.append(tmp_features[jj])
        nv = len(sup_vars)
        assert (len(features) == nv)

    ######################  Pre-processing:  #####################
    if num_test < nv:
        sample_seed_row = np.array(random.sample(list(range(nv)), num_test))
    else:
        sample_seed_row = np.array(np.random.choice(list(range(nv)), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{save_dir}/{name}.csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 5 and args[0] == '-bench':
        bench_name = args[1]

        with open(f"{bench_name}_list.txt", 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            data_file = f"datasets/{name}.csv"

            if args[4] == '-dt':
                if args[2] == '-inst':
                    dt_gen_tested_insts(name, data_file, int(args[3]), bench_name)
                elif args[2] == '-feat':
                    dt_md = DecisionTree.from_file(f"models/dts/{bench_name}/{name}.pkl")
                    dt_gen_tested_feats(name, dt_md, int(args[3]), bench_name)
            elif args[4] == '-obdd':
                obdd = OBDD.from_file(f"models/obdds/{bench_name}/{name}.pkl")
                if args[2] == '-inst':
                    obdd_gen_tested_insts(name, obdd, int(args[3]), bench_name)
                elif args[2] == '-feat':
                    obdd_gen_tested_feats(name, obdd, int(args[3]), bench_name)
            elif args[4] == '-sdd':
                circuit_sdd = f"models/sdds/{bench_name}/{name}.txt"
                circuit_sdd_vtree = f"models/sdds/{bench_name}/{name}_vtree.txt"
                feats_list = None
                if bench_name == 'density-estimation':
                    feats_list = f"models/sdds/{bench_name}/{name}_feats.txt"
                    # not to distinguish original features and generated features
                if args[2] == '-inst':
                    sdd_gen_tested_insts(circuit_sdd, circuit_sdd_vtree, name, int(args[3]), bench_name)
                elif args[2] == '-feat':
                    sdd_gen_tested_feats(circuit_sdd, circuit_sdd_vtree, name, int(args[3]), bench_name)
            print(name)

    exit(0)
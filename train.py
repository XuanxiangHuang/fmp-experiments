#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   1) Train Decision Trees
#   2) Compile OBDDs
from __future__ import print_function
import sys,os,getopt
from classifiers.decision_tree import DecisionTree
from classifiers.obdd import OBDD,parse_cnf_header


class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        self.command = command
        self.max_depth = 10
        self.test_split = 0.2
        self.train_classifier = 'dt'
        self.train_threshold = 0.5
        self.test_threshold = 0.5
        self.output = ''
        self.file = None

        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """
        try:
            opts, args = getopt.getopt(command[1:], 'c:d:f:hl:o:t:v:',
                                       ['classifier=',
                                        'depth=',
                                        'file=',
                                        'help',
                                        'learn=',
                                        'output=',
                                        'test-split='
                                        ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-d', '--depth'):
                max_depth = int(arg)
                if max_depth <= 0:
                    print('wrong parameter: -d (depth)')
                    sys.exit(1)
                self.max_depth = max_depth
            elif opt in ('-c', '--classifier'):
                train_classifier = str(arg)
                if train_classifier not in ('dt', 'obdd'):
                    print('wrong parameter: -c (classifier)')
                    sys.exit(1)
                self.train_classifier = train_classifier
            elif opt in ('-t', '--test-split'):
                test_split = float(arg)
                if test_split > 1.0 or test_split < 0.2:
                    print('wrong parameter: -t (test-split)')
                    sys.exit(1)
                self.test_split = test_split
            elif opt in ('-l', '--learn'):
                thresholds = arg.split(':')
                train_threshold = float(thresholds[0])
                test_threshold = float(thresholds[1])
                if (train_threshold < 0.5 or train_threshold > 1) or \
                        (test_threshold < 0.5 or test_threshold > 1):
                    print('wrong parameter: -l (learn) threshold')
                    sys.exit(1)
                self.train_threshold = train_threshold
                self.test_threshold = test_threshold
            elif opt in ('-f', '--file'):
                self.file = arg
            elif opt in ('-o', '--output'):
                self.output = str(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            else:
                assert False, 'unhandled option: {0} {1}'.format(opt, arg)

        if not self.file:
            print('error: no input file')
            self.usage()
            sys.exit(1)

    def usage(self):
        print('Usage: ' + os.path.basename(self.command[0]) + ' [options]')
        print('Options:')
        print(' -d,  --depth=<int>       Maximal depth of a tree (default 8);')
        print(' -l,  --learn=<float>:<float> Learn dt with threshold. (default: train >= 0.5, test >= 0.5);')
        print('                              no applicable for obdd;')
        print(' -t,  --test-split=<float>    Training and test sets split (default 0.2 ∈ [0.2, 1.0]);')
        print(' -c,  --classifier=<string>   Type of classifier. (default: dt ∈ {dt, obdd}).')
        print('                              For sdd, please use SDD package: http://reasoning.cs.ucla.edu/sdd/;')
        print(' -f,  --file=<string>         Input file (CSV file or CNF file);')
        print(' -o,  --output=<string>       Provide a NAME (without suffix) for saving train classifier in .pkl;')
        print(' --help                       Show this message.')


def train_dt(dataset, max_depth, train_threshold, test_threshold, learning_rate, save_name=None):
    dt = DecisionTree(dataset, max_depth=max_depth, learning_rate=learning_rate)
    acc_train, acc_test = dt.train()

    if acc_train < train_threshold or acc_test < test_threshold:
        print(f'DT: train accuracy {acc_train} < {train_threshold}'
              f' or test accuracy {acc_test} < {test_threshold}')
        return
    else:
        print(f"DT, Train accuracy: {acc_train * 100.0}%")
        print(f"DT, Test accuracy: {acc_test * 100.0}%")

    if save_name:
        dt.save_model(dt, save_name)


##########################################################################################
if __name__ == '__main__':
    options = Options(sys.argv)
    input_file = options.file
    basename = os.path.splitext(os.path.basename(options.file))[0]
    print(f"### train {basename} ###")

    if options.train_classifier == 'dt':
        if options.output:
            train_dt(options.file, options.max_depth,
                     options.train_threshold, options.test_threshold, options.test_split,
                     options.output + '.pkl')
        else:
            train_dt(options.file, options.max_depth,
                     options.train_threshold, options.test_threshold, options.test_split)

    elif options.train_classifier == 'obdd':
        feats, claus = parse_cnf_header(options.file)
        obdd = OBDD()
        obdd.compile(in_model=claus, in_type='CNF', features=feats)
        if options.output:
            obdd.save_model(obdd, filename=options.output + '.pkl')

# FMP-experiments
feature membership for SDD and XpG

## Getting Started
To run the scripts, you need to install the following Python packages:
- [PySAT](https://github.com/pysathq/pysat): (0.1.7.dev15) for implementing SAT encoding.
- [PySDD](https://github.com/wannesm/PySDD): for loading SDDs and encoding.
- [Orange3](https://github.com/biolab/orange3) : (ver 3.31.1) for learning and loading DTs.
- [dd](https://github.com/tulip-control/dd): for compiling OBDDs (integrated with CUDD).
- [SDD](http://reasoning.cs.ucla.edu/sdd/): for compiling SDDs.
- [XpG](https://github.com/yizza91/xpg): for mapping DTs/OBDDs into XpG's.

## Benchmarks:
* [Penn ML Benchmarks](https://epistasislab.github.io/pmlb/), should be compatible with Orange3.
* [Density Estimation Datasets](https://github.com/UCLA-StarAI/Density-Estimation-Datasets)
* CNF files for compiling OBDD: [flat30-60](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html).
* CNF files for compiling SDD: [circuit](http://www.cril.univ-artois.fr/KC/benchmarks.html).

## Examples:
Training DTs/OBDDs:
```
python3 train.py -c dt -f datasets/adult.csv
python3 train.py -c obdd -f benchmark/flat30-60/flat30-1.cnf
```
Generating tested instances/features:
```
python3 gen_samples.py -bench iscas89 -inst 100 -sdd
python3 gen_samples.py -bench iscas89 -feat 100 -sdd
```
Experiment of answering FM queries:
```
python3 exp-answer.py -bench pmlb -dt
python3 exp-answer.py -bench flat30-60 -obdd
python3 exp-answer.py -bench iscas93 -sdd
python3 exp-answer.py -bench iscas89 -sdd
python3 exp-answer.py -bench density-estimation -sdd
```
Experiment of answering FM queries and return on AXp (two-step method):
```
python3 exp-find-one.py -bench pmlb -dt
python3 exp-find-one.py -bench flat30-60 -obdd
python3 exp-find-one.py -bench iscas93 -sdd
python3 exp-find-one.py -bench iscas89 -sdd
python3 exp-find-one.py -bench density-estimation -sdd
```
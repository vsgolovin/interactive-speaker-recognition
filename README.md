# Interactive Speaker Recognition

This repository explores the approach to speaker recognition described in paper
["A Machine of Few Words -- Interactive Speaker Recognition with Reinforcement Learning"](https://arxiv.org/abs/2008.03127).

## Installation

This project is implemented as a `pip` package. To install, clone this
repository and then run
```sh
pip install .
```

Add `-e` flag if you would like to modify package files (`src/isr`).

You will probably want to use GPU, therefore it's recommended you install
`torch` with CUDA prior to installing this package.

## Usage

In order to prepare data, train and test models you will need scripts from `src/`
directory. First you will need to download the TIMIT dataset and place a link to
it in `data/`
```sh
ln -s data/TIMIT PATH_TO_TIMIT
```

In order to use x-vector embeddings you will have to download and install
[`kaldi`](http://kaldi-asr.org/). To create files necessary for `kaldi` first
make sure your shell session has `KALDI_ROOT` variable defined, then run
```sh
python3 src/data_processing.py kaldi-data-prep
```
This will create files in `data/kaldi` directory. You will need those to extract
x-vector embeddings with `kaldi`, as well as `extract_kaldi_xvectors.sh`. See the
comments in this file for more information.

---
*NOTE*:

X-vector embeddings are not particularly great, so consider using a different 
pretrained model.

---

To convert extracted embeddings to numpy arrays run
```sh
python3 src/data_processing.py kaldi-to-numpy
```

Now you can finally train and test models. Here is a pipeline example:
```sh
mkdir output models
python3 src/guesser.py train
python3 src/guesser.py test
cp output/guesser.pth models/
python3 src/enquirer.py train
python3 src/enquirer.py test
cp output/actor.pth models/enquirer.pth
python3 src/select_words.py
cp output/word_scores_val.csv models/word_scores.csv
python3 src/heuristic_agent.py -w 3
```

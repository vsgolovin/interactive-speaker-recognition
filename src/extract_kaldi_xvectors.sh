#!/usr/bin/env bash

# This script extracts 512-dimensional x-vector embeddings for TIMIT recordings.
# Use it alongside files from Kaldi SRE16 example:
# * copy `kaldi/egs/sre16/v2` to a separate folder
# * extract pretrained model (http://www.kaldi-asr.org/models/m3)
# * add `--allow-downsample=true` option to `.../conf/mfcc.conf`
# * copy `data/kaldi` (created by `TimitCorpus.kaldi_data_prep`) contents to `.../data`
# * copy and execute this script
# * copy generated files (`xvectors_*` directories) to `data/`

# extract features
for datadir in 'train' 'test' 'words'; do
  steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc.conf \
    data/${datadir}
  utils/fix_data_dir.sh data/${datadir}
  sid/compute_vad_decision.sh --nj 30 \
    data/${datadir}
  utils/fix_data_dir.sh data/${datadir}
done

# extract x-vectors
for data in 'train' 'test' 'words'; do
  sid/nnet3/xvector/extract_xvectors.sh --nj 30 \
    exp/xvector_nnet_1a data/${data} \
    data/xvectors_${data}
done
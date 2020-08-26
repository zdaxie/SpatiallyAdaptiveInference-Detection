#!/usr/bin/env bash
pip install --user mmcv==0.2.8 numpy==1.16 matplotlib cython pillow
./compile.sh
python setup.py develop --user
pip install tensorflow --user
pip install tensorboardX --user
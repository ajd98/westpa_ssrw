#!/bin/bash

WEST_ROOT=~/development/westpa

#${WEST_ROOT}/bin/w_assign \
#    --bins-from-file BINS \
#    --states-from-file STATES

PATH_AFFIX="$WEST_ROOT/lib/blessings:$WEST_ROOT/lib/h5py:$WEST_ROOT/lib/wwmgr:$WEST_ROOT/src:$WEST_ROOT/lib/west_tools"
PYTHONPATH="${PATH_AFFIX}:${PYTHONPATH}"

${WEST_ROOT}/bin/west ${WEST_ROOT}/lib/west_tools/w_postanalysis_matrix.py 

${WEST_ROOT}/bin/west ${WEST_ROOT}/lib/west_tools/w_postanalysis_reweight.py \
    -e cumulative \
    --step-iter 1 



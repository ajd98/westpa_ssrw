#!/bin/sh
# env.sh
# 

export WEST_ROOT=~/development/westpa
if [[ -z "$WEST_ROOT" ]]; then
    echo "Environmental variable WEST_ROOT not set!"
    exit
fi
export WEST_PYTHON=$(which python2.7)
export WM_WORK_MANAGER=serial
export PYTHONPATH="$WEST_ROOT/lib/blessings:$WEST_ROOT/lib/h5py:$WEST_ROOT/lib/wwmgr:$WEST_ROOT/src:$WEST_ROOT/lib/west_tools"


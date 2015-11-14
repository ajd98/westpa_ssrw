#!/usr/bin/env python
#
# Testing script for functions in w_steadystate_reweight.py
#
# Alex DeGrave, 11/13/15
import w_steadystate_reweight
import h5py 


def test_get_transmat_and_obsmat_sum():
    ## Test 1 ##
    # Open the flux matrix file
    fluxmatH5 = h5py.File('test/flux_matrices.h5', 'r') 
    tsum, osum = w_steadystate_reweight.get_transmat_and_obsmat_sum(
                                                [fluxmatH5], 1, 100, 42, None
                                                                    )
    print(tsum[...])
    print(osum[...])

if __name__ == '__main__':
    test_get_transmat_and_obsmat_sum()
                                       

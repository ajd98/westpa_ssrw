
# Import statements, unedited from w_postanalysis_reweight.py
from __future__ import print_function, division; __metaclass__ = type
import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
import h5py

from collections import Counter

import westpa
from west.data_manager import weight_dtype, n_iter_dtype
from westtools import (WESTTool, WESTDataReader, IterRangeSelection,
                       ProgressIndicatorComponent)
from westpa import h5io
from westtools.dtypes import iter_block_ci_dtype as ci_dtype

log = logging.getLogger('westtools.w_postanalysis_reweight')

'''
-------------------------------------------------------------------------------
w_steadystate_reweight.py

W0rk in progress, by Alex DeGrave

Code is largely taken from w_postanalysis_reweight.py, part of the WESTPA 
toolkit.
-------------------------------------------------------------------------------

Combine data from multiple steady-state simulations (with recycling conditions)
to estimate the equilibrium probability distribution over a given set of bins.

Using information on bin-to-bin transitions output from 
w_postanalysis_matrix.py, this tool builds a matrix, where each element ``i,j``
of the matrix represents the probability that a walker in a given bin ``i`` 
will transition to another bin ``j`` in a given lagtime, tau. For simulations
involving a recycling condition, transitions out of bins denoted as recycling
bins are ignored (this may occur if the lag time for building the matrix is 
shorter than the time between split/merge/recycle events).

The user should specify input files for this tool in a YAML-format (``.yaml``)
file.  For each simulation included in the analysis, the user should provide
athe name of the output file from w_postanalyis_matrix, as well as the indices
of bins subject to recycling during the simulaiton (ie, the bins where, if a 
simulation is in that bin at the end of an iteration, it will be recycled). 

QUESTION REMAINING: when a walker is recycled, how does w_postanalysis_matrix 
account for the transition?  Since w_postanalysis_matrix only looks at 
transitions within each iteration (this should be valid, as the last timepoint
of iteration N is the same as the first timepoint of iteration N+1), I do NOT
need to do anything extra to account for recycling events within this code. 
-------------------------------------------------------------------------------
'''

# Copied from w_postanalysis_reweight.py.
# Added correction to make sure sum is actually 1 (NOT just within machine 
# precision.
def normalize(m):
    '''
    Normalize the matrix ``m`` along axis 1 and return a new matrix (with a new 
    address in memory).
    Thus, the returned matrix is right stochastic, with rows summing to 1.
    '''
    nm = m.copy()

    row_sum = m.sum(1)
    # Find the indices of elements of ``row_sum`` that are not equal to zero.
    # (We can't divide by zero to normalize to 1; that wouldn't make sense). 
    ii = np.nonzero(row_sum)[0]
    # Finally, do the actual normalization.
    nm[ii,:] = m[ii,:] / row_sum[ii][:, np.newaxis]

    return nm


def steadystate_solve(K):
    '''
    Given a matrix K, representing the (possibly unnormalized) transition
    probabilities
    '''
    # Reformulate K to remove sink/source states
    n_components, component_assignments = csgraph.connected_components(
                                                  K, 
                                                  connection="strong"
                                                                       )

    largest_component = Counter(component_assignments).most_common(1)[0][0]
    components = np.where(component_assignments == largest_component)[0]

    ii = np.ix_(components, components)
    K_mod = K[ii]
    K_mod = normalize(K_mod)

    # For RIGHT stochastic matrix, rows sum to 1, and we right multiply a 
    # probability vector by the matrix to obtain a new probability distribution
    # np.linalg.eig gives right eigenvectors of K_mod.T
    # This is the same as the left eigenvectors of K_mod 
    # For left stochastic matrix, columns sum to 1, and we left multiply a 
    # probabilty vecotr by the matrix to obtain a new probabilty distribution.
    # In summary, we want LEFT eigenvectors for a RIGHT stochastic matrix; since
    # np.linalg.eig gives RIGHT eigenvectors, we pass it the transpose of K_mod.
    eigvals, eigvecs = np.linalg.eig(K_mod.T)
    eigvals = np.real(eigvals)
    # Now, the eigenvectors we want are given by eigvecs[:,i]
    eigvecs = np.real(eigvecs)
    
    # Deal with the case that we don't have an eigenvalue of of 1. If this is
    # the case, then the matrix is not a stochastic matrix, so something is
    # arry.
    maxi = np.argmax(eigvals)
    if not np.allclose(np.abs(eigvals[maxi]), 1.0):
        print('WARNING: Steady-state undetermined for current iteration')
        bin_prob = K.diagonal().copy()
        bin_prob = bin_prob / np.sum(bin_prob)
        return bin_prob

    # Get the probability estimate for each bin.  First, normalize the
    # eigenvector with corresponding to the maximum eigenvalue for K_mod. 
    sub_bin_prob = eigvecs[:, maxi] / np.sum(eigvecs[:, maxi])

    # Now, plug these values into the bins to which they correspond.  Here,
    # we account for bins that we removed in finding the strongest connected
    # subgraph described by K. 
    bin_prob = np.zeros(K.shape[0])
    bin_prob[components] = sub_bin_prob

    return bin_prob

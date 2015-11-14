#   i!/usr/bin/env python
# Import statements, unedited from w_postanalysis_reweight.py
# Go back and delete those not needed, eventually
from __future__ import print_function, division; __metaclass__ = type
import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
import h5py

from collections import Counter

#import westpa
#from west.data_manager import weight_dtype, n_iter_dtype
#from westtools import (WESTTool, WESTDataReader, IterRangeSelection,
#                       ProgressIndicatorComponent)
#from westpa import h5io
#from westtools.dtypes import iter_block_ci_dtype as ci_dtype

log = logging.getLogger('westtools.w_postanalysis_reweight')

'''
--------------------------------------------------------------------------------
w_steadystate_reweight.py

Work in progress, by Alex DeGrave

Code is largely taken from w_postanalysis_reweight.py, part of the WESTPA 
toolkit.
--------------------------------------------------------------------------------
Combine data from multiple WESTPA simulations to find an average transition
matrix, as well as bin populations and state-to-state rates based on analysis
of this matrix. Additionally, this tool can accommodate steady-state weighted
ensemble simulations (with recycling conditions). 

Using information on bin-to-bin transitions output from running 
w_postanalysis_matrix.py on a set of simulations, this tool builds a matrix, 
where each element ``i,j`` of the matrix represents the probability that a 
walker in a given bin ``i`` will transition to another bin ``j`` in a given 
lagtime, tau. For simulations involving a recycling condition, transitions 
out of bins denoted as recycling bins are ignored (such transitions may be
observed if the lag time for building the matrix is shorter than the time 
between split/merge/recycle events).

The user should specify input files for this tool in a YAML-format (``.yaml``)
file.  For each simulation included in the analysis, the user should provide
athe name of the output file from w_postanalyis_matrix, as well as the indices
of bins subject to recycling during the simulaiton, if any (ie, the bins where,
if a simulation is in that bin at the end of an iteration, it will be recycled). 
--------------------------------------------------------------------------------
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
    # probabilty vector by the matrix to obtain a new probabilty distribution.
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

def get_transmat_and_obsmat_sum(fluxmatH5_list, start_iter, stop_iter, nbins, 
                                recycling_bin_list=None):
    '''
    Find the sum of transition matrices from a set of simulations, for a given 
    set of timepoints. 

    Return: 
    (1) a numpy array representing the sum of the transition matrices for each 
    simulation, for each timepoint for which the transition probability is 
    well-defined (ie, if the bin in which the transition originates is not 
    subject to recycling for the given simulation, and if this bin is not 
    unoccupied. 
    Elements for which no observations were made are set
    to ``NaN`` (not a number). 
    (2) a numpy array where each element represents the count of all timepoints
    for all simulations in which the transition probably represented in (1) was
    well-defined.

    fluxmatH5_list: a list of flux matrix h5 files
    start_iter: integer; first weighted ensemble iteration to include in the
            average.
    stop_iter: integer; iteration at which to stop averaging.  Averages will 
            include UP TO stop_iter, BUT NOT stop_iter itself. 
    nbins: integer; the number of bins used in building the flux matrices. This
            should match the dimensions of the flux matrices in each HDF5 file 
            of fluxmatH5_list
    recycling_bin_list: A list of numpy arrays.  The order of the arrays should
            be the same as fluxmatH5_list. Each array should specify the indices
            of bins that were subject to recycling during the corresponding 
            weighted ensemble simulation. If all simulations were not subject
            to recycling (default), this option may be simply set to ``None``.
            If only select simulations were not subject to recycling, ``None`` 
            may be instead specified at the corresponding index of 
            recycling_bin_list. 
    '''
    
    # Array for holding the running sum of transition matrix
    transmat_sum = np.empty((nbins, nbins), np.float64)
    transmat_sum.fill(np.nan)
    temp_array = np.zeros((nbins, nbins), np.float64)
    # Array for holding the count of elements added to the transmat_sum 
    obsmat_sum = np.zeros((nbins, nbins), np.int64)
    
    # Load all the populations into physical memory ahead of time.  This should
    # be small enough to store without running into memory issues. Only read in
    # the specified iteration range. Note that the iteration indexing of these 
    # vectors is offset from the h5file itself!
    pops_list = []
    for fluxmatH5 in fluxmatH5_list:
        pops_list.append(
                np.array(fluxmatH5['bin_populations'][start_iter:stop_iter, :])
                         ) 

    np.seterr(divide='ignore')
    # Iterate over all the iterations and all the simulations
    for iiter in xrange(start_iter, stop_iter):
        for isim, fluxmatH5 in enumerate(fluxmatH5_list):
            iter_group = fluxmatH5['iterations/iter_{:08d}'.format(iiter)]
            # Reset the temporary matrix to all zeros.
            temp_array[...] = 0 
            # Load flux matrix from the HDF5 file. The matrix is stored based
            # on the ``coo_matrix`` format from the sparse module of the SciPy 
            # library. The row index, column index, and associated data is 
            # stored in vectors, for nonzero elements of the array.
            row_idxs = iter_group['rows']
            col_idxs = iter_group['cols']
            flux_data = iter_group['flux']
            # Reconstruct the (dense) matrix.
            temp_array[row_idxs, col_idxs] = flux_data
            if iiter == 35:
                print(temp_array[0])
            # Divide each row of the matrix by the population in corresponding
            # bin.  This gives ``NaN`` for values where the population of the 
            # corresponding bin is zero. 
            # CHECK THAT THIS IS DOING WHAT IT IS SUPPOSED TO!!!
           
            temp_array = np.divide(temp_array, 
                                   pops_list[isim][iiter-start_iter][:,np.newaxis]
                                   )
            # Set rows corresponding to bins involved in recycling to NaN, as 
            # well. 
            # CHECK THAT THIS IS DOING WHAT IT IS SUPPOSED TO!!!
            if recycling_bin_list is not None:
                if recycling_bin_list[isim] is not None:
                    temp_array[recycling_bin_list[isim]] = np.nan 
            # Make a mask that lets through the values that are not 
            # ``Not a Number``
            good_value_mask = np.isfinite(temp_array) 
            # Add the good values to the running sum
            # We could also experiment with weighting elements by the number of
            # observations here!
            # CHECK THIS!!!
            if np.any(np.isinf(temp_array[0])):
                print('\n %d'%iiter)
                print(temp_array[0])
                print(pops_list[isim][iiter-start_iter])
            transmat_sum = np.nansum(np.dstack((transmat_sum, temp_array)),
                                     axis=2)
            obsmat_sum += good_value_mask
    return transmat_sum, obsmat_sum

def get_average_transition_mat(fluxmatH5_list, start_iter, stop_iter, nbins, 
                               recycling_bin_list=None): 
    '''
    Find an average transition matrix from a set of simulations, for a given 
    set of timepoints. Return a numpy matrix representing the average 
    transition matrix, with elements for which no observations were made set
    to ``NaN`` (not a number). 

    fluxmatH5_list: a list of flux matrix h5 files
    start_iter: integer; first weighted ensemble iteration to include in the
            average.
    stop_iter: integer; iteration at which to stop averaging.  Averages will 
            include UP TO stop_iter, BUT NOT stop_iter itself. 
    nbins: integer; the number of bins used in building the flux matrices. This
            should match the dimensions of the flux matrices in each HDF5 file 
            of fluxmatH5_list
    recycling_bin_list: A list of numpy arrays.  The order of the arrays should
            be the same as fluxmatH5_list. Each array should specify the indices
            of bins that were subject to recycling during the corresponding 
            weighted ensemble simulation. If all simulations were not subject
            to recycling (default), this option may be simply set to ``None``.
            If only select simulations were not subject to recycling, ``None`` 
            may be instead specified at the corresponding index of 
            recycling_bin_list. 
    '''
    transmat_sum, obsmat_sum = get_transmat_and_obsmat_sum(
                                        fluxmatH5_list, start_iter, stop_iter, 
                                        nbins, 
                                        recycling_bin_list=recycling_bin_list
                                                           )
    
    # Finally, get the average transition matrix and return it.
    transmat = numpy.divide(transmat_sum, obsmat_sum)
    return transmat
            
def transmat_cumulative_mean_generator(fluxmatH5_list, start_iter, stop_iter, 
                                       step_iter, nbins, 
                                       recycling_bin_list=None):
    '''
    Generator for fast calculations of cumulatively averaged transition matrix 
    estimates.

    Find an average transition matrix from a set of simulations, for a given 
    set of timepoints. Return a numpy matrix representing the average 
    transition matrix, with elements for which no observations were made set
    to ``NaN`` (not a number). 

    fluxmatH5_list: a list of flux matrix h5 files
    start_iter: integer; first weighted ensemble iteration to include in the
            average.
    stop_iter: integer; iteration at which to stop averaging.  Averages will 
            include UP TO stop_iter, BUT NOT stop_iter itself. 
    step_iter: for each cumulative mean, include another step_iter iterations
    nbins: integer; the number of bins used in building the flux matrices. This
            should match the dimensions of the flux matrices in each HDF5 file 
            of fluxmatH5_list
    recycling_bin_list: A list of numpy arrays.  The order of the arrays should
            be the same as fluxmatH5_list. Each array should specify the indices
            of bins that were subject to recycling during the corresponding 
            weighted ensemble simulation. If all simulations were not subject
            to recycling (default), this option may be simply set to ``None``.
            If only select simulations were not subject to recycling, ``None`` 
            may be instead specified at the corresponding index of 
            recycling_bin_list. 
    '''
    # Build first cumulative_sum, from start_iter to start_iter + step_iter 
    # CHANGE: raise a more descriptive/accurate error here.
    if start_iter + step_iter >= stop_iter:
        raise ValueError
    transmat_sum, obsmat_sum = get_transmat_and_obsmat_sum(
                                        fluxmatH5_list, start_iter, 
                                        start_iter+step_iter, nbins, 
                                        recycling_bin_list=recycling_bin_list
                                                           )
    transmat = numpy.divide(transmat_sum, obsmat_sum)
    # Yield the mean transition matrix from the first averaging window.
    yield transmat
    
    # From here on out, we can iterate.
    for cumsum_stop_iter in xrange(start_iter+step_iter, stop_iter, step_iter):
        # Get the sums between cumsum_stop_iter-step_iter and cumsum_stop_iter 
        temp_transmat_sum, temp_obsmat_sum = get_transmat_and_obsmat_sum(
                                        fluxmatH5_list, 
                                        cumsum_stop_iter-step_iter, 
                                        cumsum_stop_iter, nbins, 
                                        recycling_bin_list=recycling_bin_list
                                                                         )
        transmat_sum = np.nansum(transmat_sum, temp_transmat_sum)
        obsmat_sum += temp_obsmat_sum

        transmat = numpy.divide(transmat_sum, obsmat_sum)
        # Yield the mean transition matrix from the  averaging window.
        yield transmat
        

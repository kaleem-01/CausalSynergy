__author__ = 'rquax'

'''
Owner of this project:

    Rick Quax
    https://staff.fnwi.uva.nl/r.quax
    University of Amsterdam

You are free to use this package only for your own non-profit, academic work. All I ask is to be suitably credited.
'''

try:
    from collections import Sequence, Iterable, Callable, Hashable
except ImportError:
    from collections.abc import Sequence, Iterable, Callable, Hashable  # from Python 3.10 this changed
import numpy as np
import itertools

from scipy.optimize import minimize

import warnings

# TODO: move all functions here, iron out wrinkles
# TODO: let the notebook use this file instead of its own defined functinos
# TODO: let jointpmf use this too

# note: I define these here even though they are also defined in jointpmf, just to see if I can prevent
# a circular import by having to import jointpmf here.
_prob_error_tol = 1e-7  # absolute tolerance for errors in (sums of) probabilities
_mi_error_tol = 1e-6  # absolute tolerance for errors in (sums of) mutual information or entropy quantities

### HELPER functions for the input variables

def joint_probability_multiple_inputs(xixs: Sequence[int], xvals: Sequence[int], ps: Sequence[np.ndarray]) -> float:
    """Compute the product of marginal input value probabilities (thus assuming independence of inputs).

    Args:
        xixs (Sequence[int]): list of variable indices
        xvals (Sequence[int]): _description_
        ps (np.ndarray, optional): _description_. Defaults to ps.

    Returns:
        float: _description_
    """
    assert len(xixs) == len(xvals)
    assert len(set(xixs)) == len(xixs), 'should not doubly multiply with the same variable\'s probability.'

    # return ps[tuple(zip(xixs, xvals))].prod()
    # return ps[xixs, xvals].prod()
    return np.prod([ps[xix][xv] for xix, xv in zip(xixs, xvals)])


def joint_probability_all_inputs(xvals: Sequence[int], ps: Sequence[np.ndarray]) -> float:
    """Compute the product of all marginal input value probabilities (thus assuming independence of inputs).

    Args:
        xvals (Sequence[int]): a value for each input variable (zero-indexed)
        ps (np.ndarray, optional): _description_. Defaults to ps.

    Returns:
        float: joint probability of the given joint state
    """
    # assert len(xvals) == nx, f'{xvals=}, {nx=}'
    # assert len(srv_ps.shape) == nx + ns
    # assert len(ps) == nx
    if __debug__:
        local_kxs = tuple(map(len, ps))
        assert np.all(np.less(xvals, local_kxs)), 'each value in `xvals` should be in the range 0,..,kx-1 where `kx` is the number of possible values for that input variable.'
        assert len(xvals) == len(ps), f'{len(xvals)=} is not equal to {len(ps)=}'

    # return ps[tuple(zip(range(nx), xvals))].prod()
    # return ps[range(nx), xvals].prod()
    return np.prod([ps[xix][xv] for xix, xv in enumerate(xvals)])

### HELPER functions for the SRV variable

def marginalize_single_input(xix: int, srv_ps, ps) -> np.ndarray:
    """Integrate out one input variable from the conditional SRV PMF. (Not in-place.)

    Args:
        xix (int): index of the input variable. Should be less then `len(np.shape(srv_ps))-1`.

    Returns:
        np.ndarray: conditional SRV PMF with one dimension less (one axis integrated out)
    """
    assert xix < len(np.shape(srv_ps))-1, 'at index len(np.shape(srv_ps))-1 is the SRV variables\' probabilities'

    local_kxs = tuple(map(len, ps))

    # note: there may be a more efficient, numpy-native way of doing this, but I could not find it quickly
    return sum([ps[xix][xval] * condition_single_input(xix, xval, srv_ps=srv_ps)
                for xval in range(local_kxs[xix])])


def condition_single_input(xix: int, xval: int, srv_ps) -> np.ndarray:
    """Take out one input variable from the conditional SRV PMF by selecting a value for it.

    Args:
        xix (int): index of the input variable. Should be less then `len(np.shape(srv_ps))-1`.
        xval (int): value for the input variable.
        srv_ps (np.ndarray, optional): conditional PMF of the SRV encoded as numpy array. Defaults to srv_ps.

    Returns:
        np.ndarray: conditional SRV PMF with one dimension less, basically P(S|X[xix]=xval)
    """
    assert xix < len(np.shape(srv_ps))-1, 'at index len(np.shape(srv_ps))-1 is the SRV variables\' probabilities'

    return srv_ps.take(xval, axis=xix)


def condition_multiple_inputs(xixs: Sequence[int], xvals: Sequence[int], srv_ps: np.ndarray) -> np.ndarray:
    """Take out multiple input variables from the conditional SRV PMF by selecting values for them.

    Essentially this function makes a series of calls to `specify_single_input`.

    Args:
        xixs (Sequence[int]): _description_
        xvals (Sequence[int]): _description_
        srv_ps (np.ndarray, optional): conditional PMF of the SRV encoded as numpy array. Defaults to srv_ps.

    Returns:
        np.ndarray: conditional SRV PMF with fewer dimensions, basically P(S|X[xixs]=xvals)
    """
    assert len(xixs) == len(xvals)
    assert len(set(xixs)) == len(xixs)

    # sort xixs in descending order while letting the corresponding xvals values move along accordingly
    xix_xval_pairs = list(zip(xixs, xvals))
    xix_xval_pairs = sorted(xix_xval_pairs, key=lambda pair: pair[0], reverse=True)

    srv_ps_res = srv_ps

    for xix, xval in xix_xval_pairs:
        srv_ps_res = condition_single_input(xix, xval, srv_ps_res)

    # commented out because it passed for a while, seems not needed:
    # if __debug__:
    #     if len(xixs) > 0:
    #         assert len(srv_ps_res.shape) < len(srv_ps.shape)
    #         assert len(srv_ps_res.shape) == len(srv_ps.shape) - len(xixs)
    
    return srv_ps_res


def condition_all_inputs(xvals: Sequence[int], srv_ps: np.ndarray) -> np.ndarray:
    """Take out all input variables from the conditional SRV PMF by selecting values for them.

    Essentially this function makes a series of `nx` calls to `specify_single_input`. This is 
    a slightly more efficient implementation than using `condition_multiple_inputs` (but barely I suspect).

    Args:
        xvals (Sequence[int]): one value per input variable, assumed sorted (starting with variable 0)
        srv_ps (np.ndarray, optional): conditional PMF of the SRV encoded as numpy array. Defaults to srv_ps.

    Returns:
        np.ndarray: conditional SRV PMF, P(S|X[xixs]=xvals) where `xixs` is `range(nx)`.
    """
    # assert len(xvals) == nx
    # assert len(srv_ps.shape) == nx + ns
    assert len(xvals) == len(np.shape(srv_ps)) - 1

    # sort xixs (0, 1, ...) in descending order while letting the corresponding xvals values move along accordingly
    xix_xval_pairs = list(zip(range(len(xvals)), xvals))
    xix_xval_pairs = reversed(xix_xval_pairs)

    srv_ps_res = srv_ps

    for xix, xval in xix_xval_pairs:
        srv_ps_res = condition_single_input(xix, xval, srv_ps_res)

    # commented out because it passed for a while, seems not needed:
    # if __debug__:
    #     if len(xixs) > 0:
    #         assert len(srv_ps_res.shape) < len(srv_ps.shape)
    #         assert len(srv_ps_res.shape) == len(srv_ps.shape) - len(xixs)
    
    return srv_ps_res


def marginal_srv_ps(srv_ps, ps) -> np.ndarray:
    assert len(ps) == len(np.shape(srv_ps)) - 1, f'Did you accidentally switch around `srv_ps` and `ps`? {len(ps)=}, {np.shape(srv_ps)=}'
    
    return sum([joint_probability_all_inputs(xvals, ps) * condition_all_inputs(xvals, srv_ps)
                for xvals in itertools.product(*map(range, map(len, ps)))])


### ENTROPY AND MUTUAL INFORMATION FUNCTIONS


def H(p_arr: Sequence[float]) -> float:
    """Generic entropy function"""
    if __debug__:
        np.testing.assert_almost_equal(np.sum(p_arr), 1.0, 5, "`p_arr` should sum to unity since it is a prob. distribution")

    with np.errstate(divide='ignore'):
        logps = np.log2(np.array(p_arr, dtype=float))
    return -sum(p_arr * np.where(np.isneginf(logps), 0.0, logps))


def entropy_srv(srv_ps: np.ndarray, ps: np.ndarray, _precomp_marginal_probs=None) -> float:
    assert len(srv_ps) > 0, f'looking for bug: {srv_ps=}'

    if _precomp_marginal_probs is None:
        _precomp_marginal_probs = marginal_srv_ps(ps=ps, srv_ps=srv_ps)
    else:
        pass
        # if ns == 1:
            # assert len(_precomp_marginal_probs) == ks
        # else:  # untested
        #     assert np.all(np.equal(np.shape(_precomp_marginal_probs), ks))
        
    return H(_precomp_marginal_probs)


def entropy_input(xix: int, ps) -> float:
    return H(ps[xix])


def entropy_inputs(ps) -> float:
    return np.apply_along_axis(H, -1, ps)


def conditional_entropy_srv_all_inputs(srv_ps: np.ndarray, ps: np.ndarray) -> float:
    """Average conditional entropy of the SRV given all its inputs."""
    kxs = tuple(len(p) for p in ps)  # shorthand
    return sum([joint_probability_all_inputs(xvals, ps=ps) * H(condition_all_inputs(xvals, srv_ps=srv_ps))
                for xvals in itertools.product(*map(range, kxs))])


def mutual_information_srv_all_inputs(srv_ps: np.ndarray, ps: np.ndarray, _precomp_marginal_srv_probs=None) -> float:
    """Mutual information between the SRV and all the inputs, i.e., I(S:X) where X=[X_1,...,X_nx]."""
    totH = entropy_srv(srv_ps=srv_ps, ps=ps, _precomp_marginal_probs=_precomp_marginal_srv_probs)
    condH = conditional_entropy_srv_all_inputs(srv_ps=srv_ps, ps=ps)

    assert totH >= -1e7, 'entropy should be non-negative'
    assert condH >= -1e7, 'average conditional entropy should be non-negative'
    assert totH >= condH, 'H(X) should be larger or equal to H(X|Y) for any Y'

    return totH - condH


# TODO: in all function argument lists, let `srv_ps` always occur before `ps`, for uniformity. But
# right now I cannot find the refactor button for that....
def conditional_srv_given_single_input(xix: int, xval: int, ps: Sequence, srv_ps: np.ndarray) -> np.ndarray:
    """Calculate p(S|X[xix]=xval).

    Args:
        xix (int): index of input variable to be conditioning for
        xval (int): value of the input variable
        ps (Sequence): a list of input distributions (PMFs)
        srv_ps (np.ndarray): a numpy array specifying a PMF for the SRV, indexed by input variable values 
         like `srv_ps[xval1]...[xvaln]` where `n==len(ps)`.

    Returns:
        np.ndarray: a PMF for the SRV, no longer conditioning on any input variables (variable `xix` is conditioned
         on and the rest is integrated out).
    """
    return marginal_srv_ps(condition_single_input(xix, xval, srv_ps), 
                           [ps[i] for i in range(len(ps)) if not i == xix])


def conditional_entropy_srv_given_single_input_value(xix: int, xval: int, ps: Sequence, srv_ps: np.ndarray) -> np.ndarray:
    """Calculate H(S|X[xix]=xval).

    Args:
        xix (int): index of input variable to be conditioning for
        xval (int): value of the input variable
        ps (Sequence): a list of input distributions (PMFs)
        srv_ps (np.ndarray): a numpy array specifying a PMF for the SRV, indexed by input variable values 
         like `srv_ps[xval1]...[xvaln]` where `n==len(ps)`.

    Returns:
        np.ndarray: a PMF for the SRV, no longer conditioning on any input variables (variable `xix` is conditioned
         on and the rest is integrated out).
    """
    p_S_given_xix_equals_xval = conditional_srv_given_single_input(xix, xval, ps, srv_ps)

    return H(p_S_given_xix_equals_xval)


def mutual_information_srv_given_single_input(xix: int, ps: Sequence, srv_ps: np.ndarray) -> np.ndarray:
    H_S = entropy_srv(srv_ps=srv_ps, ps=ps)
    H_S_given_Xi = sum([ps[xix][xval] * conditional_entropy_srv_given_single_input_value(xix, xval, ps=ps, srv_ps=srv_ps) 
                        for xval in range(len(ps[xix]))])
    
    return H_S - H_S_given_Xi


### FINDING A DIRECT SRV


# helper function
def unit_vector(i: int | Sequence[int], n: int, dtype=int):
    """Construct a 1D vector with a single 1-value in one place and 0-values in all other places."""
    vec = np.zeros(n, dtype=dtype)
    if isinstance(i, int):
        vec[i] = 1
    else:
        vec[tuple(i)] = 1
    return vec


def compute_direct_srv_assuming_independence(ps: Sequence[Sequence[float]], numvalues=None, method='joint',
                                             first_xix: int = None, p_S: Sequence[float] = None, 
                                             on_violation='warning', return_also_violation=False, verbose=0) -> np.ndarray:
    kxs = tuple(len(p) for p in ps)
    
    # all marginal input PMFs should sum to 1, roughly
    np.testing.assert_allclose(np.sum(ps, axis=1), np.ones(len(kxs)), atol=_prob_error_tol)
    assert first_xix is None or method == 'marginal', f'makes no sense to set first_xix if not doing the marginal method'

    ks = numvalues if not numvalues is None else max(kxs)  # choose a number of states for the SRV variable
    srv_ps_shape = srv_ps_shape = kxs + (ks,)

    srv_ps_direct = np.zeros(srv_ps_shape, dtype=float)  # initialize all zero probabilities

    if method == 'joint':
        tuples_xvals_jointprob = [(xvals, joint_probability_all_inputs(xvals, ps)) for xvals in itertools.product(*map(range, kxs))]
        np.testing.assert_almost_equal(sum([t[-1] for t in tuples_xvals_jointprob]), 1.0)
        tuples_xvals_jointprob = sorted(tuples_xvals_jointprob, key=lambda tuple: tuple[-1])  # sort from low to high joint probability of inputs
    elif method == 'marginal':
        # here I find the index of the input variable that has the smallest probability for any of its values. The goal of this is
        # to be able to iterate over the row/column that is associated to this minimum probability (in the contingency table) so that
        # it is guaranteed that the procedure below always results in a proper SRV.
        if first_xix is None:
            min_prob_input_ix, min_prob = min(zip(range(len(kxs)), list(map(min, ps))), key=lambda pair: pair[-1])
        else:
            min_prob_input_ix, min_prob = first_xix, min(ps[first_xix])

            first_xix = 0  # this variable will be the first input variable after reordering below

        # TODO: this if-statement should move up for efficiency, now some computation above gets lost
        # HACK: there is some bug somewhere I think in the ordering this way, not sure actually, but
        # sometimes reordering the input variables seems to do the trick. So let's do that.
        if min_prob_input_ix > 0:
            # NOTE: right now the rest of the input variables do not get reordered in any way, and I am not sure
            # if they should or not (maybe sort the indices by their minimum prob. or so?)
            new_ps_ixs = [min_prob_input_ix] + [xix for xix in range(len(ps)) if not xix == min_prob_input_ix]
            ps_reordered = np.take(ps, new_ps_ixs, axis=0)

            if verbose >= 2:
                print(f'hack: the input variable with minimum probability for one of its states is variable {min_prob_input_ix}, which is not 0, so I reorder the input probabilities because there is a bug somewhere.')
            
            srv_ps_direct = compute_direct_srv_assuming_independence(ps_reordered, numvalues=numvalues, method=method, first_xix=first_xix, p_S=p_S, on_violation=on_violation, return_also_violation=return_also_violation, verbose=verbose)

            if return_also_violation:
                srv_ps_direct, total_violation = srv_ps_direct  # a tuple will have been returned

            # reorder the axes to match the original `ps` (now it matches `ps_reordered`)
            srv_ps_direct = srv_ps_direct.transpose(*(new_ps_ixs + [len(ps)]))

            if return_also_violation:
                return srv_ps_direct, total_violation
            else:
                return srv_ps_direct

        # helper function
        def input_variable_sorted_values(kix: int):
            """Return all possible values of the input variable at index `kix` in ascending order of marginal probability."""
            return map(lambda tuple: tuple[0], sorted(zip(range(kxs[kix]), ps[kix]), key=lambda pair: pair[-1]))

        tuples_xvals_jointprob = []
        for xval_min_loop_ix, xval_min_prob_input_var in enumerate(input_variable_sorted_values(min_prob_input_ix)):
            # note: this complicated itertools expression is basically equal to a nested for-loop, like
            # [[... for j in kxs[1]] for i in kxs[0]], but then generalized to any length of `kxs`, so any number of inputs.
            for xvals_rest in itertools.product(*[input_variable_sorted_values(kix) for kix, k in enumerate(kxs) if not kix == min_prob_input_ix]):
                xvals = xvals_rest[:min_prob_input_ix] + (xval_min_prob_input_var,) + xvals_rest[min_prob_input_ix:]

                tuples_xvals_jointprob.append((xvals, joint_probability_all_inputs(xvals, ps)))
    else:
        raise NotImplementedError(f'unknown {method=}')


    # for xval_min_loop_ix, xval_min_prob_input_var in enumerate(input_variable_sorted_values(min_prob_input_ix)):
    #     # note: this complicated itertools expression is basically equal to a nested for-loop, like
    #     # [[... for j in kxs[1]] for i in kxs[0]], but then generalized to any length of `kxs`, so any number of inputs.
    #     for xvals_rest in itertools.product(*[input_variable_sorted_values(kix) for kix, k in enumerate(kxs) if not kix == min_prob_input_ix]):
    #         xvals = xvals_rest[:min_prob_input_ix] + (xval_min_prob_input_var,) + xvals_rest[min_prob_input_ix:]

    # as long as p_S is not yet known, set it to all 1s, indicating essentially no constraint (upper bound) in the loop
    if p_S is None:
        p_S = np.ones(ks)  # * np.max(ps)
        is_p_S_set = False  # should become true as soon as one of the rows/columns gets fully populated with probabilities
    else:
        np.testing.assert_almost_equal(np.sum(p_S), 1.0)
        pass  # use whatever the user provideds
        is_p_S_set = True  # should become true as soon as one of the rows/columns gets fully populated with probabilities

    # this will contain the sum of probability mass that could not be assigned in a way that makes the
    # SRV 'perfect' in the below procedure. Ideally this is zero, but sometimes it is not. Might be 
    # convenient for the caller to know about this (see also the `return_also_violation` argument).
    total_violation = 0.0

    # here I make sure I iterate over all input states (`xvals`) in such a way that the entire row/column associated to the 
    # minimum probability is populated first. In general, the values of the input variables are here
    # sorted in ascending order of probability, so that low-probability states are populated first (this seems to minimize the chance of errors).
    for xvals_ix, (xvals, jointprob) in enumerate(tuples_xvals_jointprob):
        if verbose >= 3:
            print(f'DEBUG: before starting loop ({xvals=}, p(xvals)={jointprob}):\nsrv_ps_direct=\n{srv_ps_direct}')

        # see explanation at `max_possible_cond_prob_per_srv_val` below; this stores minimum of all maximum possible probs
        max_possible_srv_probs_overall = np.inf * np.ones(np.shape(srv_ps_direct[xvals]))

        for xix, xval in enumerate(xvals):
            np.testing.assert_allclose(srv_ps_direct[xvals], np.zeros(np.shape(srv_ps_direct[xvals])), 
                                    err_msg=f'I did not set these SRV probabilities yet, so they should be still zero. {srv_ps_direct[xvals]=}')
            cond_srv = conditional_srv_given_single_input(xix, xval, ps, srv_ps_direct)
            if total_violation == 0.0:
                if on_violation == 'raise':
                    np.testing.assert_array_less(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)), f'no p(S|X[i]=j) should be able to exceed p(S). {xix=}, {xval=},\n----\nsrv_ps_direct:\n----\n{srv_ps_direct}\n')
                elif on_violation == 'warning':
                    if not np.all(np.less(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)))):
                        warnings.warn(f'no p(S|X[i]=j) should be able to exceed p(S)={p_S}, but for {xix=}, {xval=} I get p(S|xix=xval)={cond_srv}\n')
                elif on_violation == 'continue':
                    pass

            # set to an invalid (1, ..., 1) conditional probability p(S|\vec{X}=xvals), just to be able to compute what are the maximum
            # p(S|X[xix]=xval) we could achieve
            srv_ps_direct[xvals] = np.ones(np.shape(srv_ps_direct[xvals]))
            # NOTE: `cond_srv_max` may well be larger than p_S in some places, which is of course not allowed,
            # but we just use this to compute which conditional probability for which value we can set for the SRV.
            cond_srv_max = conditional_srv_given_single_input(xix, xval, ps, srv_ps_direct)
            srv_ps_direct[xvals] = np.zeros(np.shape(srv_ps_direct[xvals]))  # revert

            # this is not meant as a PMF but does have the same shape as `srv_ps_direct[xvals]`; it stores
            # the maximum possible conditional probability p(S=i|X[xix]=xval), for each i
            max_possible_cond_prob_per_srv_val = (p_S - cond_srv) / (cond_srv_max - cond_srv)

            if total_violation > 0.0:
                # when there are violations then elements of `cond_srv` could be larger than `p_S`,
                # but that does not mean that we should consider negative probability masses. So correct it.
                max_possible_cond_prob_per_srv_val = np.max([max_possible_cond_prob_per_srv_val, np.zeros(np.shape(max_possible_cond_prob_per_srv_val))], axis=0)
            elif not (method == 'marginal' and not first_xix is None):  # this should only potentially happen when first_xix is explicitly set and the marginal method is used
                # TODO: not sure if this is actually a bug or just a necessity (that sometimes even the 
                # 'joint' algorithm hits this), but for now I made it into a warning
                if on_violation == 'raise':
                    assert np.min(max_possible_cond_prob_per_srv_val) >= -_prob_error_tol, f'probabilities should not be negative: {max_possible_cond_prob_per_srv_val=}'
                elif on_violation == 'warning':
                    if not np.min(max_possible_cond_prob_per_srv_val) >= -_prob_error_tol:
                        warnings.warn(f'probabilities should not be negative: {max_possible_cond_prob_per_srv_val=}')
                elif on_violation == 'continue':
                    pass
                else:
                    raise NotImplementedError(f'unknown {on_violation=}')

            # now we have the maximum possible prob per SRV value, when considering this particular 'direction': for input
            # variable `xix` and its value `xval`. One could see this as a row or column in the (2 or more)-dimensional table
            # of SRV PMFs where each dimension is one of the input variable's possible values.
            max_possible_srv_probs_overall = np.min([max_possible_srv_probs_overall, max_possible_cond_prob_per_srv_val], axis=0)

        # create a list of tuples like [(SRV_states1, maxprob1), (SRV_states2, maxprob2), ....], sorted in descending order on
        # the maxprobs.
        max_possible_prob_tuples = zip(itertools.product(*map(lambda x: range(x), np.shape(max_possible_srv_probs_overall))), max_possible_srv_probs_overall)
        max_possible_prob_tuples = sorted(max_possible_prob_tuples, key=lambda tup: tup[-1], reverse=True)

        assigned_total_prob_mass = 0.0  # amount of probability mass already placed in `srv_ps_direct`
        for srv_vals, maxprob in max_possible_prob_tuples:
            if maxprob > 1.:
                maxprob = 1.
            elif maxprob < 0.:
                maxprob = 0.

            assert 0.0 <= maxprob <= 1.0, f'maxprob should be a probability, but: {maxprob=}'
            assert assigned_total_prob_mass <= 1.0, 'todo: can be removed if it passes a few times'

            # prob_to_assign = min(1.0 - assigned_total_prob_mass, maxprob)
            if maxprob < 1.0 - assigned_total_prob_mass:
                srv_ps_direct[xvals][srv_vals] = maxprob
                assigned_total_prob_mass += maxprob

                if __debug__:
                    cond_srv = conditional_srv_given_single_input(xix, xval, ps, srv_ps_direct)
                    if not total_violation > 0.0 and not (not first_xix is None and method == 'marginal'):
                        # note: when first_xix is set then it can well be that the marginalization
                        # in different axes (of `srv_ps_direct`) cannot work well, so only check
                        # this if not set.
                        if on_violation == 'raise':
                            np.testing.assert_array_less(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)), f'no p(S|X[i]=j) should be able to exceed p(S). {xix=}, {xval=},\n----\nsrv_ps_direct:\n----\n{srv_ps_direct}\n')
                        elif on_violation == 'warning':
                            if not np.all(np.less_equal(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)))):
                                warnings.warn(f'no p(S|X[i]=j) should be able to exceed p(S). {xix=}, {xval=},\n----\nsrv_ps_direct:\n----\n{srv_ps_direct}\n')
                        elif on_violation == 'continue':
                            pass
            else:
                srv_ps_direct[xvals][srv_vals] = 1.0 - assigned_total_prob_mass
                assigned_total_prob_mass = 1.0

                if __debug__:
                    cond_srv = conditional_srv_given_single_input(xix, xval, ps, srv_ps_direct)
                    if not total_violation > 0.0 and not (method == 'marginal' and not first_xix is None):
                        if on_violation == 'raise':
                            np.testing.assert_array_less(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)), f'no p(S|X[i]=j) should be able to exceed p(S). {xix=}, {xval=},\n----\nsrv_ps_direct:\n----\n{srv_ps_direct}\n')
                        elif on_violation == 'warning':
                            if not np.all(np.less_equal(cond_srv, p_S + 1e-5*np.ones(np.shape(p_S)))):
                                warnings.warn(f'no p(S|X[i]=j) should be able to exceed p(S). {xix=}, {xval=},\n----\nsrv_ps_direct:\n----\n{srv_ps_direct}\n')
                        elif on_violation == 'continue':
                            pass
            
                    np.testing.assert_almost_equal(assigned_total_prob_mass, 1.0, 
                                    err_msg=f'should have assigned a total probability mass of 1.0, but: {assigned_total_prob_mass=}')
                    np.testing.assert_almost_equal(np.sum(srv_ps_direct[xvals]), 1.0, 
                                    err_msg=f'should have assigned a total probability mass of 1.0, but: {assigned_total_prob_mass=}')

                break  # we are done: assigned a total probability mass of 1.0
        
        if not abs(assigned_total_prob_mass - 1.0) < _prob_error_tol:
            if on_violation == 'raise':
                raise UserWarning(f'Could not assign all probability for {xvals=}. {assigned_total_prob_mass=}.\n{srv_ps_direct=}\n{list(max_possible_prob_tuples)=}')
            elif on_violation in ('warning', 'continue'):
                total_violation += 1.0 - assigned_total_prob_mass  # keep track of how much violation there has been

                if on_violation == 'warning':
                    warnings.warn(f'Could not assign all probability for {xvals=}. {assigned_total_prob_mass=}.\n{srv_ps_direct=}\n{list(max_possible_prob_tuples)=}')
                
                # reuse the iterator over the SRV values but now randomly permuted; then try to
                # 'get rid' of the excess probability mass
                # for srv_vals, _ in np.random.permutation(list(max_possible_prob_tuples)):
                for srv_vals, _ in max_possible_prob_tuples:
                    if 1.0 - srv_ps_direct[xvals][srv_vals] >= 1.0 - assigned_total_prob_mass:
                        prob_mass_to_stash = 1.0 - assigned_total_prob_mass
                        assigned_total_prob_mass = 0.0
                    else:
                        prob_mass_to_stash = 1.0 - srv_ps_direct[xvals][srv_vals]
                        assigned_total_prob_mass -= prob_mass_to_stash
                    
                    srv_ps_direct[xvals][srv_vals] += prob_mass_to_stash

                    if assigned_total_prob_mass <= 0.0:
                        break
                
                np.testing.assert_almost_equal(assigned_total_prob_mass, 0.0)
            else:
                raise NotImplementedError(f'unknown {on_violation=}')

        if xvals_ix >= min(kxs) - 1 and not is_p_S_set:
            for xix, xval in enumerate(xvals):
                cond_srv = conditional_srv_given_single_input(xix, xval, ps, srv_ps_direct)

                if abs(sum(cond_srv) - 1.0) < _prob_error_tol:
                    p_S = cond_srv  # all other conditions should lead to the same distribution for SRV, so it becomes p(S)
                    is_p_S_set = True

                    if verbose >= 2:
                        print(f'debug: it is now determined: {p_S=}')

    if np.min(np.sum(srv_ps_direct, axis=2)) < 1.0 - _prob_error_tol:
        raise UserWarning(f'ERROR: not all conditional SRV probabilities sum to exactly 1: minimum probs. sum: {np.min(np.sum(srv_ps_direct, axis=2))}')
    else:
        if not return_also_violation:
            return srv_ps_direct
        else:
            return (srv_ps_direct, total_violation)


### NAIVE CONSTRAINT-BASED OPTIMIZATION


# helper function for constraint-based optimization (naive)
def constraint_srv(srv_ps: np.ndarray, ps: np.ndarray, logarithm=False) -> float:
    """Returns a value that grows with the extent that the single-input cond. SRV distributions are not equal.

    This function is meant to be used with the SLSQP optimization (or similar) algorithm of scipy.optimize. A return
    value of 0 means that the given conditional distribution `srv_ps` is actually an SRV, in the sense that it has
    zero information about any individual input. The higher the return value, the larger the violation.

    Args:
        srv_ps (np.ndarray, optional): an array with shape kx + ks, where the last axis encodes the conditional probabilities
         of the SRV given values for all its `len(kx)` inputs. So, encoding P(S|X). Defaults to srv_ps.
        ps (np.ndarray, optional): a twodimensional array encoding [P(X1), P(X2), ...]. Implies is that the joint input
         variable distribution factorizes, so P(X) = P(X1) * P(X2) * ... Defaults to ps.

    Returns:
        float: degree of violation of the equality constraint of P(S|Xi=j) being equal for all i,j.
    """
    kxs = tuple(len(p) for p in ps)  # shorthand
    p_S = marginal_srv_ps(srv_ps=srv_ps, ps=ps)

    p_S_given_X_minus_i = [marginal_srv_ps(condition_single_input(xix, xval, srv_ps), [p for pix, p in enumerate(ps) if not pix == xix])
                           for xix in range(len(ps))
                           for xval in range(kxs[xix])]
    
    violation = sum([np.linalg.norm(p_S - psrv) for psrv in p_S_given_X_minus_i])

    if logarithm:
        return np.log(violation)
    else:
        return violation


# helper function
def single_ps_to_unit_cube_coord(p_arr: np.ndarray) -> np.ndarray:
    """Convert a one-dimensional PMF (sequence of floats summing to 1) to a unit cube coordinate.

    Args:
        p_arr (np.ndarray): _description_

    Returns:
        np.ndarray: an array of `len(p_arr)-1` of floats, all between 0 and 1. 
    """
    ret = np.zeros(len(p_arr) - 1)  # pre-alloc

    warnings.filterwarnings("error", category=RuntimeWarning)  # finding a bug
    
    remaining_prob_mass = 1.0
    for ix, p in enumerate(p_arr[:-1]):
        if p > 0:  # else `remaining_prob_mass` can be also zero and then we get divide by zero
            try:
                ret[ix] = p / remaining_prob_mass
            except RuntimeWarning as e:
                print(f'{p=}, {remaining_prob_mass=}, {p_arr=}')
                assert False
            remaining_prob_mass -= p
    
    np.testing.assert_almost_equal(remaining_prob_mass, p_arr[-1], 5)

    return ret


# helper function
def single_unit_cube_coord_to_ps(coord: np.ndarray) -> np.ndarray:
    """Retrieving back a one-dimensional PMF (sequence of floats summing to 1) from a unit cube coordinate.

    Args:
        coord (np.ndarray): Sequence of floats, all between 0 and 1. A coordinate in a hypercube.

    Returns:
        np.ndarray: A sequence of floats of length `len(coord)+1` that sums to 1.
    """
    ret = np.zeros(len(coord) + 1)  # pre-alloc

    remaining_prob_mass = 1.0
    for ix, c in enumerate(coord):
        ret[ix] = remaining_prob_mass * c
        remaining_prob_mass -= ret[ix]
    
    assert -1e5 < remaining_prob_mass <= 1.0

    ret[-1] = remaining_prob_mass

    return ret


# helper function
def srv_ps_to_unit_cube(srv_ps: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(single_ps_to_unit_cube_coord, axis=-1, arr=srv_ps)


# helper function
def unit_cube_to_srv_ps(srv_coords: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(single_unit_cube_coord_to_ps, axis=-1, arr=srv_coords)


# helper function
_log_constraint_fun_values = []  # a log of constraint values, for later plotting 
def constraint_srv_1d_cubed(cubed_srv_ps_1d: np.ndarray, srv_ps_shape: Sequence[int], 
                            ps: Sequence[Sequence[float]], log=False) -> float:
    """Helper function for scipy.optimize.minimize, because that can only handle 1D arrays, so I have to convert.
    
    This function assumes that a one-dimensional version of the 'cubed' (converted to hypercube coordinate) 
    SRV probabilities array is passed, like what `srv_ps_to_unit_cube(srv_ps).reshape(-1)` would produce. It 
    then converts back to the shape of `srv_ps` containing probabilities, and then calls `constraint_srv` 
    to return its value.
    """
    # this is the shape of srv_ps_to_unit_cube(srv_ps), basically the last dimension gets one element less
    cubed_srv_ps_shape = srv_ps_shape[:-1] + (srv_ps_shape[-1]-1,)

    constr = constraint_srv(unit_cube_to_srv_ps(np.reshape(cubed_srv_ps_1d, cubed_srv_ps_shape)), ps)

    if log: 
        _log_constraint_fun_values.append(constr)

    return constr


_log_objective_fun_values = []  # a log of objective function values, for later plotting 
def objective_total_mi_1d_cubed(cubed_srv_ps_1d: np.ndarray, srv_ps_shape: Sequence[int], 
                                ps: Sequence[Sequence[float]], log=False) -> float:
    """Helper function for scipy.optimize.minimize, because that can only handle 1D arrays, so I have to convert.
    
    This function assumes that a one-dimensional version of the 'cubed' SRV probabilities array is passed, like
    `srv_ps_to_unit_cube(srv_ps).reshape(-1)` would produce. It then converts back to the shape of `srv_ps` containing
    probabilities, and then calls `mutual_information_srv_all_inputs` to return its negative value (for minimization).
    """
    # this is the shape of srv_ps_to_unit_cube(srv_ps), basically the last dimension gets one element less
    cubed_srv_ps_shape = srv_ps_shape[:-1] + (srv_ps_shape[-1]-1,)

    fun = mutual_information_srv_all_inputs(unit_cube_to_srv_ps(np.reshape(cubed_srv_ps_1d, cubed_srv_ps_shape)), ps)

    if log:
        _log_objective_fun_values.append(fun)

    return -fun


def optimize_srv_using_scipy(init_srv_ps: np.ndarray, ps: Sequence[Sequence[float]], 
                             method='SLSQP', tol=None, verbose=0) -> np.ndarray:
    """Use a constraint-based optimization to find an SRV

    Args:
        init_srv_ps (np.ndarray): _description_
        method (str, optional): _description_. Defaults to 'SLSQP'.
        verbose (int, optional): _description_. Defaults to 0.

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: _description_
    """
    global _log_objective_fun_values
    global _log_constraint_fun_values

    srv_ps_shape = np.shape(init_srv_ps)
    # this is the shape of srv_ps_to_unit_cube(srv_ps), basically the last dimension gets one element less
    cubed_srv_ps_shape = srv_ps_shape[:-1] + (srv_ps_shape[-1]-1,)

    x0 = srv_ps_to_unit_cube(init_srv_ps).reshape(-1)
    bounds = [(0, 1)]*len(x0)

    np.testing.assert_allclose(unit_cube_to_srv_ps(np.reshape(x0, cubed_srv_ps_shape)), init_srv_ps, rtol=1,
                               err_msg=f'------------{unit_cube_to_srv_ps(np.reshape(x0, cubed_srv_ps_shape))=}\n::::::::::::::::::\n{init_srv_ps}----------------')

    if method.lower() == 'slsqp':
        constraints = [{'type': 'eq', 'fun': lambda cubed_srv_ps_1d: constraint_srv_1d_cubed(cubed_srv_ps_1d, srv_ps_shape, ps, True)}]

        _log_objective_fun_values = []  # reset the (global) log of objective function values
        _log_constraint_fun_values = []  # also constraint values

        optres = minimize(objective_total_mi_1d_cubed, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                        args=(srv_ps_shape, ps, True), options={'maxiter': 1000}, tol=tol)
    else:
        raise NotImplementedError(f'unknown {method=}')

    srv_ps_opt = unit_cube_to_srv_ps(np.reshape(optres.x, cubed_srv_ps_shape))

    if verbose > 0:
        print(f'{optres.success=}')
        print(f'{optres.message=}')
        print(f'Optimal SRV found: {srv_ps_opt=}')
        print(f'Constraint SRV (should be zero): {constraint_srv(srv_ps_opt, ps)=}')
        print(f'Objective function (should be as high as possible): {mutual_information_srv_all_inputs(srv_ps_opt, ps)=}')
        print(f'Upper bound (not necessarily tight) of the objective function: {min(entropy_inputs(ps))}')
    
    return srv_ps_opt


def plot_most_recent_objective_function_curve_from_log(srv_ps_opt: np.ndarray = None, ps: np.ndarray = None):
    global _log_constraint_fun_values
    global _log_objective_fun_values
    
    import matplotlib.pyplot as plt

    # NOTE: there can be excursions to (very) high values in this plot, but remember, this is only showing the
    # objective function value, not the extent of constraint violation. In other words, the objective value might
    # sometimes be large but the constraints violated strongly, such that the final outcome (dashed line) is lower than 
    # some excursions.
    # TODO: also keep a log of the constraint values and plot those as well?
    fig, ax = plt.subplots()

    ax.plot(_log_objective_fun_values, '-k')
    ax.set_xlabel('Epoch (function call order)')
    ax.set_ylabel('Objective function value (higher is better)')

    ax.set_ylim([0, ax.get_ylim()[-1]])  # make sure the axis starts at zero

    if not srv_ps_opt is None and not ps is None:
        final_obj_val = mutual_information_srv_all_inputs(srv_ps_opt, ps)
        plt.plot([final_obj_val]*len(_log_objective_fun_values), '--k', alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(_log_constraint_fun_values, '-k', alpha=0.5, linewidth=2)
    ax2.set_ylabel('Constraint function value (0=perfect)')

    plt.show()


### TESTING FUNCTIONS


def test_specify_multiple_inputs() -> None:
    """Testing consistency between `condition_multiple_inputs` and `condition_single_input`."""
    srv_ps = np.random.random((3, 3, 3))  # an SRV with three states that conditions on two inputs of each 3 states
    srv_ps /= srv_ps.sum(axis=-1, keepdims=True)  # normalize each Pr(S | X=x)
    
    orig_shape_srv_ps = srv_ps.shape

    assert np.all(condition_multiple_inputs([1], [1]) == condition_single_input(1, 1))

    s = condition_single_input(1, 1)
    assert np.all(condition_multiple_inputs([0,1], [1,1]) == condition_single_input(0, 1, srv_ps=s))

    assert np.all(np.equal(orig_shape_srv_ps, srv_ps.shape)), 'srv_ps is passed by reference (pointer), it got changed while it should not have'


def test_unit_cube_coordinate_conversion() -> None:
    """Convert from a onedimensional PMF to a hypercube coordinate and back. Should be the same."""
    srv_ps = np.random.random((3, 3, 3))  # an SRV with three states that conditions on two inputs of each 3 states
    srv_ps /= srv_ps.sum(axis=-1, keepdims=True)  # normalize each Pr(S | X=x)

    np.testing.assert_allclose(single_ps_to_unit_cube_coord([0.60, 0.10, 0.30]), [0.6 , 0.25])
    np.testing.assert_allclose(single_unit_cube_coord_to_ps(single_ps_to_unit_cube_coord([0.60, 0.10, 0.30])), [0.60, 0.10, 0.30])

    np.testing.assert_allclose(srv_ps, unit_cube_to_srv_ps(srv_ps_to_unit_cube()))
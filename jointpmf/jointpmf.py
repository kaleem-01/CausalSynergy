__author__ = 'rquax'


'''
Owner of this project:

    Rick Quax
    https://staff.fnwi.uva.nl/r.quax
    University of Amsterdam

You are free to use this package only for your own non-profit, academic work. All I ask is to be suitably credited.
'''

import jointpmf.direct_srv as ds 

import csv
import numpy as np
import itertools
import copy
from scipy.optimize import minimize
import warnings
from funcy import flatten
try:
    from collections import Sequence, Iterable, Callable, Hashable
except ImportError:
    from collections.abc import Sequence, Iterable, Callable, Hashable  # from Python 3.10 this changed
from collections import deque
import time
from abc import abstractmethod, ABCMeta  # for requiring certain methods to be overridden by subclasses
from numbers import Integral, Number, Real
import matplotlib.pyplot as plt
import sys
import pathos.multiprocessing as mp
import networkx as nx
import scipy.stats as ss
# from astroML.visualization import hist  # for Bayesian blocks: automatic determining of variable-size binning of data
from numpy import histogram_bin_edges

# imports for doing optimization:
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
from poap.controller import SerialController

import logging
# suggested by Cillian to suppress anoying messages from pySOT
logging.getLogger("pySOT.strategy.surrogate_strategy").setLevel(logging.WARNING)

import logging
logging.getLogger("pySOT.strategy.srbf_strategy").setLevel(logging.WARNING)
# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd

from pgmpy.estimators import MmhcEstimator  # for estimating a structure for a BN
import warnings

_prob_error_tol = 1e-6
_mi_error_tol = 1e-7

# these are used e.g. in np.testing.assert_almost_equal:
_prob_tol_decimal = -int(np.log10(_prob_error_tol))  # note: for 1e6 this becomes 6
_mi_tol_decimal = -int(np.log10(_mi_error_tol))

# I use this instead of -np.inf as result of log(0), which whould be -inf but then 0 * -inf = NaN, whereas by
# common assumption in information theory: 0 log 0 = 0. So I make it finite here so that the result is indeed 0
# (since it gets multiplied by zero).
_finite_inf = sys.float_info.max / 1000

def maximum_depth(seq):
    """
    Helper function, e.g. maximum_depth([1,2,[2,4,[[4]]]]) == 4.
    :param seq: sequence, like a list of lists
    :rtype: int
    """
    seq = iter(seq)
    try:
        for level in itertools.count():
            seq = itertools.chain([next(seq)], seq)
            seq = itertools.chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level


# helper function,
# from http://stackoverflow.com/questions/2267362/convert-integer-to-a-string-in-a-given-numeric-base-in-python
def int2base(x, b, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    """

    :param x: int
    :type x: int
    :param b: int
    :param b: int
    :param alphabet:
    :rtype : str
    """

    # convert an integer to its string representation in a given base
    if b<2 or b>len(alphabet):
        if b==64: # assume base64 rather than raise error
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        else:
            raise AssertionError("int2base base out of range")

    if isinstance(x,complex): # return a tuple
        return ( int2base(x.real,b,alphabet) , int2base(x.imag,b,alphabet) )

    if x<=0:
        if x==0:
            return alphabet[0]
        else:
            return '-' + int2base(-x,b,alphabet)

    # else x is non-negative real
    rets=''

    while x>0:
        x,idx = divmod(x,b)
        rets = alphabet[idx] + rets

    return str(rets)


def apply_permutation(lst, permutation):
    """
    Return a new list where the element at position ix in <lst> will be at a new position permutation[ix].
    :param lst: list
    :type lst: array_like
    :param permutation:
    :return:
    """
    new_list = [-1]*len(lst)

    assert len(permutation) == len(lst)

    for ix in range(len(permutation)):
        new_list[permutation[ix]] = lst[ix]

    if __debug__:
        if not -1 in lst:
            assert not -1 in new_list

    return new_list


# each variable in a JointProbabilityMatrix has a label, and if not provided then this label is used
_default_variable_label = 'variable'


# any derived class from this interface is supposed
# to replace the dictionaries now used for conditional pdfs (namely, dict where keys are
# tuples of values for all variables and the values are JointProbabilityMatrix objects). For now it will be
# made to be superficially equivalent to a dict, then hopefully a derived class can have a different
# inner workings (not combinatorial explosion of the number of keys) while otherwise indistinguishable
# note: use this class name in isinstance(obj, ConditionalProbabilityMatrix)
class ConditionalProbabilities(object, metaclass=ABCMeta):

    # for enabling the decorator 'abstractmethod', which means that a given function MUST be overridden by a
    # subclass. I do this because all functions depend heavily on the data structure of cond_pdf, but derived
    # classes will highly likely have different structures, because that would be the whole point of inheritance:
    # currently the cond_pdf encodes the full conditional pdf using a dictionary, but the memory usage of that
    # scales combinatorially.
    cond_pdf = {}


    @abstractmethod
    def __init__(self, cond_pdf=None):
        assert False, 'should have been implemented'


    @abstractmethod
    def generate_random_conditional_pdf(self, num_given_variables, num_output_variables, num_values=2):
        assert False, 'should have been implemented'


    @abstractmethod
    def __getitem__(self, item):
        assert False, 'should have been implemented'


    @abstractmethod
    def iteritems(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def itervalues(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def iterkeys(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def num_given_variables(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def num_output_variables(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def num_conditional_probabilities(self):
        assert False, 'should have been implemented'


    @abstractmethod
    def update(self, partial_cond_pdf):
        assert False, 'should have been implemented'


class ConditionalProbabilityMatrix(ConditionalProbabilities):

    # a dict which fully explicitly encodes the conditional pdf,
    # namely, dict where keys are
    # tuples of values for all variables and the values are JointProbabilityMatrix objects
    cond_pdf = {}
    numvalues = 0  # note: this is the total number of joint states, so if each PMF consists of three bits, then this will store 2**3 = 8.
    numvariables = 0
    # number of values that each conditioned-upon variable can take (can also be None is it is not 
    # computed for computational efficiency reasons; atm it is foreseen to be only used for assertions)
    num_given_values = []

    def __init__(self, cond_pdf=None, num_given_values=None):
        if cond_pdf is None:
            self.cond_pdf = {}
            self.numvalues = 0
            self.numvariables = 0
            self.num_given_values = []
        elif type(cond_pdf) == dict:
            assert not np.isscalar(next(iter(cond_pdf.keys()))), 'should be tuples, even if conditioning on 1 var'

            self.cond_pdf = cond_pdf
            self.numvalues = self.num_states()
            self.numvariables = self.num_output_variables()
            # do not compute because it takes potentially time (for large state spaces) while it is not
            # needed currently for anything (if you need it then use the function `self.num_given_values()`).
            self.num_given_values = None  
        elif isinstance(cond_pdf, JointProbabilityMatrix) or isinstance(cond_pdf, BayesianNetwork):  # assume independence
            # self.cond_pdf = {states: cond_pdf for states in cond_pdf.statespace()}
            # self.numvalues = cond_pdf.numvalues
            # self.numvariables = cond_pdf.numvariables
            if num_given_values is None:
                raise ValueError('I don\'t know what the statespace of the given variables is. (Then this function needs an additional argument for that.)')
            else:
                self.cond_pdf = {states: cond_pdf for states in itertools.product(*[range(ngv) for ngv in num_given_values])}
                self.numvalues = cond_pdf.numvalues
                self.numvariables = cond_pdf.numvariables
                self.num_given_values = num_given_values
        else:
            raise NotImplementedError('unknown type for cond_pdf')


    def __getitem__(self, item):
        return self.cond_pdf[tuple(item)]


    def __len__(self):
        return self.num_output_variables()


    def __eq__(self, other):
        if isinstance(other, str):
            return False
        elif isinstance(other, Number):
            return False
        else:
            assert hasattr(other, 'num_given_variables'), 'should be ConditionalProbabilityMatrix'
            assert hasattr(other, 'iteritems'), 'should be ConditionalProbabilityMatrix'

            for states, pdf in self.items():
                if not other[states] == pdf:
                    return False

            return True

    def generate_random_conditional_pdf(self, num_given_variables, num_output_variables, num_values=2, num_given_values=None, method='dirichlet'):
        if num_given_values is None:
            num_given_values = num_values

        if np.isscalar(num_given_values):
            num_given_values = [num_given_values]*num_given_variables
        else:
            assert len(num_given_values) == num_given_variables

        states_iter = itertools.product(*[range(ngv) for ngv in num_given_values])

        self.cond_pdf = {states: JointProbabilityMatrix(num_output_variables, num_values, joint_probs=method)
                        for states in states_iter}
        self.numvalues = num_values
        self.numvariables = num_output_variables
        self.num_given_values = num_given_values


    def __getitem__(self, item):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return self.cond_pdf[item]


    def iteritems(self):
        return iter(self.cond_pdf.items())


    def itervalues(self):
        return iter(self.cond_pdf.values())


    def iterkeys(self):
        return iter(self.cond_pdf.keys())


    def num_given_variables(self) -> int:
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(next(iter(self.cond_pdf.keys())))

    def num_given_values(self) -> list:
        # this function prevents constructing the whole state space of the given variables
        # (would potentially cost a lot of memory)
        statespaces = [set() for _ in self.num_given_variables()]

        for given_states in self.cond_pdf.keys():  # given_states is a tuple of values, one for each conditioned-upon variable
            for gix, gval in enumerate(given_states):
                statespaces[gix].add(gval)
        
        num_given_values = [len(s) for s in statespaces]

        return num_given_values

    def num_output_variables(self):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(next(iter(self.cond_pdf.values())))

    def statespace(self, variables: Sequence[int] = None, only_size=False):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return next(iter(self.cond_pdf.values())).statespace(variables, only_size=only_size)

    def num_states(self) -> int:
        """Number of possible states of any conditional probability distribution.

        This is supposed to be more efficient than len(list(self.statespace())).

        Returns:
            int: number of states
        """
        # NOTE: this is buffered as a self.numvalues when constructed, so use that instead of this function if you want performance.
        ret = next(iter(self.cond_pdf.values())).num_states()

        return ret

    def matrix2params(self):
        nested_params = [list(pdf.matrix2params()) for pdf in self.cond_pdf.values()]

        assert np.shape(nested_params) == (len(self.cond_pdf), self.num_states() - 1)

        return np.array(list(flatten(nested_params)))
    
    def params2matrix(self, params):
        if len(np.shape(params)) == 1:
            nested_params = np.reshape(params, (len(self.cond_pdf), len(list(self.statespace())) - 1))
        else:
            assert len(np.shape(params)) == 2, 'expected dimension: (# given values, |statespace per variable| - 1)'
            assert np.shape(params) == (len(self.cond_pdf), self.num_states() - 1), 'expected dimension: (# given values, |statespace per variable| - 1)'
            nested_params = params
        
        # NOTE: according to https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects, 
        # the order of dict.keys() (in self.matrix2params) and dict.keys() (here) is guaranteed in the
        # same order (if nothing in between changed). So I assume that I can just iterate through the
        # keys and set the values with the corresponding parameters that self.matrix2params() generated.

        for ix, k in enumerate(self.cond_pdf.keys()):
            self.cond_pdf[k].params2matrix(nested_params[ix])

    def num_conditional_probabilities(self):
        return len(self.cond_pdf)

    def to_dict(self) -> dict:
        try:
            return {c: list(v.joint_probabilities.joint_probabilities) for c, v in self.iteritems()}
        except AttributeError:
            return {c: list(v.joint_probabilities().values()) for c, v in self.iteritems()}

    def __str__(self) -> str:
        return str(self.to_dict())


    def update(self, partial_cond_pdf):
        if type(partial_cond_pdf) == dict:
            # check only if already initialized with at least one conditional probability
            if __debug__ and len(self.cond_pdf) > 0:
                assert len(next(iter(partial_cond_pdf.keys()))) == self.num_given_variables(), 'partial cond. pdf is ' \
                                                                                              'conditioned on a different' \
                                                                                              ' number of variables'
                assert len(next(iter(partial_cond_pdf.values()))) == self.num_output_variables(), \
                    'partial cond. pdf has a different number of output variables'

            self.cond_pdf.update(partial_cond_pdf)
        elif isinstance(partial_cond_pdf, ConditionalProbabilities):
            self.cond_pdf.update(partial_cond_pdf.cond_pdf)
        else:
            raise NotImplementedError('unknown type for partial_cond_pdf')


_type_prob = np.longdouble  # type to use for storing probabilities
_type_state = np.int8  # type to store the state of a variable (0, 1, ..., `numstates-1`)
_essential_zero_prob = 1e-7  # below this value we could consider a probability to be 0 (for log(0) problems)

# this class is supposed to override member joint_probabilities of JointProbabilityMatrix, which is currently
# taken to be a nested numpy array. Therefore this class is made to act as much as possible as a nested numpy array.
# However, this class could be inherited and overriden to save e.g. memory storage, e.g. by assuming independence
# among the variables, but on the outside still act the same.
# note: use this class name for isinstance(., .) calls, because it will also be true for derived classes
class NestedArrayOfProbabilities(object, metaclass=ABCMeta):

    joint_probabilities = np.array([], dtype=_type_prob)

    @abstractmethod
    def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def num_variables(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def num_values(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def __getitem__(self, item):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def __setitem__(self, key, value):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def sum(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def flatiter(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def clip_all_probabilities(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def flatten(self):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def reset(self, jointprobs):  # basically __init__ but can be called even if object already exists
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def duplicate(self, other):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def __sub__(self, other):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'


    @abstractmethod
    def generate_random_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def generate_dirichlet_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'


# class BayesianNetwork(NestedArrayOfProbabilities):

#     # TODO: joint_probabilities should be either removed or replaced by a class
#     # whose __getitem__ is overloaded
#     joint_probabilities = np.array([], dtype=_type_prob)

#     # sequence of (conditional) probability distributions, one per variable
#     pdfs = []  # length: self.num_variables()
#     # number of values for each pdf
#     numvals = []  # length: self.num_variables()

#     @abstractmethod
#     def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
#         if isinstance(joint_probabilities, (np.ndarray, list, tuple)):
#             if len(joint_probabilities) == 0:
#                 self.pdfs = []
#                 self.numvals = []
#             else:
#                 numvals = np.shape(joint_probabilities)
#                 assert len(numvals) >= 1

#                 initial_pdf = 

#                 for nv in numvals:



#     @abstractmethod
#     def num_variables(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def num_values(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def __getitem__(self, item):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def __setitem__(self, key, value):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def sum(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def flatiter(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def clip_all_probabilities(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def flatten(self):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def reset(self, jointprobs):  # basically __init__ but can be called even if object already exists
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def duplicate(self, other):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def __sub__(self, other):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def generate_uniform_joint_probabilities(self, numvariables, numvalues):
#         assert False, 'must be implemented by subclass'


#     @abstractmethod
#     def generate_random_joint_probabilities(self, numvariables, numvalues):
#         assert False, 'must be implemented by subclass'

#     @abstractmethod
#     def generate_dirichlet_joint_probabilities(self, numvariables, numvalues):
#         assert False, 'must be implemented by subclass'


# this class is supposed to override member joint_probabilities of JointProbabilityMatrix, which is currently
# taken to be a nested numpy array. Therefore this class is made to act as much as possible as a nested numpy array.
# However, this class could be inherited and overriden to save e.g. memory storage, e.g. by assuming independence
# among the variables, but on the outside still act the same.
class FullNestedArrayOfProbabilities(NestedArrayOfProbabilities):

    # type: np.array
    joint_probabilities = np.array([], dtype=_type_prob)

    def __init__(self, joint_probabilities: np.ndarray = None):
        # assert isinstance(joint_probabilities, np.ndarray), 'should pass numpy array?'

        if joint_probabilities is None:
            # a single possible event with prob. 1 (probably a placeholder that the user will quickly replace,
            # e.g., by some optimization procedure or something else)
            joint_probabilities = np.array([1.], dtype=_type_prob)
        
        self.reset(joint_probabilities)


    def num_variables(self):
        return np.ndim(self.joint_probabilities)


    def num_values(self):
        return np.shape(self.joint_probabilities)[-1]


    def __getitem__(self, item):
        assert len(item) == self.joint_probabilities.ndim, 'see if this is only used to get single joint probs'

        if type(item) == int:
            return FullNestedArrayOfProbabilities(joint_probabilities=self.joint_probabilities[item])
        elif hasattr(item, '__iter__'):  # tuple for instance
            ret = self.joint_probabilities[item]

            if hasattr(ret, '__iter__'):
                return FullNestedArrayOfProbabilities(joint_probabilities=ret)
            else:
                assert -0.000001 <= ret <= 1.000001  # this seems the only used case

                return ret  # return a single probability
        elif isinstance(item, slice):
            return FullNestedArrayOfProbabilities(joint_probabilities=self.joint_probabilities[item])


    def __setitem__(self, key, value):
        assert False, 'see if used at all?'

        # let numpy array figure it out
        self.joint_probabilities[key] = value


    def sum(self):
        return self.joint_probabilities.sum()


    def flatiter(self):
        return self.joint_probabilities.flat


    def clip_all_probabilities(self):
        """
        Make sure all probabilities in the joint probability matrix are in the range [0.0, 1.0], which could be
        violated sometimes due to floating point operation roundoff errors.
        """
        self.joint_probabilities = np.array(np.minimum(np.maximum(self.joint_probabilities, 0.0), 1.0),
                                            dtype=_type_prob)

        if np.random.random() < 0.05 and __debug__:
            try:
                np.testing.assert_almost_equal(np.sum(self.joint_probabilities), 1.0)
            except AssertionError as e:
                print(('error message: ' + str(e)))

                print(('error: len(self.joint_probabilities) =', len(self.joint_probabilities)))
                print(('error: shape(self.joint_probabilities) =', np.shape(self.joint_probabilities)))
                if len(self.joint_probabilities) < 30:
                    print(('error: self.joint_probabilities =', self.joint_probabilities))

                raise AssertionError(e)


    def flatten(self):
        return self.joint_probabilities.flatten()


    def reset(self, jointprobs):  # basically __init__ but can be called even if object already exists
        if isinstance(jointprobs, (np.ndarray, list, tuple)):
            self.joint_probabilities = np.array(jointprobs, dtype=_type_prob)

            np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)
        else:
            # assert isinstance(jointprobs.joint_probabilities, np.ndarray), 'nested reset problem'

            self.duplicate(jointprobs)


    def duplicate(self, other):
        assert isinstance(other.joint_probabilities, np.ndarray), 'nesting problem'

        self.joint_probabilities = np.array(other.joint_probabilities, dtype=_type_prob)


    def __sub__(self, other):
        return np.subtract(self.joint_probabilities, np.array(other.joint_probabilities, dtype=_type_prob))


    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        self.joint_probabilities = np.array(np.zeros([numvalues]*numvariables), dtype=_type_prob)
        self.joint_probabilities = self.joint_probabilities + 1.0 / np.power(numvalues, numvariables)


    def generate_random_joint_probabilities(self, numvariables, numvalues):
        # todo: this does not result in random probability densities... Should do recursive
        self.joint_probabilities = np.random.random([numvalues]*numvariables)
        self.joint_probabilities /= np.sum(self.joint_probabilities)

    def generate_dirichlet_joint_probabilities(self, numvariables, numvalues):
        assert numvariables > 0
        assert numvalues > 0

        # todo: this does not result in random probability densities... Should do recursive
        self.joint_probabilities = np.random.dirichlet([1]*(numvalues**numvariables)).reshape((numvalues,)*numvariables)

        if __debug__:
            np.testing.assert_almost_equal(np.sum(self.joint_probabilities), 1.0)


# todo: move params2matrix and matrix2params to the NestedArray classes? Is specific per subclass...
class IndependentNestedArrayOfProbabilities(NestedArrayOfProbabilities, metaclass=ABCMeta):

    """
    note: each sub-array sums to 1.0
    type: np.array of np.array of float
    """
    marginal_probabilities = np.array([])  # this initial value indicates: zero variables

    def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
        self.reset(joint_probabilities)


    def num_variables(self):
        """
        The number of variables for which this object stores a joint pdf.
        """
        return len(self.marginal_probabilities)


    def num_values(self):
        assert len(self.marginal_probabilities) > 0, 'not yet initialized, so cannot determine num_values'

        return len(self.marginal_probabilities[0])


    def __getitem__(self, item):
        """
        If you supply an integer then it specifies the value of the first variable, so I will return the remaining
        PDF for the rest of the variables. If you supply a tuple then I will repeat this for every integer in the
        tuple.
        :param item: value or tuple of values for the first N variables, where N is the length of <item>.
        :type item: int or tuple
        """
        if isinstance(item, Integral):
            assert False, 'seems not used?'

            return IndependentNestedArrayOfProbabilities(self.marginal_probabilities[0][item]
                                                         * self.marginal_probabilities[len(item):])
        else:
            if len(item) == len(self.marginal_probabilities):
                return np.product([self.marginal_probabilities[vix][item[vix]] for vix in range(len(item))])
            else:
                assert len(item) < len(self.marginal_probabilities), 'supplied more values than I have variables'

                return IndependentNestedArrayOfProbabilities(np.product([self.marginal_probabilities[vix][item[vix]]
                                                                         for vix in range(len(item))])
                                                             * self.marginal_probabilities[len(item):])


    def __setitem__(self, key, value):
        assert False, 'is this used? if so, make implementation consistent with getitem? So that getitem(setitem)' \
                      ' yields back <value>? Now thinking of it, I don\'t think this is possible, because how would' \
                      ' I set the value of a joint state? It is made up of a multiplication of |X| probabilities,' \
                      ' and setting these probabilities to get the supplied joint probability is an ambiguous task.'


    def sum(self):
        """
        As far as I know, this is only used to verify that probabilities sum up to 1.
        """
        return np.average(np.sum(self.marginal_probabilities, axis=1))


    def flatiter(self):
        return self.marginal_probabilities.flat


    def clip_all_probabilities(self):
        self.marginal_probabilities = np.minimum(np.maximum(self.marginal_probabilities, 0.0), 1.0)


    def flatten(self):
        return self.marginal_probabilities.flatten()


    def reset(self, joint_probabilities):  # basically __init__ but can be called even if object already exists
        if isinstance(joint_probabilities, np.ndarray):
            shape = np.shape(joint_probabilities)

            # assume that the supplied array is in MY format, so array of single-variable prob. arrays
            if len(shape) == 2:
                assert joint_probabilities[0].sum() <= 1.000001, 'should be (marginal) probabilities'

                self.marginal_probabilities = np.array(joint_probabilities, dtype=_type_prob)
            elif len(joint_probabilities) == 0:
                # zero variables
                self.marginal_probabilities = np.array([], dtype=_type_prob)
            else:
                raise ValueError('if you want to supply a nested array of probabilities, create a joint pdf'
                                 ' from it first and then pass it to me, because I would need to marginalize variables')
        elif isinstance(joint_probabilities, JointProbabilityMatrix):
            self.marginal_probabilities = np.array([joint_probabilities.marginalize_distribution([vix])
                                                     .joint_probabilities.matrix2vector()
                                                 for vix in range(len(joint_probabilities))], dtype=_type_prob)
        else:
            raise NotImplementedError('unknown type of argument: ' + str(joint_probabilities))


    def duplicate(self, other):
        """
        Become a copy of <other>.
        :type other: IndependentNestedArrayOfProbabilities
        """
        self.marginal_probabilities = np.array(other.marginal_probabilities, dtype=_type_prob)


    def __sub__(self, other):
        """
        As far as I can tell, only needed to test e.g. for np.testing.assert_array_almost_equal.
        :type other: IndependentNestedArrayOfProbabilities
        """
        return self.marginal_probabilities - other.marginal_probabilities


    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        self.marginal_probabilities = np.array(np.zeros([numvariables, numvalues]), dtype=_type_prob)
        self.marginal_probabilities = self.marginal_probabilities + 1.0 / numvalues


    def generate_random_joint_probabilities(self, numvariables, numvalues):
        self.marginal_probabilities = np.array(np.random.random([numvariables, numvalues]), dtype=_type_prob)
        # normalize:
        self.marginal_probabilities /= np.transpose(np.tile(np.sum(self.marginal_probabilities, axis=1), (numvalues,1)))


class CausalImpactResponse(object):
    perturbed_variables = None

    mi_orig = None
    mi_nudged_list = None
    mi_diffs = None
    avg_mi_diff = None
    std_mi_diff = None

    impacts_on_output = None
    avg_impact = None
    std_impact = None

    correlations = None
    avg_corr = None
    std_corr = None

    residuals = None
    avg_residual = None
    std_residual = None

    nudges = None
    upper_bounds_impact = None

    def __init__(self):
        self.perturbed_variables = None

        self.mi_orig = None
        self.mi_nudged_list = None
        self.mi_diffs = None
        self.avg_mi_diff = None
        self.std_mi_diff = None

        self.impacts_on_output = None
        self.avg_impact = None
        self.std_impact = None

        self.correlations = None
        self.avg_corr = None
        self.std_corr = None

        self.residuals = None
        self.avg_residual = None
        self.std_residual = None

        self.nudges = None
        self.upper_bounds_impact = None


class JointProbabilityMatrix(object):
    # nested list of probabilities. For instance for three binary variables it could be:
    # [[[0.15999394, 0.06049343], [0.1013956, 0.15473886]], [[ 0.1945649, 0.15122334], [0.11951818, 0.05807175]]].
    joint_probabilities: FullNestedArrayOfProbabilities = None  # FullNestedArrayOfProbabilities

    numvariables = 0
    numvalues = 0

    # note: at the moment I only store the labels here as courtesy, but the rest of the functions do not operate
    # on labels at all, only variable indices 0..N-1.
    labels = []

    _type_prob = _type_prob

    def __init__(self, numvariables, numvalues, joint_probs='dirichlet', labels=None,
                 create_using=FullNestedArrayOfProbabilities):

        self.numvariables = numvariables
        self.numvalues = numvalues

        assert self.numvalues > 0, f'{self.numvalues=} should be >0'

        if labels is None:
            self.labels = [_default_variable_label]*numvariables
        else:
            self.labels = labels

        if joint_probs is None or numvariables == 0:
            self.joint_probabilities = create_using()
            self.generate_random_joint_probabilities()
        # todo: this 'isinstance' is probably costly, but if I don't do it then I get warnings... 
        # make a separate argument for strings and keep 'joint_probs' only for numpy arrays?
        elif isinstance(joint_probs, str):
            if joint_probs == 'uniform':
                self.joint_probabilities = create_using()
                self.generate_uniform_joint_probabilities(numvariables, numvalues)
            elif joint_probs == 'random':
                # this is BIASED! I.e., if generating bits then the marginal p of one bit being 1 is not uniform
                self.joint_probabilities = create_using()
                self.generate_random_joint_probabilities()
            elif joint_probs in ('unbiased', 'dirichlet'):
                self.joint_probabilities = create_using()
                self.generate_dirichlet_joint_probabilities()  # just to get valid params and pass the debugging tests
                numparams = len(self.matrix2params_incremental(True))
                assert numparams > 0 or numvalues == 1, f'{numparams=} is strange, should be >0. Even a binary variable has 1 probability to choose; {self.joint_probabilities.joint_probabilities=}'
                if numvalues > 1:
                    self.params2matrix_incremental(np.random.random(numparams))
                else:
                    if __debug__:
                        np.testing.assert_almost_equal(np.sum(self.joint_probabilities.joint_probabilities), 1.0)
            else:
                raise ValueError('don\'t know what to do with joint_probs=' + str(joint_probs))
        else:
            self.joint_probabilities = create_using(joint_probs)

        # note: I used to put this here for certainty but now I trust the above generators... and it saves computation time
        # self.clip_all_probabilities()

        if self.numvariables > 0 and __debug__:
            assert self.joint_probabilities.num_values() == self.numvalues
            assert self.joint_probabilities.num_variables() == self.numvariables

            np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)
        else:
            pass
            # warnings.warn('numvariables == 0, not sure if it is supported by all code (yet)! Fingers crossed.')


    def copy(self):
        """
        Deep copy.
        :rtype : JointProbabilityMatrix
        """
        return copy.deepcopy(self)


    def generate_sample(self) -> Sequence[int]:
        rand_real = np.random.random()

        for input_states in self.statespace():
            rand_real -= self.joint_probability(input_states)

            if rand_real < 0.0:
                return input_states

        assert False, 'should not happen? or maybe foating point roundoff errors built up.. or so'

        return self.statespace(len(self.numvariables))[-1]


    def generate_samples(self, n, nprocs=1, method='fast'):
        """
        Generate samples from the distribution encoded by this PDF object.
        :param n: number of samples
        :param nprocs: since this procedure is quite slow, from about >=1e5 samples you may want parallel
        :return: list of tuples, shape (n, numvariables)
        :rtype: list
        """
        if nprocs == 1:
            return [self.generate_sample() for _ in range(n)]
        elif method == 'slow':
            def worker_gs(i):
                return self.generate_sample()

            pool = mp.Pool(nprocs)

            ret = pool.map(worker_gs, range(n))

            pool.close()
            pool.terminate()

            return ret
        elif method == 'fast':
            input_states_list = [input_states for input_states in self.statespace(self.numvariables)]

            state_probs = np.array([self.joint_probability(input_states)
                                    for iix, input_states in enumerate(input_states_list)])

            if nprocs == 1:
                state_ixs = np.random.choice(len(input_states_list), size=n, p=state_probs)
            else:
                def worker_gs(numsamples):
                    return np.random.choice(len(input_states_list), size=numsamples, p=np.array(state_probs,
                                                                                                dtype=np.float64))

                numsamples_list = [int(round((pi+1)/float(nprocs) * n)) - int(round(pi/float(nprocs) * n))
                                   for pi in range(nprocs)]

                pool = mp.Pool(nprocs)

                ret = pool.map(worker_gs, numsamples_list)

                pool.close()
                pool.terminate()

                state_ixs = list(itertools.chain.from_iterable(ret))

            return np.array(input_states_list)[state_ixs]
        else:
            raise NotImplementedError('don\'t know this method')


    def transition_table_deterministic(self, variables_input, variables_output):
        """
        Create a transition table for the given variables, where for each possible sequence of values for the input
        variables I find the values for the output variables which have maximum conditional probability. I marginalize
        first over all other variables not named in either input or output.
        :type variables_input: list of int
        :type variables_output: list of int
        :return: a list of lists, where each sublist is [input_states, max_output_states, max_output_prob], where the
        first two are lists and the latter is a float in [0,1]
        """
        variables = list(variables_input) + list(variables_output)

        pdf_temp = self.marginalize_distribution(variables)

        assert len(pdf_temp) == len(variables)

        trans_table = []

        # for each possible sequence of input states, find the output states which have maximum probability
        for input_states in self.statespace(len(variables_input)):
            max_output_states = []
            max_output_prob = -1.0

            for output_states in self.statespace(len(variables_output)):
                prob = pdf_temp(list(input_states) + list(output_states))

                if prob > max_output_prob:
                    max_output_prob = prob
                    max_output_states = output_states

            # assert max_output_prob >= 1.0 / np.power(self.numvalues, len(variables_output)), \
            #     'how can this be max prob?: ' + str(max_output_prob) + '. I expected at least: ' \
            #     + str(1.0 / np.power(self.numvalues, len(variables_output)))

            trans_table.append([input_states, max_output_states, max_output_prob])

        return trans_table


    def generate_uniform_joint_probabilities(self, numvariables, numvalues, create_using=None):
        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_uniform_joint_probabilities(numvariables, numvalues)


    def generate_random_joint_probabilities(self, create_using=None):
        # jp = np.random.random([self.numvalues]*self.numvariables)
        # jp /= np.sum(jp)

        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_random_joint_probabilities(self.numvariables, self.numvalues)

    def generate_dirichlet_joint_probabilities(self, create_using=None):
        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_dirichlet_joint_probabilities(self.numvariables, self.numvalues)


    def random_samples(self, n=1):
        # sample_indices = np.random.multinomial(n, [i for i in self.joint_probabilities.flatiter()])

        # todo: I don't think this generalizes well for different numvalues per variable. Just generate 1000 value lists
        # and pick one according to their relative probabilities? Much more general and above all much more scalable!

        assert np.isscalar(self.numvalues), 'refactor this function, see todo above'

        flat_joint_probs = [i for i in self.joint_probabilities.flatiter()]

        sample_indices = np.random.choice(len(flat_joint_probs), p=flat_joint_probs, size=n)

        values_str_per_sample = [int2base(smpl, self.numvalues).zfill(self.numvariables) for smpl in sample_indices]

        assert len(values_str_per_sample) == n
        assert len(values_str_per_sample[0]) == self.numvariables, 'values_str_per_sample[0] = ' \
                                                                   + str(values_str_per_sample[0])
        assert len(values_str_per_sample[-1]) == self.numvariables, 'values_str_per_sample[-1] = ' \
                                                                   + str(values_str_per_sample[-1])

        values_list_per_sample = [[int(val) for val in valstr] for valstr in values_str_per_sample]

        assert len(values_list_per_sample) == n
        assert len(values_list_per_sample[0]) == self.numvariables
        assert len(values_list_per_sample[-1]) == self.numvariables

        return values_list_per_sample


    def __call__(self, values):
        """
        Joint probability of a list or tuple of values, one value for each variable in order.
        :param values: list of values, each value is an integer in [0, numvalues)
        :type values: list
        :return: joint probability of the given values, in order, for all variables; in [0, 1]
        :rtype: float
        """
        return self.joint_probability(values=values)


    def set_labels(self, labels):
        self.labels = labels


    def get_labels(self):
        return self.labels


    def get_label(self, variable_index):
        return self.labels[variable_index]


    def set_label(self, variable_index, label):
        self.labels[variable_index] = label


    def has_label(self, label):
        return label in self.labels


    def retain_only_labels(self, retained_labels, dict_of_unwanted_label_values):
        unwanted_labels = list(set(self.labels).difference(retained_labels))

        # unwanted_label_values = [dict_of_unwanted_label_values[lbl] for lbl in unwanted_labels]

        unwanted_vars = self.get_variables_with_label_in(unwanted_labels)
        unwanted_var_labels = list(map(self.get_label, unwanted_vars))
        unwanted_var_values = [dict_of_unwanted_label_values[lbl] for lbl in unwanted_var_labels]

        self.duplicate(self.conditional_probability_distribution(unwanted_vars, unwanted_var_values))



    def clip_all_probabilities(self):
        """
        Make sure all probabilities in the joint probability matrix are in the range [0.0, 1.0], which could be
        violated sometimes due to floating point operation roundoff errors.
        """
        self.joint_probabilities.clip_all_probabilities()


    def estimate_from_data_from_file(self, filename, discrete=True, method='empirical'):
        """
        Same as estimate_from_data but read the data first from the file, then pass data to estimate_from_data.
        :param filename: string, file format will be inferred from the presence of e.g. '.csv'
        :param discrete:
        :param method:
        :raise NotImplementedError: if file format is not recognized
        """
        if '.csv' in filename:
            fin = open(filename, 'rb')
            csvr = csv.reader(fin)

            repeated_measurements = [tuple(row) for row in csvr]

            fin.close()
        else:
            raise NotImplementedError('unknown file format: ' + str(filename))

        self.estimate_from_data(repeated_measurements=repeated_measurements, discrete=discrete, method=method)

    def probs_from_max_numbin(sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return [float(z) / len(sx) for z in list(d.values())]


    # helper function
    @staticmethod
    def probs_by_bayesian_blocks(array, bins='auto'):
        assert len(array) > 0

        if np.isscalar(array[0]):
            # bb_bins = hist(array, bins='blocks')  # this gave import errors at some point so commented out (was from astroML)
            # bb_bins = np.histogram_bin_edges(array, bins=bins)
            bb_counts, bb_bins = np.histogram(array, bins=bins)
            assert len(bb_counts) > 0
            bb_probs = bb_counts / np.sum(bb_counts)

            assert len(bb_probs) > 0
            assert np.isscalar(bb_probs[0])

            return bb_bins, bb_probs
        else:
            assert np.rank(array) == 2

            vectors = np.transpose(array)

            return list(map(JointProbabilityMatrix.probs_by_bayesian_blocks, vectors))


    @staticmethod
    def probs_by_equiprobable_bins(sx, k='auto'):

        if np.isscalar(sx[0]):
            if isinstance(k, str):  # like 'auto' or 'freedman', does not really matter
                # k = float(int(np.sqrt(n1)))  # simple heuristic
                # Freedman-Diaconis rule:
                h = 2. * (np.percentile(sx, 75) - np.percentile(sx, 25)) / np.power(len(sx), 1/3.)
                k = int(round((max(sx) - min(sx)) / float(h)))
            else:
                k = float(k)  # make sure it is numeric

            percs = [np.percentile(sx, p) for p in np.linspace(0, 100, k + 1)]
            ##    for i in xrange(n2):
            ##      assert percs[i] == sorted(percs[i]), str('error: np.percentile (#'+str(i)+') did not return a sorted list of values:\n'
            ##                                               +str(percs))
            ##      assert len(percs[i]) == len(set(percs[i])), 'error: entropyd: numbins too large, I get bins of width zero.'
            # remove bins of size zero
            # note: bins of size zero would not contribute to entropy anyway, since 0 log 0 is set
            # to result in 0. For instance, it could be that the data only has 6 unique values occurring,
            # but a numbins=10 is given. Then 4 bins will be empty.
            ##    print('percs: = ' + str(percs))
            percs_fixed = sorted(list(set(percs)))
            ##    probs = np.divide(np.histogramdd(np.hstack(sx), percs)[0], len(sx))
            print(('debug: percs_fixed =', percs_fixed))
            probs = np.divide(np.histogramdd(sx, [percs_fixed])[0], len(sx))
            probs1d = list(probs.flat)

            return percs_fixed, probs1d
        else:
            samples_list = np.transpose(sx)

            rets = [JointProbabilityMatrix.probs_by_equiprobable_bins(si, k=k) for si in samples_list]

            return np.transpose(rets)


    @staticmethod
    def convert_continuous_data_to_discrete(samples, numvalues='auto'):

        if np.isscalar(samples[0]):
            if numvalues in ('auto', 'bayes'):
                limits, probs = JointProbabilityMatrix.probs_by_bayesian_blocks(samples)
            else:
                # note: if e.g. k=='freedman' (anything other than 'auto') then a heuristic will be used to compute k
                limits, probs = JointProbabilityMatrix.probs_by_equiprobable_bins(samples, k=numvalues)

            def convert(sample):
                return np.greater_equal(sample, limits[1:-1]).sum()

            return list(map(convert, samples))
        else:
            samples_list = np.transpose(samples)

            samples_list = list(map(JointProbabilityMatrix.convert_continuous_data_to_discrete, samples_list))

            return np.transpose(samples_list)


    @staticmethod
    def discretize_data(repeated_measurements, numvalues='auto', return_also_fitness_curve=False, maxnumvalues=20,
                        stopafterdeclines=5, method='equiprobable'):

        pdf = JointProbabilityMatrix(2, 2)  # pre-alloc, values passed are irrelevant

        # assert numvalues == 'auto', 'todo'

        timeseries = np.transpose(repeated_measurements)

        fitness = []  # tuples of (numvals, fitness)

        # print 'debug: len(rep) = %s' % len(repeated_measurements)
        # print 'debug: len(ts) = %s' % len(timeseries)
        # print 'debug: ub =', xrange(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), 100))

        if numvalues == 'auto':
            possible_numvals = range(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), maxnumvalues))
        else:
            possible_numvals = [numvalues]  # try only one

        for numvals in possible_numvals:
            if method in ('equiprobable', 'adaptive'):
                bounds = [[np.percentile(ts, p) for p in np.linspace(0, 100., numvals + 1)[:-1]] for ts in timeseries]
            elif method in ('equidistant', 'fixed'):
                bounds = [[np.min(ts) + p * np.max(ts) for p in np.linspace(0., 1., numvals + 1)[:-1]] for ts in timeseries]
            else:
                raise NotImplementedError('unknown method: %s' % method)

            # NOTE: I suspect it is not entirely correct to discretize the entire timeseries and then compute
            # likelihood of right side given left...

            disc_timeseries = [[np.sum(np.less_equal(bounds[vix], val)) - 1 for val in ts]
                               for vix, ts in enumerate(timeseries)]

            assert np.min(disc_timeseries) >= 0, 'discretized values should be non-negative'
            assert np.max(disc_timeseries) < numvals, \
                'discretized values should be max. numvals-1, max=%s, numvals=%s' % (np.max(disc_timeseries), numvals)

            # estimate pdf on FIRST HALF of the data to predict the SECOND HALF (two-way cross-validation)
            pdf.estimate_from_data(np.transpose(disc_timeseries)[:int(len(repeated_measurements)/2)], numvalues=numvals)
            fitness_nv = pdf.loglikelihood(np.transpose(disc_timeseries)[int(len(repeated_measurements)/2):])
            # correction for number of variables?
            # this is the expected likelihood for independent variables if the pdf fits perfect
            fitness_nv /= np.log(1.0/np.power(numvals, len(pdf)))

            # estimate pdf on SECOND HALF of the data to predict the FIRST HALF
            pdf.estimate_from_data(np.transpose(disc_timeseries)[int(len(repeated_measurements) / 2):], numvalues=numvals)
            fitness_nv_r = pdf.loglikelihood(np.transpose(disc_timeseries)[:int(len(repeated_measurements) / 2)])
            # correction for number of variables?
            # this is the expected likelihood for independent variables if the pdf fits perfect
            fitness_nv_r /= np.log(1.0 / np.power(numvals, len(pdf)))

            fitness_nv = fitness_nv + fitness_nv_r  # combined (avg) fitness

            fitness.append((numvals, np.transpose(disc_timeseries), fitness_nv))
            # NOTE: very inefficient to store all potential disc_timeseries
            # todo: fix that

            assert np.max(disc_timeseries) <= numvals, \
                'there should be %s values but there are %s' % (numvals, np.max(disc_timeseries))

            print('debug: processed %s, fitness=%s' % (numvals, fitness_nv))
            if fitness_nv <= 0.0:
                print('debug: disc. timeseries:', np.transpose(disc_timeseries)[:10])
                print('debug: repeated_measurements:', np.transpose(repeated_measurements)[:10])
                print('debug: bounds:', np.transpose(bounds)[:10])

            if True:
                fitness_values = [x[-1] for x in fitness]

                if len(fitness) > 7:
                    if fitness_nv < max(fitness_values) * 0.5:
                        print('debug: not going to improve, will break the for-loop')
                        break

                if len(fitness) > stopafterdeclines:
                    if list(fitness_values[-stopafterdeclines:]) == sorted(fitness_values[-stopafterdeclines:]):
                        print('debug: fitness declined %s times in a row, will stop' % stopafterdeclines)
                        break

        max_ix = np.argmax([x[-1] for x in fitness], axis=0)

        if not return_also_fitness_curve:
            return fitness[max_ix][1]
        else:
            return fitness[max_ix][1], [(x[0], x[-1]) for x in fitness]


    def generate_samples_mixed_gaussian(self, n, sigma=0.2, mu=1.):

        if np.isscalar(n):
            samples = self.generate_samples(n)  # each value is integer in 0..numvalues-1 (x below)

            numvals = self.numvalues
        else:
            assert np.ndim(n) == 2, 'expected n to be a list of samples (which are lists of [integer] values)'
            samples = copy.deepcopy(n)

            assert np.min(samples) >= 0, 'sample values are assumed integer values >=0'

            numvals = int(np.max(samples) + 1)

        numvars = np.shape(n)[1]

        # note: assumption is now that the centers and sigmas of each variable are the same
        if np.ndim(mu) < 2:
            mus = [np.arange(numvals) * mu if np.isscalar(mu) else mu for _ in range(numvars)]
        else:
            assert np.ndim(mu) == 2, 'expected mu[varix][valix]'
            mus = mu

        if np.ndim(sigma) < 2:
            sigmas = [[sigma]*numvals if np.isscalar(sigma) else sigma for _ in range(numvars)]
        else:
            assert np.ndim(sigma) == 2, 'expected sigma[varix][valix]'
            sigmas = sigma

        try:
            samples = [[np.random.normal(mus[xix][x], sigmas[xix][x])
                        for xix, x in enumerate(sample)] for sample in samples]
        except IndexError as e:
            print('debug: np.shape(mus) = %s' % str(np.shape(mus)))
            print('debug: np.shape(sigmas) = %s' % str(np.shape(sigmas)))
            print('debug: np.ndim(mus) = %s' % np.ndim(mu))
            print('debug: np.ndim(sigmas) = %s' % np.ndim(sigma))
            print('debug: np.isscalar(mus) = %s' % np.isscalar(mu))
            print('debug: np.isscalar(sigmas) = %s' % np.isscalar(sigma))
            print('debug: np.shape(samples) = %s' % str(np.shape(samples)))

            raise IndexError(e)

        return samples

    def estimate_from_data(self, repeated_measurements, numvalues='auto', discrete=True, method='empirical',
                           prob_dtype=float):
        """
        From a list of co-occurring values (one per variable) create a joint probability distribution by simply
        counting. For instance, the list [[0,0], [0,1], [1,0], [1,1]] would result in a uniform distribution of two
        binary variables.
        :param repeated_measurements: list of lists, where each sublist is of equal length (= number of variables)
        :type repeated_measurements: list of list
        :param discrete: do not change
        :param method: do not change
        """
        if not discrete and method == 'empirical':
            method = 'equiprobable'  # 'empirical' makes no sense for continuous data, so pick most sensible alternative

        assert discrete and method == 'empirical' or not discrete and method in ('equiprobable', 'equiprob'), \
            'method/discreteness combination not supported'

        assert len(repeated_measurements) > 0, 'no data given'

        numvars = len(repeated_measurements[0])

        assert numvars <= len(repeated_measurements), 'fewer measurements than variables, probably need to transpose' \
                                                      ' repeated_measurements.'

        if discrete and method == 'empirical':
            # todo: store the unique values as a member field so that later algorithms know what original values
            # corresponded to the values 0, 1, 2, ... and for estimating MI based on samples rather than 'true' dist.

            if numvalues == 'relabel':
                # we will later relabel node values such that the node values are contiguous
                all_unique_values = sorted(set(np.array(repeated_measurements).flat))

                numvals = len(all_unique_values)
            else:
                # keep the original node values (so e.g. value '3' remains '3')
                maxval = max(np.array(repeated_measurements).flat)
                if numvalues == 'auto':
                    numvals = maxval + 1
                else:
                    numvals = int(numvalues)

                    assert maxval < numvals, f'you specified {numvalues=} but this should imply that the maximum possible value is {numvalues-1}, but it is {maxval}.'

                # for v in range(numvals):
                #     if not v in all_unique_values:
                #         # add bogus values to ensure `dict_val_to_index` is constructed correctly
                #         all_unique_values.append(v)
                #     else:
                #         break
                # this ensures that `dict_val_to_index` will just be a map that doesn't change any values:
                all_unique_values = list(range(numvals))
                
                assert len(all_unique_values) == numvals, 'should be fixed by the loop above here'

            # for converting the unique values given to strictly the range 0, 1, ..., n-1 (if not already)
            dict_val_to_index = {all_unique_values[valix]: valix for valix in range(numvals)}

            new_joint_probs = np.zeros([numvals]*numvars, dtype=prob_dtype)

            # todo: when you generalize self.numvalues to an array then here also set the array instead of int

            for values in repeated_measurements:
                value_indices = tuple((dict_val_to_index[val] for val in values))

                try:
                    new_joint_probs[value_indices] += 1.0 / len(repeated_measurements)
                except IndexError as e:
                    print('error: value_indices =', value_indices)
                    print('error: type(value_indices) =', type(value_indices))

                    raise IndexError(e)

            self.reset(numvars, numvals, joint_prob_matrix=new_joint_probs)
        elif not discrete and method in ('equiprobable', 'equiprob'):
            disc_timeseries = self.discretize_data(repeated_measurements, numvalues, return_also_fitness_curve=False)

            assert np.shape(repeated_measurements) == np.shape(disc_timeseries)

            self.estimate_from_data(disc_timeseries, numvalues=numvalues, discrete=True, method='empirical')
        else:
            raise NotImplementedError('unknown combination of discrete and method.')


    def loglikelihood(self, repeated_measurements, ignore_zeros=True):
        if not ignore_zeros:
            return np.sum(np.log(list(map(self, repeated_measurements))))
        else:
            return np.sum(np.log([p for p in map(self, repeated_measurements) if p > 0.]))


    def marginal_probability(self, variables, values):

        if len(set(variables)) == len(self):
            # speedup, save marginalization step, but is functionally equivalent to else clause
            return self.joint_probability(values)
        else:
            return self[variables](values)


    def joint_probability(self, values: Sequence) -> float:
        """Return the joint probability of a configuration (value for each variable in this PMF).

        Args:
            values (Sequence): a sequence of integers. Important: values are labeled 0, 1, ..., <numvalues - 1>,
                but you can also specify e.g. -1, which means the same as <numvalues - 1>.

        Returns:
            float: _description_
        """
        assert len(values) == self.numvariables, 'should specify one value per variable'

        if len(self) == 0:
            if len(values) == 0:
                return 1
            else:
                return 0
        else:
            assert np.all(np.less(values, self.numvalues)), f'variable can only take values [0, {self.numvalues-1}] but I got {values=}'

            joint_prob = self.joint_probabilities[tuple(values)]

            assert -1e5 <= joint_prob <= 1.0 + 1e5, 'not a probability? ' + str(joint_prob)

            return joint_prob


    def get_variables_with_label_in(self, labels):
        return [vix for vix in range(self.numvariables) if self.labels[vix] in labels]


    def marginalize_distribution_retaining_only_labels(self, retained_labels):
        """
        Return a pdf of variables which have one of the labels in the retained_labels set; all other variables will
        be summed out.
        :param retained_labels: list of labels to retain; the rest will be summed over.
        :type retained_labels: sequence
        :rtype: JointProbabilityMatrix
        """

        variable_indices = [vix for vix in range(self.numvariables) if self.labels[vix] in retained_labels]

        return self.marginalize_distribution(variable_indices)


    def marginalize_distribution(self, retained_variables: Sequence[int]):
        """
        Return a pdf of only the given variable indices, summing out all others
        :param retained_variables: variables to retain, all others will be summed out and will not be a variable anymore
        :type: array_like
        :rtype : JointProbabilityMatrix
        """
        if len(retained_variables) == 0:
            return JointProbabilityMatrix(0, self.numvalues)
        else:
            assert max(retained_variables) < len(self), f'{retained_variables=} while {len(self)=}'

        lists_of_possible_states_per_variable = [list(range(self.numvalues)) for variable in range(self.numvariables)]

        assert hasattr(retained_variables, '__len__'), 'marginalize_distribution(): argument should be list or so, not int or other scalar'

        marginalized_joint_probs = np.zeros([self.numvalues]*len(retained_variables))  # pre-alloc

        # if len(variables):
        #     marginalized_joint_probs = np.array([marginalized_joint_probs])

        assert np.all(list(map(np.isscalar, retained_variables))), 'each variable identifier should be int in [0, numvalues)'
        assert len(retained_variables) <= self.numvariables, 'cannot marginalize more variables than I have'
        assert len(set(retained_variables)) <= self.numvariables, 'cannot marginalize more variables than I have'

        # not sure yet about doing this:
        # variables = sorted(list(set(variables)))  # ensure uniqueness?

        # if np.all(sorted(list(set(retained_variables))) == list(range(self.numvariables))):
        #     return self.copy()  # you ask all variables back so I have nothing to do
        # else:
        if True:  # I avoid the above test (which includes sorting) for speed
            for values in itertools.product(*lists_of_possible_states_per_variable):
                marginal_values = [values[varid] for varid in retained_variables]

                marginalized_joint_probs[tuple(marginal_values)] += self.joint_probability(values)

            if __debug__:
                np.testing.assert_almost_equal(np.sum(marginalized_joint_probs), 1.0)

            marginal_joint_pdf = JointProbabilityMatrix(len(retained_variables), self.numvalues,
                                                        joint_probs=marginalized_joint_probs)

            return marginal_joint_pdf


    # helper function
    def appended_joint_prob_matrix(self, num_added_variables, values_so_far=[], added_joint_probabilities=None):
        if len(values_so_far) == self.numvariables:
            joint_prob_values = self.joint_probability(values_so_far)

            # submatrix must sum up to joint probability
            if added_joint_probabilities is None:
                # todo: does this add a BIAS? for a whole joint pdf it does, but not sure what I did here (think so...)
                added_joint_probabilities = np.array(np.random.random([self.numvalues]*num_added_variables),
                                                     dtype=self._type_prob)
                added_joint_probabilities /= np.sum(added_joint_probabilities)
                added_joint_probabilities *= joint_prob_values

                assert joint_prob_values <= 1.0
            else:
                if __debug__:
                    np.testing.assert_almost_equal(np.sum(added_joint_probabilities), joint_prob_values)

                assert np.ndim(added_joint_probabilities) == num_added_variables
                assert len(added_joint_probabilities[0]) == self.numvalues
                assert len(added_joint_probabilities[-1]) == self.numvalues

            return list(added_joint_probabilities)
        elif len(values_so_far) < self.numvariables:
            if len(values_so_far) > 0:
                return [self.appended_joint_prob_matrix(num_added_variables,
                                                        values_so_far=list(values_so_far) + [val],
                                                        added_joint_probabilities=added_joint_probabilities)
                        for val in range(self.numvalues)]
            else:
                # same as other case but np.array converted, since the joint pdf matrix is always expected to be that
                return np.array([self.appended_joint_prob_matrix(num_added_variables,
                                                               values_so_far=list(values_so_far) + [val],
                                                               added_joint_probabilities=added_joint_probabilities)
                                 for val in range(self.numvalues)])
        else:
            raise RuntimeError('should not happen?')


    def append_variables(self, num_added_variables, added_joint_probabilities=None):
        assert num_added_variables > 0

        if isinstance(added_joint_probabilities, JointProbabilityMatrix) \
                or isinstance(added_joint_probabilities, NestedArrayOfProbabilities):
            added_joint_probabilities = added_joint_probabilities.joint_probabilities

        new_joint_pdf = self.appended_joint_prob_matrix(num_added_variables,
                                                        added_joint_probabilities=added_joint_probabilities)

        if __debug__:
            assert np.ndim(new_joint_pdf) == self.numvariables + num_added_variables
            if self.numvariables + num_added_variables >= 1:
                assert len(new_joint_pdf[0]) == self.numvalues
                assert len(new_joint_pdf[-1]) == self.numvalues
            np.testing.assert_almost_equal(np.sum(new_joint_pdf), 1.0)

        self.reset(self.numvariables + num_added_variables, self.numvalues, new_joint_pdf)


    def append_variables_using_state_transitions_table(self, state_transitions):
        """
        Append one or more stochastic variables to this joint pdf, whose conditional pdf is defined by the provided
        'state transitions table'. In the rows of this table the first <self.numvariables> values are the values for
        the pre-existing stochastic variables; the added values are taken to be the deterministically determined
        added variable values, i.e., Pr(appended_vars = X, current_vars) = Pr(current_vars) so that
        Pr(appended_vars = X | current_vars) = 1 for whatever X you appended, where X is a list of values.
        :param state_transitions: list of lists, where each sublist is of
        length self.numvariables + [num. new variables] and is a list of values, each value in [0, numvalues),
        where the first self.numvariables are the existing values ('input') and the remaining are the new variables'
        values ('output').

        Can also provide a function f(values, num_values) which returns a list of values for the to-be-appended
        stochastic variables, where the argument <values> is a list of values for the existing variables (length
        self.numvariables).
        :type state_transitions: list or function
        """

        lists_of_possible_given_values = [list(range(self.numvalues)) for _ in range(self.numvariables)]

        if hasattr(state_transitions, '__call__'):
            state_transitions = [list(existing_vars_values) + list(state_transitions(existing_vars_values,
                                                                                     self.numvalues))
                                 for existing_vars_values in itertools.product(*lists_of_possible_given_values)]

        extended_joint_probs = np.zeros([self.numvalues]*len(state_transitions[0]))

        # todo this for debugging? cycle through all possible values for self.numvariables and see if it is present
        # in the state_transitions
        # lists_of_possible_states_per_variable = [range(self.numvalues) for variable in xrange(self.numvariables)]

        # one row should be included for every possible set of values for the pre-existing stochastic variables
        assert len(state_transitions) == np.power(self.numvalues, self.numvariables)

        for states_row in state_transitions:
            assert len(states_row) > self.numvariables, 'if appending then more than self.numvariables values ' \
                                                        'should be specified'
            assert len(states_row) == len(state_transitions[0]), 'not all state rows of equal length; ' \
                                                                 'appending how many variables? Make up your mind. '

            # probability that the <self.numvariables> of myself have the values <state_transitions[:self.numvariables]>
            curvars_prob = self(states_row[:self.numvariables])

            assert 0.0 <= curvars_prob <= 1.0, 'probability not in 0-1'

            # set Pr(appended_vars = X | current_vars) = 1 for one set of values for the appended variables (X) and 0
            # otherwise (which is already by default), so I setting here
            # Pr(appended_vars = X, current_vars) = Pr(current_vars)
            extended_joint_probs[tuple(states_row)] = curvars_prob

        if __debug__:
            assert np.ndim(extended_joint_probs) == len(state_transitions[0])
            if len(state_transitions[0]) > 1:
                assert len(extended_joint_probs[0]) == self.numvalues
                assert len(extended_joint_probs[-1]) == self.numvalues
            np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

        self.reset(len(state_transitions[0]), self.numvalues, extended_joint_probs)


    def reverse_reordering_variables(self, variables):

        varlist = list(variables)
        numvars = self.numvariables

        # note: if a ValueError occurs then you did not specify every variable index in <variables>, like
        # [1,2] instead of [1,2,0].
        reverse_ordering = [varlist.index(ix) for ix in range(numvars)]

        self.reorder_variables(reverse_ordering)



    def reorder_variables(self, variables):
        """
        Reorder the variables, for instance if self.numvariables == 3 then call with variables=[2,1,0] to reverse the
        order of the variables. The new joint probability matrix will be determined completely by the old matrix.
        It is also possible to duplicate variables, e.g. variables=[0,1,2,2] to duplicate the last variable (but
        not sure if that is what you want, it will simply copy joint probs, so probably not).
        :param variables: sequence of int
        """
        assert len(variables) >= self.numvariables, 'I cannot reorder if you do\'nt give me the new ordering completely'
        # note: the code is mostly written to support also duplicating a variable

        num_variables_new = len(variables)

        joint_probs_new = np.zeros([self.numvalues]*num_variables_new) - 1

        lists_of_possible_states_per_variable = [list(range(self.numvalues)) for _ in range(num_variables_new)]

        for values_new in itertools.product(*lists_of_possible_states_per_variable):
            values_old_order = [-1]*self.numvariables

            for new_varix in range(len(variables)):
                assert variables[new_varix] < self.numvariables, 'you specified the order of a variable index' \
                                                                 ' >= N (non-existent)'

                # can happen if a variable index is mentioned twice or more in 'variables' but in the current 'values_new'
                # they have different values. This is of course not possible, the two new variables should be equivalent
                # and completely redundant, so then I will set this joint prob. to zero and continue
                if values_old_order[variables[new_varix]] >= 0 \
                        and values_old_order[variables[new_varix]] != values_new[new_varix]:
                    assert len(variables) > self.numvariables, 'at least one variable index should be mentioned twice'

                    joint_probs_new[tuple(values_new)] = 0.0

                    break
                else:
                    # normal case
                    values_old_order[variables[new_varix]] = values_new[new_varix]

            if joint_probs_new[tuple(values_new)] != 0.0:
                assert not -1 in values_old_order, 'missing a variable index in variables=' + str(variables) \
                                                   + ', how should I reorder if you don\'t specify the new order of all' \
                                                   +  ' variables'

                assert joint_probs_new[tuple(values_new)] == -1, 'should still be unset element of joint prob matrix'

                joint_probs_new[tuple(values_new)] = self(values_old_order)
            else:
                pass  # joint prob already set to 0.0 in the above inner loop

        assert not -1 in joint_probs_new, 'not all new joint probs were computed'

        # change myself to the new joint pdf; will also check for being normalized etc.
        self.reset(num_variables_new, self.numvalues, joint_probs_new)


    def __eq__(self, other):  # approximate to 7 decimals
        if self.numvariables != other.numvariables or self.numvalues != other.numvalues:
            return False
        else:
            try:
                np.testing.assert_array_almost_equal(self.joint_probabilities, other.joint_probabilities)

                return True
            except AssertionError as e:
                assert 'not almost equal' in str(e), 'don\'t know what other assertion could have failed'

                return False


    # todo: implement __sub__, but for this it seems necessary to give each variable a unique ID at creation (__init__)
    # and keep track of them as you do operations such as marginalizing. Otherwise, subtraction is ambiguous,
    # i.e., subtracting a 2-variable pdf from a 5-variable pdf should result in a 3-variable conditional pdf, but
    # it is not clear which 3 variables should be chosen. Can choose this to always be the first 3 and let the
    # user be responsible to reorder them before subtraction, but seems prone to error if user does not know this or
    # forgets? Well can test equality by marginalizing the 5-variable pdf...
    def __sub__(self, other):
        assert len(self) >= len(other), 'cannot compute a conditional pdf consisting of a negative number of variables'

        if len(self) == len(other):
            assert self[list(range(len(other)))] == other, 'my first ' + str(len(other)) + ' variables are not the same as ' \
                                                                                     'that of \'other\''

            return JointProbabilityMatrix(0, self.numvalues)  # return empty pdf...
        elif len(self) > len(other):
            assert self[list(range(len(other)))] == other, 'my first ' + str(len(other)) + ' variables are not the same as ' \
                                                                                     'that of \'other\''

            return self.conditional_probability_distributions(list(range(len(other))))
        else:
            raise ValueError('len(self) < len(other), '
                             'cannot compute a conditional pdf consisting of a negative number of variables')


    # todo: implement __setitem__ for either pdf or cond. pdfs


    def __len__(self):
        return self.numvariables


    def __getitem__(self, item):
        # if item == 'all':
        #     return self
        # elif not hasattr(item, '__iter__'):
        #     return self.marginalize_distribution([item])
        # else:
        #     return self.marginalize_distribution(item)

        # note: the above might make the function slow to run, so made it simpler (ONLY ONE ITEM):
        assert not hasattr(item, '__iter__'), '__getitem__ only takes a single index, not a list; ' \
            + 'use marginalize_distribution() directly if you want to use a list of multiple indices.'
        return self.marginalize_distribution([item])


    def __iadd__(self, other):
        """

        :param other: can be JointProbabilityMatrix or a conditional distribution (dict of JointProbabilityMatrix)
        :type other: JointProbabilityMatrix or ConditionalProbabilities or dict
        """

        self.append_variables_using_conditional_distributions(other)


    def matrix2vector(self):
        return self.joint_probabilities.flatten()


    def vector2matrix(self, list_probs, clip=False):
        if __debug__:
            np.testing.assert_almost_equal(np.sum(list_probs), 1.0)

        assert np.ndim(list_probs) == 1

        self.joint_probabilities.reset(np.reshape(list_probs, [self.numvalues]*self.numvariables))

        if clip:
            self.clip_all_probabilities()


    def params2matrix(self, parameters):
        assert len(parameters) == np.power(self.numvalues, self.numvariables) - 1

        vector_probs = [-1.0]*(np.power(self.numvalues, self.numvariables))

        remaining_prob_mass = 1.0

        for pix in range(len(parameters)):
            # note: small rounding errors will be fixed below by clipping
            assert -0.000001 <= parameters[pix] <= 1.000001, 'parameters should be in [0, 1]: ' + str(parameters[pix])

            # clip the parameter to the allowed range. If a rounding error is fixed by this in the parameters then
            # possibly a rounding error will appear in the probabilities?... Not sure though
            parameters[pix] = min(max(parameters[pix], 0.0), 1.0)

            vector_probs[pix] = remaining_prob_mass * parameters[pix]

            remaining_prob_mass = remaining_prob_mass * (1.0 - parameters[pix])

        assert vector_probs[-1] < 0.0, 'should still be unset by the above loop'

        # last parameter is irrelevant, must always be 1.0 is also a way to look at it
        vector_probs[-1] = remaining_prob_mass

        if __debug__:
            np.testing.assert_almost_equal(np.sum(vector_probs), 1.0)

        self.vector2matrix(vector_probs)


    def from_params(self, parameters):  # alternative constructor
        self.params2matrix(parameters)

        return self


    def matrix2params(self):
        vector_probs = self.matrix2vector()

        remaining_prob_mass = 1.0

        parameters = [-1.0]*(len(vector_probs) - 1)

        if __debug__:
            np.testing.assert_almost_equal(np.sum(vector_probs), 1.0)

        for pix in range(len(parameters)):
            if remaining_prob_mass > 0:
                assert remaining_prob_mass <= 1.0, 'remaining prob mass: ' + str(remaining_prob_mass)
                assert vector_probs[pix] <= remaining_prob_mass + 0.00001, \
                    'vector_probs[pix]=' + str(vector_probs[pix]) + ', remaining_prob_mass=' + str(remaining_prob_mass)

                parameters[pix] = vector_probs[pix] / remaining_prob_mass

                assert -0.1 <= parameters[pix] <= 1.1, \
                    'parameters should be in [0, 1]: ' + str(parameters[pix]) \
                    + ', sum probs = ' + str(np.sum(self.joint_probabilities.joint_probabilities))

                # sometimes this happens I think due to rounding errors, but when I sum the probabilities they
                # still seem to sum to exactly 1.0 so probably is due to some parameters being 0 or 1, so clip here
                # parameters[pix] = max(min(parameters[pix], 1.0), 0.0)  # is already below
            elif remaining_prob_mass == 0:
                parameters[pix] = 0
            else:
                if not remaining_prob_mass > -0.000001:
                    print('debug: remaining_prob_mass =', remaining_prob_mass)
                    print('debug: pix =', pix, 'out of', len(parameters))

                    # todo: if just due to floating operation error, so small, then clip to zero and go on?
                    raise ValueError('remaining_prob_mass = ' + str(remaining_prob_mass)
                                     + ' < 0, which should not happen?')
                else:
                    # seems that it was intended to reach zero but due to floating operation roundoff it got just
                    # slightly under. Clip to 0 will do the trick.
                    remaining_prob_mass = 0.0  # clip to zero, so it will stay that way

                parameters[pix] = 0  # does not matter

                assert -0.1 <= parameters[pix] <= 1.1, \
                    'parameters should be in [0, 1]: ' + str(parameters[pix]) \
                    + ', sum probs = ' + str(np.sum(self.joint_probabilities.joint_probabilities))

            # sometimes this happens I think due to rounding errors, but when I sum the probabilities they
            # still seem to sum to exactly 1.0 so probably is due to some parameters being 0 or 1, so clip here
            parameters[pix] = max(min(parameters[pix], 1.0), 0.0)

            # parameters[pix] = min(max(parameters[pix], 0.0), 1.0)

            remaining_prob_mass -= remaining_prob_mass * parameters[pix]

        return parameters


    def __add__(self, other):
        """
        Append the variables defined by the (conditional) distributions in other.
        :type other: dict of JointProbabilityMatrix | JointProbabilityMatrix
        :rtype: JointProbabilityMatrix
        """

        pdf = self.copy()
        pdf.append_variables_using_conditional_distributions(other)

        return pdf

    def matrix2params_incremental(self, return_flattened=True, verbose=False):
        if self.numvariables > 1:
            # get the marginal pdf for the first variable
            pdf1 = self.marginalize_distribution([0])

            # first sequence of parameters, rest is added below here
            parameters = pdf1.matrix2params()

            pdf_conds = self.conditional_probability_distributions([0])

            # assert len(pdf_conds) == self.numvalues, 'should be one pdf for each value of first variable'

            for val in range(self.numvalues):
                pdf_cond = pdf_conds[tuple([val])]

                added_params = pdf_cond.matrix2params_incremental(return_flattened=False, verbose=verbose)

                if verbose > 0:
                    print('debug: matrix2params_incremental: recursed: for val=' + str(val) + ' I got added_params=' \
                          + str(added_params) + '.')
                    print('debug: matrix2params_incremental: old parameters =', parameters)

                # instead of returning a flat list of parameters I make it nested, so that the structure (e.g. number of
                # variables and number of values) can be inferred, and also hopefully it can be inferred to which
                # variable which parameters belong.
                # CHANGE123
                parameters.append(added_params)

                if verbose > 0:
                    print('debug: matrix2params_incremental: new parameters =', parameters)

            if return_flattened:
                # flatten the tree structure to a list of scalars, which is sorted on the variable id
                parameters = self.scalars_up_to_level(parameters)

            return parameters
        elif self.numvariables == 1:
            return self.matrix2params()
        else:
            raise ValueError('no parameters for 0 variables')

    _debug_params2matrix = False  # internal variable, used to debug a debug statement, can be removed in a while


    # todo: should this be part of e.g. FullNestedArrayOfProbabilities instead of this class?
    def params2matrix_incremental(self, parameters):
        """
        Takes in a row of floats in range [0.0, 1.0] and changes <self> to a new PDF which is characterized by the
        parameters. Benefit: np.random.random(M**N - 1) results in an unbiased sample of PDF, wnere M is numvalues
        and N is numvariables.
        :param parameters: list of floats, length equal to what matrix2params_incrmental() returns (M**N - 1)
        :type parameters: list of float
        """
        if __debug__:
            # store the original provided list of scalars
            original_parameters = list(parameters)

        # I suspect that both a tree-like input and a list of scalars should work... (add to unit test?)
        if np.all(list(map(np.isscalar, parameters))):
            assert min(parameters) > -0.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                              + str(parameters)
            assert min(parameters) < 1.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                              + str(parameters)

            # clip each parameter value to the allowed range. above I check already whether the error is not too large
            parameters = [min(max(pi, 0.0), 1.0) for pi in parameters]

            parameters = self.imbalanced_tree_from_scalars(parameters, self.numvalues)

            # verify that the procedure to make the tree out of the list of scalars is reversible and correct
            # (looking for bug)
            if __debug__ and self._debug_params2matrix:
                original_parameters2 = self.scalars_up_to_level(parameters)

                np.testing.assert_array_almost_equal(original_parameters, original_parameters2)

        if self.numvariables > 1:
            # first (numvalues - 1) values in the parameters tree structure should be scalars, as they will be used
            # to make the first variable's marginal distribution
            assert np.all(list(map(np.isscalar, parameters[:(self.numvalues - 1)])))

            ### start already by setting the pdf of the first variable...

            pdf_1 = JointProbabilityMatrix(1, self.numvalues)
            pdf_1.params2matrix(parameters[:(len(pdf_1.joint_probabilities.flatiter()) - 1)])

            assert (len(pdf_1.joint_probabilities.flatiter()) - 1) == (self.numvalues - 1), 'assumption directly above'

            assert len(pdf_1.joint_probabilities.flatiter()) == self.numvalues

            assert len(list(flatten(parameters))) == len(self.joint_probabilities.flatiter()) - 1, \
                'more or fewer parameters than needed: ' \
                  'need ' + str(len(self.joint_probabilities.flatiter()) - 1) + ', got ' + str(len(list(flatten(parameters)))) \
                  + '; #vars, #vals = ' + str(self.numvariables) + ', ' + str(self.numvalues)

            if __debug__ and self._debug_params2matrix:
                # remove this (expensive) check after it seems to work a few times?
                # note: for the conditions of no 1.0 or 0.0 prior probs, see the note in params2matrix_incremental
                if not 0.0 in pdf_1.matrix2params() and not 1.0 in pdf_1.matrix2params():
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params(),
                                                         self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params_incremental(),
                                                         self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))

            # remove the used parameters from the list
            parameters = parameters[(len(pdf_1.joint_probabilities.flatiter()) - 1):]
            assert len(parameters) == self.numvalues  # one subtree per conditional pdf

            pdf_conds = dict()

            ### now add other variables...

            for val in range(self.numvalues):
                # set this conditional pdf recursively as defined by the next sequence of parameters
                pdf_cond = JointProbabilityMatrix(self.numvariables - 1, self.numvalues)

                # note: parameters[0] is a sublist
                assert not np.isscalar(parameters[0])

                assert not np.isscalar(parameters[0])

                # todo: changing the parameters list is not necessary, maybe faster if not?

                # pdf_cond.params2matrix_incremental(parameters[:(len(pdf_cond.joint_probabilities.flatiter()) - 1)])
                pdf_cond.params2matrix_incremental(parameters[0])

                # conditional pdf should have the same set of parameters as the ones I used to create it
                # (todo: remove this expensive check if it seems to work for  while)
                if self._debug_params2matrix and __debug__:  # seemed to work for a long time...
                    try:
                        if np.random.randint(20) == 0:
                            np.testing.assert_array_almost_equal(pdf_cond.matrix2params_incremental(),
                                                                 self.scalars_up_to_level(parameters[0]))
                    except AssertionError as e:
                        # print 'debug: parameters[0] =', parameters[0]
                        # print 'debug: len(pdf_cond) =', len(pdf_cond)
                        # print 'debug: pdf_cond.joint_probabilities =', pdf_cond.joint_probabilities

                        pdf_1_duplicate1 = pdf_cond.copy()
                        pdf_1_duplicate2 = pdf_cond.copy()

                        pdf_1_duplicate1._debug_params2matrix = False  # prevent endless recursion
                        pdf_1_duplicate2._debug_params2matrix = False  # prevent endless recursion

                        pdf_1_duplicate1.params2matrix_incremental(self.scalars_up_to_level(parameters[0]))
                        pdf_1_duplicate2.params2matrix_incremental(pdf_cond.matrix2params_incremental())

                        pdf_1_duplicate1._debug_params2matrix = True
                        pdf_1_duplicate2._debug_params2matrix = True

                        assert pdf_1_duplicate1 == pdf_cond
                        assert pdf_1_duplicate2 == pdf_cond

                        del pdf_1_duplicate1, pdf_1_duplicate2

                        # note: the cause seems to be as follows. If you provide the parameters e.g.
                        # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                        # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                        # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                        # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                        # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                        # parameter values I may very well get a different parameter list from that pdf, even though
                        # the pdf built is the one intended. I don't see a way around this because even if this class
                        # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                        # but then still a user or optimization procedure can provide any list of parameters, so
                        # then also the uniformized parameter list will differ from the user-supplied.

                        # raise AssertionError(e)

                        # later add check. If this check fails then for sure there is something wrong. See also the
                        # original check below.
                        assert 0.0 in self.scalars_up_to_level(parameters) or \
                               1.0 in self.scalars_up_to_level(parameters), 'see story above. ' \
                                                                               'self.scalars_up_to_level(parameters) = ' \
                                                                               + str(self.scalars_up_to_level(parameters))

                        # original check. This check failed once, but the idea was to see if there are 0s or 1s in the
                        # prior probability distribution, which precedes the conditional probability distribution for which
                        # apparently the identifying parameter values have changed. But maybe I am wrong in that
                        # parameters[0] is not the prior only, and some prior prob. information is in all of parameters,
                        # I am not sure anymore so I added the above check to see whether that one is hit instead
                        # of this one (so above check is of course more stringent than this one....)
                        # assert 0.0 in self.scalars_up_to_level(parameters[0]) or \
                        #        1.0 in self.scalars_up_to_level(parameters[0]), 'see story above. ' \
                        #                                                        'self.scalars_up_to_level(parameters[0]) = ' \
                        #                                                        + str(self.scalars_up_to_level(parameters[0]))

                if __debug__:
                    np.testing.assert_almost_equal(pdf_cond.joint_probabilities.sum(), 1.0)

                parameters = parameters[1:]

                # add the conditional pdf
                pdf_conds[(val,)] = pdf_cond.copy()

            assert len(parameters) == 0, 'all parameters should be used to construct joint pdf'

            pdf_1.append_variables_using_conditional_distributions(pdf_conds)

            if __debug__ and self._debug_params2matrix:
                # remove this (expensive) check after it seems to work a few times?
                try:
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params_incremental(),
                                                         self.scalars_up_to_level(original_parameters))
                except AssertionError as e:
                    ### I have the hunch that the above assertion is hit but that it is only if a parameter is 1 or 0,
                    ### so that the parameter may be different but that it does not matter. still don't understand
                    ### why it happens though...

                    pdf_1_duplicate = pdf_1.copy()

                    pdf_1_duplicate._debug_params2matrix = False  # prevent endless recursion

                    pdf_1_duplicate.params2matrix_incremental(self.scalars_up_to_level(original_parameters))

                    pdf_1_duplicate._debug_params2matrix = True

                    if not pdf_1_duplicate == pdf_1:
                        print('error: the pdfs from the two different parameter lists are also not equivalent')

                        del pdf_1_duplicate

                        raise AssertionError(e)
                    else:
                        # warnings.warn('I found that two PDF objects can have the same joint prob. matrix but a'
                        #               ' different list of identifying parameters. This seems to be due to a variable'
                        #               ' having 0.0 probability on a certain value, making the associated conditional'
                        #               ' PDF of other variables 0 and therefore those associated parameters irrelevant.'
                        #               ' Find a way to make these parameters still uniform? Seems to happen in'
                        #               ' "pdf_1.append_variables_using_conditional_distributions(pdf_conds)"...')

                        # note: (duplicated) the cause seems to be as follows. If you provide the parameters e.g.
                        # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                        # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                        # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                        # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                        # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                        # parameter values I may very well get a different parameter list from that pdf, even though
                        # the pdf built is the one intended. I don't see a way around this because even if this class
                        # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                        # but then still a user or optimization procedure can provide any list of parameters, so
                        # then also the uniformized parameter list will differ from the user-supplied.

                        del pdf_1_duplicate

            assert pdf_1.numvariables == self.numvariables
            assert pdf_1.numvalues == self.numvalues

            self.duplicate(pdf_1)  # make this object (self) be the same as pdf_1
        elif self.numvariables == 1:
            self.params2matrix(parameters)
        else:
            assert len(parameters) == 0, 'at the least 0 parameters should be given for 0 variables...'

            raise ValueError('no parameters for 0 variables')


    def duplicate(self, other_joint_pdf):
        """

        :type other_joint_pdf: JointProbabilityMatrix
        """
        self.reset(other_joint_pdf.numvariables, other_joint_pdf.numvalues, other_joint_pdf.joint_probabilities)


    def reset(self, numvariables, numvalues, joint_prob_matrix, labels=None):
        """
        This function is intended to completely reset the object, so if you add variables which determine the
        behavior of the object then add them also here and everywhere where called.
        :type numvariables: int
        :type numvalues: int
        :type joint_prob_matrix: NestedArrayOfProbabilities
        """

        self.numvariables = numvariables
        self.numvalues = numvalues
        if type(joint_prob_matrix) == np.ndarray:
            self.joint_probabilities.reset(np.array(joint_prob_matrix, dtype=self._type_prob))
        else:
            self.joint_probabilities.reset(joint_prob_matrix)

        if labels is None:
            self.labels = [_default_variable_label]*numvariables
        else:
            self.labels = labels

        # assert np.ndim(joint_prob_matrix) == self.numvariables, 'ndim = ' + str(np.ndim(joint_prob_matrix)) + ', ' \
        #                                                         'self.numvariables = ' + str(self.numvariables) + ', ' \
        #                                                         'joint matrix = ' + str(joint_prob_matrix)
        assert self.joint_probabilities.num_variables() == self.numvariables, \
            'ndim = ' + str(np.ndim(joint_prob_matrix)) + ', ' \
                                                          'self.numvariables = ' + str(self.numvariables) + ', ' \
                                                                'joint matrix = ' + str(self.joint_probabilities.joint_probabilities) \
            + ', joint_probabilities.num_variables() = ' + str(self.joint_probabilities.num_variables())
        assert self.joint_probabilities.num_values() == self.numvalues

        # # maybe this check should be removed, it is also checked in clip_all_* below, but after clipping, which
        # # may be needed to get this condition valid again?
        # if np.random.random() < 0.01:  # make less frequent
        #     np.testing.assert_array_almost_equal(np.sum(joint_prob_matrix), 1.0)

        # I removed this clipping because it takes computation time, I replace it by two tests below
        # self.clip_all_probabilities()

        assert np.max(self.joint_probabilities.joint_probabilities) <= 1. + _prob_error_tol
        assert np.min(self.joint_probabilities.joint_probabilities) >= -_prob_error_tol
        np.testing.assert_almost_equal(np.sum(self.joint_probabilities.joint_probabilities), 1.0, decimal=_prob_tol_decimal)


    def statespace(self, variables: Sequence[int] = None, only_size=False):
        if variables is None:  # -1 indicates 'all'
            variables = range(self.numvariables)
        
        assert hasattr(variables, '__contains__'), '\'variables\' should be an integer or a list/set/tuple/array (something acting as a set)'
        
        lists_of_possible_joint_values = [list(range(self.numvalues)) for _ in variables]

        if only_size:
            return int(np.product(list(map(len, lists_of_possible_joint_values))))
        else:
            return itertools.product(*lists_of_possible_joint_values)


    def num_states(self):
        return self.numvalues ** self.numvariables


    def append_redundant_variables(self, num_redundant_variables):
        self.append_variables_using_state_transitions_table(lambda my_values_tuple, numvals:
                                                            [sum(my_values_tuple) % numvals]*num_redundant_variables)


    def append_copy_of(self, variables):
        self.append_variables_using_state_transitions_table(lambda vals, nv: vals)
        # note to self: p(A)*p(B|A) == p(B)*p(A|B) leads to: p(A)/p(B) == p(A|B)/p(B|A)
        #   -->  p(A)/p(B) == [p(A)*p(B|A)/p(B)]/p(B|A)


    def append_nudged(self, variables, epsilon=0.01, num_repeats=5, verbose=False):
        pdf_P = self[variables]
        pdf_P_copy = pdf_P.copy()
        nudge = pdf_P_copy.nudge(list(range(len(variables))), [], epsilon=epsilon)
        desired_marginal_probabilities = pdf_P_copy.joint_probabilities.joint_probabilities

        # a global variable used by the cost_func() to use for quantifying the cost, without having to re-create a new
        # pdf object every time
        pdf_new = pdf_P.copy()
        pdf_new.append_variables(len(variables))

        max_mi = pdf_P.entropy()

        def cost_func(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))
            pdf_P_nudged = pdf_new[list(range(int(len(pdf_new) / 2), len(pdf_new)))]  # marginalize the nudged version
            prob_diffs = pdf_P_nudged.joint_probabilities.joint_probabilities - desired_marginal_probabilities

            cost_marginal = np.sum(np.power(prob_diffs, 2))
            # percentage reduction in MI, to try to bring to the same order of magnitude as cost_marginal
            cost_mi = (max_mi - pdf_new.mutual_information(list(range(int(len(pdf_new) / 2))),
                                                           list(range(int(len(pdf_new) / 2), len(pdf_new))))) / max_mi
            cost_mi = epsilon * np.power(cost_mi, 2)
            return cost_marginal + cost_mi

        pdf_P.append_optimized_variables(len(variables), cost_func, num_repeats=num_repeats, verbose=verbose)

        # get the actually achieved marginal probabilities of the append variables, which are hopefully close to
        # desired_marginal_probabilities
        obtained_marginal_probabilities = pdf_P[list(range(len(variables), 2*len(variables)))].joint_probabilities.joint_probabilities
        # the array obtained_nudge is hopefully very close to ``nudge''
        obtained_nudge = obtained_marginal_probabilities - pdf_P[list(range(len(variables)))].joint_probabilities.joint_probabilities

        cond_P_nudged = pdf_P.conditional_probability_distributions(list(range(len(variables))),
                                                                    list(range(len(variables), 2*len(variables))))

        self.append_variables_using_conditional_distributions(cond_P_nudged, variables)

        return nudge, obtained_nudge


    def append_nudged_direct(self, variables, epsilon=0.01):
        self.append_copy_of(variables)
        nudge = self.nudge(variables, np.setdiff1d(list(range(len(self))), variables), epsilon=epsilon)

        print('debug: MI:', self.mutual_information(variables, list(range(len(self) - len(variables), len(self)))))

        ordering = list(range(len(self)))
        for vix in range(len(variables)):
            ordering[variables[vix]] = ordering[len(self) - len(variables) + vix]
            ordering[len(self) - len(variables) + vix] = variables[vix]

        self.reorder_variables(ordering)

        return nudge


    def append_independent_variables(self, joint_pdf):
        """

        :type joint_pdf: JointProbabilityMatrix
        """
        assert not type(joint_pdf) in (int, float, str), 'should pass a JointProbabilityMatrix object'

        self.append_variables_using_conditional_distributions({(): joint_pdf})


    def append_variables_using_conditional_distributions(self, cond_pdf, given_variables=None):
        """


        :param cond_pdf: dictionary of JointProbabilityMatrix objects, one for each possible set of values for the
        existing <self.numvariables> variables. If you provide a dict with one empty tuple as key then it is
        equivalent to just providing the pdf, or calling append_independent_variables with that joint pdf. If you
        provide a dict with tuples as keys which have fewer values than self.numvariables then the dict will be
        duplicated for all possible values for the remaining variables, and this function will call itself recursively
        with complete tuples.
        :type cond_pdf dict of tuple or JointProbabilityMatrix or ConditionalProbabilities
        :param given_variables: If you provide a dict with tuples as keys which have fewer values than
        self.numvariables then it is assumed that the tuples specify the values for the consecutive highest indexed
        variables (so always until variable self.numvariables-1). Unless you specify this list, whose length should
        be equal to the length of the tuple keys in pdf_conds
        :type given_variables: list of int
        """

        if type(cond_pdf) == dict:
            cond_pdf = ConditionalProbabilityMatrix(cond_pdf)  # make into conditional pdf object

        if isinstance(cond_pdf, JointProbabilityMatrix):
            cond_pdf = ConditionalProbabilityMatrix({tuple(my_values): cond_pdf for my_values in self.statespace()})
        elif isinstance(cond_pdf, ConditionalProbabilities):
            # num_conditioned_vars = len(cond_pdf.keys()[0])
            num_conditioned_vars = cond_pdf.num_given_variables()

            assert num_conditioned_vars <= len(self), 'makes no sense to condition on more variables than I have?' \
                                                      ' %s <= %s' % (num_conditioned_vars, len(self))

            if num_conditioned_vars < len(self):
                # the provided conditional pdf conditions on fewer variables (m) than I currently exist of, so appending
                # the variables is strictly speaking ill-defined. What I will do is assume that the conditional pdf
                # is conditioned on the last m variables, and simply copy the same conditional pdf for the first
                # len(self) - m variables, which implies independence.

                pdf_conds_complete = ConditionalProbabilityMatrix()

                num_independent_vars = len(self) - num_conditioned_vars

                if __debug__:
                    # finding bug, check if all values of the dictionary are pdfs
                    for debug_pdf in cond_pdf.itervalues():
                        assert isinstance(debug_pdf, JointProbabilityMatrix), 'debug_pdf = ' + str(debug_pdf)

                statespace_per_independent_variable = [list(range(self.numvalues))
                                                       for _ in range(num_independent_vars)]

                if given_variables is None:
                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({(tuple(indep_vals) + tuple(key)): value
                                                   for key, value in cond_pdf.items()})
                else:
                    assert len(given_variables) == len(next(iter(cond_pdf.iterkeys()))), \
                        'if conditioned on ' + str(len(given_variables)) + 'then I also expect a conditional pdf ' \
                        + 'which conditions on ' + str(len(given_variables)) + ' variables.'

                    not_given_variables = np.setdiff1d(list(range(self.numvariables)), given_variables)

                    assert len(not_given_variables) + len(given_variables) == self.numvariables

                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({tuple(apply_permutation(indep_vals + tuple(key),
                                                                           list(not_given_variables)
                                                                           + list(given_variables))): value
                                                   for key, value in cond_pdf.iteritems()})

                assert len(next(iter(pdf_conds_complete.iterkeys()))) == len(self)
                assert pdf_conds_complete.num_given_variables() == len(self)

                # recurse once with a complete conditional pdf, so this piece of code should not be executed again:
                self.append_variables_using_conditional_distributions(cond_pdf=pdf_conds_complete)

                return

        assert isinstance(cond_pdf, ConditionalProbabilities)

        num_added_variables = cond_pdf[(0,)*self.numvariables].numvariables

        assert cond_pdf.num_output_variables() == num_added_variables  # checking after refactoring

        # assert num_added_variables > 0, 'makes no sense to append 0 variables?'
        if num_added_variables == 0:
            return  # nothing needs to be done, 0 variables being added

        assert self.numvalues == cond_pdf[(0,)*self.numvariables].numvalues, 'added variables should have same #values'

        # see if at the end, the new joint pdf has the expected list of identifying parameter values
        if __debug__ and len(self) == 1:
            _debug_parameters_before_append = self.matrix2params()

            # note: in the loop below the sublists of parameters will be added

        statespace_per_variable = [list(range(self.numvalues))
                                          for _ in range(self.numvariables + num_added_variables)]

        extended_joint_probs = np.zeros([self.numvalues]*(self.numvariables + num_added_variables))

        for values in itertools.product(*statespace_per_variable):
            existing_variables_values = values[:self.numvariables]
            added_variables_values = values[self.numvariables:]

            assert len(added_variables_values) == cond_pdf[tuple(existing_variables_values)].numvariables, 'error: ' \
                    'len(added_variables_values) = ' + str(len(added_variables_values)) + ', cond. numvariables = ' \
                    '' + str(cond_pdf[tuple(existing_variables_values)].numvariables) + ', len(values) = ' \
                    + str(len(values)) + ', existing # variables = ' + str(self.numvariables) + ', ' \
                    'num_added_variables = ' + str(num_added_variables)

            prob_existing = self(existing_variables_values)
            prob_added_cond_existing = cond_pdf[tuple(existing_variables_values)](added_variables_values)

            assert 0.0 <= prob_existing <= 1.0 + _mi_error_tol, f'prob not normalized: {prob_existing=}'
            assert 0.0 <= prob_added_cond_existing <= 1.0 + _mi_error_tol, f'prob not normalized: {prob_added_cond_existing=}'

            extended_joint_probs[tuple(values)] = prob_existing * prob_added_cond_existing

            if __debug__ and len(self) == 1:
                _debug_parameters_before_append.append(cond_pdf[tuple(existing_variables_values)].matrix2params_incremental)

        if __debug__ and np.random.randint(10) == 0:
            np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

        self.reset(self.numvariables + num_added_variables, self.numvalues, extended_joint_probs)

        # if __debug__ and len(self) == 1:
        #     # of course this test depends on the implementation of matrix2params_incremental, currently it should
        #     # work
        #     np.testing.assert_array_almost_equal(self.scalars_up_to_level(_debug_parameters_before_append),
        #                                          self.matrix2params_incremental(return_flattened=True))


    def conditional_probability_distribution(self, given_variables, given_values):
        """

        :param given_variables: list of integers
        :param given_values: list of integers
        :rtype: JointProbabilityMatrix
        """
        assert len(given_values) == len(given_variables)
        assert len(given_variables) < self.numvariables, 'no variables left after conditioning'

        lists_of_possible_states_per_variable = [list(range(self.numvalues)) for variable in range(self.numvariables)]

        # overwrite the 'state spaces' for the specified variables, to the specified state spaces
        for gix in range(len(given_variables)):
            assert np.isscalar(given_values[gix]), 'assuming specific value, not list of possibilities'

            lists_of_possible_states_per_variable[given_variables[gix]] = \
                [given_values[gix]] if np.isscalar(given_values[gix]) else given_values[gix]

        conditioned_variables = [varix for varix in range(self.numvariables) if not varix in given_variables]

        conditional_probs = np.zeros([self.numvalues]*len(conditioned_variables))

        assert len(conditional_probs) > 0, 'you want me to make a conditional pdf of 0 variables?'

        assert len(given_variables) + len(conditioned_variables) == self.numvariables

        for values in itertools.product(*lists_of_possible_states_per_variable):
            values_conditioned_vars = [values[varid] for varid in conditioned_variables]

            assert conditional_probs[tuple(values_conditioned_vars)] == 0.0, 'right?'

            # note: here self.joint_probability(values) == Pr(conditioned_values | given_values) because the
            # given_variables == given_values constraint is imposed on the set
            # itertools.product(*lists_of_possible_states_per_variable); it is only not yet normalized
            conditional_probs[tuple(values_conditioned_vars)] += self.joint_probability(values)

        summed_prob_mass = np.sum(conditional_probs)

        # testing here if the summed prob. mass equals the marginal prob of the given variable values
        if __debug__:
            # todo: can make this test be run probabilistically, like 10% chance or so, pretty expensive?
            if np.all(list(map(np.isscalar, given_values))):
                pdf_marginal = self.marginalize_distribution(given_variables)

                prob_given_values = pdf_marginal(given_values)

                np.testing.assert_almost_equal(prob_given_values, summed_prob_mass)

        assert np.isscalar(summed_prob_mass), 'sum of probability mass should be a scalar, not a list or so: ' \
                                              + str(summed_prob_mass)
        assert np.isfinite(summed_prob_mass)

        assert summed_prob_mass >= 0.0, 'probability mass cannot be negative'

        if summed_prob_mass > 0.0:
            conditional_probs /= summed_prob_mass
        else:
            # note: apparently the probability of the given condition is zero, so it does not really matter
            # what I substitute for the probability mass here. I will add some probability mass so that I can
            # normalize it.

            # # I think this warning can be removed at some point...
            # warnings.warn('conditional_probability_distribution: summed_prob_mass == 0.0 (can be ignored)')

            if __debug__:
                # are all values zero?
                try:
                    np.testing.assert_almost_equal(np.min(conditional_probs), 0.0)
                except ValueError as e:
                    print('debug: conditional_probs =', conditional_probs)
                    print('debug: min(conditional_probs) =', min(conditional_probs))

                    raise ValueError(e)

                np.testing.assert_almost_equal(np.max(conditional_probs), 0.0)

            conditional_probs *= 0
            conditional_probs += 1.0  # create some fake mass, making it a uniform distribution

            conditional_probs /= np.sum(conditional_probs)

        conditional_joint_pdf = JointProbabilityMatrix(len(conditioned_variables), self.numvalues,
                                                    joint_probs=conditional_probs)

        return conditional_joint_pdf


    def conditional_probability_distributions(self, given_variables, object_variables='auto', nprocs=1):
        """

        :param given_variables:
        :return: dict of JointProbabilityMatrix, keys are all possible values for given_variables
        :rtype: dict of JointProbabilityMatrix
        """
        if len(given_variables) == self.numvariables:  # 'no variables left after conditioning'
            warnings.warn('conditional_probability_distributions: no variables left after conditioning')

            lists_of_possible_given_values = [list(range(self.numvalues)) for variable in range(len(given_variables))]

            dic = {tuple(given_values): JointProbabilityMatrix(0, self.numvalues)
                   for given_values in itertools.product(*lists_of_possible_given_values)}

            return ConditionalProbabilityMatrix(dic)
        else:
            if object_variables in ('auto', 'all'):
                object_variables = list(np.setdiff1d(list(range(len(self))), given_variables))
            else:
                object_variables = list(object_variables)
                ignored_variables = list(np.setdiff1d(list(range(len(self))), list(given_variables) + object_variables))

                pdf = self.copy()
                pdf.reorder_variables(list(given_variables) + object_variables + ignored_variables)
                # note: I do this extra step because reorder_variables does not support specifying fewer variables
                # than len(self), but once it does then it can be combined into a single reorder_variables call.
                # remove ignored_variables
                pdf = pdf.marginalize_distribution(list(range(len(given_variables + object_variables))))

                return pdf.conditional_probability_distributions(list(range(len(given_variables))))

            lists_of_possible_given_values = [list(range(self.numvalues)) for variable in range(len(given_variables))]

            if nprocs == 1:
                dic = {tuple(given_values): self.conditional_probability_distribution(given_variables=given_variables,
                                                                                       given_values=given_values)
                        for given_values in itertools.product(*lists_of_possible_given_values)}
            else:
                def worker_cpd(given_values):  # returns a 2-tuple
                    return (tuple(given_values),
                            self.conditional_probability_distribution(given_variables=given_variables,
                                                                      given_values=given_values))

                pool = mp.Pool(nprocs)
                dic = pool.map(worker_cpd, itertools.product(*lists_of_possible_given_values))
                dic = dict(dic)  # make a dictionary out of it
                pool.close()
                pool.terminate()

            return ConditionalProbabilityMatrix(dic)


    # todo: try to make this entropy() function VERY efficient, maybe compiled (weave?), or C++ binding or something,
    # it is a central
    # function in all information-theoretic quantities and especially it is a crucial bottleneck for optimization
    # procedures such as in append_orthogonal* which call information-theoretic quantities a LOT.
    def entropy(self, variables=None, base=2):

        if variables is None:
            if np.random.random() < 0.1:
                np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)

            probs = self.joint_probabilities.flatten()
            probs = np.select([probs != 0], [probs], default=1)

            if base == 2:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * np.log2(probs)
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)
            elif base == np.e:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * np.log(probs)
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)
            else:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * (np.log(probs) / np.log(base))
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)

            assert joint_entropy >= 0.0

            return joint_entropy
        else:
            assert hasattr(variables, '__iter__')

            if len(variables) == 0:  # hard-coded this because otherwise I have to support empty pdfs (len() = 0)
                return 0.0
            else:
                assert max(variables) < len(self), f'{variables=} but {len(self)=}'

            marginal_pdf = self.marginalize_distribution(retained_variables=variables)

            return marginal_pdf.entropy(base=base)


    def conditional_entropy(self, variables, given_variables=None):
        assert hasattr(variables, '__iter__'), 'variables1 = ' + str(variables)
        assert hasattr(given_variables, '__iter__') or given_variables is None, 'variables2 = ' + str(given_variables)

        assert max(variables) < self.numvariables, 'variables are 0-based indices, so <= N - 1: variables=' \
                                                   + str(variables) + ' (N=' + str(self.numvariables) + ')'

        if given_variables is None:
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables) if not varix in variables]

            assert len(set(variables)) + len(given_variables) == self.numvariables, 'variables=' + str(variables) \
                                                                                    + ', given_variables=' \
                                                                                    + str(given_variables)

        # H(Y) + H(X|Y) == H(X,Y)
        condent = self.entropy(list(set(list(variables) + list(given_variables)))) - self.entropy(given_variables)

        assert np.isscalar(condent)
        assert np.isfinite(condent)

        assert condent >= 0.0, 'conditional entropy should be non-negative'

        return condent


    def mutual_information_labels(self, labels1, labels2):
        variables1 = self.get_variables_with_label_in(labels1)
        variables2 = self.get_variables_with_label_in(labels2)

        return self.mutual_information(variables1, variables2)


    def conditional_mutual_informations(self, variables1, variables2, given_variables='all', nprocs=1):

        if given_variables == 'all':
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables)
                               if not varix in list(variables1) + list(variables2)]

        # pZ = self.marginalize_distribution(given_variables)

        cond_pXY_given_z = self.conditional_probability_distributions(given_variables, nprocs=nprocs)

        varXnew = list(variables1)
        varYnew = list(variables2)

        # compute new indices of X and Y variables in conditioned pdfs, cond_pXY_given_z
        for zi in sorted(given_variables, reverse=True):
            for ix in range(len(varXnew)):
                assert varXnew[ix] != zi
                if varXnew[ix] > zi:
                    varXnew[ix] -= 1
            for ix in range(len(varYnew)):
                assert varYnew[ix] != zi
                if varYnew[ix] > zi:
                    varYnew[ix] -= 1

        mi_given_z = dict()

        # todo: also parallellize over nprocs cpus?
        for z in self.statespace(range(len(given_variables))):
            mi_given_z[z] = cond_pXY_given_z[z].mutual_information(varXnew, varYnew)

        return mi_given_z


    def conditional_mutual_information(self, variables1, variables2, given_variables='all'):

        if given_variables == 'all':
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables)
                               if not varix in list(variables1) + list(variables2)]

        condmis = self.conditional_mutual_informations(variables1, variables2, given_variables)
        p3 = self.marginalize_distribution(given_variables)

        return sum([p3(z) * condmis[z] for z in p3.statespace()])


    def mutual_information(self, variables1, variables2, base=2):
        assert hasattr(variables1, '__iter__'), 'variables1 = ' + str(variables1)
        assert hasattr(variables2, '__iter__'), 'variables2 = ' + str(variables2)

        if len(variables1) == 0 or len(variables2) == 0:
            mi = 0  # trivial case, no computation needed
        elif len(variables1) == len(variables2):
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            if np.equal(sorted(variables1), sorted(variables2)).all():
                mi = self.entropy(variables1, base=base)  # more efficient, shortcut
            else:
                # this one should be equal to the outer else-clause (below), i.e., the generic case
                mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                     - self.entropy(list(set(list(variables1) + list(variables2))), base=base)
        else:
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                 - self.entropy(list(set(list(variables1) + list(variables2))), base=base)

        assert np.isscalar(mi)
        assert np.isfinite(mi)

        # due to floating-point operations it might VERY sometimes happen that I get something like -4.4408920985e-16
        # here, so to prevent this case from firing an assert I clip this to 0:
        if -0.000001 < mi < 0.0:
            mi = 0.0

        assert mi >= 0, 'mutual information should be non-negative: ' + str(mi)

        return mi


    def synergistic_information_naive(self, variables_Y, variables_X, return_also_relerr=False):
        """
        Estimate the amount of synergistic information contained in the variables (indices) <variables_Y> about the
        variables (indices) <variables_X>. It is a naive estimate which works best if <variables_SRV> is (approximately)
        an SRV, i.e., if it already has very small MI with each individual variable in <variables_X>.

        Also referred to as the Whole-Minus-Sum (WMS) algorithm.

        Note: this is not compatible with the definition of synergy used by synergistic_information(), i.e., one is
        not an unbiased estimator of the other or anything. Very different.

        :param variables_SRV:
        :param variables_X:
        :param return_also_relerr: if True then a tuple of 2 floats is returned, where the first is the best estimate
         of synergy and the second is the relative error of this estimate (which is preferably below 0.1).
        :rtype: float or tuple of float
        """

        indiv_mis = [self.mutual_information(list(variables_Y), list([var_xi])) for var_xi in variables_X]
        total_mi = self.mutual_information(list(variables_Y), list(variables_X))

        syninfo_lowerbound = total_mi - sum(indiv_mis)
        syninfo_upperbound = total_mi - max(indiv_mis)

        if not return_also_relerr:
            return (syninfo_lowerbound + syninfo_upperbound) / 2.0
        else:
            best_estimate_syninfo = (syninfo_lowerbound + syninfo_upperbound) / 2.0
            uncertainty_range = syninfo_upperbound - syninfo_lowerbound

            return (best_estimate_syninfo, uncertainty_range / best_estimate_syninfo)


    def unique_individual_information(self, variables_Y, variables_X, tol_nonunq=0.05, verbose=True,
                                      num_repeats_per_append=3, assume_independence=False):

        """

        :param variables_Y:
        :param variables_X:
        :param tol_nonunq:
        :param verbose:
        :param num_repeats_per_append:
        :param assume_independence: if all variables_X are independent then this function greatly simplifies to the
        sum of mutual information terms.
        :return:
        """

        if assume_independence:
            return sum([self.mutual_information(variables_Y, [x]) for x in variables_X])
        else:
            pdf_c = self.copy()
            pdf_X = self.marginalize_distribution(variables_X)

            xixs = np.random.permutation(list(range(len(variables_X))))

            for xid in range(len(xixs)):
                pdf_X.append_unique_individual_variable(xixs[xid], verbose=verbose, tol_nonunique=tol_nonunq,
                                                        num_repeats=num_repeats_per_append,
                                                        ignore_variables=(None if xid <= 0 else xixs[:xid]))
                # note: agnostic_about is only set in case there are at least two other unique variables since
                # in that case there could be synergistic information about other unique information. For only one
                # other unique variable added this is not the case and the optimization itself already discounts for
                # unique information about others
                # CHECK: actually it discounts the total information with all other variables so

            # append the URVs to the original pdf_c so that we can compute MI(Y:URVs)
            cond_urvs = pdf_X.conditional_probability_distributions(list(range(len(variables_X))))
            pdf_c.append_variables_using_conditional_distributions(cond_urvs, variables_X)

            # sum up individual MI's with URVs. If done all URVs simultaneously then synergistic information arises,
            # like Y=XOR(X1,X2) would have 1.0 unique information then (X1,X2 independent).
            mi = np.sum([pdf_c.mutual_information([urvix], variables_Y) for urvix in range(len(self), len(pdf_c))])

            return mi


    def synergistic_information(self, variables_Y, variables_X, tol_nonsyn_mi_frac=0.05, verbose=False,
                                minimize_method=None, num_repeats_per_srv_append=3, break_when_srv_found=True,
                                initial_guess_summed_modulo=False, inplace=False):
        """Estimate the amount of synergistic information that variables_X share about variables_Y.

        Args:
            variables_Y (list): _description_
            variables_X (list): _description_
            tol_nonsyn_mi_frac (float, optional): _description_. Defaults to 0.05.
            verbose (bool, optional): _description_. Defaults to False.
            minimize_method (_type_, optional): _description_. Defaults to None.
            num_repeats_per_srv_append (int, optional): _description_. Defaults to 3.
            break_when_srv_found (bool, optional): _description_. Defaults to True.
            initial_guess_summed_modulo (bool, optional): _description_. Defaults to False.
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            float: amount of synergistic information (non-negative).
        """
        original_len = len(self)  # in case inplace=True

        if not max(variables_Y) < original_len: 
            raise UserWarning('target variables (Y) have to exist: variables_Y=' + str(variables_Y) \
                + ' but len(self)=' + str(len(self)))

        if inplace:
            pdf_with_srvs = self
        else:
            pdf_with_srvs = self.copy()

        # TODO: improve the retrying of finding and adding srvs, in different orders?

        syn_entropy = self.synergistic_entropy_upper_bound(variables_X)
        max_ent_per_var = np.log2(self.numvalues)
        max_num_srv_add_trials = int(round(syn_entropy / max_ent_per_var * 2 + 0.5))  # heuristic
        # ent_X = self.entropy(variables_X)

        # note: currently I constrain to add SRVs which consist each of only 1 variable each. I guess in most cases
        # this will work fine, but I do not know for sure that this will always be sufficient (i.e., does there
        # ever an MSRV exist with bigger entropy, and if so, is this MSRV not decomposable into parts such that
        # this approach here still works?)

        total_syn_mi = 0

        for i in range(max_num_srv_add_trials):
            try:
                agnostic_about = list(range(original_len, len(pdf_with_srvs)))  # new SRV must not correlate with previous SRVs
                pdf_with_srvs.append_synergistic_variables(1, initial_guess_summed_modulo=initial_guess_summed_modulo,
                                                           subject_variables=variables_X,
                                                           num_repeats=num_repeats_per_srv_append,
                                                           agnostic_about=agnostic_about,
                                                           minimize_method=minimize_method,
                                                           break_when_srv_found=break_when_srv_found)
            except UserWarning as e:
                assert 'minimize() failed' in str(e), 'only known reason for this error'

                warnings.warn(str(e) + '. Will now skip this sample in synergistic_information(). [attempt ' + str(i) + '/' + str(max_num_srv_add_trials) + ']')

                continue

            total_mi = pdf_with_srvs.mutual_information([-1], variables_X)
            indiv_mi_list =  [pdf_with_srvs.mutual_information([-1], [xid]) for xid in variables_X]
            new_syn_info = total_mi - sum(indiv_mi_list)  # lower bound, actual amount might be higher (but < total_mi)

            if new_syn_info < syn_entropy * 0.01:  # very small added amount of syn. info., so stop after this one
                if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-syn. information?
                    pdf_with_srvs.marginalize_distribution(list(range(len(pdf_with_srvs) - 1)))  # remove last srv

                    if verbose > 0:
                        print('debug: synergistic_information: a last SRV was found but with too much non-syn. info.')
                else:
                    if verbose > 0:
                        print('debug: synergistic_information: a last (small) ' \
                              'SRV was found, a good one, and now I will stop.')

                    total_syn_mi += new_syn_info

                break  # assume that no more better SRVs can be found from now on, so stop (save time)
            else:
                # if you reach here then it is not really a problem per se. The idea was to check if we
                # 'converge', meaning that we could expect that when we have found all the synergy there is
                # then we would find only very small SRVs. But depending on the optimization algorithm,
                # I suppose it is also possible that the optimization algorithm finds nothing at all.
                # So we reached here because the SRV we found is not very small, but it does not mean that
                # we necessarily did not 'converge'. Probably the earlier cycles of the for loop did not result
                # in anything and therefore we only found an SRV now. If the previous cycles had such trouble
                # finding an SRV to begin with, one could hope that finding even more SRVs is even harder and
                # therefore not worth trying.

                if i == max_num_srv_add_trials - 1:
                    if verbose > 0:  # this warning is only sent when verbose because it is not REALLY problematic...
                        warnings.warn('synergistic_information: never converged to adding SRV with ~zero synergy')

                if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(list(range(len(pdf_with_srvs) - 1)))  # remove last srv

                    if verbose > 0:
                        print(('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                              + str((new_syn_info) / total_mi) + ' was found, which will be removed again because'
                                                                     ' it does not meet the tolerance of '
                              + str(tol_nonsyn_mi_frac)))
                else:
                    if verbose > 0:
                        print('debug: synergistic_information: an SRV with new_syn_info/total_mi = ' \
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = ' \
                                  + str((new_syn_info) / total_mi) + ' was found, which satisfies the tolerance of ' \
                                  + str(tol_nonsyn_mi_frac))

                        if len(agnostic_about) > 0:
                            agn_mi = pdf_with_srvs.mutual_information([-1], agnostic_about)

                            print('debug: agnostic=%s, agn. mi = %s (should be close to 0)' % (agnostic_about, agn_mi))

                    total_syn_mi += new_syn_info

            if total_syn_mi >= syn_entropy * 0.99:
                if verbose > 0:
                    print('debug: foundof the upper bound of synergistic entropy which is high so I stop.')

                break  # found enough SRVs, cannot find more (except for this small %...)

        if verbose > 0:
            print('debug: synergistic_information: number of SRVs:', len(range(original_len, len(pdf_with_srvs))))
            print('debug: synergistic_information: entropy of SRVs:', \
                pdf_with_srvs.entropy(list(range(original_len, len(pdf_with_srvs)))))
            print('debug: synergistic_information: H(SRV_i) =', \
                [pdf_with_srvs.entropy([srv_id]) for srv_id in range(original_len, len(pdf_with_srvs))])
            print('debug: synergistic_information: I(Y, SRV_i) =', \
                [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in range(original_len, len(pdf_with_srvs))])

        syn_info = sum([pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in range(original_len, len(pdf_with_srvs))])

        return syn_info


    def synergistic_entropy(self, variables_X, tolerance_nonsyn_mi=0.05, verbose=True):

        assert False, 'todo: change from information to entropy here, just return different value and do not ask for Y'

        pdf_with_srvs = self.copy()

        syn_entropy = self.synergistic_entropy_upper_bound(variables_X)
        max_ent_per_var = np.log2(self.numvalues)
        max_num_srv_add_trials = int(round(syn_entropy / max_ent_per_var * 2))

        # note: currently I constrain to add SRVs which consist each of only 1 variable each. I guess in most cases
        # this will work fine, but I do not know for sure that this will always be sufficient (i.e., does there
        # ever an MSRV exist with bigger entropy, and if so, is this MSRV not decomposable into parts such that
        # this approach here still works?)

        for i in range(max_num_srv_add_trials):
            pdf_with_srvs.append_synergistic_variables(1, initial_guess_summed_modulo=False,
                                                       subject_variables=variables_X, num_repeats=3,
                                                       agnostic_about=list(range(len(variables_Y) + len(variables_X),
                                                                            len(pdf_with_srvs))))

            # todo: can save one MI calculation here
            new_syn_info = self.synergistic_information_naive([-1], variables_X)
            total_mi = self.mutual_information([-1], variables_X)

            if new_syn_info < syn_entropy * 0.01:
                if (total_mi - new_syn_info) / total_mi > tolerance_nonsyn_mi:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(list(range(len(pdf_with_srvs) - 1)))  # remove last srv

                    if verbose > 0:
                        print('debug: synergistic_information: a last SRV was found but with too much non-syn. info.')
                else:
                    if verbose > 0:
                        print('debug: synergistic_information: a last (small) ' \
                              'SRV was found, a good one, and now I will stop.')

                break  # assume that no more better SRVs can be found from now on, so stop (save time)
            else:
                if i == max_num_srv_add_trials - 1:
                    warnings.warn('synergistic_information: never converged to adding SRV with zero synergy')

                if (total_mi - new_syn_info) / total_mi > tolerance_nonsyn_mi:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(list(range(len(pdf_with_srvs) - 1)))  # remove last srv

                    if verbose > 0:
                        print(('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                                  + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                                  + str((new_syn_info) / total_mi) + ' was found, which will be removed again because'
                                                                     ' it does not meet the tolerance of '
                                  + str(tolerance_nonsyn_mi)))
                else:
                    if verbose > 0:
                        print('debug: synergistic_information: an SRV with new_syn_info/total_mi = ' \
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = ' \
                                  + str((new_syn_info) / total_mi) + ' was found, which satisfies the tolerance of ' \
                                  + str(tolerance_nonsyn_mi))

        # if verbose > 0:
        #     print 'debug: synergistic_information: number of SRVs:', len(xrange(len(self), len(pdf_with_srvs)))
        #     print 'debug: synergistic_information: entropy of SRVs:', \
        #         pdf_with_srvs.entropy(range(len(self), len(pdf_with_srvs)))
        #     print 'debug: synergistic_information: H(SRV_i) =', \
        #         [pdf_with_srvs.entropy([srv_id]) for srv_id in xrange(len(self), len(pdf_with_srvs))]
        #     print 'debug: synergistic_information: I(Y, SRV_i) =', \
        #         [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
        #                 for srv_id in xrange(len(self), len(pdf_with_srvs))]

        syn_info = sum([pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in range(len(self), len(pdf_with_srvs))])

        return syn_info


    def synergistic_entropy_upper_bound(self, variables=None):
        """
        For a derivation, see Quax et al. (2015). No other stochastic variable can have more than this amount of
        synergistic information about the stochastic variables defined by this pdf object, or the given subset of
        variables in <variables>.
        :type variables: list of int
        :rtype: float
        """
        if variables is None:
            variables = list(range(len(self)))

        indiv_entropies = [self.entropy([var]) for var in variables]

        return float(self.entropy() - max(indiv_entropies))


    # todo: add optional numvalues? so that the synergistic variables can have more possible values than the
    # current variables (then set all probabilities where the original variables exceed their original max to 0)
    def append_synergistic_variables(self, num_synergistic_variables : int, initial_guess_summed_modulo=True, verbose=False,
                                     subject_variables=None, agnostic_about=None, num_repeats=1, minimize_method=None,
                                     tol_nonsyn_mi_frac=0.05, tol_agn_mi_frac=0.05, min_mi_with_subject=0.025, 
                                     break_when_srv_found=True):
        """
        Append <num_synergistic_variables> variables in such a way that they are agnostic about any individual
        existing variable (one of self.numvariables thus) but have maximum MI about the set of self.numvariables
        variables taken together.
        :param minimize_method:
        :param tol_nonsyn_mi_frac: set to None for computational speed
        :param tol_agn_mi_frac: set to None for computational speed
        :return:
        :param agnostic_about: a list of variable indices to which the new synergistic variable set should be
        agnostic (zero mutual information). This can be used to find a 'different', second SRV after having found
        already a first one. If already multiple SRVs have been found then you can choose between either being agnostic
        about these previous SRVs jointly (so also possibly any synergy among them), or being 'pairwise' agnostic
        to each individual one, in which case you can pass a list of lists, then I will compute the added cost for
        each sublist and sum it up.
        :param num_repeats:
        :param subject_variables: the list of variables the <num_synergistic_variables> should be synergistic about;
        then I think the remainder variables the <num_synergistic_variables> should be agnostic about. This way I can
        support adding many UMSRVs (maybe make a new function for that) which then already are orthogonal among themselves,
        meaning I do not have to do a separate orthogonalization for the MSRVs as in the paper's theoretical part.
        :param num_synergistic_variables:
        :param initial_guess_summed_modulo: if True then the initial guess for the SRVs is XOR 9or parity for non-binary variables).
        For two uniform binary variables this leads to a speedup of a factor of roughly 2x faster. For variables with three values
        it is even 20x faster. If 'auto' then only for adding 1 SRV it is True, otherwise False (since for the second, third, etc.
        SRV the initial guess is likely to correlate completely with the first SRV guess, which is against the idea of SRVs being
        orthogonal).
        :param verbose:
        :return:
        """

        if not agnostic_about is None:
            if len(agnostic_about) == 0:
                agnostic_about = None  # treat as if not supplied

        if __debug__:  # looking for bug
            assert max(self.matrix2params_incremental()) < 1.0000001, 'param is out of bound: ' \
                                                                      + str(max(self.matrix2params_incremental()))
            assert min(self.matrix2params_incremental()) > -0.0000001, 'param is out of bound: ' \
                                                                      + str(min(self.matrix2params_incremental()))

        parameter_values_before = list(self.matrix2params_incremental())

        # make an initial pdf of the correct final size so that we know how large the new parameter space is
        # (a pdf with XOR[s] as [first] appended variable[s] (already MSRV for binary variables) just in case initial_guess_summed_modulo=True
        pdf_with_srvs = self.copy()
        # pdf_with_srvs.append_variables_using_state_transitions_table(
        #     state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_synergistic_variables)
        if num_synergistic_variables >= 1:
            pdf_with_srvs.append_variables_using_state_transitions_table(
                state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))] + [np.random.randint(mv)]*(num_synergistic_variables - 1))
        else:
            raise ValueError('you are asking me to add %i SRVs, which makes no sense.' % num_synergistic_variables)

        assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

        parameter_values_after = pdf_with_srvs.matrix2params_incremental()

        assert len(parameter_values_after) > len(parameter_values_before), 'should be additional free parameters'
        if np.random.random() < 0.01 and __debug__:  # reduce slowdown from this by chance of executing the testing
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                 parameter_values_after[:len(parameter_values_before)])

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        assert num_synergistic_variables == 0 or num_free_parameters > 0

        if initial_guess_summed_modulo:
            # note: this is xor for binary variables, and parity for other variables
            initial_guess = parameter_values_after[len(parameter_values_before):]
        else:
            initial_guess = np.random.random(num_free_parameters)

        if verbose and __debug__:
            debug_pdf_with_srvs = pdf_with_srvs.copy()
            debug_pdf_with_srvs.params2matrix_incremental(list(parameter_values_before) + list(initial_guess))

            # store the synergistic information before the optimization procedure (after procedure should be higher...)
            debug_before_syninfo = debug_pdf_with_srvs.synergistic_information_naive(
                variables_Y=list(range(self.numvariables, pdf_with_srvs.numvariables)),
                variables_X=list(range(self.numvariables)))

        assert len(initial_guess) == num_free_parameters

        pdf_with_srvs_for_optimization = pdf_with_srvs.copy()

        if not subject_variables is None:
            pdf_subjects_syns_only = pdf_with_srvs_for_optimization.marginalize_distribution(
                list(subject_variables) + list(range(len(pdf_with_srvs) - num_synergistic_variables, len(pdf_with_srvs))))

            pdf_subjects_only = pdf_subjects_syns_only.marginalize_distribution(list(range(len(subject_variables))))

            if __debug__ and np.random.random() < 0.01:
                debug_pdf_subjects_only = pdf_with_srvs.marginalize_distribution(subject_variables)

                assert debug_pdf_subjects_only == pdf_subjects_only

            num_free_parameters_synonly = len(pdf_subjects_syns_only.matrix2params_incremental()) \
                                          - len(pdf_subjects_only.matrix2params_incremental())

            parameter_values_static = pdf_subjects_only.matrix2params_incremental()

            initial_guess = np.random.random(num_free_parameters_synonly)

            # pdf_subjects_syns_only should be the new object that fitness_func operates on, instead of
            # pdf_with_srvs_for_optimization
        else:
            # already like this, so simple renaming
            pdf_subjects_syns_only = pdf_with_srvs_for_optimization

            parameter_values_static = parameter_values_before

            num_free_parameters_synonly = num_free_parameters

            # subject_variables = range(len(self))

        upper_bound_synergistic_information = self.synergistic_entropy_upper_bound(subject_variables)
        if not agnostic_about is None:
            # upper_bound_agnostic_information is only used to normalize the cost term for non-zero MI with
            # the agnostic_variables (evidently a SRV is sought that has zero MI with these)
            if np.ndim(agnostic_about) == 1:
                upper_bound_agnostic_information = self.entropy(agnostic_about)
            elif np.ndim(agnostic_about) == 2:
                # in this case the cost term is the sum of MIs of the SRV with the sublists, so max cost term is this..
                upper_bound_agnostic_information = sum([self.entropy(ai) for ai in agnostic_about])
        else:
            upper_bound_agnostic_information = 0  # should not even be used...

        # todo: should lower the upper bounds if the max possible entropy of the SRVs is less...

        assert upper_bound_synergistic_information != 0.0, 'can never find any SRV!'

        # in the new self, these indices will identify the synergistic variables that will be added
        synergistic_variables = list(range(len(self), len(self) + num_synergistic_variables))

        # todo: shouldn't the cost terms in this function not be squared for better convergence?
        def cost_func_subjects_only(free_params, parameter_values_before, extra_cost_rel_error=True):
            """
            This cost function searches for a Pr(S,Y,A,X) such that X is synergistic about S (subject_variables) only.
            This fitness function also taxes any correlation between X and A (agnostic_variables), but does not care
            about the relation between X and Y.
            :param free_params:
            :param parameter_values_before:
            :return:
            """
            assert len(free_params) == num_free_parameters_synonly

            if min(free_params) < -0.00001 or max(free_params) > 1.00001:
                warnings.warn('scipy\'s minimize() is violating the parameter bounds 0...1 I give it: '
                              + str(free_params))

                # high cost for invalid parameter values
                # note: maximum cost normally from this function is about 2.0
                return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
                                             + [np.abs(p) for p in free_params if p < 0.0])

            # assert max(free_params) <= 1.00001, \
            #     'scipy\'s minimize() is violating the parameter bounds 0...1 I give it: ' + str(free_params)

            # removed to save computation:
            # free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors

            pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
            # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
            if not subject_variables is None:
                assert pdf_subjects_syns_only.numvariables == len(subject_variables) + num_synergistic_variables

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if not extra_cost_rel_error:
                    cost = (upper_bound_synergistic_information - pdf_subjects_syns_only.synergistic_information_naive(
                        variables_Y=list(range(len(subject_variables), len(pdf_subjects_syns_only))),
                        variables_X=list(range(len(subject_variables))))) / upper_bound_synergistic_information
                else:
                    tot_mi = pdf_subjects_syns_only.mutual_information(
                        list(range(len(subject_variables), len(pdf_subjects_syns_only))),
                        list(range(len(subject_variables))))

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           list(range(len(subject_variables),
                                                                                 len(pdf_subjects_syns_only))))
                                 for var in range(len(subject_variables))]

                    syninfo_naive = tot_mi - sum(indiv_mis)

                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    cost = (upper_bound_synergistic_information - syninfo_naive) \
                           / upper_bound_synergistic_information

                    # add an extra cost term for the fraction of 'individual' information versus the total information
                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    if tot_mi != 0:
                        cost += sum(indiv_mis) / tot_mi
                    else:
                        cost += sum(indiv_mis)
            else:
                assert pdf_subjects_syns_only.numvariables == len(self) + num_synergistic_variables

                if not extra_cost_rel_error:
                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    cost = (upper_bound_synergistic_information - pdf_subjects_syns_only.synergistic_information_naive(
                        variables_Y=list(range(len(self), len(pdf_subjects_syns_only))),
                        variables_X=list(range(len(self))))) / upper_bound_synergistic_information
                else:
                    tot_mi = pdf_subjects_syns_only.mutual_information(
                        list(range(len(self), len(pdf_subjects_syns_only))),
                        list(range(len(self))))

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           list(range(len(self),
                                                                                 len(pdf_subjects_syns_only))))
                                 for var in range(len(self))]

                    syninfo_naive = tot_mi - sum(indiv_mis)

                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    cost = (upper_bound_synergistic_information - syninfo_naive) \
                           / upper_bound_synergistic_information

                    # add an extra cost term for the fraction of 'individual' information versus the total information
                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    if tot_mi != 0:
                        cost += sum(indiv_mis) / tot_mi
                    else:
                        cost += sum(indiv_mis)

            # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
            if not agnostic_about is None:
                assert not subject_variables is None, 'how can all variables be subject_variables and you still want' \
                                                      ' to be agnostic about certain (other) variables? (if you did' \
                                                      ' not specify subject_variables, do so.)'

                # make a conditional distribution of the synergistic variables conditioned on the subject variables
                # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
                # agnostic constraint
                cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                    list(range(len(subject_variables)))
                )

                assert type(cond_pdf_syns_on_subjects) == dict \
                       or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

                pdf_with_srvs_for_agnostic = self.copy()
                pdf_with_srvs_for_agnostic.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                            subject_variables)

                if np.ndim(agnostic_about) == 1:
                    # note: cost term for agnostic is in [0,1]
                    cost += (pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agnostic_about)) \
                            / upper_bound_agnostic_information
                else:
                    assert np.ndim(agnostic_about) == 2, 'expected list of lists, not more... made a mistake?'

                    assert False, 'I don\'t think this case should happen, according to my 2017 paper should be just ' \
                                  'I(X:A) so ndim==1'

                    for agn_i in agnostic_about:
                        # note: total cost term for agnostic is in [0,1]
                        cost += (1.0 / float(len(agnostic_about))) * \
                                pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agn_i) \
                                / upper_bound_agnostic_information

            assert np.isscalar(cost)
            assert np.isfinite(cost)

            return float(cost)

        param_vectors_trace = []

        # these options are altered mainly to try to lower the computation time, which is considerable.
        minimize_options = {'ftol': 1e-6}

        if True:
        # if num_repeats == 1:
        #     optres = minimize(cost_func_subjects_only, initial_guess, bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
        #                       callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
        #                       args=(parameter_values_static,), method=minimize_method, options=minimize_options)
        # else:
            optres_list = []

            for ix in range(num_repeats):
                optres_ix = minimize(cost_func_subjects_only,
                                     np.random.random(num_free_parameters_synonly) if ix > 0 else initial_guess,
                                     bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
                                     callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                                     args=(parameter_values_static,), method=minimize_method, options=minimize_options)

                if verbose > 0:
                    print('note: finished a repeat. success=' + str(optres_ix.success) + ', cost=' \
                          + str(optres_ix.fun))

                pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_static) + list(optres_ix.x))

                num_subject_vars = len(pdf_subjects_syns_only) - num_synergistic_variables

                if not subject_variables is None:
                    assert num_subject_vars == len(subject_variables)

                tot_mi = pdf_subjects_syns_only.mutual_information(
                                    list(range(num_subject_vars, len(pdf_subjects_syns_only))),
                                    list(range(num_subject_vars)))
                
                if not min_mi_with_subject is None:
                    if tot_mi < min_mi_with_subject:
                        continue  # too small MI with inputs, we'll not bother

                if not tol_nonsyn_mi_frac is None:
                    if subject_variables is None:
                        if __debug__ and verbose:
                            print('debug: will set subject_variables=%s' % (list(range(len(self)))))
                            subject_variables = list(range(len(self)))

                    assert not pdf_subjects_syns_only is None

                    

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           list(range(num_subject_vars,
                                                                                 len(pdf_subjects_syns_only))))
                                 for var in range(num_subject_vars)]

                    if sum(indiv_mis) / float(tot_mi) > tol_nonsyn_mi_frac:
                        if verbose > 0:
                            print('debug: in iteration %s I found an SRV but with total MI %s and indiv. MIs %s it ' \
                                  'violates the tol_nonsyn_mi_frac=%s' % (ix, tot_mi, indiv_mis, tol_nonsyn_mi_frac))

                        continue  # don't add this to the list of solutions

                if not tol_agn_mi_frac is None and not agnostic_about is None:
                    if len(agnostic_about) > 0:
                        cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                                                        list(range(num_subject_vars))
                                                    )

                        # I also need the agnostic variables, which are not in pdf_subjects_syns_only, so construct
                        # the would-be final result (the original pdf with the addition of the newly found SRV)
                        pdf_with_srvs = self.copy()
                        pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                       given_variables=subject_variables)

                        srv_ixs = list(range(len(self), len(pdf_with_srvs)))
                        srv_ent = pdf_with_srvs.entropy(srv_ixs)

                        if srv_ent <= min_mi_with_subject:
                            continue

                        agn_mi = pdf_with_srvs.mutual_information(agnostic_about, srv_ixs)
                        agn_ent = self.entropy(agnostic_about)

                        if agn_mi / agn_ent > tol_agn_mi_frac:
                            if verbose > 0:
                                print('verbose: in iteration %s I found an SRV but with agn_mi=%s and agn_ent=%s it ' \
                                      'violates the tol_agn_mi_frac=%s' % (ix, agn_mi, agn_ent, tol_nonsyn_mi_frac))

                            continue  # don't add this to the list of solutions

                # if True:
                #     print('debug: in iteration ix=' + str(ix) + ' of num_repeats=' + str(num_repeats) \
                #           + ' I found a potential solution that I will append to optres. The fraction' \
                #             + ' of indiv MI over total MI is ' + str(sum(indiv_mis) / float(tot_mi)) \
                #                 + '. len(optres)=' + str(len(optres_list)))

                optres_list.append(optres_ix)

                if break_when_srv_found:
                    break

            if verbose and __debug__:
                print('debug: num_repeats=' + str(num_repeats) + ', all cost values were: ' \
                      + str([resi.fun for resi in optres_list]))
                print('debug: successes =', [resi.success for resi in optres_list])

            optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

            if len(optres_list) == 0:
                raise UserWarning('all ' + str(num_repeats) + ' optimizations using minimize() failed.')

            costvals = [res.fun for res in optres_list]
            min_cost = min(costvals)
            optres_ix = costvals.index(min_cost)

            assert optres_ix >= 0 and optres_ix < len(optres_list)

            optres = optres_list[optres_ix]

        if subject_variables is None:
            assert len(optres.x) == num_free_parameters
        else:
            assert len(optres.x) == num_free_parameters_synonly

        assert max(optres.x) <= 1.0000001, 'parameter bound significantly violated, ' + str(max(optres.x))
        assert min(optres.x) >= -0.0000001, 'parameter bound significantly violated, ' + str(min(optres.x))

        # todo: reuse the .append_optimized_variables (or so) instead, passing the cost function only? would also
        # validate that function.

        # clip to valid range
        optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

        # optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)
        # pdf_with_srvs.params2matrix_incremental(optimal_parameters_joint_pdf)

        assert len(pdf_subjects_syns_only.matrix2params_incremental()) == len(parameter_values_static) + len(optres.x)

        pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_static) + list(optres.x))

        if not subject_variables is None:
            cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                list(range(len(subject_variables)))
            )

            assert isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)
            assert cond_pdf_syns_on_subjects.num_given_variables() > 0, 'conditioned on 0 variables?'

            # if this hits then something is unintuitive with conditioning on variables...
            assert cond_pdf_syns_on_subjects.num_given_variables() == len(subject_variables)

            pdf_with_srvs = self.copy()
            pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                           given_variables=subject_variables)
        else:
            pdf_with_srvs = pdf_subjects_syns_only  # all variables are subject

        assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

        if __debug__:
            parameter_values_after2 = pdf_with_srvs.matrix2params_incremental()

            assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'

            if not 1.0 in parameter_values_after2 and not 0.0 in parameter_values_after2:
                # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
                # have to optimize the latter part of parameter_values_after
                np.testing.assert_array_almost_equal(parameter_values_before,
                                                     parameter_values_after2[:len(parameter_values_before)])
                np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                     optres.x)
            else:
                # it can happen that some parameters are 'useless' in the sense that they defined conditional
                # probabilities in the case where the prior (that is conditioned upon) has zero probability. The
                # resulting pdf is then always the same, no matter this parameter value. This can only happen if
                # there is a 0 or 1 in the parameter list, (sufficient but not necessary) so skip the test then...
                pass

            # store the synergistic information before the optimization procedure (after procedure should be higher...)
            debug_after_syninfo = pdf_with_srvs.synergistic_information_naive(variables_Y=list(range(self.numvariables,
                                                                                         pdf_with_srvs.numvariables)),
                                                                              variables_X=list(range(self.numvariables)))

            if verbose > 0:
                print('debug: append_synergistic_variables: I started from synergistic information =', \
                    debug_before_syninfo, 'at initial guess. After optimization it became', debug_after_syninfo, \
                    '(should be higher). Optimal params:', \
                    parameter_values_after2[len(parameter_values_before):])

        self.duplicate(pdf_with_srvs)


    def susceptibility_local_single(self, var_id, num_output_variables, perturbation_size=0.01, ntrials=25,
                                    impact_measure='abs', nudge_method='fixed', also_return_pdf_after=False,
                                    auto_reorder=True):
        """

        :return: nudges_array, suscs_array
        :rtype: tuple
        """
        nudges = []
        suscs = []
        pdf_after = []  # optional

        # output marginal pdf for assessing impact, possibly
        pdf_output = self[list(range(len(self) - num_output_variables, len(self)))]
        # cond_pdf_out = self.conditional_probability_distributions(range(len(self) - num_output_variables))

        num_input_variables = len(self) - num_output_variables

        if not var_id == num_input_variables - 1 and auto_reorder:
            pdf_c = self.copy()
            other_input_ixs = [i for i in range(num_input_variables) if not i == var_id]
            pdf_c.reorder_variables(other_input_ixs + [var_id] + list(range(num_input_variables, len(self))))

            # make sure the input variable perturbed is the last listed of the inputs so that it does not have
            # any causal effect on the other inputs, which would overestimate the impact on the output potentially
            return pdf_c.susceptibility_local_single(var_id=num_input_variables - 1,
                                                     num_output_variables=num_output_variables,
                                                     perturbation_size=perturbation_size, ntrials=ntrials,
                                                     impact_measure=impact_measure, nudge_method=nudge_method,
                                                     also_return_pdf_after=also_return_pdf_after,
                                                     auto_reorder=auto_reorder)

        for trial in range(ntrials):
            pdf = self.copy()
            nudge = pdf.nudge([var_id], list(range(len(self) - num_output_variables, len(self))), method=nudge_method,
                              epsilon=perturbation_size)

            if impact_measure in ('sq',):
                pdf_out_nudged = pdf[list(range(len(self) - num_output_variables, len(self)))]

                impact = np.sum(np.power(np.subtract(pdf_output.joint_probabilities,
                                                     pdf_out_nudged.joint_probabilities), 2))
            elif impact_measure in ('abs',):
                pdf_out_nudged = pdf[list(range(len(self) - num_output_variables, len(self)))]

                impact = np.sum(np.abs(np.subtract(pdf_output.joint_probabilities,
                                                     pdf_out_nudged.joint_probabilities)))
            elif impact_measure in ('prob',):
                pdf_out_nudged = pdf[list(range(len(self) - num_output_variables, len(self)))]

                impact = np.subtract(pdf_output.joint_probabilities, pdf_out_nudged.joint_probabilities)
            elif impact_measure in ('kl', 'kld', 'kldiv'):
                pdf_out_nudged = pdf[list(range(len(self) - num_output_variables, len(self)))]

                impact = pdf_output.kullback_leibler_divergence(pdf_out_nudged)
            elif impact_measure in ('hell', 'h', 'hellinger'):
                pdf_out_nudged = pdf[list(range(len(self) - num_output_variables, len(self)))]

                impact = pdf_output.hellinger_distance(pdf_out_nudged)
            else:
                raise NotImplementedError('impact_measure=%s is unknown' % impact_measure)

            suscs.append(impact)
            nudges.append(nudge)
            if also_return_pdf_after:
                pdf_after.append(pdf.copy())

        if not also_return_pdf_after:
            return nudges, suscs
        else:
            return nudges, suscs, pdf_after


    def susceptibilities_local(self, num_output_variables, perturbation_size=0.1, ntrials=25, impact_measure='abs',
                               auto_reorder=True):
        return [self.susceptibility_local_single(varid, num_output_variables, perturbation_size=perturbation_size,
                                                 impact_measure=impact_measure, ntrials=ntrials,
                                                 auto_reorder=auto_reorder)
                for varid in range(len(self) - num_output_variables)]


    # def susceptibility_local(self, num_output_variables, perturbation_size=0.1, ntrials=25, auto_reorder=True):
    #     return np.mean([self.susceptibility_local_single(varid, num_output_variables,
    #                                                      perturbation_size=perturbation_size,
    #                                                      ntrials=ntrials, auto_reorder=auto_reorder)
    #                     for varid in xrange(len(self) - num_output_variables)])


    def susceptibility_global(self, num_output_variables, perturbation_size=0.1, ntrials=25,
                              impact_measure='hellinger'):
        """
        Perturb the current pdf Pr(X,Y) by changing Pr(X) slightly to Pr(X'), but keeping Pr(Y|X) fixed. Then
        measure the relative change in mutual information I(X:Y). Do this by Monte Carlo using <ntrials> repeats.
        :param num_output_variables: the number of variables at the end of the list of variables which are considered
        to be Y. The first variables are taken to be X. If your Y is mixed with the X, first do reordering.
        :param perturbation_size:
        :param ntrials:
        :return: expected relative change of mutual information I(X:Y) after perturbation
        :rtype: float
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables > 0, 'makes no sense to compute resilience with an empty set'

        original_mi = self.mutual_information(list(range(num_input_variables)),
                                              list(range(num_input_variables, num_input_variables + num_output_variables)))

        pdf_input_only = self.marginalize_distribution(list(range(num_input_variables)))

        affected_params = pdf_input_only.matrix2params_incremental()  # perturb Pr(X) to Pr(X')
        static_params = self.matrix2params_incremental()[len(affected_params):]  # keep Pr(Y|X) fixed

        pdf_perturbed = self.copy()  # will be used to store the perturbed Pr(X)Pr(Y'|X)

        def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
            return max(min(num, 1), 0)

        susceptibilities = []

        pdf_output_only = self[list(range(num_input_variables, len(self)))]

        for i in range(ntrials):
            perturbation = np.random.random(len(affected_params))
            perturbation = perturbation / np.linalg.norm(perturbation) * perturbation_size  # normalize vector norm

            pdf_perturbed.params2matrix_incremental(list(map(clip_to_unit_line, np.add(affected_params, perturbation)))
                                                    + static_params)

            # susceptibility = pdf_perturbed.mutual_information(range(num_input_variables),
            #                                                   range(num_input_variables,
            #                                                         num_input_variables + num_output_variables)) \
            #                  - original_mi

            pdf_perturbed_output_only = pdf_perturbed[list(range(num_input_variables, len(self)))]

            if impact_measure in ('kl', 'kld', 'kldiv'):
                susceptibility = pdf_output_only.kullback_leibler_divergence(pdf_perturbed_output_only)
            elif impact_measure in ('hell', 'h', 'hellinger'):
                susceptibility = pdf_output_only.hellinger_distance(pdf_perturbed_output_only)
            else:
                raise NotImplementedError('unknown impact_measure=%s' % impact_measure)

            susceptibilities.append(abs(susceptibility))

        return np.mean(susceptibilities) / original_mi


    def susceptibility_non_local(self, output_variables, variables_X1, variables_X2,
                                 perturbation_size=0.1, ntrials=25):

        if list(variables_X1) + list(variables_X2) + list(output_variables) == list(range(max(output_variables)+1)):
            # all variables are already nice ordered, so call directly susceptibility_non_local_ordered, maybe
            # only extraneous variables need to be deleted

            if len(self) > max(output_variables) + 1:
                pdf_lean = self[:(max(output_variables) + 1)]

                # should be exactly the same function call as in 'else', but on different pdf (removing extraneous
                # variables)
                return pdf_lean.susceptibility_non_local_ordered(len(output_variables),
                                                         num_second_input_variables=len(variables_X2),
                                                         perturbation_size=perturbation_size, ntrials=ntrials)
            else:
                return self.susceptibility_non_local_ordered(len(output_variables),
                                                     num_second_input_variables=len(variables_X2),
                                                     perturbation_size=perturbation_size, ntrials=ntrials)
        else:
            # first reorder, then remove extraneous variables, and then call susceptibility_non_local_ordered

            reordering_relevant = list(variables_X1) + list(variables_X2) + list(output_variables)

            extraneous_vars = np.setdiff1d(list(range(len(self))), reordering_relevant)

            reordering = reordering_relevant + list(extraneous_vars)

            pdf_reordered = self.copy()
            pdf_reordered.reorder_variables(reordering)

            pdf_reordered = pdf_reordered[:len(reordering_relevant)]  # keep only relevant variables (do in reorder?)

            return pdf_reordered.susceptibility_non_local_ordered(len(output_variables),
                                                                  num_second_input_variables=len(variables_X2),
                                                                  perturbation_size=perturbation_size, ntrials=ntrials)


    def kullback_leibler_divergence(self, other_pdf):

        div = 0.0

        assert len(other_pdf) == len(self), 'must have same number of variables (marginalize first)'

        for states in self.statespace():
            assert len(states) == len(other_pdf)

            if other_pdf(states) > 0:
                div += self(states) * np.log2(self(states) / other_pdf(states))
            else:
                if self(states) > 0:
                    div += 0.0
                else:
                    div = np.inf  # divergence becomes infinity if the support of Q is not that of P

        return div


    def hellinger_distance(self, other_pdf):

        div = 0.0

        assert len(other_pdf) == len(self), 'must have same number of variables (marginalize first)'

        for states in self.statespace():
            assert len(states) == len(other_pdf)

            div += np.power(np.sqrt(self(states)) - np.sqrt(other_pdf(states)), 2)

        return 1/np.sqrt(2) * np.sqrt(div)


    def susceptibility_non_local_ordered(self, num_output_variables, num_second_input_variables=1,
                                 perturbation_size=0.1, ntrials=25):
        """
        Perturb the current pdf Pr(X1,X2,Y) by changing Pr(X2|X1) slightly, but keeping Pr(X1) and Pr(X2) fixed. Then
        measure the relative change in mutual information I(X:Y). Do this by Monte Carlo using <ntrials> repeats.
        :param num_second_input_variables: number of variables making up X2.
        :param num_output_variables: the number of variables at the end of the list of variables which are considered
        to be Y. The first variables are taken to be X. If your Y is mixed with the X, first do reordering.
        :param perturbation_size:
        :param ntrials:
        :return: expected relative change of mutual information I(X:Y) after perturbation
        :rtype: float
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables >= 2, 'how can I do non-local perturbation if only 1 input variable?'

        assert num_input_variables > 0, 'makes no sense to compute resilience with an empty set'

        original_mi = self.mutual_information(list(range(num_input_variables)),
                                              list(range(num_input_variables, num_input_variables + num_output_variables)))

        susceptibilities = []

        pdf_inputs = self[list(range(num_input_variables))]

        cond_pdf_output = self - pdf_inputs

        pdf_outputs = self[list(range(num_input_variables, len(self)))]  # marginal Y

        for i in range(ntrials):
            # perturb only among the input variables
            resp = pdf_inputs.perturb_non_local(num_second_input_variables, perturbation_size)

            pdf_perturbed = resp.pdf + cond_pdf_output

            pdf_perturbed_outputs = pdf_perturbed[list(range(num_input_variables, len(self)))]  # marginal Y'

            # susceptibility = pdf_perturbed.mutual_information(range(num_input_variables),
            #                                                   range(num_input_variables,
            #                                                         num_input_variables + num_output_variables)) \
            #                  - original_mi

            # TODO: this compares the marginals of the outputs, but the MI between the old and new output which
            # would be better, but is more involved.
            susceptibility = pdf_outputs.kullback_leibler_divergence(pdf_perturbed_outputs)

            susceptibilities.append(abs(susceptibility))

        return np.mean(susceptibilities) / original_mi


    class PerturbNonLocalResponse(object):
        pdf = None  # JointProbabilityMatrix object
        # cost_same_output_marginal = None  # float, should be as close to zero as possible.
        # cost_different_relation = None  # float, should be as close to zero as possible
        perturb_size = None  # norm of vector added to params


    def susceptibility(self, variables_Y, variables_X='all', perturbation_size=0.1, only_non_local=False,
                       impact_measure='midiff'):

        # note: at the moment I only perturb per individual variable in X, not jointly; need to supprt this in
        # self.perturb()

        """
        :param impact_measure: 'midiff' for nudge control impact, 'relative' for normalized
        :type impact_measure: str
        """
        if variables_X in ('all', 'auto'):
            variables_X = list(np.setdiff1d(list(range(len(self))), variables_Y))

        assert len(variables_X) > 0, 'makes no sense to measure susceptibility to zero variables (X)'

        if impact_measure == 'relative':
            pdf_X = self[variables_X]  # marginalize X out of Pr(X,Y)
            cond_pdf_Y = self.conditional_probability_distributions(variables_X)

            if __debug__:
                ent_X = pdf_X.entropy()

            pdf_XX = pdf_X.copy()
            pdf_XX.append_variables_using_state_transitions_table(lambda x, nv: x)  # duplicate X

            if __debug__:
                # duplicating X should not increase entropy in any way
                np.testing.assert_almost_equal(ent_X, pdf_XX.entropy())

            for x2id in range(len(variables_X), 2*len(variables_X)):
                pdf_XX.perturb([x2id], perturbation_size=perturbation_size, only_non_local=only_non_local)

            if __debug__:
                # I just go ahead and assume that now the entropy must have increased, since I have exerted some
                # external (noisy) influence
                # note: hitting this assert is highly unlikely but could still happen, namely if the perturbation(s)
                # fail to happen
                # note: could happen if some parameter(s) are 0 or 1, because then there are other parameter values which
                # are irrelevant and have no effect, so those could be changed to satisfy the 'perturbation' constraint
                if not 0 in pdf_XX.matrix2params_incremental() and not 1 in pdf_XX.matrix2params_incremental():
                    assert ent_X != pdf_XX.entropy(), 'ent_X=' + str(ent_X) + ', pdf_XX.entropy()=' + str(pdf_XX.entropy())

            pdf_XXYY = pdf_XX.copy()
            pdf_XXYY.append_variables_using_conditional_distributions(cond_pdf_Y, list(range(len(variables_X))))
            pdf_XXYY.append_variables_using_conditional_distributions(cond_pdf_Y, list(range(len(variables_X), 2*len(variables_X))))


            ent_Y = pdf_XXYY.entropy(list(range(2*len(variables_X), 2*len(variables_X) + len(variables_Y))))
            mi_YY = pdf_XXYY.mutual_information(list(range(2*len(variables_X), 2*len(variables_X) + len(variables_Y))),
                                                list(range(2*len(variables_X) + len(variables_Y),
                                                      2*len(variables_X) + 2*len(variables_Y))))

            impact = 1.0 - mi_YY / ent_Y
        elif impact_measure in ('midiff',):
            pdf2 = self.copy()

            pdf2.perturb(variables_X, perturbation_size=perturbation_size, only_non_local=only_non_local)

            impact = np.nan  # pacify IDE
            assert False, 'todo: finish implementing, calc MI in the two cases and return diff'

        return impact


    # helper function
    # NOTE: the preferred function now is `generate_nudge_within_bounds`; this function is naive and can fail especially for conditional
    # variables which have a large dependency to other variables.
    # TODO: replace this with `generate_nudge_within_bounds` and rename to this again.
    def generate_nudge(self, epsilon, shape):
        nudge_vec = np.random.dirichlet([1] * np.product(shape))
        nudge_vec -= 1. / np.product(shape)
        norm = np.linalg.norm(nudge_vec)
        nudge_vec = np.reshape(nudge_vec / norm * epsilon, newshape=shape)

        return nudge_vec

    # proposed replacement of the `generate_nudge` helper function, which always returns a solution within bounds OR reports that it is not possible
    @staticmethod
    def generate_nudge_within_bounds(nudge_norm, min_cond_probs, max_cond_probs, rel_tol_norm=0.1, tol_zero_mean=1e-2 * _essential_zero_prob, rel_weight_zero_dev=1.0, scipy_tol_factor=1e-10,
                                     max_retries=5, verbose=0, scipy_min_method=None):
        shape = np.shape(min_cond_probs)

        assert np.shape(max_cond_probs) == shape, 'min and max conditional probabilities should be the same shape'
        assert np.max(max_cond_probs) <= 1.0 + _mi_error_tol
        assert np.min(max_cond_probs) >= 0.0
        assert np.max(min_cond_probs) <= 1.0 + _mi_error_tol
        assert np.min(min_cond_probs) >= 0.0

        tol_norm = rel_tol_norm * nudge_norm

        # individual values of the nudge vector should not be larger than +epsilon, or smaller if that would make
        # the corresponding probability (possibly) out of bounds (>1):
        max_nudge_vals = np.min([np.ones(shape) * nudge_norm, 1.0 - max_cond_probs], axis=0)
        # individual values of the nudge vector should not be lower than -epsilon, or higher (closer to 0) if that would make
        # the corresponding probability (possibly) out of bounds (<0):
        min_nudge_vals = np.max([-1.0 * np.ones(shape) * nudge_norm, -min_cond_probs], axis=0)

        if verbose > 0:
            print(f'debug: generate_nudge_within_bounds: {min_nudge_vals=}')
            print(f'debug: generate_nudge_within_bounds: {max_nudge_vals=}')
        
        # some (non-tight) upper and lower bound checks to see if finding a nudge vector is at all possible.
        # `max_nudge_vals` is not a tight upper bound of the nudge vector because a nudge vector cannot consist
        # only of positive values; at least one of the entries must be negative such that the sum of the vector
        # is zero. But this is a quick enough check for at least the most pathological cases (such as conditional
        # probabilities consisting of both 0s and 1s, meaning that no non-zero nudge vector could be added to both).

        # CILLIAN commented and Uncommented
        if np.linalg.norm(max_nudge_vals) < nudge_norm - tol_norm:
            raise UserWarning(f'impossible to find a nudge vector, because the (not tight) upper bound of the nudge vector is {max_nudge_vals} which has norm {np.linalg.norm(max_nudge_vals)}, which is not close enough to {tol_norm}.')
        if np.linalg.norm(min_nudge_vals) < nudge_norm - tol_norm:
            raise UserWarning(f'impossible to find a nudge vector, because the (not tight) lower bound of the nudge vector is {min_nudge_vals} which has norm {np.linalg.norm(min_nudge_vals)}, which is not within {tol_norm} of {nudge_norm}.')
        

        # cost function to be minimized
        def f_cost_nudge_vec(nudge_vec_1d, rel_weight_zero_dev=1.0):
            nudge_vec = np.reshape(nudge_vec_1d, newshape=shape)

            # todo: remove after a while:
            assert np.all(np.less_equal(nudge_vec, max_nudge_vals)), 'proposed vector by scipy.optimize.minimize is out of the specified bounds (should not be)'
            assert np.all(np.greater_equal(nudge_vec, min_nudge_vals)), 'proposed vector by scipy.optimize.minimize is out of the specified bounds (should not be)'

            norm = np.linalg.norm(nudge_vec)
            mean_zero_deviation = np.sum(nudge_vec)

            return (norm - nudge_norm)**2.0 + np.sqrt(len(nudge_vec_1d)) * rel_weight_zero_dev * mean_zero_deviation**2.0
        
        # formulate the bounds of the nudge vector for scipy's minimize
        bounds = np.transpose([np.reshape(min_nudge_vals, newshape=np.product(shape)), 
                               np.reshape(max_nudge_vals, newshape=np.product(shape))])  # now shaped as requested by scipy's minimize()
    
        for trial in range(max_retries):  # 'minimize' sometimes finds no solution but then re-run it does; so run it a few times if needed
            if trial == 0:
                # try first to start from the zero vector once
                initial_guess = np.zeros(shape=np.product(shape))  # flatten the epsilon vector because scipy minimize can only handle that
            else:
                # formulate an initial guess for the nudge vector; it is at least within bounds and it is [not-necessarily-]near-zero mean, but incorrect norm
                # (since rescaling the initial guess might make it go out of bounds for at least one of the entries; let scipy deal with it)
                initial_guess = np.random.dirichlet([1] * np.product(shape))
                initial_guess -= 1. / np.product(shape)
                initial_guess = np.reshape(initial_guess, newshape=shape)
                initial_guess = min_nudge_vals + (max_nudge_vals - min_nudge_vals) * initial_guess

                initial_guess = np.reshape(initial_guess, newshape=np.product(shape))  # flatten the epsilon vector because scipy minimize can only handle that

            optres = minimize(f_cost_nudge_vec, initial_guess, bounds=bounds, args=(rel_weight_zero_dev,), 
                              tol=scipy_tol_factor * (tol_norm**2.0 + rel_weight_zero_dev * tol_zero_mean**2.0), method=scipy_min_method)

            if optres.success:
                # only if the norm condition is also satisfied then stop the search process (note: the "zero mean" condition is not tested
                # here but if that is the only violating condition then it is heuristically corrected below anyway)
                if not abs(np.linalg.norm(optres.x) - nudge_norm) > tol_norm:
                    break
                else:
                    if verbose > 1:
                        print(f'debug: minimize() succeeded but the resulting nudge norm is not within tolerance')
            else:
                if verbose > 1:
                    print(f'debug: scipy minimize failed. Error message: {optres.message}')

        if not optres.success:
            raise UserWarning(f'scipy.optimize.minimize() failed to find any suitable nudge vector. Error message: {optres.message}')
        else:
            nudge_vec = np.reshape(optres.x, newshape=shape)

            if verbose > 0:
                print(f'debug: generate_nudge_within_bounds: {nudge_vec=}')

            assert np.all(np.less_equal(nudge_vec, max_nudge_vals)), 'solution vector by scipy.optimize.minimize is out of the specified bounds (should not be)'
            assert np.all(np.greater_equal(nudge_vec, min_nudge_vals)), 'solution vector by scipy.optimize.minimize is out of the specified bounds (should not be)'

            norm = np.linalg.norm(nudge_vec)
            sum_zero_deviation = np.sum(nudge_vec)  # this is preferably 0.0

            # CILLIAN Uncommented and changed elif to if --- Then undone this
            if abs(norm - nudge_norm) > tol_norm:
                raise UserWarning(f'the returned solution (nudge vector) of scipy.optimize.minimize() has {norm=} (desired: {nudge_norm}), which is a larger ' \
                                  + 'difference than the allowed tolerance of {tol_norm}.')
            elif abs(sum_zero_deviation) > tol_zero_mean:
                ### a post-processing hack to make the mean closer to zero by adding/subtracting somewhere a little correction

                if sum_zero_deviation < 0.0:
                    corr_factor = -1.0

                    diffs = max_nudge_vals - nudge_vec  # this is a vector showing our 'budget'; how much can still be added to nudge_vec
                else:
                    corr_factor = 1.0

                    diffs = nudge_vec - min_nudge_vals  # this is a vector showing our 'budget'; how much can still be subtracted from nudge_vec

                if verbose > 0:
                    print(f'error: {sum_zero_deviation=}: will try to fix it. Budget: {max_nudge_vals - nudge_vec=}')

                # let's simplify by only trying to add to the index/indices with maximum 'budget' (this is not perfect but convenient)
                max_diff = np.max(diffs)
                diffs = np.array(diffs == max_diff, dtype=float)

                nummax = diffs.sum()

                if max_diff * nummax >= sum_zero_deviation * corr_factor:  # good news: it is possible to add/subtract the `sum_zero_deviation` to/from `nudge_vec`
                    diffs *= sum_zero_deviation / nummax

                    # note: if `sum_zero_deviation` is negative then here we add since diffs will have negative entries; otherwise we subtract
                    nudge_vec -= diffs

                    if __debug__:
                        old_sum_zero_deviation = sum_zero_deviation

                    sum_zero_deviation = np.sum(nudge_vec)  # this should now basically be 0.0

                    if verbose > 0:
                        print(f'{diffs=} was added to nudge_vec; this is the new {sum_zero_deviation=}, which should now basically be 0.0.')

                    assert abs(sum_zero_deviation) <= tol_zero_mean, f'I did a correction so should now be zero: the new {sum_zero_deviation=}; {tol_zero_mean=}, {old_sum_zero_deviation=}'

                    if abs(norm - nudge_norm) > tol_norm:
                        # sum_zero_deviation is usually very small, like 1e-09, so hopefully this does not happen too often (violating the
                        # norm while before this correction the norm was fine):
                        raise UserWarning(f'after fixing the mean: the new solution (nudge vector) now has {norm=} (desired: {nudge_norm}), which is a larger difference than ' \
                                          + 'the allowed tolerance of {tol_norm} (while before correcting it was not). On the other hand, now: the new {sum_zero_deviation=}, {tol_zero_mean=}.')
                else:
                    if verbose > 0:
                        print(f'{sum_zero_deviation=} cannot easily be added to nudge_vec because {max_nudge_vals - nudge_vec=}')
                
                    raise UserWarning(f'the returned solution (nudge vector) of scipy.optimize.minimize() has a mean of {sum_zero_deviation} (desired: 0.0), which is a ' \
                                      + f' larger difference than the allowed tolerance of {tol_zero_mean}. [{norm=}, desired={nudge_norm}]. ' \
                                      + f'I tried to fix it heuristically but I could not.')
            
            return nudge_vec


    def nudge_single(self, perturbed=0, eps_norm=0.01, method='invariant', verbose=0):
        """
        In a bivariate distribution p(X,Y), nudge a second variable Y's marginal probabilities
        without affecting the first (X). It does this by changing the conditional probabilities p(y|x).

        :param perturbed: which variable index to nudge. Must be an index in the range 0..numvariables-1.
        :param eps_norm: norm of zero-sum vector to be added to Y's marginal probability vector.
        :type ntrials: int
        :returns: nudge vector applied.
        :param method: 'invariant' means that a single nudge vector is generated and applied to all conditional
        distributions p(Y|X=x) for all x
        :rtype: np.array
        """

        assert method == 'invariant', 'no other method implemented yet'

        # in the notation: X = variable(s) which are not nudged AND not effected by the nudge (causal predecessors);
        # Y = perturbed variable(s). Z = downstream variables potentially influenced by the nudge (not nudged themselves)

        # note: code below assumes perfect ordering (X, Y, Z) and contiguous ranges of indices (xix+yix+zix=range(n))
        xix = list(range(perturbed))
        yix = list(range(perturbed, perturbed + 1))
        zix = list(range(perturbed + 1, self.numvariables))

        pdfXYZ = self  # shorthand
        pdfXY = self.marginalize_distribution(xix + yix)  # shorthand, readable

        # assert len(zix) == 0, 'currently assumed by the code, but not hard to implement (store cond. pdf and then add back)'
        assert len(yix) > 0, 'set of variables to nudge should not be empty'

        pX = pdfXY.marginalize_distribution(xix)
        pY_X = pdfXY.conditional_probability_distributions(xix, yix)
        pY = pdfXY.marginalize_distribution(yix)

        # remove Z but later add it/them back by means of their conditional pdf
        pZ_XY = pdfXYZ.conditional_probability_distributions(xix + yix, zix)

        nY = len(list(pY.statespace()))  # shorthand

        # make pY_X in list format, condprobs[yi][xi] is p(yi|xi)
        condprobs = np.array([[pY_X[xi](yi) for xi in pX.statespace()] for yi in pY.statespace()], dtype=_type_prob)

        # np.testing.assert_array_almost_equal(np.sum(condprobs, axis=0), np.ones(nY))

        # note: in total, sum_xi condprobs[yi][xi] must change by amount epsilon[yi], but we
        # have to divide this change into |Y| subterms which sum to epsilon[yi]...

        pXprobs = np.array(pX.joint_probabilities.joint_probabilities, dtype=_type_prob)

        # np.testing.assert_array_almost_equal(
        #     [np.sum(condprobs[yix] * pXprobs) for yix, yi in enumerate(pY.statespace())],
        #     pY.joint_probabilities.joint_probabilities)
        # np.testing.assert_array_almost_equal(np.sum(pXprobs * condprobs, axis=1),
        #                                      pY.joint_probabilities.joint_probabilities)

        # across all conditional distributions p(Y|X=x) these are the min. and max. probabilities for Y, so the
        # nudge vector (which will be added to all vectors p(Y|X=x) for all x) cannot exceed these bounds since
        # otherwise there will be probabilities out of the range [0,1]
        min_cond_probs_Y = np.min(condprobs, axis=1)
        max_cond_probs_Y = np.max(condprobs, axis=1)

        if False:  # working on replacing this with a different helper function that just returns a suitable vector; raises an error otherwise
            tol_sum_eps = 1e-10
            ntrials_clip = 30
            tol_norm_rel = 0.1  # relative error allowed for the norm of the epsilon vector (0.1=10% error)
            ntrials_norm = 20

            for j in range(ntrials_norm):
                # generate a nudge vector
                epsilon = pY.generate_nudge(eps_norm, np.shape(pY.joint_probabilities.joint_probabilities))

                epsilon = np.array(epsilon, dtype=_type_prob)  # try to reduce roundoff errors below

                # clip the nudge vector to stay within probabilities [0,1]
                # TODO: generate directly a nudge vector within these bounds? in the helper function?
                for i in range(ntrials_clip):  # try to clip the nudge vector within a number of trials
                    epsilon = np.max([epsilon, -min_cond_probs_Y], axis=0)
                    epsilon = np.min([epsilon, 1.0 - max_cond_probs_Y], axis=0)

                    if np.abs(np.sum(epsilon)) < tol_sum_eps:
                        # print 'debug: took %s trials to find a good epsilon after clipping' % (i + 1)
                        break
                    else:
                        epsilon -= np.mean(epsilon)

                if not np.abs(np.sum(epsilon)) < tol_sum_eps:
                    # raise UserWarning('error: did not manage to make a nudge vector (epsilon) sum to zero after clipping')
                    pass  # next loop iteration
                elif np.linalg.norm(epsilon) < eps_norm * (1.0 - tol_norm_rel):
                    # raise UserWarning('error: did not manage to keep the norm within tolerance (%s not close enough to %s)' % (np.linalg.norm(epsilon), eps_norm))
                    pass  # next loop iteration
                else:
                    # print 'debug: epsilon: %s -- norm: %s -- sum %s (trial %s)' % (epsilon, np.linalg.norm(epsilon), np.sum(epsilon), j+1)
                    break  # successfully found an epsilon vector that matches the desired norm and sums to approx. zero

            if not np.abs(np.sum(epsilon)) < tol_sum_eps:
                raise UserWarning('debug: did not manage to make a nudge vector sum to zero after clipping (%s)' % np.sum(epsilon))
            elif np.linalg.norm(epsilon) < eps_norm * (1.0 - tol_norm_rel):
                raise UserWarning('debug: did not manage to keep the norm within tolerance (%s)' % np.linalg.norm(epsilon))
        else:
            tol_norm_rel = 0.1  # relative error allowed for the norm of the epsilon vector (0.1=10% error)
            tol_sum_eps = 1e-10

            epsilon = JointProbabilityMatrix.generate_nudge_within_bounds(eps_norm, min_cond_probs_Y, max_cond_probs_Y, rel_tol_norm=tol_norm_rel, tol_zero_mean=tol_sum_eps, verbose=verbose-1)

        nudged_pdf = pX.copy()  # first add the (unchanged) X

        new_pY_x = lambda x: pdfXY.conditional_probability_distribution(xix, x).joint_probabilities.joint_probabilities + epsilon

        nudged_pdf.append_variables_using_conditional_distributions({x: JointProbabilityMatrix(len(yix), nY, new_pY_x(x))
                                                                     for x in pX.statespace()}, xix)

        # add back the Z through the conditional pdf (so p(Z) may now be changed if p(Z|Y) != p(Z))
        nudged_pdf.append_variables_using_conditional_distributions(pZ_XY)

        self.duplicate(nudged_pdf)

        return epsilon


    # WARNING: nudge_single() is now preferred!
    # todo: remove this and use only nudge_single()
    def nudge(self, perturbed_variables, output_variables, epsilon=0.01, marginalize_prior_ignored=False,
              method='fixed', nudge=None, verbose=True):
        """

        :param perturbed_variables:
        :param output_variables:
        :param epsilon:
        :param marginalize_prior_ignored:
        :param method:
            'random': a nudge is drawn uniformly random from the hypercube [-epsilon, +epsilon]^|P| or whatever
            subspace of that cube that still leaves all probabilities in the range [0,1]
            'fixed': a nudge is a random vector with norm epsilon, whereas for 'random' the norm can be (much) smaller
        :return:
        """
        perturbed_variables = list(sorted(set(perturbed_variables)))
        output_variables = list(sorted(set(output_variables)))
        ignored_variables = list(np.setdiff1d(list(range(len(self))), perturbed_variables + output_variables))

        # caution: trial of a new way to do a nudge if there are causal predecessors to the perturbed variables,
        # namely: first marginalize out the perturbed variables, do the nudge, and then glue back together by
        # slightly changing the conditional
        # if min(perturbed_variables) > 0:


        # this performs a nudge on each conditional pdf of the perturbed variables, given the values for variables
        # which appear on the lefthand side of them (these are causal predecessors, which should not be affected by
        # the nudge)
        if min(perturbed_variables + output_variables) > 0 and marginalize_prior_ignored:
            # these variables appear before all perturbed and output variables so there should be no causal effect on
            # them (marginal probabilities the same, however the conditional pdf with the perturbed variables
            # may need to change to make the nudge happen, so here I do that (since nudge() by itself tries to fixate
            # both marginals and conditionals of prior variables, which is not always possible (as e.g. for a copied variable)
            ignored_prior_variables = list(range(min(perturbed_variables + output_variables)))

            pdf_prior = self[ignored_prior_variables]  # marginal of prior variables

            # conditional pdfs of the rest
            cond_pdfs = self.conditional_probability_distributions(ignored_prior_variables)

            # the pdf's in the conditional cond_pdfs now are missing the first len(ignored_prior_variables)
            new_pv = np.subtract(perturbed_variables, len(ignored_prior_variables))
            new_ov = np.subtract(output_variables, len(ignored_prior_variables))
            new_iv = np.subtract(ignored_variables, len(ignored_prior_variables))

            nudge_dict = dict()

            for k in cond_pdfs.keys():
                # do an independent and random nudge for each marginal pdf given the values for the prior variables...
                # for each individual instance this may create a correlation with the prior variables, but on average
                # (or for large num_values) it will not.
                nudge_dict[k] = cond_pdfs[k].nudge(perturbed_variables=new_pv, output_variables=new_ov, epsilon=epsilon)

            new_pdf = pdf_prior + cond_pdfs

            self.duplicate(new_pdf)

            return nudge_dict
        # code assumes
        # if max(perturbed_variables) == len(perturbed_variables) - 1 \
        #         and max(output_variables) == len(perturbed_variables) + len(output_variables) - 1:
        # todo: doing this rearrangement seems wrong? because the causal predecessors should be first in the list
        # of variables, so there should be no causation from a 'later' variable back to an 'earlier' variable?
        # elif max(perturbed_variables) == len(perturbed_variables) - 1:
        elif True:  # see if this piece of code can also handle ignored variables to precede the perturbed variables
            pdf_P = self[perturbed_variables]  # marginalize X out of Pr(X,Y)
            pdf_I = self[ignored_variables]
            pdf_IO_P = self.conditional_probability_distributions(perturbed_variables)
            pdf_P_I = self.conditional_probability_distributions(ignored_variables,
                                                                 object_variables=perturbed_variables)
            pdf_O_IP = self.conditional_probability_distributions(ignored_variables + perturbed_variables,
                                                                 object_variables=output_variables)

            # initial limits, to be refined below
            max_nudge = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
            min_nudge = -np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon

            for sx in pdf_P.statespace():
                for sy in iter(pdf_IO_P.values()).next().statespace():
                    if pdf_IO_P[sx](sy) != 0.0:
                        # follows from equality (P(p) + nudge(p)) * P(rest|p) == 1
                        # max_nudge[sx] = min(max_nudge[sx], 1.0 / pdf_IO_P[sx](sy) - pdf_P(sx))
                        # I think this constraint is always superseding the above one, so just use this:
                        max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P(sx))
                    else:
                        pass  # this pair of sx and sy is impossible so no need to add a constraint for it?
                # note: I removed the division because in the second line it adds nothing
                # min_nudge[sx + sy] = 0.0 / pdf_IO_P[sx](sy) - pdf_P(sx)
                min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P(sx))

                # same nudge will be added to each P(p|i) so must not go out of bounds for any pdf_P_I[si]
                for pdf_P_i in pdf_P_I.values():
                    max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P_i(sx))
                    min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P_i(sx))

                # I think this should not happen. Worst case, max_nudge[sx + sy] == min_nudge[sx + sy], like
                # when P(p)=1 for I=1 and P(p)=0 for I=1, then only a nudge of 0 could be added to P(b).
                # small round off error?
                # alternative: consider making the nudge dependent on I, which gives more freedom but I am not sure
                # if then e.g. the expected correlation is still zero. (Should be, right?)
                assert max_nudge[sx] >= min_nudge[sx], 'unsatisfiable conditions for additive nudge!'

                # note: this is simply a consequence of saying that a nudge should change only a variable's own
                # probabilities, not the conditional probabilities of this variable given other variables
                # mechanism)
                # NOTE: although... it seems inevitable that conditional probabilities change?
                # NOTE: well, in the derivation you assume that the connditional p(B|A) does not change -- at the moment
                # (try it?)
                if max_nudge[sx] == min_nudge[sx]:
                    warnings.warn('max_nudge[sx] == min_nudge[sx], meaning that I cannot find a single nudge'
                                  ' epsilon_a for all ' + str(len(list(pdf_P_I.values())))
                                  + ' pdf_P_i of pdf_P_I=P(perturbed_variables | ignored_variables='
                                  + str(ignored_variables) + '); sx=' + str(sx))
                    # NOTE: implement a different nudge for each value for ignored_variables? Then this does not happen
                    # NOTE: then the nudge may correlate with ignored_variables, which is not what you want because
                    # then you could be adding correlations?

            range_nudge = max_nudge - min_nudge

            max_num_trials_nudge_gen = 10000
            max_secs = 20.
            time_before = time.time()
            if nudge is None:
                for trial in range(max_num_trials_nudge_gen):
                    if method == 'random' or method == 'fixed':
                        nudge = np.random.random(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * range_nudge \
                                + min_nudge

                        # sum_nudge = np.sum(nudge)  # should become 0, but currently will not be
                        # correction = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) / nudge.size * sum_nudge
                        # nudge -= correction
                        nudge -= np.mean(nudge)  # make sum up to 0

                        if method == 'fixed':  # make sure the norm is right
                            nudge *= epsilon / np.linalg.norm(nudge)

                        if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                            break  # found a good nudge!
                        else:
                            if trial == max_num_trials_nudge_gen - 1:
                                raise UserWarning('max_num_trials_nudge_gen=' + str(max_num_trials_nudge_gen)
                                                  + ' was not enough to find a good nudge vector. '
                                                    'max_nudge=%s (norm: %s), min_nudge=%s'
                                                  % (max_nudge, np.linalg.norm(max_nudge), min_nudge))
                            elif time.time() - time_before > max_secs:
                                raise UserWarning('max_secs=%s was not enough to find a good nudge vector. '
                                                  'trial=%s out of %s' % (max_secs, trial, max_num_trials_nudge_gen))
                            else:
                                continue  # darn, let's try again
                    else:
                        raise NotImplementedError('unknown method: %s' % method)
                # todo: if the loop above fails then maybe do a minimize() attempt?
            else:
                nudge = np.array(nudge)

                if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                    pass  # done
                else:
                    orig_nudge = copy.deepcopy(nudge)

                    def cost(nudge):
                        overshoot = np.sum(np.max([nudge - max_nudge, np.zeros(nudge.shape)], axis=0))
                        undershoot = np.sum(np.max([min_nudge - nudge, np.zeros(nudge.shape)], axis=0))
                        # these two are actually not needed I think I was thinking of probabilities when I typed it:
                        overbound = np.sum(np.max([nudge - np.ones(nudge.shape), np.zeros(nudge.shape)], axis=0))
                        underbound = np.sum(np.max([np.zeros(nudge.shape) - nudge, np.zeros(nudge.shape)], axis=0))

                        dist = np.linalg.norm(nudge - orig_nudge)

                        return np.sum(np.power([overshoot, undershoot, overbound, underbound], 2)) \
                               + np.power(dist, 2) \
                               + np.power(np.linalg.norm(nudge) - epsilon, 1.0)  # try to get a nudge of desired norm

                    optres = minimize(cost, nudge)

                    max_num_minimize = 5
                    trial = 0
                    while not optres.success and trial < max_num_minimize:
                        trial += 1
                        optres = minimize(cost, nudge)

                    nudge = optres.x

                    if not optres.success:
                        raise UserWarning('You specified nudge=%s but it would make certain probabilities out of '
                                          'bounds (0,1). So I tried a minimize() %s times but it failed.'
                                          '\nmin_nudge=%s'
                                          '\nmax_nudge=%s' % (orig_nudge, max_num_minimize, min_nudge, max_nudge))
                    elif not np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                        raise UserWarning('You specified nudge=%s but it would make certain probabilities out of '
                                          'bounds (0,1). So I tried a minimize() step but it found a nudge still out '
                                          'of the allowed min and max nudge. cost=%s. I tried %s times.'
                                          '\nmin_nudge=%s'
                                          '\nmax_nudge=%s' % (orig_nudge, optres.fun, max_num_minimize,
                                                              min_nudge, max_nudge))
                    elif verbose > 0:
                        print('debug: you specified nudge=%s but it would make certain probabilities out of ' \
                              'bounds (0,1). So I tried a minimize() step and I got nudge=%s (norm %s, cost %s)' \
                              % (nudge, np.linalg.norm(nudge), optres.fun))

            if __debug__:
                np.testing.assert_almost_equal(np.sum(nudge), 0.0, decimal=10)  # todo: remove after a while

            # this is the point of self.type_prob ...
            assert type(next(pdf_P.joint_probabilities.joint_probabilities.flat)) == self._type_prob

            # if len(ignored_variables) > 0:  # can this become the main code, so not an if? (also works if
            if True:
                # len(ignored*) == 0?)
                # new attempt for making the new joint pdf
                new_joint_probs = -np.ones(np.shape(self.joint_probabilities.joint_probabilities))

                # print 'debug: ignored_variables = %s, len(self) = %s' % (ignored_variables, len(self))

                for s in self.statespace():
                    # shorthands: the states pertaining to the three subsets of variables
                    sp = tuple(np.take(s, perturbed_variables))
                    so = tuple(np.take(s, output_variables))
                    si = tuple(np.take(s, ignored_variables))

                    new_joint_probs[s] = pdf_I(si) * min(max((pdf_P_I[si](sp) + nudge[sp]), 0.0), 1.0) * pdf_O_IP[si + sp](so)

                    # might go over by a tiny bit due to roundoff, then just clip
                    if -1e-10 < new_joint_probs[s] < 0:
                        new_joint_probs[s] = 0
                    elif 1 < new_joint_probs[s] < 1 + 1e-10:
                        new_joint_probs[s] = 1

                    assert 0 <= new_joint_probs[s] <= 1, 'new_joint_probs[s] = ' + str(new_joint_probs[s])

                self.reset(self.numvariables, self.numvalues, new_joint_probs)
            else:
                print('debug: perturbed_variables = %s, len(self) = %s' % (perturbed_variables, len(self)))

                new_probs = pdf_P.joint_probabilities.joint_probabilities + nudge

                # this is the point of self.type_prob ...
                assert type(next(new_probs.flat)) == self._type_prob

                # # todo: remove test once it works a while
                # if True:
                #     assert np.max(new_probs) <= 1.0, 'no prob should be >1.0: ' + str(np.max(new_probs))
                #     assert np.min(new_probs) >= 0.0, 'no prob should be <0.0: ' + str(np.min(new_probs))

                # assert np.max(nudge) <= abs(epsilon), 'no nudge should be >' + str(epsilon) + ': ' + str(np.max(nudge))
                # assert np.min(nudge) >= -abs(epsilon), 'no nudge should be <' + str(-epsilon) + ': ' + str(np.min(nudge))

                # total probability mass should be unchanged  (todo: remove test once it works a while)
                if __debug__:
                    np.testing.assert_almost_equal(np.sum(new_probs), 1)

                if __debug__:
                    pdf_X_orig = pdf_P.copy()

                pdf_P.joint_probabilities.reset(new_probs)

                if len(pdf_IO_P) > 0:
                    if len(next(iter(pdf_IO_P.values()))) > 0:  # non-zero number of variables in I or O?
                        self.duplicate(pdf_P + pdf_IO_P)
                    else:
                        self.duplicate(pdf_P)
                else:
                    self.duplicate(pdf_P)

            return nudge
        else:
            # reorder the variables such that the to-be-perturbed variables are first, then call the same function
            # again (ending up in the if block above) and then reversing the reordering.

            # output_variables = list(np.setdiff1d(range(len(self)), perturbed_variables))

            # WARNING: this seems incorrect, now there is also causal effect on variables which appear before
            # all perturbed variables, which is not intended as the order in which variables appear give a partial
            # causality predecessor ordering
            # todo: remove this else-clause? Now the above elif-clause can handle everything

            self.reorder_variables(perturbed_variables + ignored_variables + output_variables)

            ret = self.nudge(list(range(len(perturbed_variables))), output_variables, epsilon=epsilon)

            self.reverse_reordering_variables(perturbed_variables + ignored_variables + output_variables)

            return ret

    # # I adopt the convention that variables are ordered in a causal predecessor partial ordering,
    # # so variable 1 cannot causally influence 0 but 0 *can* causally influence 1. This ordering should always be
    # # possible (otherwise add time to make loops). Ignored variables are marginalized out first
    # # todo: does not work satisfactorily
    # def nudge_new(self, perturbed_variables, output_variables, force_keep=None, epsilon=0.01):
    #
    #     if force_keep is None:
    #         force_keep = []
    #
    #     perturbed_variables = sorted(list(set(perturbed_variables)))
    #     output_variables = sorted(list(set(output_variables)))
    #     ignored_variables = list(np.setdiff1d(range(len(self)), perturbed_variables + output_variables))
    #     ignored_variables = list(np.setdiff1d(ignored_variables, force_keep))
    #
    #     assert len(set(perturbed_variables + output_variables)) == len(perturbed_variables) + len(output_variables), \
    #         'perturbed_variables=%s and output_variables%s should not overlap' % (perturbed_variables, output_variables)
    #
    #     if len(ignored_variables) > 0:
    #         self.duplicate(self.marginalize_distribution(perturbed_variables + output_variables))
    #
    #         perturbed_variables = [pv - np.sum(np.less(ignored_variables, pv)) for pv in perturbed_variables]
    #         output_variables = [ov - np.sum(np.less(ignored_variables, ov)) for ov in output_variables]
    #
    #         assert len(list(np.setdiff1d(range(len(self)), perturbed_variables + output_variables))) == 0, \
    #             'perturbed_variables=%s and output_variables%s should cover all variables now: %s' \
    #             % (perturbed_variables, output_variables,
    #                np.setdiff1d(range(len(self)), perturbed_variables + output_variables))
    #
    #         return self.nudge_new(perturbed_variables=perturbed_variables,
    #                               output_variables=output_variables,
    #                               epsilon=epsilon)
    #
    #     # code assumes
    #     # if max(perturbed_variables) == len(perturbed_variables) - 1 \
    #     #         and max(output_variables) == len(perturbed_variables) + len(output_variables) - 1:
    #     # todo: doing this rearrangement seems wrong? because the causal predecessors should be first in the list
    #     # of variables, so there should be no causation from a 'later' variable back to an 'earlier' variable?
    #     # if max(perturbed_variables) == len(perturbed_variables) - 1:
    #     if True:  # test
    #         pdf_P = self[perturbed_variables]  # marginalize X out of Pr(X,Y)
    #         pdf_I = self[ignored_variables]
    #         pdf_IO_P = self.conditional_probability_distributions(perturbed_variables)
    #         pdf_P_I = self.conditional_probability_distributions(ignored_variables,
    #                                                              object_variables=perturbed_variables)
    #         pdf_O_IP = self.conditional_probability_distributions(ignored_variables + perturbed_variables,
    #                                                               object_variables=output_variables)
    #
    #         # initial limits, to be refined below
    #         max_nudge = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
    #         min_nudge = -np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
    #
    #         for sx in pdf_P.statespace():
    #             for sy in pdf_IO_P.itervalues().next().statespace():
    #                 # follows from (P(p) + nudge(p)) * P(rest|p) == 1
    #                 max_nudge[sx] = min(max_nudge[sx], 1.0 / pdf_IO_P[sx](sy) - pdf_P(sx))
    #             # note: I removed the division because in the second line it adds nothing
    #             # min_nudge[sx + sy] = 0.0 / pdf_IO_P[sx](sy) - pdf_P(sx)
    #             min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P(sx))
    #
    #             # same nudge will be added to each P(p|i) so must not go out of bounds for any pdf_P_I[si]
    #             for pdf_P_i in pdf_P_I.itervalues():
    #                 max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P_i(sx))
    #                 min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P_i(sx))
    #
    #             # I think this should not happen. Worst case, max_nudge[sx + sy] == min_nudge[sx + sy], like
    #             # when P(p)=1 for I=1 and P(p)=0 for I=1, then only a nudge of 0 could be added to P(b).
    #             # small round off error?
    #             # alternative: consider making the nudge dependent on I, which gives more freedom but I am not sure
    #             # if then e.g. the expected correlation is still zero. (Should be, right?)
    #             assert max_nudge[sx] >= min_nudge[sx], 'unsatisfiable conditions for additive nudge!'
    #
    #             if max_nudge[sx] == min_nudge[sx]:
    #                 warnings.warn('max_nudge[sx] == min_nudge[sx], meaning that I cannot find a single nudge'
    #                               ' \epsilon_a for all ' + str(len(list(pdf_P_I.itervalues())))
    #                               + ' pdf_P_i of pdf_P_I=P(perturbed_variables | ignored_variables='
    #                               + str(ignored_variables) + ')')
    #                 # todo: implement a different nudge for each value for ignored_variables? Then this does not happen
    #                 # NOTE: then the nudge may correlate with ignored_variables, which is not what you want because
    #                 # then you could be adding correlations?
    #
    #         range_nudge = max_nudge - min_nudge
    #
    #         max_num_trials_nudge_gen = 1000
    #         for trial in xrange(max_num_trials_nudge_gen):
    #             nudge = np.random.random(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * range_nudge \
    #                     + min_nudge
    #
    #             sum_nudge = np.sum(nudge)  # should become 0, but currently will not be
    #
    #             correction = np.ones(
    #                 np.shape(pdf_P.joint_probabilities.joint_probabilities)) / nudge.size * sum_nudge
    #
    #             nudge -= correction
    #
    #             if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
    #                 break  # found a good nudge!
    #             else:
    #                 if trial == max_num_trials_nudge_gen - 1:
    #                     warnings.warn(
    #                         'max_num_trials_nudge_gen=' + str(max_num_trials_nudge_gen) + ' was not enough'
    #                         + ' to find a good nudge vector. Will fail...')
    #
    #                 continue  # darn, let's try again
    #
    #         np.testing.assert_almost_equal(np.sum(nudge), 0.0, decimal=12)
    #
    #         # this is the point of self.type_prob ...
    #         assert type(pdf_P.joint_probabilities.joint_probabilities.flat.next()) == self.type_prob
    #
    #         if len(ignored_variables) > 0:  # can this become the main code, so not an if? (also works if
    #             # len(ignored*) == 0?)
    #             # new attempt for making the new joint pdf
    #             new_joint_probs = -np.ones(np.shape(self.joint_probabilities.joint_probabilities))
    #
    #             for s in self.statespace():
    #                 # shorthands: the states pertaining to the three subsets of variables
    #                 sp = tuple(np.take(s, perturbed_variables))
    #                 so = tuple(np.take(s, output_variables))
    #                 si = tuple(np.take(s, ignored_variables))
    #
    #                 new_joint_probs[s] = pdf_I(si) * (pdf_P_I[si](sp) + nudge[sp]) * pdf_O_IP[si + sp](so)
    #
    #                 # might go over by a tiny bit due to roundoff, then just clip
    #                 if -1e-10 < new_joint_probs[s] < 0:
    #                     new_joint_probs[s] = 0
    #                 elif 1 < new_joint_probs[s] < 1 + 1e-10:
    #                     new_joint_probs[s] = 1
    #
    #                 assert 0 <= new_joint_probs[s] <= 1, 'new_joint_probs[s] = ' + str(new_joint_probs[s])
    #
    #             self.reset(self.numvariables, self.numvalues, new_joint_probs)
    #         else:
    #             new_probs = pdf_P.joint_probabilities.joint_probabilities + nudge
    #
    #             # this is the point of self.type_prob ...
    #             assert type(new_probs.flat.next()) == self.type_prob
    #
    #             # todo: remove test once it works a while
    #             if True:
    #                 assert np.max(new_probs) <= 1.0, 'no prob should be >1.0: ' + str(np.max(new_probs))
    #                 assert np.min(new_probs) >= 0.0, 'no prob should be <0.0: ' + str(np.min(new_probs))
    #
    #             # assert np.max(nudge) <= abs(epsilon), 'no nudge should be >' + str(epsilon) + ': ' + str(np.max(nudge))
    #             # assert np.min(nudge) >= -abs(epsilon), 'no nudge should be <' + str(-epsilon) + ': ' + str(np.min(nudge))
    #
    #             # total probability mass should be unchanged  (todo: remove test once it works a while)
    #             np.testing.assert_almost_equal(np.sum(new_probs), 1)
    #
    #             if __debug__:
    #                 pdf_X_orig = pdf_P.copy()
    #
    #             pdf_P.joint_probabilities.reset(new_probs)
    #
    #             self.duplicate(pdf_P + pdf_IO_P)
    #
    #         return nudge
    #     else:
    #         raise UserWarning('you should not mix perturbed_variables and output_variables. Not sure yet how to'
    #                           ' implement that. The output variables that occur before a perturbed variable should'
    #                           ' probably be swapped but with the condition that they become independent from the'
    #                           ' said perturbed variable(s), however the output variable could also be a causal '
    #                           ' predecessor for that perturbed variable and swapping them means that this'
    #                           ' causal relation would be lost (under the current assumption that variables are'
    #                           ' ordered as causal predecessors of each other). Maybe split this up in different'
    #                           ' nudge scenarios?')


    # helper function
    def logbase(self, x, base, replace_zeros=True):
        """
        A wrapper around np.log(p) which will return 0 if p is 0, which is useful for MI calc. because 0 log 0 = 0
        by custom.
        :type replace_zeros: bool
        """

        if replace_zeros:
            if not np.isscalar(x):
                x2 = copy.deepcopy(x)

                if replace_zeros:
                    np.place(x2, x2 == 0, 1)
            else:
                x2 = x if x != 0 else 1
        else:
            x2 = x

        if base == 2:
            ret = np.log2(x2)
        elif base == np.e:
            ret = np.log(x2)
        else:
            ret = np.log(x2) / np.log(base)

        return ret


    def causal_impact_of_nudge(self, perturbed_variables, output_variables='auto', hidden_variables=None,
                               epsilon=0.01, num_samples=20, base=2):
        """
        This function determines the *direct* causal impact of <perturbed_variables> on <output_variables>. All other
         variables are assumed 'fixed', meaning that the question becomes: if I fix all other variables' values, will
         a change in <perturbed_variables> result in a change in <output_variables>?

         This function will more specifically determine to what extent the mutual information between
         <perturbed_variables> and <output_variables> is 'causal'. If the return object is <impact> then this
         extent (fraction) can be calculated as "impact.avg_impact / (impact.avg_corr - impact.avg_mi_diff)".
        :rtype: CausalImpactResponse
        """

        # todo: add also an unobserved_variables list or so. now ignored_variables are actually considered fixed,
        # not traced out

        if hidden_variables is None:
            hidden_variables = []

        if output_variables in ('auto', 'all'):
            fixed_variables = []

            output_variables = list(np.setdiff1d(list(range(len(self))), list(perturbed_variables) + list(hidden_variables)))
        else:
            fixed_variables = list(np.setdiff1d(list(range(len(self))), list(perturbed_variables) + list(hidden_variables)
                                                + list(output_variables)))

        perturbed_variables = sorted(list(set(perturbed_variables)))
        hidden_variables = sorted(list(set(hidden_variables)))
        output_variables = sorted(list(set(output_variables)))
        fixed_variables = sorted(list(set(fixed_variables)))

        # print 'debug: output_variables =', output_variables
        # print 'debug: hidden_variables =', hidden_variables
        # print 'debug: fixed_variables =', fixed_variables
        # print 'debug: perturbed_variables =', perturbed_variables

        # assert sorted(perturbed_variables + hidden_variables + output_variables + fixed_variables)

        if not hidden_variables is None:
            if len(hidden_variables) > 0:
                pdf = self.copy()

                pdf.reorder_variables(list(perturbed_variables) + list(fixed_variables) + list(output_variables)
                                      + list(hidden_variables))

                # sum out the hidden variables
                pdf = pdf.marginalize_distribution(list(range(len(list(perturbed_variables) + list(fixed_variables)
                                                             + list(output_variables)))))

        ret = CausalImpactResponse()

        ret.perturbed_variables = perturbed_variables

        ret.mi_orig = self.mutual_information(perturbed_variables, output_variables, base=base)

        ret.mi_nudged_list = []
        ret.impacts_on_output = []
        ret.correlations = []

        ret.nudges = []
        ret.upper_bounds_impact = []

        pdf_out = self.marginalize_distribution(output_variables)
        pdf_pert = self.marginalize_distribution(perturbed_variables)

        cond_out = self.conditional_probability_distributions(perturbed_variables, output_variables)

        assert len(next(iter(cond_out.values()))) == len(output_variables), \
            'len(cond_out.itervalues().next()) = ' + str(len(next(iter(cond_out.values())))) \
            + ', len(output_variables) = ' + str(len(output_variables))

        for i in range(num_samples):
            pdf = self.copy()

            nudge = pdf.nudge(perturbed_variables, output_variables, epsilon)

            upper_bound_impact = np.sum([np.power(np.sum([nudge[a] * cond_out[a](b)
                                                          for a in pdf_pert.statespace()]), 2) / pdf_out(b)
                                         for b in pdf_out.statespace()])

            # NOTE: below I do this multiplier to make everything work out, but I do not yet understand why, but anyway
            # then I have to do it here as well -- otherwise I cannot compare the below <impact> with this upper bound
            upper_bound_impact *= 1.0 / 2.0 * 1.0 / np.log(base)

            if __debug__:
                np.testing.assert_almost_equal(np.sum(nudge), 0, decimal=12,
                                            err_msg='more strict: 0 != ' + str(np.sum(nudge)))
            # assert np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]) == 0, \
            #     'more strict to find bug: ' + str(np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]))

            ret.nudges.append(nudge)
            ret.upper_bounds_impact.append(upper_bound_impact)

            ### determine MI difference

            ret.mi_nudged_list.append(pdf.mutual_information(perturbed_variables, output_variables, base=base))

            ### compute causal impact term

            pdf_out_new = pdf.marginalize_distribution(output_variables)

            impact = np.sum([np.power(pdf_out_new(b) - pdf_out(b), 2) / pdf_out(b)
                             for b in pdf_out.statespace()])
            # NOTE: I do not yet understand where the factor 1/2 comes from!
            # (note to self: the upper bound of i_b is easily derived, but for clearly 100% causal relations
            # this <impact> falls  short of the upper bound by exactly this multiplier... just a thought. Maybe
            # I would have to correct <correlations> by this, and not impact?)
            impact *= 1.0/2.0 * 1.0/np.log(base)
            ret.impacts_on_output.append(impact)

            # if __debug__:
            # the last try-catch seems to fail a lot, don't know why, maybe roundoff, but eps=0.01 seems to always
            # work, so go with that.
            if False:
                # for some reason this seems to be more prone to roundoff errors than above... strange...
                debug_impact = np.sum(np.power(pdf_out_new.joint_probabilities - pdf_out.joint_probabilities, 2)
                                      / pdf_out.joint_probabilities.joint_probabilities)

                debug_impact *= 1.0/2.0 * 1.0/np.log2(base)

                if np.random.randint(10) == 0 and __debug__:
                    array1 = np.power(pdf_out_new.joint_probabilities - pdf_out.joint_probabilities, 2) \
                             / pdf_out.joint_probabilities.joint_probabilities
                    array2 = [np.power(pdf_out_new(b) - pdf_out(b), 2) / pdf_out(b)
                              for b in pdf_out.statespace()]

                    # print 'error: len(array1) =', len(array1)
                    # print 'error: len(array2) =', len(array2)

                    np.testing.assert_almost_equal(array1.flatten(), array2)

                # debug_impact seems highly prone to roundoff errors, don't know why, but coupled with the pairwise
                # testing in the if block above and this more messy test I think it is good to know at least that
                # the two are equivalent, so probably correct
                try:
                    np.testing.assert_almost_equal(impact, debug_impact, decimal=3)
                except AssertionError as e:
                    warnings.warn('np.testing.assert_almost_equal(impact=' + str(impact)
                                  + ', debug_impact=' + str(debug_impact) + ', decimal=3): ' + str(e))

            assert impact >= 0.0

            ### determine correlation term (with specific surprise)

            if __debug__:
                # shorthand, but for production mode I am worried that such an extra function call to a very central
                # function (in Python) may slow things down a lot, so I do it in debug only (seems to take 10-25%
                # more time indeed)
                # def logbase_debug(x, base):
                #     if x != 0:
                #         if base==2:
                #             return np.log2(x)
                #         elif base==np.e:
                #             return np.log(x)
                #         else:
                #             return np.log(x) / np.log(base)
                #     else:
                #         # assuming that this log factor is part of a p * log(p) term, and 0 log 0 = 0 by
                #         # common assumption, then the result will now be 0 whereas if -np.inf then it will be np.nan...
                #         return -_finite_inf

                # looking for bug, sometimes summing nudges in this way results in non-zero
                try:
                    np.testing.assert_almost_equal(np.sum([nudge[pvs] for pvs in pdf_pert.statespace()]), 0,
                                                   err_msg='shape nudge = ' + str(np.shape(nudge)))
                except IndexError as e:
                    print('error: shape nudge = ' + str(np.shape(nudge)))

                    raise IndexError(e)
                # assert np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]) == 0, 'more strict to find bug'

                if __debug__:
                    if np.any(pdf_out.joint_probabilities.joint_probabilities == 0):
                        warnings.warn('at least one probability in pdf_out is zero, so will be an infinite term'
                                      ' in the surprise terms since it has log(... / pdf_out)')

                debug_surprise_terms_fast = np.array([np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                               * self.logbase(cond_out[pvs].joint_probabilities.joint_probabilities
                                                         / pdf_out.joint_probabilities.joint_probabilities, base))
                                               for pvs in cond_out.keys()])

                debug_surprise_sum_fast = np.sum(debug_surprise_terms_fast)

                pdf_pert_new = pdf.marginalize_distribution(perturbed_variables)
                np.testing.assert_almost_equal(nudge, pdf_pert_new.joint_probabilities - pdf_pert.joint_probabilities)

                debug_surprise_terms = np.array([np.sum([cond_out[pvs](ovs)
                                                         * (self.logbase(cond_out[pvs](ovs)
                                                            / pdf_out(ovs), base)
                                                            if cond_out[pvs](ovs) > 0 else 0)
                                          for ovs in pdf_out.statespace()])
                     for pvs in cond_out.keys()])
                debug_surprise_sum = np.sum(debug_surprise_terms)
                debug_mi_orig = np.sum(
                    [pdf_pert(pvs) * np.sum([cond_out[pvs](ovs)
                                             * (self.logbase(cond_out[pvs](ovs) / pdf_out(ovs), base)
                                                if cond_out[pvs](ovs) > 0 else 0)
                             for ovs in pdf_out.statespace()])
                     for pvs in cond_out.keys()])

                if not np.isnan(debug_surprise_sum_fast):
                    np.testing.assert_almost_equal(debug_surprise_sum, debug_surprise_sum_fast)
                if not np.isnan(debug_mi_orig):
                    np.testing.assert_almost_equal(debug_mi_orig, ret.mi_orig)

                assert np.all(debug_surprise_terms >= -0.000001), \
                    'each specific surprise s(a) should be non-negative right? ' + str(np.min(debug_surprise_terms))
                if not np.isnan(debug_surprise_sum_fast):
                    np.testing.assert_almost_equal(debug_surprise_terms_fast, debug_surprise_terms)
                    assert np.all(debug_surprise_terms_fast >= -0.000001), \
                        'each specific surprise s(a) should be non-negative right? ' + str(np.min(debug_surprise_terms_fast))

            def logdiv(p, q, base=2):
                if q == 0:
                    return 0  # in sum p(q|p) log p(q|p) / p(q) this will go alright and prevents an error for NaN
                elif p == 0:
                    return 0  # same
                else:
                    return np.log(p / q) / np.log(base)

            # todo: precompute cond_out[pvs] * np.log2(cond_out[pvs] / pdf_out) matrix?
            if base == 2:
                # note: this is the explicit form, should check if a more implicit numpy array form (faster) will be
                # equivalent
                method_correlation = 'fast'
                # method_correlation = 'slow'
                if not method_correlation == 'fast' \
                        or len(fixed_variables) > 0 \
                        or len(hidden_variables) > 0:  # else-block is not yet adapted for this case, not sure yet how
                    correlation = np.array(
                        [nudge[pvs] * np.sum([cond_out[pvs](ovs) * logdiv(cond_out[pvs](ovs), pdf_out(ovs), base)
                                 for ovs in pdf_out.statespace()]) for pvs in cond_out.keys()], dtype=_type_prob)

                    assert np.all(np.isfinite(correlation)), \
                        'correlation contains non-finite scalars, like NaN probably...'

                    correlation = np.sum(correlation)

                    assert np.isfinite(correlation)
                else:
                    assert len(fixed_variables) == 0, 'not yet supported by this else block, must tile over ' \
                                                        'ignored_variables as well, but not sure if it matters how...'

                    # allprobs = self.joint_probabilities.joint_probabilities  # shorthand
                    try:
                        allprobs = np.reshape(np.array([cond_out[a].joint_probabilities.joint_probabilities
                                              for a in pdf_pert.statespace()],
                                              dtype=_type_prob),
                                              np.shape(pdf_pert.joint_probabilities.joint_probabilities)
                                              + np.shape(iter(cond_out.values()).next().joint_probabilities.joint_probabilities))
                    except ValueError as e:
                        print('error: np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities) =', \
                            np.shape(iter(cond_out.values()).next().joint_probabilities.joint_probabilities))
                        print('error: np.shape(self.joint_probabilities.joint_probabilities) =', np.shape(self.joint_probabilities.joint_probabilities))
                        print('error: np.shape(cond_out.iterkeys().next().joint_probabilities.joint_probabilities) =', np.shape(iter(cond_out.keys()).next().joint_probabilities.joint_probabilities))
                        print('error: np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities) =', np.shape(iter(cond_out.values()).next().joint_probabilities.joint_probabilities))

                        raise ValueError(e)
                    pdf_out_tiled = np.tile(pdf_out.joint_probabilities.joint_probabilities,
                                            [pdf_pert.numvalues] * len(pdf_pert) + [1])  # shorthand

                    correlation = np.sum(nudge * np.sum(allprobs * (self.logbase(allprobs, base) - self.logbase(pdf_out_tiled, base)),
                                                        axis=tuple(range(len(perturbed_variables), len(self)))))

                    assert np.isfinite(correlation)

                    # if __debug__:
                    #     if num_samples <= 10:
                    #         print 'debug: correlation =', correlation

                # todo: try to avoid iterating over nudge[pvs] but instead do an implicit numpy operation, because
                # this way I get roundoff errors sometimes

                # correlation = np.sum([nudge[pvs] * cond_out[pvs].joint_probabilities.joint_probabilities
                #               * (np.log2(cond_out[pvs].joint_probabilities.joint_probabilities)
                #               - np.log2(pdf_out.joint_probabilities.joint_probabilities))
                #               for pvs in cond_out.iterkeys()])

                if __debug__ and np.random.randint(10) == 0:
                    correlation2 = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                  * (self.logbase(cond_out[pvs].joint_probabilities.joint_probabilities, 2)
                                  - self.logbase(pdf_out.joint_probabilities.joint_probabilities, 2)))
                                  for pvs in cond_out.keys()])

                    if not np.isnan(correlation2):
                        np.testing.assert_almost_equal(correlation, correlation2)
            elif base == np.e:
                correlation = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                      * (np.log(cond_out[pvs].joint_probabilities.joint_probabilities)
                                         - np.log(pdf_out.joint_probabilities.joint_probabilities)))
                                      for pvs in cond_out.keys()])
            else:
                correlation = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                      * (np.log(cond_out[pvs].joint_probabilities.joint_probabilities)
                                         - np.log(pdf_out.joint_probabilities.joint_probabilities)) / np.log(base))
                                      for pvs in cond_out.keys()])

            if __debug__ and not method_correlation == 'fast':
                debug_corr = np.sum(
                    [nudge[pvs] * np.sum([cond_out[pvs](ovs) * self.logbase(cond_out[pvs](ovs) / pdf_out(ovs), base)
                                          for ovs in pdf_out.statespace()])
                     for pvs in cond_out.keys()])

                # should be two ways of computing the same
                np.testing.assert_almost_equal(debug_corr, correlation)

            assert np.isfinite(correlation), 'correlation should be a finite number'

            ret.correlations.append(correlation)

        ret.avg_mi_diff = np.mean(np.subtract(ret.mi_nudged_list, ret.mi_orig))
        ret.avg_impact = np.mean(ret.impacts_on_output)
        ret.avg_corr = np.mean(ret.correlations)

        ret.std_mi_diff = np.std(np.subtract(ret.mi_nudged_list, ret.mi_orig))
        ret.std_impact = np.std(ret.impacts_on_output)
        ret.std_corr = np.std(ret.correlations)

        ret.mi_diffs = np.subtract(ret.mi_nudged_list, ret.mi_orig)

        ret.mi_nudged_list = np.array(ret.mi_nudged_list)  # makes it easier to subtract and such by caller
        ret.impacts_on_output = np.array(ret.impacts_on_output)
        ret.correlations = np.array(ret.correlations)

        assert np.all(np.isfinite(ret.correlations))
        assert not np.any(np.isnan(ret.correlations))

        ret.residuals = ret.impacts_on_output - (ret.correlations - ret.mi_diffs)
        ret.avg_residual = np.mean(ret.residuals)
        ret.std_residual = np.std(ret.residuals)

        # return [avg, stddev]
        # return np.mean(np.subtract(mi_nudged_list, mi_orig)), np.std(np.subtract(mi_nudged_list, mi_orig))
        return ret


    # todo: rename to perturb(..., only_non_local=False)
    def perturb(self, perturbed_variables, perturbation_size=0.1, only_non_local=False):
        """

        """

        subject_variables = np.setdiff1d(list(range(len(self))), list(perturbed_variables))

        assert len(subject_variables) + len(perturbed_variables)

        # todo: for the case only_non_local=False this function makes no sense, there is then still an optimization
        # procedure, but I think it is desired to have a function which adds a *random* vector to the parameters,
        # and minimize() is not guaranteed to do that in absence of cost terms (other than going out of unit cube)
        if only_non_local == False:
            warnings.warn('current perturb() makes little sense with only_non_local=False, see todo above in code')

        if max(subject_variables) < min(perturbed_variables):
            # the variables are a contiguous block of subjects and then a block of perturbs. Now we can use the property
            # of self.matrix2params_incremental() in that its parameters are ordered to encode dependencies to
            # lower-numbered variables.

            pdf_subs_only = self[list(range(len(subject_variables)))]

            params_subs_only = list(pdf_subs_only.matrix2params_incremental())

            params_subs_perturbs = list(self.matrix2params_incremental())

            num_static_params = len(params_subs_only)
            num_free_params = len(params_subs_perturbs) - num_static_params

            free_params_orig = list(params_subs_perturbs[num_static_params:])

            # using equation: a^2 + b^2 + c^2 + ... = p^2, where p is norm and a..c are vector elements.
            # I can then write s_a * p^2 + s_b(p^2 - s_a * p^2) + ... = p^2, so that a = sqrt(s_a * p^2), where all
            # s_{} are independent coordinates in range [0,1]
            def from_sphere_coords_to_vec(coords, norm=perturbation_size):
                accounted_norm = np.power(norm, 2)

                vec = []

                for coord in coords:
                    if -0.0001 <= coord <= 1.0001:
                        coord = min(max(coord, 0.0), 1.0)
                    assert 0 <= coord <= 1.0, 'invalid coordinate'

                    if accounted_norm < 0.0:
                        assert accounted_norm > -0.0001, 'accounted_norm dropped below zero, seems not just rounding' \
                                                         ' error: ' + str(accounted_norm)

                    accounted_norm_i = coord * accounted_norm

                    vec.append(np.sqrt(accounted_norm_i))

                    accounted_norm -= accounted_norm_i

                # add last vector element, which simply consumes all remaining 'norm' quantity
                vec.append(np.sqrt(accounted_norm))

                # norm of resulting vector should be <norm>
                np.testing.assert_almost_equal(np.linalg.norm(vec), norm)

                return vec

            pdf_perturbs_only = self[list(range(len(subject_variables), len(subject_variables) + len(perturbed_variables)))]
            params_perturbs_only = list(pdf_perturbs_only.matrix2params_incremental())

            def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
                return max(min(num, 1), 0)

            pdf_new = self.copy()

            # note: to find a perturbation vector of <num_free_params> length and of norm <perturbation_size> we
            # now just have to find <num_free_params-1> independent values in range [0,1] and retrieve the
            # perturbation vector by from_sphere_coords_to_vec
            # rel_weight_out_of_bounds: the higher the more adversion to going out of hypercube (and accepting the
            # ensuing necessary clipping, which makes the norm of the perturbation vector less than requested)
            def cost_perturb_vec(sphere_coords, return_cost_list=False, rel_weight_out_of_bounds=10.0):
                new_params = params_subs_perturbs  # old param values
                perturb_vec = from_sphere_coords_to_vec(sphere_coords)
                new_params = np.add(new_params, [0]*num_static_params + perturb_vec)
                new_params_clipped = list(map(clip_to_unit_line, new_params))

                # if the params + perturb_vec would go out of the unit hypercube then this number will become nonzero
                # and increase with 'severity'. should use this as added, high cost
                missing_param_mass = np.sum(np.power(np.subtract(new_params, new_params_clipped), 2))

                pdf_new.params2matrix_incremental(new_params_clipped)

                pdf_new_perturbs_only = pdf_new[list(range(len(subject_variables),
                                                      len(subject_variables) + len(perturbed_variables)))]
                params_new_perturbs_only = pdf_new_perturbs_only.matrix2params_incremental()

                marginal_pdf_perturbs_diff = np.sum(np.power(np.subtract(params_perturbs_only, params_new_perturbs_only), 2))

                # cost term for going out of the hypercube, which we don't want
                cost_out_of_bounds = np.sqrt(missing_param_mass) / perturbation_size

                if only_non_local or return_cost_list:
                    # note: assumed to be maximally <perturbation_size, namely by a vector on the rim of the hypercube
                    # and then in the perpendicular direction outward into 'unallowed' space (like negative)
                    cost_diff_marginal_perturbed = np.sqrt(marginal_pdf_perturbs_diff) / perturbation_size
                else:
                    # we don't care about a cost term for keeping the marginal the same
                    cost_diff_marginal_perturbed = 0

                if not return_cost_list:
                    return float(rel_weight_out_of_bounds * cost_out_of_bounds + cost_diff_marginal_perturbed)
                else:
                    # this is for diagnostics only, not for any optimization procedure
                    return (cost_out_of_bounds, cost_diff_marginal_perturbed)

            num_sphere_coords = num_free_params - 1  # see description for from_sphere_coords_to_vec()

            initial_perturb_vec = np.random.random(num_sphere_coords)

            optres = minimize(cost_perturb_vec, initial_perturb_vec, bounds=[(0.0, 1.0)]*num_sphere_coords)

            assert optres.success, 'scipy\'s minimize() failed'

            assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

            # convert optres.x to new parameters
            new_params = params_subs_perturbs  # old param values
            perturb_vec = from_sphere_coords_to_vec(optres.x)
            new_params = np.add(new_params, [0]*num_static_params + perturb_vec)
            new_params_clipped = list(map(clip_to_unit_line, new_params))

            self.params2matrix_incremental(new_params_clipped)

            # get list of individual cost terms, for better diagnostics by caller
            cost = cost_perturb_vec(optres.x, return_cost_list=True)

            resp = self.PerturbNonLocalResponse()
            resp.pdf = self  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
            resp.cost_out_of_bounds = float(cost[0])  # cost of how different marginal P(Y) became (bad)
            resp.cost_diff_marginal_perturbed = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference
            resp.perturb_size = np.linalg.norm(np.subtract(params_subs_perturbs, new_params_clipped))

            return resp
        else:
            pdf_reordered = self.copy()

            retained_vars = list(subject_variables) + list(perturbed_variables)
            ignored_vars = list(np.setdiff1d(list(range(len(self))), retained_vars))

            if len(ignored_vars) > 0:
                cond_pdf_ignored = self.conditional_probability_distributions(ignored_vars)

            pdf_reordered.reorder_variables(list(subject_variables) + list(perturbed_variables))

            resp = pdf_reordered.perturb(list(range(len(subject_variables),
                                               len(subject_variables) + len(perturbed_variables))))

            # note: pdf_reordered is now changed in place, the perturbed values

            if len(ignored_vars) > 0:
                pdf_reordered.append_variables_using_conditional_distributions(cond_pdf_ignored)

            reverse_ordering = [-1]*len(self)

            for new_six in range(len(subject_variables)):
                orig_six = subject_variables[new_six]
                reverse_ordering[orig_six] = new_six

            for new_pix in range(len(subject_variables), len(subject_variables) + len(perturbed_variables)):
                orig_pix = perturbed_variables[new_pix - len(subject_variables)]
                reverse_ordering[orig_pix] = new_pix

            for new_iix in range(len(subject_variables) + len(perturbed_variables),
                                  len(subject_variables) + len(perturbed_variables) + len(ignored_vars)):
                orig_pix = perturbed_variables[new_pix - len(subject_variables) - len(perturbed_variables)]
                reverse_ordering[orig_pix] = new_pix

            pdf_reordered.reorder_variables(reverse_ordering)

            self.duplicate(pdf_reordered)  # change in-place this object now (could remove the use of pdf_reordered?)

            return resp


        # assert False, 'todo'
        #
        # num_input_variables = len(self) - num_output_variables
        #
        # assert num_input_variables > 0, 'makes no sense to perturb a relation with an empty set'
        #
        # original_params = self.matrix2params_incremental()
        #
        # static_params = list(self[range(num_input_variables)].matrix2params_incremental())
        #
        # num_free_params = len(original_params) - len(static_params)
        #
        # marginal_output_pdf = self[range(num_input_variables, len(self))]
        # assert len(marginal_output_pdf) == num_output_variables, 'programming error'
        # marginal_output_pdf_params = marginal_output_pdf.matrix2params_incremental()
        #
        # pdf_new = self.copy()  # just to create an object which I can replace everytime in cost function
        #
        # def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
        #     return max(min(num, 1), 0)
        #
        # def cost_perturb_non_local(free_params, return_cost_list=False):
        #     new_params = static_params + list(map(clip_to_unit_line, free_params))
        #
        #     pdf_new.params2matrix_incremental(new_params)
        #
        #     marginal_output_pdf_new = pdf_new[range(num_input_variables, len(self))]
        #     marginal_output_pdf_new_params = marginal_output_pdf_new.matrix2params_incremental()
        #
        #     cost_same_output_marginal = np.linalg.norm(np.subtract(marginal_output_pdf_new_params,
        #                                                            marginal_output_pdf_params))
        #     cost_different_relation = np.linalg.norm(np.subtract(free_params, original_params[len(static_params):]))
        #
        #     if not return_cost_list:
        #         cost = np.power(cost_same_output_marginal - 0.0, 2) \
        #                + np.power(cost_different_relation - perturbation_size, 2)
        #     else:
        #         cost = [np.power(cost_same_output_marginal - 0.0, 2),
        #                 np.power(cost_different_relation - perturbation_size, 2)]
        #
        #     return cost
        #
        # initial_guess_perturb_vec = np.random.random(num_free_params)
        # initial_guess_perturb_vec /= np.linalg.norm(initial_guess_perturb_vec)
        # initial_guess_perturb_vec *= perturbation_size
        #
        # initial_guess = np.add(original_params[len(static_params):],
        #                        initial_guess_perturb_vec)
        # initial_guess = map(clip_to_unit_line, initial_guess)  # make sure stays in hypercube's unit volume
        #
        # optres = minimize(cost_perturb_non_local, initial_guess, bounds=[(0.0, 1.0)]*num_free_params)
        #
        # assert optres.success, 'scipy\'s minimize() failed'
        #
        # assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'
        #
        # pdf_new.params2matrix_incremental(list(static_params) + list(optres.x))
        #
        # # get list of individual cost terms, for better diagnostics by caller
        # cost = cost_perturb_non_local(optres.x, return_cost_list=True)
        #
        # resp = self.PerturbNonLocalResponse()
        # resp.pdf = pdf_new  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
        # resp.cost_same_output_marginal = float(cost[0])  # cost of how different marginal P(Y) became (bad)
        # resp.cost_different_relation = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference
        #
        # return resp


    def perturb_non_local(self, num_output_variables, perturbation_size=0.001):  # todo: generalize
        """
        Perturb the pdf P(X,Y) by changing P(Y|X) without actually changing P(X) or P(Y), by numerical optimization.
        Y is assumed to be formed by stochastic variables collected at the end of the pdf.
        :param num_output_variables: |Y|.
        :param perturbation_size: the higher, the more different P(Y|X) will be from <self>
        :rtype: PerturbNonLocalResponse
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables > 0, 'makes no sense to perturb a relation with an empty set'

        original_params = self.matrix2params_incremental()

        static_params = list(self[list(range(num_input_variables))].matrix2params_incremental())

        num_free_params = len(original_params) - len(static_params)

        marginal_output_pdf = self[list(range(num_input_variables, len(self)))]
        assert len(marginal_output_pdf) == num_output_variables, 'programming error'
        marginal_output_pdf_params = marginal_output_pdf.matrix2params_incremental()

        pdf_new = self.copy()  # just to create an object which I can replace everytime in cost function

        def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
            return max(min(num, 1), 0)

        def cost_perturb_non_local(free_params, return_cost_list=False):
            new_params = static_params + list(map(clip_to_unit_line, free_params))

            pdf_new.params2matrix_incremental(new_params)

            marginal_output_pdf_new = pdf_new[list(range(num_input_variables, len(self)))]
            marginal_output_pdf_new_params = marginal_output_pdf_new.matrix2params_incremental()

            cost_same_output_marginal = np.linalg.norm(np.subtract(marginal_output_pdf_new_params,
                                                                   marginal_output_pdf_params))
            cost_different_relation = np.linalg.norm(np.subtract(free_params, original_params[len(static_params):]))

            if not return_cost_list:
                cost = np.power(cost_same_output_marginal - 0.0, 2) \
                       + np.power(cost_different_relation - perturbation_size, 2)
            else:
                cost = [np.power(cost_same_output_marginal - 0.0, 2),
                        np.power(cost_different_relation - perturbation_size, 2)]

            return cost

        initial_guess_perturb_vec = np.random.random(num_free_params)
        initial_guess_perturb_vec /= np.linalg.norm(initial_guess_perturb_vec)
        initial_guess_perturb_vec *= perturbation_size

        initial_guess = np.add(original_params[len(static_params):],
                               initial_guess_perturb_vec)
        initial_guess = list(map(clip_to_unit_line, initial_guess))  # make sure stays in hypercube's unit volume

        optres = minimize(cost_perturb_non_local, initial_guess, bounds=[(0.0, 1.0)]*num_free_params)

        assert optres.success, 'scipy\'s minimize() failed'

        assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

        pdf_new.params2matrix_incremental(list(static_params) + list(optres.x))

        # get list of individual cost terms, for better diagnostics by caller
        cost = cost_perturb_non_local(optres.x, return_cost_list=True)

        resp = self.PerturbNonLocalResponse()
        resp.pdf = pdf_new  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
        resp.cost_same_output_marginal = float(cost[0])  # cost of how different marginal P(Y) became (bad)
        resp.cost_different_relation = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference

        return resp


    def append_globally_resilient_variables(self, num_appended_variables, target_mi):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        def cost_func_resilience_and_mi(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(list(range(len(self))), list(range(len(self), len(pdf_new))))

            susceptibility = pdf_new.susceptibility_global(num_appended_variables)

            return np.power(abs(target_mi - mi) / target_mi + susceptibility, 2)

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_resilience_and_mi,
                                        initial_guess=np.random.random(num_free_parameters))

        return


    def append_variables_with_target_mi(self, num_appended_variables, target_mi, relevant_variables='all',
                                        verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        if relevant_variables in ('all', 'auto'):
            relevant_variables = list(range(len(self)))
        else:
            assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
            assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'

        if target_mi == 0.0:
            raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
                              ' Therefore you should call append_independent_variables instead and specify explicitly'
                              ' which PDFs you want to add independently.')

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        def cost_func_target_mi(free_params, parameter_values_before):

            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(relevant_variables, list(range(len(self), len(pdf_new))))

            return np.power((target_mi - mi) / target_mi, 2)

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)

        return  # nothing, in-place


    def append_variables_with_target_mi_and_marginal(self, num_appended_variables, target_mi, marginal_probs,
                                                     relevant_variables='all', verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        if relevant_variables in ('all', 'auto'):
            relevant_variables = list(range(len(self)))
        else:
            assert len(relevant_variables) < len(self), 'cannot be relative to more variables than I originally had'
            assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'

        if target_mi == 0.0:
            raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
                              ' Therefore you should call append_independent_variables instead and specify explicitly'
                              ' which PDFs you want to add independently.')

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        def cost_func_target_mi2(free_params, parameter_values_before):

            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(relevant_variables, list(range(len(self), len(pdf_new))))
            pdf_B = pdf_new[list(range(len(pdf_new)-num_appended_variables, len(pdf_new)))]
            diff_prob_cost = np.mean(np.power(pdf_B.joint_probabilities.joint_probabilities
                                              - marginal_probs, 2))

            return np.power((target_mi - mi) / target_mi, 2) + diff_prob_cost

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi2,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)

        return  # nothing, in-place


    def append_unique_individual_variable(self, about_variable_ix, verbose=True, tol_nonunique=0.05,
                                          num_repeats=3, agnostic_about=None, ignore_variables=None):
        """

        :param about_variable_ix:
        :param verbose:
        :param tol_nonunique:
        :param num_repeats: seems necessary to repeat so maybe 3 or 5?
        :param agnostic_about: I thought this was needed but now not sure (see note inside unique_individual_information())
        :param ignore_variables: used by unique_individual_information() for I_{unq}(X --> Y) to find URVs for only X
        :return:
        """
        assert not np.isscalar(agnostic_about), 'agnostic_about should be a list of ints like [3,4]'

        exponent = 1  # for individual cost terms

        pdf_new = self.copy()
        pdf_new.append_variables(1) # just to get the correct size

        if ignore_variables is None:
            ignore_variables = []

        assert not about_variable_ix in ignore_variables, 'makes no sense'

        def cost_func_unique(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi_indiv = pdf_new.mutual_information([about_variable_ix], [len(self)])  # this is a good thing
            # this is not a good thing:
            mi_nonunique = pdf_new.mutual_information([i for i in range(len(self))
                                                       if i != about_variable_ix and not i in ignore_variables],
                                                      [len(self)])

            if agnostic_about is None:
                # I treat the two terms equally important even though mi_nonunique can typically be much higher,
                # so no normalization for the number of 'other' variables.
                # note: I want to have max `tol_nonunique` fraction of nonunique MI so try to get this by weighting:
                cost = -np.power(mi_indiv, exponent) * tol_nonunique + np.power(mi_nonunique, exponent) * (1. - tol_nonunique)
            else:
                mi_agn = pdf_new.mutual_information([len(self)], agnostic_about)

                cost = -np.power(mi_indiv, exponent) * tol_nonunique + 0.5 * (np.power(mi_nonunique, exponent) + np.power(mi_agn, exponent)) * (1. - tol_nonunique)

            # note: if exponent==1 then cost should be <0 to be 'acceptable'
            return cost

        pdf_c = self.copy()
        optres = pdf_c.append_optimized_variables(1, cost_func_unique, verbose=verbose, num_repeats=num_repeats)

        if optres.success:
            self.duplicate(pdf_c)
            return optres
        else:
            raise UserWarning('optimization was unsuccessful: ' + str(optres))


    def append_optimized_variables(self, num_appended_variables, cost_func, initial_guess=None, verbose=True,
                                   num_repeats=None):
        """
        Append variables in such a way that their conditional pdf with the existing variables is optimized in some
        sense, for instance they can be synergistic (append_synergistic_variables) or orthogonalized
        (append_orthogonalized_variables). Use the cost_func to determine the relation between the new appended
        variables and the pre-existing ones.
        :param num_appended_variables:
        :param cost_func: a function cost_func(free_params, parameter_values_before) which returns a float.
        The parameter set list(parameter_values_before) + list(free_params) defines a joint pdf of the appended
        variables together with the pre-existing ones, and free_params by itself defines completely the conditional
        pdf of the new variables given the previous. Use params2matrix_incremental to construct a joint pdf from the
        parameters and evaluate whatever you need, and then return a float. The higher the return value of cost_func
        the more desirable the joint pdf induced by the parameter set list(parameter_values_before) + list(free_params).
        :param initial_guess: initial guess for 'free_params' where you think cost_func will return a relatively
        low value. It can also be None, in which case a random point in parameter space will be chosen. It can also
        be an integer value like 10, in which case 10 optimizations will be run each starting from a random point
        in parameter space, and the best solution is selected.
        :param verbose:
        :rtype: scipy.optimize.OptimizeResult
        """

        # these parameters should be unchanged and the first set of parameters of the resulting pdf_new
        parameter_values_before = list(self.matrix2params_incremental())

        assert min(parameter_values_before) >= -0.00000001, \
            'minimum of %s is < 0, should not be.' % parameter_values_before
        assert max(parameter_values_before) <= 1.00000001, \
            'minimum of %s is < 0, should not be.' % parameter_values_before

        if __debug__:
            debug_params_before = copy.deepcopy(parameter_values_before)

        # a pdf with XORs as appended variables (often already MSRV for binary variables), good initial guess?
        # note: does not really matter how I set the pdf of this new pdf, as long as it has the correct number of
        # paarameters for optimization below
        pdf_new = self.copy()
        pdf_new.append_variables_using_state_transitions_table(
            state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        assert num_appended_variables > 0, 'makes no sense to add 0 variables'
        assert len(parameter_values_after) > len(parameter_values_before), 'should be >0 free parameters to optimize?'
        if __debug__:
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                parameter_values_after[:len(parameter_values_before)])

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        assert num_appended_variables == 0 or num_free_parameters > 0

        # if initial_guess is None:
        #     initial_guess = np.random.random(num_free_parameters)  # start from random point in parameter space

        param_vectors_trace = []  # storing the parameter vectors visited by the minimize() function

        if num_repeats is None:
            if type(initial_guess) == int:
                num_repeats = int(initial_guess)
                initial_guess = None

                assert num_repeats > 0, 'makes no sense to optimize zero times?'
            else:
                num_repeats = 1

        optres = None

        def cost_func_wrapper(free_params, parameter_values_before):
            # note: jezus CHRIST not only does minimize() ignore the bounds I give it, it also suggests [nan, ...]!
            # assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            if not np.all(np.isfinite(free_params)):
                return np.power(np.sum(np.isfinite(free_params)), 2) * 10.
            else:
                clipped_free_params = np.max([np.min([free_params, np.ones(np.shape(free_params))], axis=0),
                                             np.zeros(np.shape(free_params))], axis=0)
                # penalize going out of bounds
                extra_cost = np.power(np.sum(np.abs(np.subtract(free_params, clipped_free_params))), 2)
                return cost_func(clipped_free_params, parameter_values_before) + extra_cost

        for rep in range(num_repeats):
            if initial_guess is None:
                initial_guess_i = np.random.random(num_free_parameters)  # start from random point in parameter space
            else:
                initial_guess_i = initial_guess  # always start from supplied point in parameter space

            assert len(initial_guess_i) == num_free_parameters
            assert np.all(np.isfinite(initial_guess_i)), 'looking for bug 55142'
            assert np.all(np.isfinite(parameter_values_before)), 'looking for bug 44142'

            if verbose > 0:
                print('debug: starting minimize() #' + str(rep) \
                      + ' at params=' + str(initial_guess_i) + ' at cost_func=' \
                      + str(cost_func_wrapper(initial_guess_i, parameter_values_before)))

            optres_i = minimize(cost_func_wrapper,
                              initial_guess_i, bounds=[(0.0, 1.0)]*num_free_parameters,
                              # callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                              args=(parameter_values_before,))

            if optres_i.success:
                if verbose > 0:
                    print('debug: successfully ended minimize() #' + str(rep) \
                          + ' at params=' + str(optres_i.x) + ' at cost_func=' \
                          + str(optres_i.fun))

                if optres is None:
                    optres = optres_i
                elif optres.fun > optres_i.fun:
                    optres = optres_i
                else:
                    pass  # this solution is worse than before, so do not change optres

        if optres is None:
            # could never find a good solution, in all <num_repeats> attempts
            raise UserWarning('always failed to successfully optimize: increase num_repeats')

        assert len(optres.x) == num_free_parameters
        assert max(optres.x) <= 1.0001, 'parameter bound significantly violated: ' + str(optres.x)
        assert min(optres.x) >= -0.0001, 'parameter bound significantly violated: ' + str(optres.x)

        # clip the parameters within the allowed bounds
        optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

        optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)

        assert min(optimal_parameters_joint_pdf) >= 0.0, \
            'minimum of %s is < 0, should not be.' % optimal_parameters_joint_pdf
        assert min(optimal_parameters_joint_pdf) <= 1.0, \
            'minimum of %s is > 1, should not be.' % optimal_parameters_joint_pdf
        assert min(parameter_values_before) >= 0.0, \
            'minimum of %s is < 0, should not be.' % parameter_values_before
        assert min(parameter_values_before) <= 1.0, \
            'minimum of %s is > 1, should not be.' % parameter_values_before

        pdf_new.params2matrix_incremental(optimal_parameters_joint_pdf)

        assert len(pdf_new) == len(self) + num_appended_variables

        if __debug__:
            parameter_values_after2 = pdf_new.matrix2params_incremental()

            assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                 parameter_values_after2[:len(parameter_values_before)])
            # note: for the if see the story in params2matrix_incremental()
            if not (0.000001 >= min(self.scalars_up_to_level(parameter_values_after2)) or \
                            0.99999 <= max(self.scalars_up_to_level(parameter_values_after2))):
                try:
                    np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                         optres.x)
                except AssertionError as e:
                    # are they supposed to be equal, but in different order?
                    print('debug: sum params after 1 =', np.sum(parameter_values_after2[len(parameter_values_before):]))
                    print('debug: sum params after 2 =', optres.x)
                    print('debug: parameter_values_before (which IS equal and correct) =', parameter_values_before)
                    # does this one below have a 1 or 0 in it? because then the error could be caused by the story in
                    # params2matrix_incremental()
                    print('debug: parameter_values_after2 =', parameter_values_after2)

                    raise AssertionError(e)
        if __debug__:
            # unchanged, not accidentally changed by passing it as reference? looking for bug
            np.testing.assert_array_almost_equal(debug_params_before, parameter_values_before)

        self.duplicate(pdf_new)

        return optres


    def append_orthogonalized_variables(self, variables_to_orthogonalize, num_added_variables_orthogonal,
                                        num_added_variables_parallel, verbose=True,
                                        num_repeats=1, randomization_per_repeat=0.01):

        """
        Let X=<variables_to_orthogonalize> and Y=complement[<variables_to_orthogonalize>]. Add two sets of variables
        X1 and X2 such that I(X1:Y)=0, I(X1:X)=H(X|Y); and I(X2:Y)=I(X2:X)=I(X:Y), and I(X1:X2)=0. In words, X is being
        decomposed into two parts: X1 (orthogonal to Y, MI=0) and X2 (parallel to Y, MI=max).

        This object itself will be expanded by <num_added_variables_orthogonal> + <num_added_variables_parallel>
        variables.

        Warning: if this pdf object also contains other variables Z which the orthogonalization can ignore, then
        first take them out of the pdf (marginalize or condition) because it blows up the number of free parameters
        that this function must optimize.

        (Do the optimization jointly, so both parallel and orthogonal together, because the function below seems not so
        effective for some reason.)
        :param variables_to_orthogonalize: list of variable indices which should be decomposed (X). The complemennt in
        range(len(self)) is then implicitly the object to decompose against ((Y in the description).
        :type variables_to_orthogonalize: list of int
        :param num_added_variables_orthogonal: |X1|
        :param num_added_variables_parallel: |X2|
        :param verbose:
        :param num_repeats: number of times to perform minimize() starting from random parameters
        and take the best result.
        :param randomization_per_repeat:
        :return: :raise ValueError:
        """
        assert num_added_variables_orthogonal > 0, 'cannot make parallel variables if I don\'t first make orthogonal'
        assert num_added_variables_parallel > 0, 'this may not be necessary, assert can be removed at some point?'

        # remove potential duplicates; and sort (for no reason)
        variables_to_orthogonalize = sorted(list(set(variables_to_orthogonalize)))

        pdf_ortho_para = self.copy()

        # these parameters should remain unchanged during the optimization, these are the variables (Y, X) where
        # X is the variable to be orthogonalized into the added (X1, X2) and Y is to be orthogonalized against.
        original_parameters = list(range(len(pdf_ortho_para.matrix2params_incremental(return_flattened=True))))

        original_variables = list(range(len(self)))  # this includes variables_to_orthogonalize
        subject_variables = list(np.setdiff1d(original_variables, variables_to_orthogonalize))

        # the rest of the function assumes that the variables_to_orthogonalize are all at the end of the
        # original_variables, so that appending a partial conditional pdf (conditioned on
        # len(variables_to_orthogonalize) variables) will be conditioned on the variables_to_orthogonalize only. So
        # reorder the variables and recurse to this function once.
        if not max(subject_variables) < min(variables_to_orthogonalize):
            pdf_reordered = self.copy()

            pdf_reordered.reorder_variables(subject_variables + variables_to_orthogonalize)

            assert len(pdf_reordered) == len(self)
            assert len(list(range(len(subject_variables), pdf_reordered.numvariables))) == len(variables_to_orthogonalize)

            # perform orthogonalization on the ordered pdf, which it is implemented for
            pdf_reordered.append_orthogonalized_variables(list(range(len(subject_variables), len(self))))

            # find the mapping from ordered to disordered, so that I can reverse the reordering above
            new_order = subject_variables + variables_to_orthogonalize
            original_order = [-1] * len(new_order)
            for varix in range(len(new_order)):
                original_order[new_order[varix]] = varix

            assert not -1 in original_order
            assert max(original_order) < len(original_order), 'there were only so many original variables...'

            pdf_reordered.reorder_variables(original_order + list(range(len(original_order), len(pdf_reordered))))

            self.duplicate(pdf_reordered)

            # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
            # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
            # are expected to be all valid
            assert min(self.matrix2params_incremental()) >= 0.0, \
                'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
            assert max(self.matrix2params_incremental()) <= 1.0, \
                'parameter(s) out of bound: ' + str(self.matrix2params_incremental())

            return

        assert len(np.intersect1d(original_variables, variables_to_orthogonalize)) == len(variables_to_orthogonalize), \
            'original_variables should include the variables_to_orthogonalize'

        pdf_ortho_para.append_variables(num_added_variables_orthogonal)
        orthogonal_variables = list(range(len(self), len(pdf_ortho_para)))

        orthogonal_parameters = list(range(len(original_parameters),
                                      len(pdf_ortho_para.matrix2params_incremental(return_flattened=True))))

        pdf_ortho_para.append_variables(num_added_variables_parallel)
        parallel_variables = list(range(len(pdf_ortho_para) - num_added_variables_parallel, len(pdf_ortho_para)))

        parallel_parameters = list(range(len(orthogonal_parameters) + len(original_parameters),
                                    len(pdf_ortho_para.matrix2params_incremental(return_flattened=True))))

        assert len(np.intersect1d(orthogonal_parameters, parallel_parameters)) == 0, \
            'error: orthogonal_parameters = ' + str(orthogonal_parameters) + ', parallel_parameters = ' \
            + str(parallel_parameters)
        assert len(np.intersect1d(orthogonal_variables, parallel_variables)) == 0

        free_parameters = list(orthogonal_parameters) + list(parallel_parameters)

        initial_params_list = np.array(list(pdf_ortho_para.matrix2params_incremental(return_flattened=True)))

        # todo: make this more efficient by having only free parameters X1,X2 as p(X1,X2|X), not also conditioned on Y?
        # this would make it a two-step thing maybe again naively, but maybe there is a way to still do it simultaneous
        # (of course the cost function DOES depend on Y, but X1,X2=f(X) only and the point is to find an optimal f(),)

        # todo: START of optimization

        # let the X1,X2 parameterization only depend on X (=variables_to_orthogonalize) to reduce the parameter space
        # greatly
        pdf_X = self.marginalize_distribution(variables_to_orthogonalize)

        pdf_X_X1_X2 = pdf_X.copy()
        pdf_X_X1_X2.append_variables(num_added_variables_orthogonal)
        pdf_X_X1_X2.append_variables(num_added_variables_parallel)

        free_params_X1_X2_given_X = list(range(len(pdf_X.matrix2params_incremental()),
                                          len(pdf_X_X1_X2.matrix2params_incremental())))
        # parameter values at these parameter indices should not change:
        static_params_X = list(range(len(pdf_X.matrix2params_incremental())))
        static_params_X_values = list(np.array(pdf_X.matrix2params_incremental())[static_params_X])

        # check if building a complete joint pdf using the subset-conditional pdf works as expected
        if __debug__:
            cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(list(range(len(variables_to_orthogonalize))))

            assert cond_pdf_X1_X2_given_X.num_output_variables() == num_added_variables_orthogonal \
                                                              + num_added_variables_parallel

            pdf_test_Y_X_X1_X2 = self.copy()
            pdf_test_Y_X_X1_X2.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

            # test if first so-many params are the same as pdf_ortho_para's
            np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                                 pdf_test_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])
            # still, it would be immensely unlikely that the two pdfs are the same, since the conditional pdf(X1,X2|X)
            # is independently and randomly generated for both pdfs
            assert pdf_test_Y_X_X1_X2 != pdf_ortho_para

        # used repeatedly in the cost function below, prevent recomputing it every time
        ideal_H_X1 = self.conditional_entropy(variables_to_orthogonalize, subject_variables)
        ideal_mi_X2_Y = self.mutual_information(variables_to_orthogonalize, subject_variables)

        default_weights = (1, 1, 1, 1, 1, 1)  # used for cost_func_minimal_params and to know the number of weights

        # cost function to be used in scipy's minimize() procedure
        def cost_func_minimal_params(proposed_params, rel_weights=default_weights):
            assert len(proposed_params) == len(free_params_X1_X2_given_X)

            # relative weight coefficients for the different terms that contribute to the cost function below.
            wIso, wIvo, wIvp, wIsp, wHop, wIop = list(map(abs, rel_weights))

            pdf_X_X1_X2.params2matrix_incremental(static_params_X_values + list(proposed_params))

            cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(list(range(len(variables_to_orthogonalize))))
            pdf_proposed_Y_X_X1_X2 = self.copy()
            pdf_proposed_Y_X_X1_X2.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

            if __debug__:
                # test if first so-many params are the same as pdf_ortho_para's
                np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                                    pdf_proposed_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])

            # note: unwanted effects should be positive terms ('cost'), and desired MIs should be negative terms
            # note: if you change the cost function here then you should also change the optimal value below
            # cost = wIso * pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, orthogonal_variables) \
            #        - wIsp * pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, parallel_variables) \
            #        - wHop * pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables) \
            #        + wIop * pdf_proposed_Y_X_X1_X2.mutual_information(orthogonal_variables, parallel_variables)

            # - wHop * pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables + parallel_variables) \

            # note: each term is intended to be normalized to [0, 1], where 0 is best and 1 is worst. Violation of this
            # is possible though, but they are really bad solutions.

            cost_terms = dict()
            cost_terms['Iso'] = \
                wIso * abs(pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, orthogonal_variables) - 0.0) \
                / ideal_mi_X2_Y
            cost_terms['Ivo'] = \
                wIvo * abs(pdf_proposed_Y_X_X1_X2.mutual_information(variables_to_orthogonalize, orthogonal_variables)
                           - ideal_H_X1) / ideal_H_X1
            # question: is the following one necessary?
            # cost_terms['Ivp'] = \
            #     wIvp * abs(pdf_proposed_Y_X_X1_X2.mutual_information(variables_to_orthogonalize, parallel_variables)
            #                - ideal_mi_X2_Y) / ideal_mi_X2_Y
            cost_terms['Ivp'] = 0.0
            cost_terms['Isp'] = \
                wIsp * abs(pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, parallel_variables)
                           - ideal_mi_X2_Y) / ideal_mi_X2_Y
            # cost_terms['Hop'] = wHop * abs(pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables) - ideal_H_X1)
            cost_terms['Hop'] = \
                wHop * 0.0  # let's see what happens now (trying to improve finding global optimum)
            cost_terms['Iop'] = \
                wIop * abs(pdf_proposed_Y_X_X1_X2.mutual_information(orthogonal_variables, parallel_variables) - 0.0) \
                / ideal_H_X1

            # sum of squared errors, or norm of vector in error space, to make a faster convergence hopefully
            cost = float(np.sum(np.power(list(cost_terms.values()), 2)))

            assert np.isfinite(cost)
            assert np.isscalar(cost)

            return float(cost)


        if __debug__:
            # for each term in the cost function above I determine what would be the optimal value,
            # note: if you change the cost function above then you should also change this optimal value
            debug_optimal_cost_value = 0

            # hack, I should actually perform some min()'s in this but this is just a rough guide:
            # note: assuming orthogonal_variables = variables_to_orthogonalize and H(parallel_variables) = 0
            debug_worst_cost_value = ideal_mi_X2_Y + abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1) \
                                     + ideal_mi_X2_Y + ideal_mi_X2_Y \
                                     + abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1)
                                                   # + ideal_mi_X2_Y \
                               # + 2 * abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1) + ideal_mi_X2_Y

            debug_random_time_before = time.time()

            debug_num_random_costs = 20
            debug_random_cost_values = [cost_func_minimal_params(np.random.random(len(free_params_X1_X2_given_X)))
                                  for _ in range(debug_num_random_costs)]

            debug_random_time_after = time.time()

            # for trying to eye-ball whether the minimize() is doing significantly better than just random sampling,
            # getting a feel for the solution space
            debug_avg_random_cost_val = np.mean(debug_random_cost_values)
            debug_std_random_cost_val = np.std(debug_random_cost_values)
            debug_min_random_cost_val = np.min(debug_random_cost_values)
            debug_max_random_cost_val = np.max(debug_random_cost_values)

            if verbose > 0:
                print('debug: cost values of random parameter vectors:', debug_random_cost_values, '-- took', \
                    debug_random_time_after - debug_random_time_before, 'seconds for', debug_num_random_costs, \
                    'vectors.')

        initial_guess_minimal = list(np.array(pdf_X_X1_X2.matrix2params_incremental())[free_params_X1_X2_given_X])

        # todo: bring optimal_cost_value to release code (not __debug__) and use it as criterion to decide if the
        # solution is acceptable? Maybe pass a tolerance value in which case I would raise an exception if exceeded?

        if verbose and __debug__:
            print('debug: append_orthogonalized_variables: BEFORE optimization, cost func =', \
                cost_func_minimal_params(initial_guess_minimal), '(optimal=' + str(debug_optimal_cost_value) \
                + ', a worst=' + str(debug_worst_cost_value) + ', avg~=' + str(debug_avg_random_cost_val) \
                                                                 + '+-' + str(debug_std_random_cost_val) + ')')

            # mutual informations
            print('debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables))
            print('debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'orthogonal_variables + parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'subject_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, subject_variables))

            # entropies
            print('debug: pdf_ortho_para.entropy(subject_variables) =', \
                pdf_ortho_para.entropy(subject_variables))
            print('debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
                pdf_ortho_para.entropy(variables_to_orthogonalize))
            print('debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
                pdf_ortho_para.entropy(orthogonal_variables))
            print('debug: pdf_ortho_para.entropy(parallel_variables) =', \
                pdf_ortho_para.entropy(parallel_variables))

            print('debug: (num free parameters for optimization:', len(free_params_X1_X2_given_X), ')')

            time_before = time.time()

        if num_repeats == 1:
            optres = minimize(cost_func_minimal_params, initial_guess_minimal,
                              bounds=[(0.0, 1.0)]*len(free_params_X1_X2_given_X),
                              args=((1,)*len(default_weights),))
        elif num_repeats > 1:
            # perform the minimization <num_repeats> times starting from random points in parameter space and select
            # the best one (lowest cost function value)

            # note: the args= argument passes the relative weights of the different (now 4) terms in the cost function
            # defined above. At every next iteration it allows more and more randomization around the value (1,1,1,1)
            # which means that every term would be equally important.
            optres_list = [minimize(cost_func_minimal_params, np.random.random(len(initial_guess_minimal)),
                                    bounds=[(0.0, 1.0)]*len(free_params_X1_X2_given_X),
                                    args=(tuple(1.0 + np.random.randn(len(default_weights))
                                                * randomization_per_repeat * repi),))
                           for repi in range(num_repeats)]

            if verbose and __debug__:
                print('debug: num_repeats=' + str(num_repeats) + ', all cost values were: ' \
                      + str([resi.fun for resi in optres_list]))
                print('debug: successes =', [resi.success for resi in optres_list])

            optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

            assert len(optres_list) > 0, 'all ' + str(num_repeats) + ' optimizations using minimize() failed...?!'

            costvals = [res.fun for res in optres_list]
            min_cost = min(costvals)
            optres_ix = costvals.index(min_cost)

            assert optres_ix >= 0 and optres_ix < len(optres_list)

            optres = optres_list[optres_ix]
        else:
            raise ValueError('cannot repeat negative times')

        assert optres.success, 'scipy\'s minimize() failed'

        assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

        # build the most optimal PDF then finally:
        pdf_X_X1_X2.params2matrix_incremental(static_params_X_values + list(optres.x))
        cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(list(range(len(variables_to_orthogonalize))))
        pdf_ortho_para = self.copy()
        pdf_ortho_para.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

        if __debug__:
            # test if first so-many params are the same as pdf_ortho_para's
            np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                                pdf_test_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])

        if verbose and __debug__:
            print('debug: append_orthogonalized_variables: AFTER optimization, cost func =', optres.fun)

            # mutual informations
            print('debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables), \
                '(optimal=' + str(debug_optimal_cost_value) + ')')
            print('debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'orthogonal_variables + parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables))
            print('debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'subject_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, subject_variables))

            # entropies
            print('debug: pdf_ortho_para.entropy(subject_variables) =', \
                pdf_ortho_para.entropy(subject_variables))
            print('debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
                pdf_ortho_para.entropy(variables_to_orthogonalize))
            print('debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
                pdf_ortho_para.entropy(orthogonal_variables))
            print('debug: pdf_ortho_para.entropy(parallel_variables) =', \
                pdf_ortho_para.entropy(parallel_variables))

            time_after = time.time()

            print('debug: the optimization took', time_after - time_before, 'seconds in total.')

        self.duplicate(pdf_ortho_para)

        return

        # todo: END of optimization

        # # note: the code below is 'old', in the sense that it should be equivalent above but using an optimization
        # # step in a (much) larger parameter space
        #
        # def cost_func2(proposed_params):
        #     assert len(free_parameters) == len(proposed_params)
        #
        #     params_list = list(initial_params_list[original_parameters]) + list(proposed_params)
        #
        #     pdf_ortho_para.params2matrix_incremental(params_list)  # in-place
        #
        #     # note: unwanted effects should be positive terms ('cost'), and desired MIs should be negative terms
        #     cost = pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) \
        #            - pdf_ortho_para.mutual_information(subject_variables, parallel_variables) \
        #            - pdf_ortho_para.mutual_information(variables_to_orthogonalize,
        #                                                orthogonal_variables + parallel_variables)
        #
        #     assert np.isfinite(cost)
        #     assert np.isscalar(cost)
        #
        #     return float(cost)
        #
        #
        # initial_guess = list(initial_params_list[free_parameters])
        #
        # if verbose and __debug__:
        #     print 'debug: append_orthogonalized_variables: BEFORE optimization, cost func =', cost_func2(initial_guess)
        #
        #     # mutual informations
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
        #           'orthogonal_variables + parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
        #
        #     # entropies
        #     print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
        #         pdf_ortho_para.entropy(subject_variables)
        #     print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
        #         pdf_ortho_para.entropy(variables_to_orthogonalize)
        #     print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
        #         pdf_ortho_para.entropy(orthogonal_variables)
        #     print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
        #         pdf_ortho_para.entropy(parallel_variables)
        #
        # optres = minimize(cost_func2, initial_guess, bounds=[(0.0, 1.0)]*len(free_parameters))
        #
        # assert optres.success, 'scipy\'s minimize() failed'
        #
        # if verbose and __debug__:
        #     print 'debug: append_orthogonalized_variables: AFTER optimization, cost func =', optres.fun
        #
        #     # mutual informations
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
        #           'orthogonal_variables + parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
        #
        #     # entropies
        #     print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
        #         pdf_ortho_para.entropy(subject_variables)
        #     print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
        #         pdf_ortho_para.entropy(variables_to_orthogonalize)
        #     print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
        #         pdf_ortho_para.entropy(orthogonal_variables)
        #     print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
        #         pdf_ortho_para.entropy(parallel_variables)
        #
        # self.duplicate(pdf_ortho_para)
        #
        # # del pdf_ortho_para


    # todo: set default entropy_cost_factor=0.1 or 0.05?
    # def append_orthogonalized_variables(self, variables_to_orthogonalize, num_added_variables_orthogonal=None,
    #                                     num_added_variables_parallel=None, entropy_cost_factor=0.1, verbose=True,
    #                                     num_repeats=3):
    #     """
    #     Orthogonalize the given set of variables_to_orthogonalize=X relative to the rest. I.e.,
    #     decompose X into two parts {X1,X2}
    #     :param num_added_variables_parallel:
    #     :param verbose:
    #     :param num_repeats: number of times that the optimization procedure of the cost function (for both
    #     the orthogonal variables and the parallel variables) is repeated, of which the best solution is then chosen.
    #     where I(X1:rest)=0 but I(X1:X)=H(X1) and I(X2:rest)=H(X2). The orthogonal set is added first and the parallel
    #     set last.
    #
    #     In theory the entropy of self as a whole should not increase, though it is not the end of the world if it does.
    #     But you can set e.g. entropy_cost_factor=0.1 to try and make it increase as little as possible.
    #     :param variables_to_orthogonalize: set of variables to orthogonalize (X)
    #     :param num_added_variables_orthogonal: number of variables in the orthogonal variable set to add. The more the
    #     better, at the combinatorial cost of memory and computation of course, though at some value the benefit should
    #     diminish to zero (or perhaps negative if the optimization procedure sucks at higher dimensions). If you also
    #     add parallel variables (num_added_variables_parallel > 0) then the quality of the parallel variable set depends
    #     on the quality of the orthogonal variable set, so then a too low number for num_added_variables_orthogonal
    #     might hurt.
    #     :type num_added_variables_orthogonal: int
    #     :type num_added_variables_parallel: int
    #     :param entropy_cost_factor: keep entropy_cost_factor < 1 or (close to) 0.0 (not negative)
    #     :type variables: list of int
    #     """
    #
    #     # remove potential duplicates; and sort (for no reason)
    #     variables_to_orthogonalize = sorted(list(set(variables_to_orthogonalize)))
    #
    #     # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
    #     # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
    #     # are expected to be all valid
    #     assert min(self.matrix2params_incremental()) >= 0.0, \
    #         'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #     assert max(self.matrix2params_incremental()) <= 1.0, \
    #         'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #
    #     # will add two sets of variables, one orthogonal to the pre-existing 'rest' (i.e. MI zero) and one parallel
    #     # (i.e. MI max). How many variables should each set contain?
    #     # note: setting to len(variables_to_orthogonalize) means that each set has at least surely enough entropy to do the job,
    #     # but given the discrete nature I am not sure if it is also always optimal. Can increase this to be more sure?
    #     # At the cost of more memory usage and computational requirements of course.
    #     if num_added_variables_orthogonal is None:
    #         num_added_variables_orthogonal = len(variables_to_orthogonalize)
    #     if num_added_variables_parallel is None:
    #         num_added_variables_parallel = len(variables_to_orthogonalize)
    #
    #     # variables to orthogonalize against
    #     remaining_variables = sorted([varix for varix in xrange(self.numvariables)
    #                                   if not varix in variables_to_orthogonalize])
    #
    #     if not max(remaining_variables) < min(variables_to_orthogonalize):
    #         pdf_reordered = self.copy()
    #
    #         pdf_reordered.reorder_variables(remaining_variables + variables_to_orthogonalize)
    #
    #         assert len(pdf_reordered) == len(self)
    #         assert len(range(len(remaining_variables), pdf_reordered.numvariables)) == len(variables_to_orthogonalize)
    #
    #         # perform orthogonalization on the ordered pdf, which it is implemented for
    #         pdf_reordered.append_orthogonalized_variables(range(len(remaining_variables),
    #                                                             pdf_reordered.numvariables))
    #
    #         # find the mapping from ordered to disordered, so that I can reverse the reordering above
    #         new_order = remaining_variables + variables_to_orthogonalize
    #         original_order = [-1] * len(new_order)
    #         for varix in xrange(len(new_order)):
    #             original_order[new_order[varix]] = varix
    #
    #         assert not -1 in original_order
    #         assert max(original_order) < len(original_order), 'there were only so many original variables...'
    #
    #         pdf_reordered.reorder_variables(original_order + range(len(original_order), len(pdf_reordered)))
    #
    #         self.duplicate(pdf_reordered)
    #
    #         # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
    #         # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
    #         # are expected to be all valid
    #         assert min(self.matrix2params_incremental()) >= 0.0, \
    #             'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #         assert max(self.matrix2params_incremental()) <= 1.0, \
    #             'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #     else:
    #         pdf_result = self.copy()  # used to store the result and eventually I will copy to self
    #
    #         ### first add the ORTHOGONAL part
    #
    #         # did not yet do minimize() or something like that so why would a parameter already be out of bound
    #         # even if by a small amount)? should be detected already at some earlier stage.
    #         assert min(pdf_result.matrix2params_incremental()) >= 0.0, 'parameter(s) out of bound, weird: ' \
    #                                                                    + str(pdf_result.matrix2params_incremental())
    #         assert max(pdf_result.matrix2params_incremental()) <= 1.0, 'parameter(s) out of bound, weird' \
    #                                                                    + str(pdf_result.matrix2params_incremental())
    #
    #         if num_added_variables_orthogonal > 0:
    #             pdf_for_ortho_only_optimization = self.copy()
    #             pdf_for_ortho_only_optimization.append_variables(num_added_variables_orthogonal)
    #
    #             num_free_parameters_ortho_only = len(pdf_for_ortho_only_optimization.matrix2params_incremental()) \
    #                                                     - len(self.matrix2params_incremental())
    #
    #             assert num_free_parameters_ortho_only > 0 or num_added_variables_orthogonal == 0
    #
    #             orthogonal_variables = range(len(self), len(self) + num_added_variables_orthogonal)
    #
    #             assert len(orthogonal_variables) == num_added_variables_orthogonal
    #
    #             # did not yet do minimize() or something like that so why would a parameter already be out of bound
    #             # even if by a small amount)? should be detected already at some earlier stage.
    #             assert min(pdf_for_ortho_only_optimization.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound, weird: ' \
    #                 + str(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #             assert max(pdf_for_ortho_only_optimization.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound, weird' \
    #                 + str(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #
    #             def cost_function_orthogonal_part(free_params, static_params, entropy_cost_factor=entropy_cost_factor):
    #                 # note: keep entropy_cost_factor < 1 or (close to) 0.0
    #
    #                 assert len(free_params) == num_free_parameters_ortho_only
    #
    #                 pdf_for_ortho_only_optimization.params2matrix_incremental(list(static_params) + list(free_params))
    #
    #                 # also try to minimize the total entropy of the orthogonal variable set, i.e., try to make the
    #                 # orthogonal part 'efficient' in the sense that it uses only as much entropy as it needs to do its
    #                 # job but no more
    #                 if entropy_cost_factor != 0.0:
    #                     entropy_cost = entropy_cost_factor * pdf_for_ortho_only_optimization.entropy(orthogonal_variables)
    #                 else:
    #                     entropy_cost = 0.0  # do not compute entropy if not used anyway
    #
    #                 # MI with 'remaining_variables' is unwanted, but MI with 'variables_to_orthogonalize' is wanted
    #                 cost_ortho = pdf_for_ortho_only_optimization.mutual_information(remaining_variables,
    #                                                                                 orthogonal_variables) \
    #                              - pdf_for_ortho_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                                   orthogonal_variables) \
    #                              + entropy_cost
    #
    #                 return float(cost_ortho)
    #
    #             # if verbose and __debug__:
    #             #     static_param_values = self.matrix2params_incremental()
    #             #     free_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()[len(self.matrix2params_incremental()):]
    #             #
    #             #     print 'debug: append_orthogonalized_variables: orthogonal cost value before optimization =', \
    #             #         cost_function_orthogonal_part(free_param_values, static_param_values), \
    #             #         '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #             pdf_result.append_optimized_variables(num_added_variables_orthogonal, cost_function_orthogonal_part,
    #                                                   initial_guess=num_repeats)
    #
    #             # why would a parameter already be out of bound
    #             # even if by a small amount)? should be detected already at some earlier stage.
    #             # append_optimized_variables should already fix this itself.
    #             assert min(pdf_result.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound, weird: ' \
    #                 + str(pdf_result.matrix2params_incremental())
    #             assert max(pdf_result.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound, weird' \
    #                 + str(pdf_result.matrix2params_incremental())
    #
    #             # if verbose and __debug__:
    #             #     static_param_values = self.matrix2params_incremental()
    #             #     free_param_values = pdf_result.matrix2params_incremental()[len(self.matrix2params_incremental()):]
    #             #
    #             #     # test whether the 'static' parameters were indeed kept static during optimization
    #             #     np.testing.assert_array_almost_equal(self.matrix2params_incremental(),
    #             #                                          pdf_result.matrix2params_incremental()[:len(self.matrix2params_incremental())])
    #             #
    #             #     print 'debug: append_orthogonalized_variables: orthogonal cost value after optimization =', \
    #             #         cost_function_orthogonal_part(free_param_values, static_param_values)
    #
    #         ### now add the PARALLEL part
    #
    #         if num_added_variables_parallel > 0:
    #             if num_added_variables_orthogonal == 0:
    #                 raise UserWarning('it is ill-defined to add \'parallel\' variables if I do not have any '
    #                                   '\'orthogonal\' variables to minimize MI against. Just also ask for '
    #                                   'orthogonal variables and then remove them (marginalize all other variables)?')
    #
    #             pdf_for_para_only_optimization = pdf_for_ortho_only_optimization.copy()
    #             # todo: it should be possible to let the parallel variables depend only on the orthogonal_variables
    #             # and the variables_to_orthogonalize, not also the remaining_variables, which would greatly
    #             # reduce the number of free parameters. But then you need to add this artial conditional pdf
    #             # to the complete pdf afterward, repeating it in some way.
    #             pdf_for_para_only_optimization.append_variables(num_added_variables_parallel)
    #
    #             num_free_parameters_para_only = len(pdf_for_para_only_optimization.matrix2params_incremental()) \
    #                                             - len(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #
    #             assert num_free_parameters_para_only > 0 or num_added_variables_parallel == 0
    #
    #             parallel_variables = range(len(pdf_for_ortho_only_optimization),
    #                                        len(pdf_for_para_only_optimization))
    #
    #             assert len(np.intersect1d(parallel_variables, orthogonal_variables)) == 0
    #             assert len(np.intersect1d(parallel_variables, remaining_variables)) == 0
    #
    #             assert len(parallel_variables) == num_added_variables_parallel
    #
    #             # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly
    #             # out of bounds, like
    #             # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this
    #             # point the parameters
    #             # are expected to be all valid
    #             assert min(pdf_for_para_only_optimization.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_for_para_only_optimization.matrix2params_incremental())
    #             assert max(pdf_for_para_only_optimization.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_for_para_only_optimization.matrix2params_incremental())
    #
    #             def cost_function_parallel_part(free_params, static_params, entropy_cost_factor=entropy_cost_factor):
    #                 # note: keep entropy_cost_factor < 1 or (close to) 0.0
    #
    #                 assert len(free_params) == num_free_parameters_para_only
    #                 assert len(free_params) > 0, 'makes no sense to optimize 0 parameters'
    #
    #                 pdf_for_para_only_optimization.params2matrix_incremental(list(static_params) + list(free_params))
    #
    #                 # also try to minimize the total entropy of the parallel variable set, i.e., try to make the
    #                 # parallel part 'efficient' in the sense that it uses only as much entropy as it needs to do its
    #                 # job but no more
    #                 if entropy_cost_factor != 0.0:
    #                     entropy_cost = entropy_cost_factor * pdf_for_para_only_optimization.entropy(parallel_variables)
    #                 else:
    #                     entropy_cost = 0.0  # do not compute entropy if not used anyway
    #
    #                 # MI with 'variables_to_orthogonalize' is wanted, but MI with 'orthogonal_variables' is unwanted
    #                 cost_para = - pdf_for_para_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                                 parallel_variables) \
    #                             + pdf_for_para_only_optimization.mutual_information(parallel_variables,
    #                                                                                  orthogonal_variables) \
    #                             + entropy_cost
    #
    #                 return float(cost_para)
    #
    #             if verbose and __debug__:
    #                 static_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()
    #                 free_param_values = pdf_for_para_only_optimization.matrix2params_incremental()[len(pdf_for_ortho_only_optimization.matrix2params_incremental()):]
    #
    #                 print 'debug: append_orthogonalized_variables: parallel cost value before optimization =', \
    #                     cost_function_parallel_part(free_param_values, static_param_values), \
    #                     '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #                 # note: this is a probabilistic check: with high probability the 0.0 value is suspected to
    #                 # lead to the asserted condition, but it is also possible that 0.0 just so happens to be due to
    #                 # randomness, however this should be with very small probability
    #                 if cost_function_parallel_part(free_param_values, static_param_values) == 0.0:
    #                     # is this because all entropy of the <variables_to_orthogonalize> is already completely
    #                     # in <orthogonal_variables>, so that for <parallel_variables> there is no more entropy left?
    #                     try:
    #                         np.testing.assert_almost_equal(
    #                             pdf_for_para_only_optimization.mutual_information(remaining_variables, orthogonal_variables)
    #                             ,
    #                             pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                         )
    #                     except AssertionError as e:
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(remaining_variables, orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(remaining_variables,
    #                                                                               orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(variables_to_orthogonalize) =', \
    #                             pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(parallel_variables, orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(parallel_variables,
    #                                                                                orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(variables_to_orthogonalize, parallel_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                               parallel_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(remaining_variables)', \
    #                             pdf_for_para_only_optimization.entropy(remaining_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.entropy(orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(parallel_variables) =', \
    #                             pdf_for_para_only_optimization.entropy(parallel_variables)
    #
    #                         raise AssertionError(e)
    #
    #             pdf_result.append_optimized_variables(num_added_variables_parallel, cost_function_parallel_part)
    #
    #             # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly
    #             # out of bounds, like
    #             # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this
    #             # point the parameters
    #             # are expected to be all valid
    #             assert min(pdf_result.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_result.matrix2params_incremental())
    #             assert max(pdf_result.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_result.matrix2params_incremental())
    #
    #             if verbose and __debug__:
    #                 static_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()
    #                 free_param_values = pdf_result.matrix2params_incremental()[len(pdf_for_ortho_only_optimization.matrix2params_incremental()):]
    #
    #                 print 'debug: append_orthogonalized_variables: parallel cost value after optimization =', \
    #                     cost_function_parallel_part(free_param_values, static_param_values), \
    #                     '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #                 # note: this is a probabilistic check: with high probability the 0.0 value is suspected to
    #                 # lead to the asserted condition, but it is also possible that 0.0 just so happens to be due to
    #                 # randomness, however this should be with very small probability
    #                 if cost_function_parallel_part(free_param_values, static_param_values) == 0.0:
    #                     # is this because all entropy of the <variables_to_orthogonalize> is already completely
    #                     # in <orthogonal_variables>, so that for <parallel_variables> there is no more entropy left?
    #                     np.testing.assert_almost_equal(
    #                         pdf_for_para_only_optimization.mutual_information(remaining_variables, orthogonal_variables)
    #                         ,
    #                         pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                     )
    #
    #         self.duplicate(pdf_result)
    #
    #
    #
    #     # todo: add some debug tolerance measures to detect if the error is too large in orthogonalization, like >10%
    #     # of entropy of orthogonal_variables is MI with parallel_variables?


    def scalars_up_to_level(self, list_of_lists, max_level=None):
        """
        Helper function. E.g. scalars_up_to_level([1,[2,3],[[4]]]) == [1], and
        scalars_up_to_level([1,[2,3],[[4]]], max_level=2) == [1,2,3]. Will be sorted on level, with highest level
        scalars first.

        Note: this function is not very efficiently implemented I think, but my concern now is that it works at all.

        :type list_of_lists: list
        :type max_level: int
        :rtype: list
        """
        # scalars = [v for v in list_of_lists if np.isscalar(v)]
        #
        # if max_level > 1 or (max_level is None and len(list_of_lists) > 0):
        #     for sublist in [v for v in list_of_lists if not np.isscalar(v)]:
        #         scalars.extend(self.scalars_up_to_level(sublist,
        #                                                 max_level=max_level-1 if not max_level is None else None))

        scalars = []

        if __debug__:
            debug_max_depth_set = (max_level is None)

        if max_level is None:
            max_level = maximum_depth(list_of_lists)

        for at_level in range(1, max_level + 1):
            scalars_at_level = self.scalars_at_level(list_of_lists, at_level=at_level)

            scalars.extend(scalars_at_level)

        if __debug__:
            if debug_max_depth_set:
                assert len(scalars) == len(list(flatten(list_of_lists))), 'all scalars should be present, and not duplicate' \
                                                                    '. len(scalars) = ' + str(len(scalars)) \
                                                                    + ', len(flatten(list_of_lists)) = ' \
                                                                    + str(len(list(flatten(list_of_lists))))

        return scalars

    def scalars_at_level(self, list_of_lists, at_level=1):
        """
        Helper function. E.g. scalars_up_to_level([1,[2,3],[[4]]]) == [1], and
        scalars_up_to_level([1,[2,3],[[4]]], max_level=2) == [1,2,3]. Will be sorted on level, with highest level
        scalars first.
        :type list_of_lists: list
        :type max_level: int
        :rtype: list
        """

        if at_level == 1:
            scalars = [v for v in list_of_lists if np.isscalar(v)]

            return scalars
        elif at_level == 0:
            warnings.warn('level 0 does not exist, I start counting from at_level=1, will return [].')

            return []
        else:
            scalars = []

            for sublist in [v for v in list_of_lists if not np.isscalar(v)]:
                scalars.extend(self.scalars_at_level(sublist, at_level=at_level-1))

            assert np.ndim(scalars) == 1

            return scalars

    def imbalanced_tree_from_scalars(self, list_of_scalars, numvalues):
        """
        Helper function.
        Consider e.g. tree =
                        [0.36227870614214747,
                         0.48474422004766832,
                         [0.34019329926554265,
                          0.40787146599658614,
                          [0.11638879037422999, 0.64823088842780996],
                          [0.33155311703042312, 0.11398958845340294],
                          [0.13824154613818085, 0.42816388506114755]],
                         [0.15806602176772611,
                          0.32551465875945773,
                          [0.25748947995256499, 0.35415524846620511],
                          [0.64896559115417218, 0.65575802084978507],
                          [0.36051945555508391, 0.40134903827671109]],
                         [0.40568439663760192,
                          0.67602830725264651,
                          [0.35103999983495449, 0.59577145940649334],
                          [0.38917741342947187, 0.44327101890582132],
                          [0.075034425516081762, 0.59660319391007388]]]

        If you first call scalars_up_to_level on this you get a list [0.36227870614214747, 0.48474422004766832,
        0.34019329926554265, 0.40787146599658614, 0.15806602176772611, ...]. If you pass this flattened list through
        this function then you should get the above imbalanced tree structure back again.

        At each level in the resulting tree there will be <numvalues-1> scalars and <numvalues> subtrees (lists).
        :type list_of_scalars: list
        :type numvalues: int
        :rtype: list
        """

        num_levels = int(np.round(np.log2(len(list_of_scalars) + 1) / np.log2(numvalues)))

        all_scalars_at_level = dict()

        list_of_scalars_remaining = list(list_of_scalars)

        for level in range(num_levels):
            num_scalars_at_level = np.power(numvalues, level) * (numvalues - 1)

            scalars_at_level = list_of_scalars_remaining[:num_scalars_at_level]

            all_scalars_at_level[level] = scalars_at_level

            list_of_scalars_remaining = list_of_scalars_remaining[num_scalars_at_level:]

        def tree_from_levels(all_scalars_at_level):
            if len(all_scalars_at_level) == 0:
                return []
            else:
                assert len(all_scalars_at_level[0]) == numvalues - 1

                if len(all_scalars_at_level) > 1:
                    assert len(all_scalars_at_level[1]) == numvalues * (numvalues - 1)
                if len(all_scalars_at_level) > 2:
                    assert len(all_scalars_at_level[2]) == (numvalues*numvalues) * (numvalues - 1), \
                        'len(all_scalars_at_level[2]) = ' + str(len(all_scalars_at_level[2])) + ', ' \
                        '(numvalues*numvalues) * (numvalues - 1) = ' + str((numvalues*numvalues) * (numvalues - 1))
                if len(all_scalars_at_level) > 3:
                    assert len(all_scalars_at_level[3]) == (numvalues*numvalues*numvalues) * (numvalues - 1)
                # etc.

                tree = list(all_scalars_at_level[0][:(numvalues - 1)])

                if len(all_scalars_at_level) > 1:
                    # add <numvalues> subtrees to this level
                    for subtree_id in range(numvalues):
                        all_scalars_for_subtree = dict()

                        for level in range(len(all_scalars_at_level) - 1):
                            num_scalars_at_level = len(all_scalars_at_level[level + 1])

                            assert np.mod(num_scalars_at_level, numvalues) == 0, 'should be divisible nu <numvalues>'

                            num_scalars_for_subtree = int(num_scalars_at_level / numvalues)

                            all_scalars_for_subtree[level] = \
                                all_scalars_at_level[level + 1][subtree_id * num_scalars_for_subtree
                                                                :(subtree_id + 1) * num_scalars_for_subtree]

                        subtree_i = tree_from_levels(all_scalars_for_subtree)

                        if len(all_scalars_for_subtree) > 1:
                            # numvalues - 1 scalars and numvalues subtrees
                            assert len(subtree_i) == (numvalues - 1) + numvalues, 'len(subtree_i) = ' \
                                                                                  + str(len(subtree_i)) \
                                                                                  + ', expected = ' \
                                                                                  + str((numvalues - 1) + numvalues)
                        elif len(all_scalars_for_subtree) == 1:
                            assert len(subtree_i) == numvalues - 1

                        tree.append(subtree_i)

                return tree

        tree = tree_from_levels(all_scalars_at_level)

        assert maximum_depth(tree) == len(all_scalars_at_level)  # should be numvariables if the scalars are parameters
        assert len(list(flatten(tree))) == len(list_of_scalars), 'all scalars should end up in the tree, and not duplicate'

        return tree

# define our own optimization problem object, since that seems necessary (?) for using pySOT
# (this was meant as an inner class in BayesianNetwork but it creates a cyclic reference)
class SynergyProblem(OptimizationProblem):
    bn = None  # store the BN to be able to reset the SRV with new params (in eval()) (type: BayesianNetwork)
    srv_ix = -1
    X_ixs = []
    agn_ixs = []

    def __init__(self, bn_with_srv: object, srv_ix: int, X_ixs: Sequence[int], agn_ixs: Sequence[int] = []):
        # only get params to obtain the dimensionality of the problem:
        params = bn_with_srv.pdfs[srv_ix].matrix2params()
        dim = len(params)

        # self.bn = copy.deepcopy(bn_with_srv)  # this object will be changed during optimization so make a copy
        self.bn = bn_with_srv  # by reference, so that changes are seen by the caller (most likely desired)
        self.srv_ix = srv_ix
        self.X_ixs = X_ixs
        self.agn_ixs = agn_ixs
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim, dtype=float)
        self.lb = np.zeros(dim, dtype=float)
        self.ub = np.ones(dim, dtype=float)
        self.int_var = np.array([])  # none are integer-valued
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional SRV optimization for BayesianNetwork: " + str(bn_with_srv)
    
    def eval(self, x) -> float:
        if __debug__:
            assert len(x) == self.dim
        
        fval = self.bn.objective_single_SRV(x, self.srv_ix, self.X_ixs, self.agn_ixs)

        return fval

# define our own optimization problem object, since that seems necessary (?) for using pySOT
# (this was meant as an inner class in BayesianNetwork but it creates a cyclic reference)
class DependenceProblem(OptimizationProblem):
    bn = None  # store the BN to be able to reset the SRV with new params (in eval()) (type: BayesianNetwork)
    drv_ix = -1
    X_ixs = []
    agn_ixs = []
    obj_f = lambda x, drv_ix, X_ixs, agn_ixs: 0.0  # objective function

    def __init__(self, bn_with_drv: object, drv_ix: int, X_ixs: Sequence[int], agn_ixs: Sequence[int] = [],
                 objective_func: Callable = None) -> None:
        """A Problem object to be used with pySOT for optimization. Objective function can be supplied.

        Args:
            bn_with_drv (object): a BayesianNetwork object.
            drv_ix (int): variable index of the dependent variable.
            X_ixs (Sequence[int]): variables (indices) to be correlated with.
            agn_ixs (Sequence[int], optional): variables (indices) to be agnostic about. Defaults to [].
            objective_func (Callable, optional): The objective function to be used. It should be a member function 
             of the BayesianNetwork object (bn_with_drv) most likely, and most likely is called something like 
             "objective_*". By default (so if None) this is self.bn.objective_single_target_mi. If user-supplied,
             it should have the same signature as this default.
        """
        # only get params to obtain the dimensionality of the problem:
        params = bn_with_drv.pdfs[drv_ix].matrix2params()
        dim = len(params)

        # self.bn = copy.deepcopy(bn_with_srv)  # this object will be changed during optimization so make a copy
        self.bn = bn_with_drv  # by reference, so that changes are seen by the caller (most likely desired)
        self.drv_ix = drv_ix
        self.X_ixs = X_ixs
        self.agn_ixs = agn_ixs
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim, dtype=float)
        self.lb = np.zeros(dim, dtype=float)
        self.ub = np.ones(dim, dtype=float)
        self.int_var = np.array([])  # none are integer-valued
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional DRV optimization for BayesianNetwork: " + str(bn_with_drv)

        if objective_func is None:
            self.obj_f = self.bn.objective_single_target_mi
        else:
            self.obj_f = objective_func

            assert np.isscalar(self.eval([0.5]*self.dim)), 'is the supplied objective function of correct signature? does it return a scalar?'
    
    def eval(self, x) -> float:
        if __debug__:
            assert len(x) == self.dim
        
        fval = self.obj_f(x, self.drv_ix, self.X_ixs, self.agn_ixs)

        return fval

class BayesianNetwork():

    pdfs = []
    # directed graph of indices into self.pdfs (1 -> 2 means 2 conditions on 1).
    # note: if a node conditions on other nodes, then the order of the keys in the corresponding
    # ConditionalProbabilityMatrix (i.e., self.pdfs[node]) is taken to be the sorted (ascending) 
    # order of the node's predecessors.
    dependency_graph = nx.empty_graph(create_using=nx.DiGraph)
    numvalues = []
    numvariables = 0  # this is the total of the number of variables in self.pdfs (in principle one pdf can describe multiple variables)
    # NOTE: for the time being, some code is hardcoded to assume that each PMF in `self.pdfs` contains only
    # one variable, so `self.numvariables` is actually redundant and should equal `len(self.numvalues)`
    # and `len(self.pdfs)`

    # NOTE: this cache is used internally to try and save some computations
    _use_cache_upto = 0  # maximum size of marginal BNs that are cached (0=none, 1=BNs of only 1 variable, ...)
    _marginal_pmf_cache = dict()  # from tuple[int] (sorted!) to BayesianNetwork

    _default_use_cache_upto = 0

    def __init__(self, use_cache_upto=None) -> None:
        if use_cache_upto is None:
            self.reset(use_cache_upto=self._default_use_cache_upto)
        else:
            self.reset(use_cache_upto=use_cache_upto)
    
    def reset(self, use_cache_upto=None) -> None:
        # BUG: sometimes, after optimization functions, a cached marginal can give different probabilities than the actual BN.
        assert use_cache_upto == 0 or use_cache_upto is None and self._use_cache_upto == 0, \
            'BUG: sometimes, after optimization functions, a cached marginal can give different probabilities than the actual BN.'  

        self.pdfs = []
        self.dependency_graph = nx.empty_graph(create_using=nx.DiGraph)  # directed graph of indices into self.pdfs
        self.numvalues = []
        self.numvariables = 0  # this is the total of the number of variables in self.pdfs (in principle one pdf could describe multiple variables)
        self._marginal_pmf_cache = dict()
        if not use_cache_upto is None:
            self._use_cache_upto = use_cache_upto
        
    
    def pop_pmf(self) -> None:
        """Remove the last PMF (highest index).

        This function is simpler than a more general `remove_pmf` because for sure this PMF does not have
        any successors, i.e., other variables dependent on it.
        """
        numvars_last_pmf = len(self.pdfs[-1])  # most likely just 1 but just in case
        self.pdfs.pop()
        ix_removed = len(self.pdfs)
        self.numvalues.pop()
        self.dependency_graph.remove_node(ix_removed)  # remove the node from the network
        self.numvariables -= numvars_last_pmf

        self.clear_cache()  # TODO: add functionality to only remove affected cached marginals

        # test some consistencies:
        assert self.numvariables >= 0
        assert len(self.numvalues) == len(self.pdfs)
        assert len(self.pdfs) == self.numvariables, 'remove this assert if PMFs in `self.pdfs can now contain also >1 variables (atm this is not possible)'


    def cache_marginal(self, variables: Sequence[int], bn: 'BayesianNetwork' = None, 
                       deepcopy=True) -> None:
        if bn is None:
            bn = self.marginal_pmf(variables)
        elif deepcopy:
            bn = copy.deepcopy(bn)
        bn._marginal_pmf_cache = dict()  # 'delete' the cache of the given BN so that we don't get endless nested caches
        self._marginal_pmf_cache[tuple(variables)] = bn  # cache the PMF

    
    def clear_cache(self, variables: Sequence[int] = None) -> None:
        if variables is None:
            self._marginal_pmf_cache = dict()  # 'clear' cache
        else:
            self._marginal_pmf_cache.pop(tuple(variables), None)

    
    def read_marginal_cache(self, variables: Sequence[int]) -> 'BayesianNetwork' or None:
        return self._marginal_pmf_cache.get(tuple(variables), None)
    

    def precompute_marginal_cache(self, upto: int = None) -> None:
        """Iterate over all possible combinations of variables, compute the marginal PMF, and cache it.

        Args:
            upto (int, optional): Iterate over tuples of variables of length 1,..,`upto`. Defaults to None,
             in which case `self._use_cache_upto` is used (set by __init__).
        """
        if upto is None:
            upto = self._use_cache_upto
        else:
            assert upto >= 0, 'mistake?'
        
        for numvars in range(1, upto + 1):
            for variables in itertools.combinations(range(len(self)), numvars):
                self.cache_marginal(variables)
    

    # TODO: add caching functionality to marginal_pmf, conditional_probabilities.


    def append_independent_variable(self, pmf : JointProbabilityMatrix | str = 'dirichlet', 
                                    numvalues : int = None, cache=False, verbose=0) -> int:
        """Append a PMF which does not depend on any existing PMFs.

        Args:
            pdf (JointProbabilityMatrix or str): If a string, then it should be accepted by the constructor
             of JointProbabilityMatrix, such as 'uniform' or 'dirichlet', and a PMF will be generated. 
             Otherwise provide a JointProbabilityMatrix object.
            numvalues (int, optional): Number of values that this PMF can take (state space). If not
             specified, the `numvalues` of the currently last variable is used. Defaults to None.

        Raises:
            UserWarning: currently it is expected that the `pdf` has only 1 variable in it.

        Returns:
            int: index of the appended PMF, which can be used in the other functions.
        """
        assert len(self.pdfs) == len(self.numvalues), 'should always be true'

        if isinstance(pmf, str):  # the user wants us to generate a pdf based on some method, like 'dirichlet'
            method_str = pmf

            if not numvalues is None:
                pmf = JointProbabilityMatrix(1, numvalues, joint_probs=method_str)
            else:
                if len(self.numvalues) >= 1:  # let's go with whatever numvalues was added last...
                    numvalues = self.numvalues[-1]
                else:
                    numvalues = 2  # let's just guess binary, why not
                
                pmf = JointProbabilityMatrix(1, self.numvalues[-1], joint_probs=method_str)
        elif isinstance(pmf, JointProbabilityMatrix):
            assert len(pmf) == 1, 'looking for bug'

            pass
        elif not np.isscalar(pmf):
            assert len(np.shape(pmf)) == 1, f'I can only deal with appending one variable at a time'
            np.testing.assert_almost_equal(np.sum(pmf), 1.0, decimal=_prob_tol_decimal)

            if verbose >= 2:
                print(f'debug: going to add {pmf=}')

            pmf = JointProbabilityMatrix(1, len(pmf), pmf)
        else:
            raise NotImplementedError(f'unknown {type(pmf)=}')

        if len(pmf) > 1:
            raise UserWarning('You are appending a PMF to a BayesianNetwork which has >1 variables in it. ' 
                              + 'Use this at your own peril; it is intended to be supported in the end, but '
                              + 'currently untested.')

        self.pdfs.append(copy.deepcopy(pmf))
        self.numvalues.append(pmf.numvalues)
        new_node_id = len(self.pdfs) - 1
        self.dependency_graph.add_node(len(self.pdfs) - 1)  # add a new node whose ID is the index into self.pdfs
        # note: no edges are added from this new node to any other node because it is independent
        self.numvariables += len(pmf)

        if self._use_cache_upto >= len(pmf) and cache:
            self.cache_marginal((new_node_id,), self.pdfs[-1], copy=False)

        return new_node_id


    def append_conditional_variable(self, cond_pmf : ConditionalProbabilityMatrix | str, 
                                    given_variables : list = None, numvalues : int = None, cache=False) -> int:
        assert len(self.pdfs) == len(self.numvalues), 'should always be true'

        if given_variables is None:
            # assume dependence is on all current variables if omitted
            given_variables = list(range(len(self)))

            if not isinstance(cond_pmf, str):
                assert len(given_variables) == cond_pmf.num_given_variables()
        else:
            assert hasattr(given_variables, '__iter__'), f'{given_variables=} is not iterable?'
            assert not isinstance(given_variables, str), f'{given_variables=} should not be a str..?'

        assert len(given_variables) <= len(self.pdfs), f'cannot condition on more variables than exist: {given_variables=}; {len(self.pdfs)=}.'
        assert max(given_variables) < len(self.pdfs), 'cannot condition on variables that do not exist'
        assert min(given_variables) >= 0, 'indices should be non-negative'

        if isinstance(cond_pmf, str):
            method_str = cond_pmf  # expected to be something like 'dirichlet' or 'uniform', that JointProbabilityMatrix() accepts
            cond_pmf = ConditionalProbabilityMatrix()

            numvalues_of_given_vars = [int(self.numvalues[gvix]) for gvix in given_variables]

            if numvalues is None:  # note that `numvalues` is of the output variable(s), not the conditioned-upon variables
                if len(self.numvalues) >= 1:  # use the maximum numvalues of the variables that this CMF will condition on (just to pick something)
                    numvalues = max(numvalues_of_given_vars)
                else:  # no variables existing yet? strange... but ok, if no error is generated, let's go with binary
                    numvalues = 2
            
            cond_pmf.generate_random_conditional_pdf(len(given_variables), 1, numvalues, num_given_values=numvalues_of_given_vars, method=method_str)
        elif isinstance(cond_pmf, ConditionalProbabilityMatrix):
            assert len(cond_pmf) == 1, 'looking for bug'

            pass
        elif not np.isscalar(cond_pmf):
            cond_pmf_shape = np.shape(cond_pmf)  # precompute for efficiency
            assert len(given_variables) == len(cond_pmf_shape) - 1, f'I assume that the first n dimensions are the' \
                                                                     + ' conditions and the last dimension is the PMF of the single variable to be added'
            cond_pmf = np.array(cond_pmf, dtype=_type_prob)
            cond_pmf = {inputvals: JointProbabilityMatrix(1, cond_pmf_shape[-1], cond_pmf[inputvals])
                        for inputvals in itertools.product(*[range(k) for k in cond_pmf_shape[:-1]])}
            cond_pmf = ConditionalProbabilityMatrix(cond_pmf, len(given_variables))

            assert cond_pmf.numvalues == cond_pmf_shape[-1]
            assert cond_pmf.numvariables == 1

            assert len(cond_pmf) == 1, 'looking for bug'
        else:
            raise NotImplementedError(f'unknown {type(cond_pmf)=}')
        
        assert isinstance(cond_pmf, ConditionalProbabilityMatrix), f'should be a CMF, but is {type(cond_pmf)=}'
        
        if len(cond_pmf) > 1:
            raise UserWarning('You are appending a CMF to a BayesianNetwork which has >1 variables in it. ' 
                              + 'Use this at your own peril; it is intended to be MAYBE supported in the end, but '
                              + 'currently not implemented and untested.')
        
        if __debug__:
            for states, pmf in cond_pmf.cond_pdf.items():
                for gix, gv in enumerate(given_variables):
                    assert states[gix] < self.numvalues[gv], f'The provided conditional PMF conditions on a state that does not exist. {states=}, {given_variables=}'
                
                if isinstance(pmf, JointProbabilityMatrix):
                    np.testing.assert_almost_equal(sum(pmf.joint_probabilities.joint_probabilities), 1.0)
                elif isinstance(pmf, BayesianNetwork):
                    np.testing.assert_almost_equal(sum(pmf.joint_probabilities().values()), 1.0)
                else:
                    raise NotImplementedError()
            
            states_cond_pmf = set(list(cond_pmf.cond_pdf.keys()))
            states_gv = set(self.statespace(given_variables))
            states_not_conditioned_on = states_gv.difference(states_cond_pmf)

            assert len(states_cond_pmf.difference(states_gv)) == 0, f'The cond. pmf. conditions on joint states that do not exist in the statespace of the {given_variables=}.'

            for hidden_states in states_not_conditioned_on:
                assert self.marginal_probability(given_variables, hidden_states) <= _prob_error_tol, f'the provided cond. pmf. does not condition on state {given_variables=} = {hidden_states}, but this state has >0 probability.'
            
        assert len(given_variables) == cond_pmf.num_given_variables()

        assert tuple(sorted(given_variables)) == tuple(given_variables), 'the conditional pdf is required to be constructed such that the variable indices it depends (conditions) on MUST be sorted in ascending order. This is implicitly assumed by e.g. self.joint_probability; self.given_variables only specifies the dependencies, not their order (could be added but at the moment is not).'

        self.pdfs.append(cond_pmf)
        pmf_ix = len(self.pdfs) - 1  # index of the new pdf into self.pdfs
        self.dependency_graph.add_node(pmf_ix)
        assert np.all([(given in self.dependency_graph) for given in given_variables]), 'each variable index that this pdf depends on should already exist in the graph'
        self.dependency_graph.add_edges_from([(given, pmf_ix) for given in given_variables])
        self.numvalues.append(cond_pmf.num_states())
        self.numvariables += len(cond_pmf)

        # compute and cache marginal pmf?
        if self._use_cache_upto >= cond_pmf.num_output_variables() and cache:
            pmf = self.marginal_pmf((pmf_ix,))
            self.cache_marginal((pmf_ix,), pmf, copy=False)

        return pmf_ix


    def matrix2params(self):
        # note: the lists in this list can be of different lengths (but always flat)
        # note: converted back to list because otherwise 'flatten' appears to not work properly
        nested_params = [list(pdf.matrix2params()) for pdf in self.pdfs]

        # test if we fully understand the different lengths in this nested_params (to be used in self.params2matrix)
        if __debug__:
            for ix, pdf in enumerate(self.pdfs):
                if self.dependency_graph.in_degree(ix) == 0:
                    # it's a JointProbabilitiesMatrix (doesn't depend on any other variables)
                    assert isinstance(pdf, JointProbabilityMatrix)
                    assert len(nested_params[ix]) == len(list(pdf.statespace())) - 1, '%i != %i' % (len(nested_params[ix]), len(list(pdf.statespace())) - 1)
                else:
                    # it's a ConditionalProbabilityMatrix
                    assert isinstance(pdf, ConditionalProbabilities)
                    assert len(nested_params[ix]) == pdf.num_conditional_probabilities() * (len(list(pdf.statespace())) - 1)
        
        return np.array(list(flatten(nested_params)))


    def params2matrix(self, params : np.ndarray):
        assert len(params) > 0
        assert np.isscalar(params[0]), 'params should not be a nested list but a flattened list'

        self.clear_cache()
        
        start_ix_params = 0  # running start index of the params to be used to re-init a pdf
        for ix, pdf in enumerate(self.pdfs):
            if self.dependency_graph.in_degree(ix) == 0: # this variable does not depend (condition) on any other variables
                assert isinstance(pdf, JointProbabilityMatrix)
                numparams = len(list(pdf.statespace())) - 1  # number of params needed to re-init this pdf
            else:
                assert isinstance(pdf, ConditionalProbabilities)
                numparams = pdf.num_conditional_probabilities() * (len(list(pdf.statespace())) - 1)
            
            assert len(params) >= start_ix_params + numparams
            
            self.pdfs[ix].params2matrix(params[start_ix_params:(start_ix_params+numparams)])

            start_ix_params += numparams
    

    def objective_single_SRV(self, params_SRV: Sequence[float], variable_SRV: int, variables_X: Sequence[int], agnostic_about: Sequence[int] = []) -> float:
        """Function to be used with global optimization procedures such as pySOT or scipy.optimize. Lower is better.

        Args:
            params_SRV (Sequence[float]): sequence of parameters (in [0,1] range) accepted by params2matrix().
            variable_SRV (int): index of the variable that already exists in <self> and is tested as an SRV for variables_X.
            variables_X (Sequence[int]): the variables that the SRV should be synergistic about.
            agnostic_about (Sequence[int]): optional: any prior SRVs that were already added for variables_X and that this SRV should be 'orthogonal' to (see paper).
        Returns:
            float: _description_
        """
        assert len(variables_X) > 0, 'cannot compute synergy with zero variables'
        
        if len(variables_X) == 1:
            return 0.0
        elif variable_SRV in variables_X:
            return 0.0
        
        assert variable_SRV < len(self)

        assert isinstance(self.pdfs[variable_SRV], ConditionalProbabilityMatrix), f'I expect a cond. PMF for SRV (an independent PMF would never be synergistic)'

        self.pdfs[variable_SRV].params2matrix(params_SRV)  # change the cond. PMF according to the given parameters
        # todo: make a clear_cache() option that only marginal PMFs which are affected by changing `variable_SRV`
        # are cleared from cache, and others left intact (for instance, simple hack could be: all marginals
        # where all variable indices are < `variable_SRV`; more fancy would be to obtain all downstream nodes)
        self.clear_cache()  # because I now change the PMF, clear the cache, if any (this does not happen automatically because here we directly change one of the sub-PMFs)

        totmi = self.mutual_information([variable_SRV], variables_X)  # we do want this
        indivmis = [self.mutual_information([variable_SRV], [xi]) for xi in variables_X]  # we don't want this
        agnostmis = [self.mutual_information([variable_SRV], [ai]) for ai in agnostic_about]  # we don't want this

        return sum(indivmis) + sum(agnostmis) - totmi

    def objective_single_target_mi(self, params_DRV: Sequence[float], variable_DRV: int, variables_X: Sequence[int], agnostic_about: Sequence[int] = [], target_mi: float = None) -> float:
        """Function to be used with global optimization procedures such as pySOT or scipy.optimize. Lower is better.

        Args:
            params_DRV (Sequence[float]): sequence of parameters (in [0,1] range) accepted by params2matrix().
            variable_DRV (int): index of the variable that already exists in <self> and is tested as a dependent random variable (DRV) for variables_X.
            variables_X (Sequence[int]): the variables that the DRV should be synergistic about.
            agnostic_about (Sequence[int]): optional: any other RVs that this DRV should not correlate with.
        Returns:
            float: I(D:A) - I(D:X) where A=agnostic and D=DRV.
        """
        assert len(variables_X) > 0, 'cannot compute MI with zero variables'
        
        if len(variables_X) == 0:
            return 0.0
        
        assert variable_DRV < len(self), 'the DRV should already be part of this PMF.'

        assert isinstance(self.pdfs[variable_DRV], ConditionalProbabilityMatrix), f'I expect a cond. PMF for DRV (an independent PMF would never correlate with anything)'

        self.pdfs[variable_DRV].params2matrix(params_DRV)  # change the cond. PMF according to the given parameters
        self.clear_cache()  # because I now change the PMF, clear the cache, if any (this does not happen automatically because here we directly change one of the sub-PMFs)

        if target_mi is None:
            totmi = self.mutual_information([variable_DRV], variables_X)  # we do want this
            if len(agnostic_about) > 0:
                agnostmi = self.mutual_information([variable_DRV], agnostic_about)  # we don't want this
            else:  # try to save a few function calls which will result in 0.0 anyway (performance)
                agnostmi = 0.0

            return agnostmi - totmi  # reminder: lower is better
        else:
            midiff = (target_mi - self.mutual_information([variable_DRV], variables_X))**2  # we don't want this
            if len(agnostic_about) > 0:
                agnostmi = self.mutual_information([variable_DRV], agnostic_about)  # we don't want this
            else:  # try to save a few function calls which will result in 0.0 anyway (performance)
                agnostmi = 0.0

            return agnostmi + midiff  # reminder: lower is better


    def append_direct_synergistic_variable(self, variables_X: Sequence[int], numvalues: int = None, 
                                           direct_srv_args={}, optimize_after=False, scipy_tol=1e-3, verbose=0) -> np.ndarray:
        """Append a 'direct SRV', assuming independence among the `variables_X`. 

        If the `variables_X` are indeed independent then the SRV is guaranteed to be purely synergistic,
        i.e., zero mutual information with any individual variable in `variables_X`.

        Args:
            variables_X (Sequence[int]): a sequence of integers indicating indices of variables ('inputs')
            numvalues (int, optional): number of values for the SRV to be constructed. Defaults to None.
            direct_srv_args (dict, optional): see the argument list of `compute_direct_srv_assuming_independence_v2`.

        Returns:
            np.ndarray: the conditional PMF of the direct SRV encoded in a numpy array, where the last
             axis encodes the conditional probabilities.
        """
        ps = [self.marginal_probabilities(xix) for xix in variables_X]
        # ps = [p.marginal_probabilities(0) for p in ps]

        if verbose >= 2:
            print(f'debug: {ps=}')
            print(f'debug: {numvalues=}')

        if __debug__:
            for p in ps:
                assert len(np.shape(p)) == 1, 'should be a single variable\'s probabilities'
                np.testing.assert_almost_equal(np.sum(p), 1.0, decimal=_prob_tol_decimal)
        
        if direct_srv_args.get('return_also_violation', False):
            direct_srv, totviol = ds.compute_direct_srv_assuming_independence(ps, numvalues, **direct_srv_args)

            if verbose >= 2:
                print(f'debug: total violation of the added SRV is {totviol}')
        else:
            direct_srv = ds.compute_direct_srv_assuming_independence(ps, numvalues, **direct_srv_args)

        if optimize_after:
            opt_direct_srv = ds.optimize_srv_using_scipy(direct_srv, ps, tol=scipy_tol, verbose=verbose-1)
        else:
            opt_direct_srv = direct_srv

        self.append_conditional_variable(opt_direct_srv, variables_X, numvalues)

        return opt_direct_srv


    def append_synergistic_variable(self, variables_X: Sequence[int], numvalues: int = None, agnostic_about=[], 
                                    max_evals=200, rel_tol=0.05, cache=False, verbose=0):
        if numvalues is None:
            # heuristic: pick the maximum number of states in X, since the maximum synergy
            # that is possible to need to store is H(x|X\x).
            # todo: check the maximum H(x|X\x) and pick `numvalues` based on that?
            numvalues = max([self.numvalues[x] for x in variables_X])

        if __debug__:
            _debug_original_len_bn = len(self)

        # first append a variable with the correct conditioning and number of values, but
        # arbitrary probabilities:
        self.append_conditional_variable('uniform', variables_X, numvalues=numvalues)
        srv_ix = len(self)-1

        # create a problem object
        # note: 'self' is passed by reference so updated in-place
        synprob = SynergyProblem(self, srv_ix, variables_X, agn_ixs=agnostic_about)

        # this bit I adapted from some examples in the pySOT github:
        rbf = RBFInterpolant(dim=synprob.dim, lb=synprob.lb, ub=synprob.ub, kernel=CubicKernel(), tail=LinearTail(synprob.dim))
        slhd = SymmetricLatinHypercube(dim=synprob.dim, num_pts=2 * (synprob.dim + 1))

        # this as well; create a strategy and a controller
        controller = SerialController(objective=synprob.eval)
        controller.strategy = SRBFStrategy(max_evals=max_evals, opt_prob=synprob, exp_design=slhd, surrogate=rbf)

        optres = controller.run()

        if optres.is_completed:
            # make sure that the appended SRV reflects the most optimal solution
            # (this might be redundant but I do not know if pySOT guarantees that the final
            # call to the objective function is to the most optimal solution [I guess not])
            self.pdfs[srv_ix].params2matrix(optres.params[0])
            self.clear_cache()

            # check if the found SRV violates any (relative) tolerances
            if not rel_tol is None:
                base_reltol_check = 2  # base of H and MI to be used here

                H_srv = self.entropy([srv_ix], base=base_reltol_check)
                H_srv_given_X = self.conditional_entropy([srv_ix], variables_X, base=base_reltol_check)
                MI_srv_X = H_srv - H_srv_given_X  # shorthand

                if MI_srv_X < rel_tol * H_srv:
                    optres._status = "out_of_relative_tol"  # this makes optres.is_completed == False

                    if verbose >= 1:
                        print(f'append_synergistic_variable: SRV optimization succeeded according to pySOT, but fails due to the provided {rel_tol=}: I(SRV:X)={MI_srv_X} < {rel_tol} * H(SRV)={H_srv} (the SRV is \'too small\').')
                else:
                    if not agnostic_about is None:
                        H_srv_given_A = self.conditional_entropy([srv_ix], agnostic_about, base=base_reltol_check)
                        MI_srv_A = H_srv - H_srv_given_A  # shorthand

                        if MI_srv_A >= rel_tol * H_srv:  # too much of the SRV correlates with the agnostic variables:
                            optres._status = "out_of_relative_tol"  # this makes optres.is_completed == False

                            if verbose >= 1:
                                print(f'append_synergistic_variable: SRV optimization succeeded according to pySOT, but fails due to the provided {rel_tol=}: I(SRV:A)={MI_srv_A} < {rel_tol} * H(SRV)={H_srv} (the SRV is \'too correlated with the to-be-agnostic-about-variables\').')
        
        # if optres.is_completed:
            # make sure that the appended SRV reflects the most optimal solution
            # (this might be redundant but I do not know if pySOT guarantees that the final
            # call to the objective function is to the most optimal solution [I guess not])
            # self.pdfs[srv_ix].params2matrix(optres.params[0])
            # self.clear_cache()

            if cache:
                self.cache_marginal((srv_ix,))
        
        if not optres.is_completed:  # note: this is not an 'else' to the above if because inside that if the optres can be made invalid again
            assert srv_ix == len(self) - 1, 'assumed by pop_pmf'
            self.pop_pmf()  # remove the SRV, since it could not be successfully calibrated OR it violated the user-specified tolerances

            assert len(self) == _debug_original_len_bn, f'after a failed SRV optimization, no variable should have been appended (or removed): {len(self)=}, {_debug_original_len_bn=}'

        return optres  # 'self' is updated in place (if successful)


    def append_dependent_variable(self, variables_X: Sequence[int], numvalues: int = FileNotFoundError, target_mi: float = None,
                                  agnostic_about: Sequence[int] = [], max_evals=100, cache=False):
        # first append a variable with the correct conditioning and number of values
        self.append_conditional_variable('uniform', variables_X, numvalues=numvalues)
        drv_ix = len(self)-1

        if numvalues is None:
            numvalues = max([self.numvalues[vix] for vix in variables_X])

        # create a problem object
        # note: 'self' is passed by reference so updated in-place
        if target_mi is None:
            depprob = DependenceProblem(self, drv_ix, variables_X, agnostic_about, self.objective_single_target_mi)
        else:
            # first construct a new function that has the required signature while specifying the target MI:
            self.obj_f_target_mi = lambda x, drv_ix, X_ixs, agn_ixs: self.objective_single_target_mi(x, drv_ix, X_ixs, agn_ixs, target_mi=target_mi)

            depprob = DependenceProblem(self, drv_ix, variables_X, agnostic_about, self.obj_f_target_mi)

        # this bit I adapted from some examples in the pySOT github:
        rbf = RBFInterpolant(dim=depprob.dim, lb=depprob.lb, ub=depprob.ub, kernel=CubicKernel(), tail=LinearTail(depprob.dim))
        slhd = SymmetricLatinHypercube(dim=depprob.dim, num_pts=2 * (depprob.dim + 1))

        # this as well; create a strategy and a controller
        controller = SerialController(objective=depprob.eval)
        controller.strategy = SRBFStrategy(max_evals=max_evals, opt_prob=depprob, exp_design=slhd, surrogate=rbf)

        optres = controller.run()

        if optres.is_completed:
            # make sure that the appended SRV reflects the most optimal solution
            # (this might be redundant but I do not know if pySOT guarantees that the final
            # call to the objective function is to the most optimal solution [I guess not])
            self.pdfs[drv_ix].params2matrix(optres.params[0])
            self.clear_cache()

            if cache:
                self.cache_marginal((drv_ix,))
        else:
            # TODO: remove the DRV and return the 'optres' anyway, since the caller can use that
            # to discover that appending a DRV failed. For now I will throw an exception.
            # raise UserWarning('DRV optimization failed')
            self.pop_pmf()  # remove the pmf we just added
        
        return optres  # 'self' is updated in place 


    def joint_probability(self, values : Sequence, variables : Sequence = None, verbose=0) -> float:
        """Return the joint probability of a sequence of values (one for each variable).

        The sum over all possible sequences of values is 1.0.

        Args:
            values (tuple or list or np.ndarray): sequence of values, one for each variable. If shorter than
                len(self) then the joint probability of the first len(values) variables is returned, ignoring
                the rest of the variables. (That is, all possible of these shorter sequences also sum to 1.0.)
            variables (Sequence): if specified, multiply only the (conditional) probabilities of these
                variable indices in the product. That is, suppose that a BN is p(A)*p(B|A)*p(C|B) but
                variables=[0,2] then actually what is returned is p(A)*p(C|B). This example shows that this makes
                no sense, UNLESS you specifically try to ignore certain variables because they are not part of
                the markov blanket or full set of ancestors of a particular set of variables. Therefore, use
                this with CAUTION. If you want to compute the probability of a subset of variables, use
                marginal_probability instead.

        Returns:
            float: probability
        """
        if variables is None:
            variables = range(len(self))
        else:
            if len(variables) > 1:
                assert len(values) > max(variables), f'should specify a value for each variable index you specified, at least: {values=}, {variables=}'
            else:
                # return 1.0  # return a default probability for an empty state (note: created an assertion error)
                pass  # let the below code deal with this situation
        
        if verbose > 0:
            print(f'{values=}')
            print(f'{variables=}')
            print(f'{len(self)=}')
        
        joint_prob = _type_prob(1.0)
        start_ix_values = 0
        for ix, pdf in enumerate(self.pdfs):
            assert start_ix_values < len(values)  # todo: remove this after a while

            if verbose > 0:
                print(f'joint_probability: loop {ix=}, {start_ix_values=}, {ix in variables=}')

            if ix in variables:
                if self.dependency_graph.in_degree(ix) == 0:  # this variable does not depend (condition) on any other variables
                    if (start_ix_values + len(pdf)) <= len(values):
                        subprob = pdf.joint_probability(values[start_ix_values:(start_ix_values+len(pdf))])

                        if verbose > 0:
                            print(f'joint_probability: p({values[start_ix_values:(start_ix_values+len(pdf))]}) = {subprob}, \t{start_ix_values=}, {ix=}')

                        joint_prob *= subprob
                    else:
                        # assert False, 'todo: return marginalized probability'  # todo
                        raise UserWarning('"values" is too short; should include values for all indices in "variables".')
                else:  # this variable conditions on other variables
                    given_variables = tuple(sorted(self.dependency_graph.predecessors(ix)))
                    assert min(given_variables) >= 0, 'should be indices into the list self.pdfs'
                    assert max(given_variables) < ix, 'variables should only depend on other variables that are defined BEFORE them, so lower index (it is a DAG).'
                    # which variables are conditioned on but not part of the variables that we have valid values for?
                    given_variables_to_sum_out = np.setdiff1d(given_variables, variables)
                    assert np.all(np.less(given_variables_to_sum_out, max(variables))), 'should not condition on variables with higher index! internal error'
                    given_variables_to_sum_out_ixs = [given_variables.index(gvix) for gvix in given_variables_to_sum_out]

                    if __debug__:
                        givvars_sum_over_values = [values[gvix] for gvix in given_variables_to_sum_out]
                        assert np.all(np.less(givvars_sum_over_values, 0)), f'it would be good practice to specify a negative (invalid) value at each index in {values=} which is not in {variables=} to indicate that you really intended not to take that value into account (since {ix=} depends on {given_variables=} but not all of those are in the aforementioned \'variables\').'

                    # note: the -1s are intended to be overwritten in the loop below (this is why it is cast to a np.array first)
                    given_values = np.array(tuple((values[gvix] if not gvix in given_variables_to_sum_out else -1 for gvix in given_variables)), dtype=int)

                    if __debug__:  # looking for bug
                        original_given_values = copy.copy(given_values)
                    
                    # note: 'pdf' is now a conditional pdf: ConditionalProbabilityMatrix (which is essentially a wrapper for a dictionary of pdfs)
                    if (start_ix_values + len(pdf)) <= len(values):
                        cond_prob = 0.0
                        # if not all variables in given_variables are in 'variables' then we must marginalize ... 
                        for states in self.statespace(given_variables_to_sum_out):
                            if not tuple(given_values) in pdf.cond_pdf:
                                # print(f'error: {len(self)=}, variable {ix=}, {self.numvalues=}, {len(self.pdfs[ix])=}, {start_ix_values=}')
                                # print(f'error: {given_variables_to_sum_out_ixs=} (used to fill in values {states=} into the array of given_values (below))')
                                # print(f'error: {pdf.cond_pdf.keys()=}')
                                # print(f'error: {tuple(given_values)=}')
                                # print(f'error: {original_given_values=}')
                                # print(f'error: {values=}')
                                # print(f'error: {given_variables=}')
                                # print(f'error: {self.dependency_graph.edges=}')

                                # raise UserWarning('not tuple(given_values) in pdf.cond_pdf')
                                # subprob = 0.0

                                # # assert marginal_prob_sum_over == 0.0, f'{marginal_prob_sum_over=}, {given_variables_to_sum_out_ixs=}, {given_values=}, {states=}, {given_variables=}, {variables=}, {values=}'
                                # marginal_prob_sum_over = 1.0  # to let the division below not generate an error
                                marginal_prob_sum_over = 0.0  # `given_values` was apparently unseen in the data
                                subprob = 0.0
                            else:
                                # given_values = np.array(tuple((values[gvix] if not gvix in given_variables_to_sum_out else -1 for gvix in given_variables)), dtype=int)
                                np.put(given_values, given_variables_to_sum_out_ixs, states)  # overwrite the -1s with the states to sum over

                                assert min(given_values) >= 0, f'given_values is used for indexing into a conditional PMF and this expects non-negative values (-1 could indicate that the caller of this function expects this value not to be used in the computation, which is then hereby violated). {values=}, {variables=}, {given_variables=}, {given_values=}, {len(self)=}.'

                                marginal_prob_sum_over = self.marginal_probability(given_variables_to_sum_out, states)

                                subprob = pdf[tuple(given_values)].joint_probability(values[start_ix_values:(start_ix_values+len(pdf))])

                            if verbose > 0:
                                print(f'joint_probability: p({values[start_ix_values:(start_ix_values+len(pdf))]} | {given_values}) = {subprob}, \t{start_ix_values=}, {ix=}')

                            cond_prob += subprob * marginal_prob_sum_over
                    else:
                        # assert False, 'todo: return marginalized probability'  # todo
                        raise UserWarning('"values" is too short; should include values for all indices in "variables".')
                    joint_prob *= cond_prob
            else:
                # let the joint prob. unchanged
                if verbose > 0:
                    print(f'joint_probability: variable {ix} not in {variables=} so ignored')

            start_ix_values += len(pdf)

            if verbose > 0:
                print(f'joint_probability: end of loop. {len(pdf)=}, {start_ix_values=}, {len(values)=}')

            assert not start_ix_values > len(values)  # huh? which values have been used above here then?

            if start_ix_values == len(values):
                break  # allow the length of 'values' to be shorter than len(self.pdfs)

        return joint_prob

    def joint_probabilities(self) -> dict:
        # TODO: this is far from efficient for larger BNs; find a more efficient way; maybe multiplying matrices?
        return {s: self.joint_probability(s) for s in self.statespace()}


    def marginal_probabilities(self, xix: int, verbose=0) -> np.ndarray:
        pmf = self.marginal_pmf([xix], verbose=verbose)

        assert len(pmf) == 1

        if isinstance(pmf, JointProbabilityMatrix):
            return pmf.joint_probabilities.joint_probabilities
        elif isinstance(pmf, BayesianNetwork):
            if isinstance(pmf.pdfs[0], JointProbabilityMatrix):
                return pmf.pdfs[0].joint_probabilities.joint_probabilities
            else:
                return pmf.pdfs[0].marginal_probabilities(0)


    def hellinger_distance(self, other: "BayesianNetwork") -> float:
        self_probs = self.joint_probabilities()
        other_probs = other.joint_probabilities()

        dist = 0.5 * np.sum([np.power(np.sqrt(self_probs[s]) - np.sqrt(other_probs[s]), 2) for s in self_probs.keys()])

        assert np.isscalar(dist)

        return dist
    

    def all_upstream_nodes(self, variables: Sequence[int], given_variables: Sequence[int] = None) -> set:
        """The full set of nodes to which the 'variables' might be causally dependent.

        The 'given_variables' represent a set of variables (indices) which are considered to be set to a 
        specific value and 'pinned' on that value. In that regard, they (and their ancestors) are ignored
        by this function since after pinning their value, they no longer have causal effect.

        This is NOT the Markov blanket. It is basically all nodes reachable upstream in the dependency graph.

        Args:
            variable_ix (int): index of the variable in question.

        Returns:
            set: set of nodes
        """
        if not given_variables is None:
            # 'remove' the given_variables nodes so that when asking for ancestors of a variable, these
            # given_variables and their ancestors are ignored
            dep_graph_given = self.dependency_graph.subgraph(np.setdiff1d(self.dependency_graph.nodes(), given_variables))

            try:
                assert len(dep_graph_given) == len(self.dependency_graph) - len(set(given_variables)), f'seems like not all indices in {given_variables=} are valid, because they do not occur in {self.dependency_graph.nodes()=}'
            except TypeError:
                print(f'error: {dep_graph_given=}')
                print(f'error: {self.dependency_graph=}')
                print(f'error: {given_variables=}')
                print(f'error: {len(dep_graph_given)=}')
                print(f'error: {len(self.dependency_graph)=}')
                print(f'error: {len(set(given_variables))=}')
        else:
            dep_graph_given = self.dependency_graph  # nothing to remove

        # note: it may happen that 'vix' is no longer in 'dep_graph_given' in case 
        # 'given_variables' and 'variables' have non-zero overlap
        ancs = [nx.ancestors(dep_graph_given, vix) if vix in dep_graph_given else [] for vix in variables]

        ancs_set = set()
        ancs_set.update(*ancs)  # add each list of ancestors to the set, to remove overlaps between ancestor lists
        ancs_set = ancs_set.difference(variables)  # the variables themselves should not be part of 'upstream' nodes

        return ancs_set

    def markov_blanket(self, variable_ixs: Iterable[int]) -> set:
        parents = set()
        parents.update(*[self.dependency_graph.predecessors(vix) for vix in variable_ixs])

        children = set()
        children.update(*[self.dependency_graph.successors(vix) for vix in variable_ixs])
        for vix in variable_ixs:
            children.discard(vix)  # remove already the variables themselves from children since these parents are already included above and the variables themselves are not part of a blanket

        children_parents = [set(self.dependency_graph.predecessors(child)) for child in children]

        blanket = parents
        blanket.update(children, *children_parents)
        for vix in variable_ixs:
            blanket.discard(vix)  # make sure the variable itself is not part of the blanket (it is one of the parents of its children so probably included here)

        return blanket

    def conditional_probability(self, variables: Sequence[int], values: Sequence[int], 
                                given_variables: Sequence[int], given_values: Sequence[int], 
                                raise_on_unseen_condition=False, prob_condition=None, verbose=0):
        prob = _type_prob(0.0)

        if len(variables) == 0:
            # ill-defined case, but there is a recursion between this function 
            # and self.marginal_probability, which terminates this way:
            return 1.0
        
        assert len(variables) == len(values), f'should be same length: {variables=}, {values=}'

        # check for inconsistent conditioning (in principle it is weird that the user asks for this)
        overlap = np.intersect1d(variables, given_variables)
        for vix in overlap:
            ix_variables = variables.index(vix)
            ix_given_variables = given_variables.index(vix)
            if not values[ix_variables] == given_values[ix_given_variables]:
                warnings.warn(f'inconsistent conditioning: variables {overlap=} but their respective values do not match (at least not for {vix})')

                return _type_prob(0.0)  # prob(X!=x | X=x) should be 0 at all times

        # we need to iterate over values of the statespace, but only up to the maximum index among
        # variables and given_variables, because all variables can only condition
        # on lower variable indices, not higher. E.g. in A -> B -> C, p({A,B}) is always independent 
        # of C (but for instance p({B,C}) does depend on A in general).
        if len(given_variables) > 0:
            max_var_ix = max(max(given_variables), max(variables))
        else:
            max_var_ix = max(variables)

        # what is the complete set of variables that 'variables' depend on? 
        try:
            ancestors = self.all_upstream_nodes(variables, given_variables)
        except nx.NetworkXError as e:
            print(f'seems like not all indices in {variables=} are valid, because they do not occur in {self.dependency_graph.nodes()=}')

            raise e

        if len(ancestors) > 0:
            assert max(ancestors) < max_var_ix, 'conditional dependence should be going in decreasing order of variable index'

        # the `given_variables` may be after the variables (i.e., their integer indices are higher),
        # or the `given_variables` may not be part of the ancestors of variables. In both these cases,
        # we also need to iterate over these ancestors-of-the-given-variables, excluding the `variables`
        # and their ancestors
        ancestors_given_vars = self.all_upstream_nodes(given_variables, variables)

        # if __debug__:
        #     for varix in ancestors.union(ancestors_given_vars):
        #         assert not (varix in variables or varix in given_variables), 'I would not expect this to happen since each `varix` comes from an ancestry of one that excludes the other'

        # these are the variable indices to sum over (marginalize)
        var_ixs_to_sum_over = sorted(ancestors.union(ancestors_given_vars))

        vars_to_compute_prob_for = sorted(list(variables) + list(var_ixs_to_sum_over) + list(given_variables))
        # vars_to_compute_prob_for = np.append(np.append(np.array(variables, dtype=int), var_ixs_to_sum_over), given_variables)
        # efficient shorthand: corresponding values for the vars_to_compute_prob_for, where the 'states' will be replaced in the loop below
        # now set to -1 to prevent unintended errors):
        # joint_state = np.append(np.append(values, np.zeros(len(var_ixs_to_sum_over), dtype=int) - 1), given_values)
        # joint_state = list(values) + [-1]*len(var_ixs_to_sum_over) + list(given_values)
        joint_state = np.zeros(max_var_ix + 1, dtype=int) - 1  # any value unset below will be -1, i.e., invalid, so errors can be spotted

        np.put(joint_state, given_variables, given_values)  # prefill these values, always the same
        np.put(joint_state, variables, values)

        if verbose >= 1:
            print(f'conditional_probability: {ancestors=}')
            print(f'conditional_probability: {ancestors_given_vars=}')
            print(f'conditional_probability: {var_ixs_to_sum_over=}')
            print(f'conditional_probability: {vars_to_compute_prob_for=}')
            print(f'conditional_probability: {max_var_ix=}')

        for states in self.statespace(var_ixs_to_sum_over):
            np.put(joint_state, var_ixs_to_sum_over, states)

            # note: not all variable indices have a valid state set in joint_state, possibly, and
            # vars_to_compute_prob_for is used to make sure that those states are ignored.

            if __debug__:
                for vix in vars_to_compute_prob_for:
                    assert joint_state[vix] < self.numvalues[vix]

            subprob = self.joint_probability(joint_state, variables=vars_to_compute_prob_for)
            # subprob = self.marginal_probability(vars_to_compute_prob_for, joint_state)  # <-- leads to infinite recursion

            # return None  # finding bug that makes kernel crash

            if verbose > 1:
                print(f'conditional_probability: \tPr({joint_state}) = {subprob} \t{vars_to_compute_prob_for=}')
            
            prob += subprob

        if len(given_variables) > 0:
            if prob_condition is None:
                given_prob = self.marginal_probability(given_variables, given_values)

                if __debug__:
                    # compute the `given_prob` in a way that does not use caching: does it match?
                    given_prob_check = self.conditional_probability(given_variables, given_values, [], [])

                    # if you hit this then there is an issue with caching:
                    np.testing.assert_approx_equal(given_prob, given_prob_check, 5, f'given_prob != given_prob_check; {self._debug_marginal_probability_used_cache_last=}')
            else:
                # skip computation and use the user-supplied marginal prob. of the condition
                given_prob = _type_prob(prob_condition)

            if verbose >= 1:
                print(f'conditional_probability: Pr({given_variables} = {given_values}) = {given_prob=}  ({prob_condition=})')
        else:
            given_prob = 1.0

            if verbose >= 1:
                print(f'{given_prob=} (because given_variables is empty)')

        if given_prob > 0.0:
            return prob / given_prob
        else:
            if raise_on_unseen_condition:
                raise ValueError(f'The condition {given_variables=}={given_values=} has prob. 0, so asking for any probability conditioned on this is ill-defined.')
            else:
                return prob  # should not be so important what to return, since the condition never occurs 
    

    _debug_marginal_probability_used_cache_last = False
    def marginal_probability(self, variables: Sequence, values: Sequence, verbose: int = 0):
        if self._use_cache_upto >= len(variables):
            cached_pmf = self.read_marginal_cache(variables)

            if not cached_pmf is None:
                assert len(cached_pmf) == len(values), 'looking for bug'
                self._debug_marginal_probability_used_cache_last = True
                # reminder: in `cached_pmf` the variables will be numbered differently than in `variables`
                prob = cached_pmf.joint_probability(values, list(range(len(cached_pmf))))
                if __debug__:  # TODO: when this check passes for a while, remove it, for performance 
                    prob_check = self.conditional_probability(variables, values, [], [], verbose=verbose)
                    np.testing.assert_approx_equal(prob, prob_check, 5, 'marginal prob != prob_check')
                return prob
                # return cached_pmf.conditional_probability(variables, values, [], [], verbose=verbose)
            else:
                # NOTE: for this single probability that is asked for we will now not compute a cached
                # marginal PMF, because that could potentially take much longer than computing this single probability.
                # So, we take no action.
                pass  # not cached, must compute
        
        self._debug_marginal_probability_used_cache_last = False

        return self.conditional_probability(variables, values, [], [], verbose=verbose)


    def conditional_probabilities_pmf(self, variables: Sequence[int], given_variables: Sequence[int]) -> ConditionalProbabilityMatrix:
        """Create a conditional PMF p(variables|given_variables) as a JointProbabilityMatrix object.

        Note that each PMF that is part of the returned Conditional PMF is a JointProbabilityMatrix object, not
        a BayesianNetwork. This means that it has only a single 'numvalues' and stores the entire joint prob.
        mass distribution in memory, even if it is actually sparse. The 'numvalues' chosen is the maximum of
        the original variables in the BayesianNetwork that it represents.

        Args:
            variables (Sequence): list of indices into self.pdfs
            given_variables (Sequence): list of indices into self.pdfs (should not overlap with 'variables')

        Returns:
            ConditionalProbabilityMatrix: conditional PMF object
        """
        numvals_vars = [self.numvalues[vix] for vix in variables]  # shorthand

        max_numval = max(numvals_vars)  # JointProbabilityMatrix can only handle one numvalues, so take the maximum

        def make_jpm(given_values):
            valid_state = lambda values: np.all(np.less(values, numvals_vars))
            jointprobs = [self.conditional_probability(variables, values, given_variables, given_values) 
                          if valid_state(values) else 0.0 
                          for values in itertools.product(*[range(max_numval)]*len(variables))]
            jointprobs = np.array(jointprobs, dtype=_type_prob)  # convert
            jointprobs = np.reshape(jointprobs, numvals_vars)  # reshape as required by JointProbabilityMatrix
            return JointProbabilityMatrix(len(variables), numvalues=max_numval, joint_probs=jointprobs)

        cond_pmf = {given_values: make_jpm(given_values)
                    for given_values in self.statespace(given_variables)}

        return ConditionalProbabilityMatrix(cond_pmf)


    def conditional_probabilities(self, variables: Sequence[int], given_variables: Sequence[int], verbose=0) -> ConditionalProbabilityMatrix:
        """Create a conditional PMF p(variables|given_variables).

        This function intends to use BayesianNetwork objects instead of JointProbabilityMatrix in the ConditionalProbabilityMatrix returned.

        Args:
            variables (Sequence[int]): list of indices into self.pdfs
            given_variables (Sequence[int]): list of indices into self.pdfs

        Returns:
            ConditionalProbabilityMatrix: conditional PMF object where each PMF is a BayesianNetwork.
        """
        assert len(np.shape(variables)) == 1, f'`{variables=}` should be a list of indices (int) of variables'
        assert len(np.shape(given_variables)) == 1, f'`{given_variables=}` should be a list of indices (int) of variables'

        if len(variables) > 1:
            # ensure that `variables` and `given_variables` are actually sorted
            assert np.all([(variables[ix+1] > variables[ix]) for ix in range(len(variables) - 1)]), f'{variables=} should be sorted! it is still on the todo-list to support unsorted arguments (which amounts to reordering variables)'
            assert np.all([(given_variables[ix+1] > given_variables[ix]) for ix in range(len(given_variables) - 1)]), f'{given_variables=} should be sorted! it is still on the todo-list to support unsorted arguments (which amounts to reordering variables)'

        predecessors_per_var = [list(self.dependency_graph.predecessors(vix)) for vix in variables]
        specified_vars = np.append(variables, given_variables).astype(int, copy=False)
        # these variables are conditioned on by one or more of the 'variables' but are not in either 'variables' or 'given_variables', which
        # means we have to marginalize over them:
        # vars_to_sum_out = [np.setdiff1d(preds, specified_vars) for ix, preds in enumerate(predecessors_per_var)]
        # in each PMF (for each conditioned state for given_variables) each variable will still condition on variables if those
        # conditioned variables are inside 'variables', otherwise it must be summed out
        preds_to_keep = [np.array(np.intersect1d(preds, variables), dtype=int)
                         for preds in predecessors_per_var]
        old_varixs_to_new_varixs = {varix: ix for ix, varix in enumerate(variables)}

        # TODO: add a functionality to shrink `given_variables` to only those variables on which 
        # `variables` actually depend, which is worst-case still `given_variables` but might be smaller
        # or empty (so the indices of which variables are depended upon must then also be returned;
        # maybe as part of the ConditionalProbabilityMatrix object?).

        if verbose > 0:
            print(f'conditional_probabilities: ({verbose=}) {predecessors_per_var=}')
            print(f'conditional_probabilities: ({verbose=}) {preds_to_keep=}')

        cond_dict_overall = dict()  # from tuples of 'given_values' to BayesianNetwork objects

        given_variables = np.array(given_variables, dtype=int)  # convert so that below we get correct data types (np.append)

        if len(given_variables) > 0:
            given_vars_pmf = self.marginal_pmf(given_variables, verbose=verbose-1)
        else:
            # break an infinite recursion
            given_vars_pmf = BayesianNetwork()

        # seems inefficient but for now will do:
        for given_values in self.statespace(given_variables):
            # prob_condition = self.marginal_probability(given_variables, given_values)
            prob_condition = given_vars_pmf.joint_probability(given_values)

            assert len(given_variables) > 0 or prob_condition == 1.0, f'if not conditioning then I think the statespace should consist of one element (empty tuple) which should have probability 1.0? {prob_condition=}, {list(self.statespace(given_variables))=}'

            if prob_condition > 0.0:
                # construct a BayesianNetwork for this case
                bn = BayesianNetwork(use_cache_upto=0)  # start empty

                for ix in range(len(variables)):
                    
                    varix = variables[ix]
                    pdf = self.pdfs[varix]  # TODO: double check: just changed this from `ix` to `varix`

                    if len(predecessors_per_var[ix]) == 0:  # it is an independent variable (does not condition on anything)
                        newpmf = pdf.copy()

                        bn.append_independent_variable(newpmf)

                        if verbose > 1:
                            print(f'conditional_probabilities: var {varix=} is independent: copied from original BN.')
                    else:  # this variable used to condition on other variables
                        assert pdf.numvariables == 1, f'the code in this block is specialized to only 1 variable per pdf in the BN (so either break this object up (at {ix=}) or fix this code to be more general)'

                        if len(preds_to_keep[ix]) == 0:  # will no longer condition on any variables in the reduced set of variables ('variables')
                            try:
                                jointprobs = [self.conditional_probability([varix], values, given_variables, given_values, 
                                                                        raise_on_unseen_condition=True, prob_condition=prob_condition) 
                                            for values in self.statespace([varix])]
                                jointprobs = np.array(jointprobs, dtype=_type_prob)  # convert
                            except ValueError as e:
                                assert False, 'should not happen anymore (remove try-except after this does not happen for a while)'
                                # assert 'condition' in str(e), 'I assume this error means that the condition given_variables=given_values has prob. zero; currently the function conditional_probability has no other way of generating ValueError.'

                                # # this condition has probability zero, so the P(variables|given_variables=given_values)
                                # # for this case is ill-defined. Since it probably does not matter what we now take
                                # # for this P (probably multiplied by P(given_Variables=given_values)=0 or used in 
                                # # mutual information where 0 log 0 = 0)
                                # jointprobs = np.ones(self.statespace([varix], only_size=True), dtype=float) / self.statespace([varix], only_size=True)

                                # # TODO: catch the ValueError in the larger code block and skip adding the condition?

                            # todo: if the sum of probs is zero then assert that the marginal prob is zero, and 
                            # act accordingly? OR: append an arbitrary cond. PMF just to not have this problem?
                            if __debug__:
                                try:
                                    np.testing.assert_almost_equal(jointprobs.sum(), 1.0)
                                except AssertionError as e:
                                    print(f'error: {given_variables=}')
                                    print(f'error: {given_values=}')
                                    print(f'error: {self.marginal_probability(given_variables, given_values, verbose=1)=}')
                                    print(f'error: {list(self.pdfs[varix].cond_pdf.keys())=}')
                                    print(f'error: {self.pdfs[varix].cond_pdf[tuple(given_values)].joint_probabilities.joint_probabilities=}')
                                    raise e

                            # jointprobs = np.reshape(jointprobs, numvals_vars)  # reshape as required by JointProbabilityMatrix
                            newpmf = JointProbabilityMatrix(1, numvalues=self.numvalues[varix], joint_probs=jointprobs)

                            bn.append_independent_variable(newpmf)

                            if verbose > 1:
                                print(f'conditional_probabilities: var {varix=} is conditional in the original BN, but not anymore in the new BN. Made into a {newpmf=}.')
                        else:
                            if len(bn) == 0 and __debug__:
                                print(f'error: {len(bn)=}; {ix=}; {preds_to_keep[ix]=} == function was called with: {variables=}; {given_variables}. MORE DETAILS:  {predecessors_per_var=};  {preds_to_keep[ix]=};  {old_varixs_to_new_varixs=}')
                                assert False, f'strange: how can I be appending a conditional PMF if there are no variables yet to condition on?'

                            cond_dict = dict()  # from tuples of values to JointProbabilityMatrix objects

                            extended_given_vars = np.append(given_variables, preds_to_keep[ix])
                            # extended_given_vals = np.append(given_values, [-1]*len(preds_to_keep[ix]))  # -1s to be replaced in the loop below
                            extended_given_vals = np.append(given_values, np.full(len(preds_to_keep[ix]), -1, dtype=int))  # -1s to be replaced in the loop below

                            # for some reason the dtype of the np.append above is not int, even though all
                            # arguments are int arrays, so make sure of correct type:
                            extended_given_vars = extended_given_vars.astype(dtype=int, copy=False)
                            extended_given_vals = extended_given_vals.astype(dtype=int, copy=False)

                            assert np.issubdtype(extended_given_vars.dtype, np.integer), 'list of variable indices should be a list of integer-type'
                            assert np.issubdtype(extended_given_vals.dtype, np.integer), 'list of values should be a list of integer-type'

                            if verbose > 1:
                                print(f'conditional_probabilities: var {varix=} is conditional on {predecessors_per_var[ix]}, in the new BN only on {preds_to_keep[ix]}.')
                                print(f'conditional_probabilities: \t{extended_given_vars=}')

                            for preds_to_keep_values in self.statespace(preds_to_keep[ix]):
                                # 'write' the new conditioned values in the pre-allocated array (hopefully efficient):
                                np.put(extended_given_vals, range(len(given_values), len(extended_given_vals)), preds_to_keep_values)

                                jointprobs = [self.conditional_probability([varix], values, extended_given_vars, extended_given_vals) 
                                            for values in self.statespace([varix])]
                                jointprobs = np.array(jointprobs, dtype=_type_prob)  # convert
                                # jointprobs = np.reshape(jointprobs, numvals_vars)  # reshape as required by JointProbabilityMatrix
                                if abs(np.sum(jointprobs)) < 1e-7:
                                    # all probabilities are zero. This can happen if the condition (extended_given_vars == extended_given_vals)
                                    # is never observed, so for p(Y|X) we have p(X)=0.
                                    # HACK: just make some kind of valid PMF just so that everything works, and hope it is not
                                    # actually used (uniform because it introduces a minimal amount of correlation hopefull, in case
                                    # it happens to still be used for something (add a property that let's any call to this PMF
                                    # throw an error?))
                                    newpmf = JointProbabilityMatrix(1, numvalues=self.numvalues[varix], joint_probs='uniform')
                                else:
                                    try:
                                        newpmf = JointProbabilityMatrix(1, numvalues=self.numvalues[varix], joint_probs=jointprobs)
                                    except AssertionError as e:
                                        print(f'error: {list(self.dependency_graph.edges())=}')
                                        print(f'error: {jointprobs=}')
                                        print(f'error: {np.sum(jointprobs)=}')
                                        print(f'error: {extended_given_vars=}')
                                        print(f'error: {extended_given_vals=}')
                                        print(f'error: {[varix]=}')
                                        print(f'error: {self.marginal_probability(extended_given_vars, extended_given_vals)=}')
                                        print(f'------')
                                        print(f'error: {self.conditional_probability([varix], [self.numvalues[varix] - 1], extended_given_vars, extended_given_vals, verbose=2)=}')
                                        print(f'------')
                                        print(f'error: {self.conditional_probability([varix], [self.numvalues[varix] - 1], [], [], verbose=2)=}')
                                        print(f'------')
                                        print(f'error: {self.conditional_probability(extended_given_vars, extended_given_vals, [], [], verbose=2)=}')
                                        print(f'------')
                                        # print(f'{self.joint_probabilities()}')
                                        jps = self.joint_probabilities()
                                        for state, prob in jps.items():
                                            print(f'error: \t{state}:\t{str(prob).replace(".", ",")}')
                                        print(f'error: \tsum: {sum(jps.values())}')
                                        raise e

                                cond_dict[preds_to_keep_values] = newpmf

                                if verbose > 2:
                                    print(f'conditional_probabilities: \t\tcond_dict[{preds_to_keep_values}] = {newpmf}')
                            
                            cond_pmf = ConditionalProbabilityMatrix(cond_dict)
                            
                            given_variables_ix = [old_varixs_to_new_varixs[ptk] for ptk in preds_to_keep[ix]]
                            if __debug__:
                                assert len(given_variables_ix) <= len(bn), f'{given_variables_ix=}; {len(bn)=}; {ix=}; {preds_to_keep[ix]=} == function was called with: {variables=}; {given_variables}. MORE DETAILS:  {predecessors_per_var=};  {preds_to_keep[ix]=};  {old_varixs_to_new_varixs=}'
                            bn.append_conditional_variable(cond_pmf, given_variables_ix)

                            if verbose > 1:
                                print(f'conditional_probabilities: \twill append {cond_pmf=}.')

                cond_dict_overall[given_values] = bn
            else:
                # NOTE: estimating a PMF based on this condition given_variables=given_values is
                # ill-defined. If `self` was estimated from data then this condition was never observed.
                pass  # do not add this condition (`given_values`) to the dict `cond_dict_overall`
        
        cond_pmf_overall = ConditionalProbabilityMatrix(cond_dict_overall)

        return cond_pmf_overall


    def pin(self, varix: int, state=None, shift=0):
        """Marginalize a variable (no incoming edges anymore) and let it have a single state with 100% probability.

        Args:
            varix (int): _description_
            state (int, optional): If desired, set the state of the variable always to this value. If not
             specified then a random choice is being made, weighted by the pre-intervention probabilities. 
             Defaults to None.
            shift (int, optional): This is added to the state that is selected or given. For instance, if with shift=0 the
             pinned state would have been 1, and `self.numvalues[varix] == 2`, then with shift=1 it would become 0, and vice versa.
             This can be used to pin the state to something counterfactual.
        """
        if not state is None:
            assert state < self.numvalues[varix], f'{state=} is not valid'

        # the variable is first marginalized and replaced by this marginal, removing all its causal predecessors (in-degree becomes zero)
        self.marginalize(varix)

        assert isinstance(self.pdfs[varix], JointProbabilityMatrix)  # for function hinting

        if state is None: 
            state = self.pdfs[varix].generate_sample()

            # change the JointProbabilityMatrix in-place by estimating it from a single sample
            self.pdfs[varix].estimate_from_data([[(s + shift) % self.numvalues[varix] for s in state]])
        else:
            # change the JointProbabilityMatrix in-place by estimating it from a single sample
            if np.isscalar(state):
                self.pdfs[varix].estimate_from_data([((state + shift) % self.numvalues[varix],)])
            else:
                self.pdfs[varix].estimate_from_data([[(s + shift) % self.numvalues[varix] for s in state]])


    def nudge(self, varix: int, nudge_norm=0.01, rel_tol_norm=0.1, tol_zero_mean=1e-2 * _essential_zero_prob, method='invariant',
              scipy_min_method=None, max_retries=5, type='soft', verbose=0) -> np.ndarray[float]:
        """Perform a (small) probabilistic intervention on the given variable, changing its probabilities in-place.

        Args:
            varix (int): index of variable to be nudged
            nudge_norm (float, optional): norm of the vector to be added to the probabilities of the variable's state. Lower values means
             smaller interventions, which also make it less likely that you run into errors (due to conditional variables which can sometimes
             have probabilities close to zero and other times close to 1, in which case no nudge vector would exist that keeps all probabilities
             always within the 0..1 range). Defaults to 0.01.
            rel_tol_norm (float, optional): Relative error that is acceptable in the norm of the nudge vector, relative to `nudge_norm`. Defaults to 0.1.
            tol_zero_mean (_type_, optional): Absolute error in the mean of the nudge vector that is acceptable (should be zero). Defaults to 0.01*_essential_zero_prob.
            method (str, optional): How the nudge is applied. If 'invariant', then the same nudge vector is added to each conditional distribution of the variable.
             Defaults to 'invariant'.
            scipy_min_method (_type_, optional): Directly passed to scipy's minimize function as `method` argument. Should be a method that can handle
             bounds. Defaults to None.
            max_retries (int, optional): Sometimes the minimize method from scipy fails to find a solution, or finds a solution that is not within the tolerances,
             but then if re-run then usually it does find a solution (if one exists). For this reason, if scipy says it fails or the norm of the nudge vector is
             not acceptable, re-try the optimization this many times. Defaults to 5.
            type (str, optional): Type of intervention to apply. If 'soft' then a nudged variable retains all its causal predecessors; if 'hard' then
             the variable is first marginalized and replaced by this marginal, removing all its causal predecessors (in-degree becomes zero), and then
             the nudge is applied.
            verbose (int, optional): Verbosity level (printing debugging and error messages). The higher, the more messages. Defaults to 0.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            np.ndarray[float]: _description_
        """
        assert method == 'invariant', f'this is the only method implemented now'

        if type == 'hard':
            # the variable is first marginalized and replaced by this marginal, removing all its causal predecessors (in-degree becomes zero)
            self.marginalize(varix)

        # determine the minimum and maximum (conditional) probabilities of the nudged variable (which has index `varix`), so that
        # the helper function can find a suitable nudge vector that does not bring any probability out of bounds (between 0 and 1).
        if isinstance(self.pdfs[varix], JointProbabilityMatrix):
            min_cond_probs = self.pdfs[varix].joint_probabilities.joint_probabilities
            max_cond_probs = min_cond_probs
        elif isinstance(self.pdfs[varix], ConditionalProbabilityMatrix):
            assert not type == 'hard', 'variable should have been marginalized?'

            # TODO: harmonize the access to the joint_probabilities for at least a single variable through BayesianNetwork; now I have to
            # keep using isinstance() and change how to access the probabilities.
            all_cond_probs = [(p.joint_probabilities.joint_probabilities if isinstance(p, JointProbabilityMatrix) 
                               else p.pdfs[0].joint_probabilities.joint_probabilities)  # then it should be a BayesianNetwork containing only one variable, which atm is a JointProbabilityMatrix object
                              for p in self.pdfs[varix].cond_pdf.values()]
            min_cond_probs = np.min(all_cond_probs, axis=0)
            max_cond_probs = np.max(all_cond_probs, axis=0)
        else:
            # I guess it could also be a BayesianNetwork object, which boils then again down to one of the above, but I'll worry about that when 
            # I hit this:
            raise NotImplementedError(f'{type(self.pdfs[varix])=}')
        
        # use the JointProbabilityMatrix's generate_nudge_vector_within_bounds
        nudge_vec = JointProbabilityMatrix.generate_nudge_within_bounds(nudge_norm, min_cond_probs, max_cond_probs, rel_tol_norm, tol_zero_mean, scipy_min_method=scipy_min_method,
                                                                        max_retries=max_retries, verbose=verbose-1)

        # 'apply' the nudge by adding the nudge vector to all conditional probability vectors
        if isinstance(self.pdfs[varix], JointProbabilityMatrix):
            self.pdfs[varix].joint_probabilities.joint_probabilities += nudge_vec

            assert np.sum(self.pdfs[varix].joint_probabilities.joint_probabilities) - 1.0 <= _essential_zero_prob, f'after nudging the probabilities don\'t add up to zero as nicely anymore: probs={np.sum(self.pdfs[varix].joint_probabilities.joint_probabilities)}'
            assert np.max(self.pdfs[varix].joint_probabilities.joint_probabilities) <= 1.0 + _essential_zero_prob, f'after nudging the max probability is not <= 1 anymore: max prob={np.max(self.pdfs[varix].joint_probabilities.joint_probabilities)}'
            assert np.min(self.pdfs[varix].joint_probabilities.joint_probabilities) >= 0.0 - _essential_zero_prob, f'after nudging the min probability is not >= 0 anymore: min prob={np.min(self.pdfs[varix].joint_probabilities.joint_probabilities)}'
        elif isinstance(self.pdfs[varix], ConditionalProbabilityMatrix):
            for cond_states in self.pdfs[varix].cond_pdf.keys():
                if isinstance(self.pdfs[varix].cond_pdf[cond_states], JointProbabilityMatrix):
                    self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities += nudge_vec

                    assert np.sum(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities) - 1.0 <= _essential_zero_prob, f'after nudging the probabilities don\'t add up to zero as nicely anymore: probs={np.sum(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                    assert np.max(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities) <= 1.0 + _essential_zero_prob, f'after nudging the max probability is not <= 1 anymore: max prob={np.max(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                    assert np.min(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities) >= 0.0 - _essential_zero_prob, f'after nudging the min probability is not >= 0 anymore: min prob={np.min(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                elif isinstance(self.pdfs[varix].cond_pdf[cond_states], BayesianNetwork):
                    assert len(self.pdfs[varix].cond_pdf[cond_states]) == 1, 'here I assume that each BayesianNetwork in the cond. PMF contains only one variable, for simplicity'
                    self.pdfs[varix].cond_pdf[cond_states].pdfs[0].joint_probabilities.joint_probabilities += nudge_vec

                    assert np.sum(self.pdfs[varix].cond_pdf[cond_states].pdfs[0].joint_probabilities.joint_probabilities) - 1.0 <= _essential_zero_prob, f'after nudging the probabilities don\'t add up to zero as nicely anymore: probs={np.sum(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                    assert np.max(self.pdfs[varix].cond_pdf[cond_states].pdfs[0].joint_probabilities.joint_probabilities) <= 1.0 + _essential_zero_prob, f'after nudging the max probability is not <= 1 anymore: max prob={np.max(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                    assert np.min(self.pdfs[varix].cond_pdf[cond_states].pdfs[0].joint_probabilities.joint_probabilities) >= 0.0 - _essential_zero_prob, f'after nudging the min probability is not >= 0 anymore: min prob={np.min(self.pdfs[varix].cond_pdf[cond_states].joint_probabilities.joint_probabilities)}'
                else:
                    raise NotImplementedError('not sure what other case I could imagine here!')
        else:
            # I guess it could also be a BayesianNetwork object, which boils then again down to one of the above, but I'll worry about that when 
            # I hit this:
            raise NotImplementedError(f'{type(self.pdfs[varix])=}')
        
        return nudge_vec  # this object was already changed in place, no further action needed; just return the nudge vector that was added


    def generate_instance_bn(self, sample : Sequence[int] = None) -> "BayesianNetwork":
        """Generate a BN which encodes 100% probability on a single sample and 0% otherwise.

        This is useful for instance to generate 'patients' who either have or do not have certain
        symptoms, signs, and diseases. The BN can be used e.g. to perform individual interventions and quantify
        the impact.

        Args:
            sample (Sequence[int], optional): a single sequence of int values, one state per variable in `self`.

        Returns:
            BayesianNetwork: _description_
        """
        if sample is None:
            sample = self.generate_sample()

        bn_patient = BayesianNetwork()
        bn_patient.infer_bn_on_dag([sample], self.dependency_graph, numvalues=self.numvalues)

        return bn_patient


    def generate_samples(self, n: int) -> Sequence[Sequence[int]]:
        assert n >= 0

        return np.array([self.generate_sample() for _ in range(n)], dtype=_type_state)
    

    def generate_continuous_samples(self, n: int, sigma=0.5) -> np.ndarray:
        """Generate discrete samples using `generate_samples` and then add normally distributed noise.

        Args:
            n (int): number of samples
            sigma (float, optional): standard deviation of the normally distributed noise. Defaults to 0.5.

        Returns:
            np.ndarray: an array of shape `(n, len(self))` with floats.
        """
        discr_data = self.generate_samples(n)
        
        return np.add(discr_data, np.random.normal(0, sigma, np.shape(discr_data)))
    

    def generate_sample(self) -> Sequence[int]:
        """Generate a single sample of states, i.e., one integer value per variable in this BN.

        Returns:
            Sequence[int]: list of integer values of length `len(self)`, in order of the ascending sorted `self.dependency_graph.nodes`
        """
        # find nodes that do not depend on any other nodes, so that we can start with those
        # to generate a random state, which should then inform states that depend on these roots, etc.
        roots = [node for node in self.dependency_graph.nodes if self.dependency_graph.in_degree(node) == 0]
        roots = deque(roots)  # make it a queue; using popleft() and append() make it fifo
        
        states = dict()  # from node id (int) to state tuples (tuple[int])

        while len(roots) > 0:
            node = roots.popleft()
            pdf = self.pdfs[node]  # shorthand
            preds = list(self.dependency_graph.predecessors(node))

            # if not all variables that this `node`` conditions on have already a state generated
            # for them, then just re-queue it and move on with other nodes
            if not np.all([pred in states.keys() for pred in preds]):
                assert len(roots) > 0, 'this would for sure gcreate an infinite loop'
                roots.append(node)
                continue  # skip this node for now, will return to it later

            if len(preds) == 0:  # root node
                # note: if you hit this assert but it is actually an instance of BayesianNetwork,
                # then I guess we have to overload the functions used below in this loop
                assert isinstance(pdf, JointProbabilityMatrix), 'root nodes are expected to be JointProbabilityMatrix, not conditional'

                state = pdf.generate_sample()  # this is a tuple, even if self.pdfs[node].numvariables == 1
                states[node] = state
            else:  # node which conditions on other nodes
                assert isinstance(pdf, ConditionalProbabilityMatrix), 'root nodes are expected to be JointProbabilityMatrix, not conditional'

                preds = sorted(preds)  # see the note with self.dependency_graph

                pred_values = [states[pred_node] for pred_node in preds]  # list of tuples

                pred_values = tuple(flatten(pred_values))

                # probs = [self.conditional_probability([node], [value], )]
                assert len(pred_values) == pdf.num_given_variables()
                pdf = pdf[pred_values]

                assert isinstance(pdf, (JointProbabilityMatrix, BayesianNetwork)), 'now should not be conditional anymore'

                state = pdf.generate_sample()  # this is a tuple, even if self.pdfs[node].numvariables == 1
                states[node] = state
            
            for succ_node in self.dependency_graph.successors(node):
                roots.append(succ_node)
        
        return list(flatten([states[node] for node in sorted(states.keys())]))
    

    def draw_graph(self, figsize=(3, 3)):
        for layer, nodes in enumerate(nx.topological_generations(self.dependency_graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                self.dependency_graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(self.dependency_graph, subset_key="layer")

        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        nx.draw_networkx(self.dependency_graph, pos=pos, ax=ax)
        ax.set_title("DAG layout in topological order (left to right)")
        fig.tight_layout()
        # plt.show()

        return fig


    def draw_correlation_matrix(self, corrmat=None, nodes=None, figsize=(7, 7),
                                title='Pairwise mutual information matrix'):
        import seaborn as sns

        if corrmat is None:
            corrmat = self.compute_correlation_matrix(nodes=nodes)
        else:
            assert nodes is None, 'you already provided `corrmat`, I don\'t need `nodes` now.'
        
        h = plt.figure(figsize=figsize)
        sns.heatmap(corrmat, annot=True)
        plt.title(title)

        return h  # maybe the caller wants to save it to file or something


    def compute_correlation_matrix(self, nodes=None) -> Sequence[Sequence[float]]:
        if nodes is None:
            nodes = list(self.dependency_graph.nodes)  # go through all
        
        corrmat = [[self.mutual_information([x], [y]) if x != y else self.entropy([x])
                    for y in nodes] 
                    for x in nodes]
        
        return corrmat
    
    @staticmethod
    def estimate_structure(df: pd.DataFrame, columns: Sequence[str] = None, method='mmhc', verbose=0) -> list[tuple]:
        if not columns is None:
            subset = df[columns]
        else:
            subset = df

        if verbose > 0:
            print(f'debug: estimate_structure: {subset.columns=}')
        
        if method == 'mmhc':
            est = MmhcEstimator(subset)

            with warnings.catch_warnings():
                # pgmpy causes some FutureWarning which is annoying
                warnings.simplefilter(action='ignore', category=FutureWarning)

                model = est.estimate()  # estimate the BN model
            
            edges = copy.copy(model.edges())
        else:
            raise NotImplementedError(f'unrecognized {method=} for estimating structure')

        if verbose > 0:
            print(f'debug: estimate_structure: result: {edges=}')

        return edges


    def infer_bn_on_dag(self, df: pd.DataFrame | Sequence[Sequence[int]], edges: Sequence[Sequence[Hashable]] | nx.DiGraph, columns: Sequence[Hashable] = None, 
                        method='mle', force_all_conditions=False, use_cache_upto=None, numvalues=None, verbose=False) -> dict[Hashable]:
        self.reset(use_cache_upto=use_cache_upto)  # make `self` empty (zero variables)

        # note: this is now a graph with column names (str) as nodes (while BayesianNetwork needs integers
        # in topological order). Will be fixed below.
        graph = nx.DiGraph(edges)

        if not hasattr(df, 'columns'):  # the user didn't supply a Pandas dataframe but probably a list of lists of states (int)
            _, num_states = np.shape(df)
            
            assert num_states == graph.number_of_nodes(), f'you provide a graph consisting of {graph.number_of_nodes()} nodes but the data samples contain {num_states} values, which does not match.'
            
            # note: here we assume that the column names are the sorted node names in `graph`
            # df = pd.DataFrame(df, columns=list(range(num_states)))
            df = pd.DataFrame(df, columns=sorted(graph.nodes))

        if not columns is None:
            subset = df[list(graph.nodes)]
        else:
            subset = df

        if __debug__:
            # check if the graph consists of nodes which are also column names in `df`
            for node in graph.nodes:
                assert node in df.columns, f'{node=} is part of the supplied `edges` but is not a column in the supplied `df`.'
            
            assert nx.is_directed_acyclic_graph(graph), 'the provided `edges` should define a DAG'
        
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        roots = deque(roots)  # make it a queue; using popleft() and append() make it fifo
        
        newid = dict()  # from 'old' column name (Hashable) to 'new' node id (int)

        if verbose >= 2:
            print(f'At the start, {roots=}')

        while len(roots) > 0:
            node = roots.popleft()

            # get the predecessors but in the same order of the variables added so far to `self`
            preds = list(graph.predecessors(node))  # shorthand
            try:
                pred_newid_pairs = sorted([(pred, newid[pred]) for pred in preds], key=lambda x: x[-1])
            except KeyError:
                # `pred` was apparently not yet processed here, so it did not get a new id in `newid`
                assert len(roots) > 0, 'this would for sure create an infinite loop'
                roots.append(node)

                continue
            preds = [pred for pred, _ in pred_newid_pairs]
            
            if node in newid:  # appended to the BN already?
                # note: a node could only be queued >1 times 'in parallel' if it has multiple predecessors:
                assert len(preds) > 1, f'node {node} should not have been handled already; maybe the given `edges` do not form a DAG?'

                continue

            # TODO: remove this after a while:
            assert np.all([pred in newid.keys() for pred in preds]), 'should already be dealt with in the above try-except'

            # if not all variables that this `node` conditions on have already a pmf generated
            # for them, then just re-queue it and move on with other nodes
            if False:  #not np.all([pred in newid.keys() for pred in preds]):
                if verbose >= 2:
                    print(f'debug: \tI popped {node=} but I re-queue it because its {preds=} are not all processed already.')

                assert len(roots) > 0, 'this would for sure create an infinite loop'
                roots.append(node)

                continue
            else:
                if verbose >= 2:
                    print(f'debug: \tI popped {node=} and will now add it to the BN, with predecessors {preds}.')
                
                # add the successors to the queue so that they are also considered later on
                for succ in graph.successors(node):
                    if verbose >= 2:
                        print(f'\t\tI will add {succ=} to the queue because it is a successor of {node=}')
                    roots.append(succ)
            
            if len(preds) == 0:  # defines a JointProbabilityMatrix
                assert 'int' in str(subset.dtypes[node]), f'column {node} is of type {subset.dtypes[node]} which is not recognized as a categorical or ordinal type (I expect integer-type).'

                data = subset.loc[:, node].to_numpy(dtype='int')
                if numvalues is None:
                    numvals = max(data) + 1  # assume that the maximum observed value is indeed the true maximum
                else:
                    numvals = numvalues[node]  # note: states are always 0, 1, ...

                pmf = JointProbabilityMatrix(1, numvals)
                pmf.estimate_from_data(np.expand_dims(data, axis=1), numvals)

                id = self.append_independent_variable(pmf)

                if verbose >= 2:
                    print(f'debug: \t{node=} was appended as independent variable to the BN with {id=}')
            else:  # we will construct a ConditionalProbabilityMatrix
                assert 'int' in str(subset.dtypes[node]), f'column {node} is of type {subset.dtypes[node]} which is not recognized as a categorical or ordinal type (I expect integer-type).'

                data = subset.loc[:, list(preds) + [node]].to_numpy(dtype='int')

                condvals = dict()  # from tuple[int] to list[int]

                if numvalues is None:
                    numvals = -np.inf  # running maximum of node values
                else:
                    # note: the user specified `numvalues[node]`, which may be larger than what we 
                    # otherwise would have found ourselves (if `numvalues` is None), but if it is lower
                    # then below it is corrected.
                    numvals = numvalues[node]

                for row in data:
                    # NOTE: the order of the predvals below
                    # should be the same as the order in which those variables ended
                    # up in the `pdf` so far (see `newid`)
                    predvals = tuple(row[:len(preds)])
                    nodeval = row[-1]

                    # append this node value to the already observed ones for this combination 
                    # of values for its predecessors
                    condvals[predvals] = condvals.get(predvals, []) + [nodeval]

                    if nodeval + 1 > numvals:
                        if not numvalues is None:
                            warnings.warn(f'infer_bn_on_dag: you specified {numvalues[node]=} where {node=}, but I found an observed value {nodeval} in the provided data for this node. Will correct it but maybe you made a mistake?')

                        numvals = nodeval + 1  # keep track of maximum value to be used below
                
                for predvals, nodedata in condvals.items():
                    # NOTE: Possible inefficiency. Suppose that `nodevals` is [0, 7]. Then `pmf` below
                    # will actually store an array of 8 probabilities, for all states 0..7, even
                    # though many of them do not occur. Values cannot be relabeled (e.g. 7 --> 1)
                    # because then the values would not be consistent anymore with other conditions
                    # (`predvals`). There is currently no easy solution I think... Perhaps in
                    # JointProbabilityMatrix the [0,1] values could be labeled [0,7], but then every
                    # where in that class we'd have to translate back and forth (might be slower).
                    # TODO: Inefficieny. It would be better to allow for each condition here to
                    # set the numvalues= argument of JointProbabilityMatrix to whatever is the `maxval+1`
                    # of _this_ condition, which may be smaller than using the maximum across
                    # all conditions. Then getters like .joint_probability should return 0 when
                    # they are asked for a value that is higher than what they store (like .get() of dict). 
                    # maxval = max(nodedata)  # JointProbabilityMatrix stores node values as a contiguous array
                    pmf = JointProbabilityMatrix(1, numvals, 'uniform')

                    pmf.estimate_from_data(np.expand_dims(nodedata, axis=1), numvals)

                    condvals[predvals] = pmf  # replace the node's conditional data by a pmf for this condition
                
                if force_all_conditions:
                    # if there are conditions for which no data was available, insert some arbitrary
                    # PMF just so that other parts of the code doesn't break (it would be nicer to 
                    # support not having per se all conditions stored in the ConditionalProbabilityMatrix
                    # but I could not figure it out and I don't have time now).
                    for predvals in itertools.product(*([range(numvals)]*len(preds))):
                        if not predvals in condvals:
                            condvals[predvals] = JointProbabilityMatrix(1, numvals, 'uniform')
                    
                    assert len(condvals) == (numvals)**len(preds), 'all conditions should have a PMF stored, even if not seen in the data'
                
                condpmf = ConditionalProbabilityMatrix(condvals, len(preds))

                if __debug__:
                    for pred in preds:
                        assert pred in newid, f'{newid=} does not contain {pred=} for {node=} (all {preds=})'

                id = self.append_conditional_variable(condpmf, sorted([newid[pred] for pred in preds]))

                if verbose >= 2:
                    print(f'debug: \t{node=} was appended as conditional variable to the BN with {id=}')
            
            newid[node] = id
            if verbose >= 2:
                print(f'debug: \t{node=} was added to {newid=}')
        
        return newid

            
    def infer_random_bn_on_dag(self, graph: nx.DiGraph | int, numvals: int | dict | list | Callable = 2, method_pdf='dirichlet', 
                                   method_cond_pdf: float | Callable | str = 'dirichlet', max_opt_evals=100,
                                   use_cache_upto=None, network_type='gn', reverse_gn=True, verbose=0) -> None:
        """Generate a random BayesianNetwork on a given or randomly generated DAG.

        This function is meant for rapid prototyping or testing.

        Args:
            graph (nx.DiGraph | int): either a DAG or an integer `n` which will be used to generate a random
             DAG using `networkx.gn_graph(n)` (after which the edges are inverted).
            numvals (int | Callable, optional): Number of values that each appended variable can take.
             If it is a function then it will be called as `numvals(node)` to generate an integer for
             variable `node`, where `node` is probably an integer index of the variable. Defaults to 2.
            method_pdf (str, optional): Method to use to generate independent pdfs (JointProbabilityMatrix). 
             Defaults to 'dirichlet'.
            method_cond_pdf (float | Callable | str, optional): Method to use to generate conditional pdfs
             (ConditionalProbabilityMatrix). If a string, it is passed to ConditionalProbabilityMatrix.generate_random_cond_pdf.
             If a float, it is taken as a target MI and the optimization procedure BayesianNetwork.append_dependent_variable
             is used. If it is Callable, it will be called `method_cond_pdf(k, m)` where `k` is the number of
             values of the to-be-appended variable and where `m` is the number of variables that the conditional 
             pdf conditions on. Defaults to 'dirichlet'.
            max_opt_evals (int, optional): Only used if method_pdf or method_cond_pdf indicates a target (MI) value.
             Indicates the maximum number of optimization steps to be performed (lower is faster but less accurate). 
             Defaults to 100.
            reverse_gn (bool, optional): Only has effect if `network_type`=='gn'. Reverses the generated edges. If True,
             then the resulting graph will only contain nodes which have 0 or 1 predecessors. If False, >1 predecessors
             also become possible (this can be interesting for generating multivariate or [partially] synergistic dependencies).
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        if np.isscalar(graph):
            if network_type == 'gn':
                graph = nx.gn_graph(int(graph), create_using=nx.DiGraph)
                if reverse_gn:
                    graph = graph.reverse()
            elif network_type == 'line':
                graph = nx.path_graph(int(graph), create_using=nx.DiGraph)
            else:
                raise NotImplementedError(f'unrecognized {network_type=} for generating a network')
        else:
            assert isinstance(graph, nx.DiGraph), '`graph` should be a directed graph'
            assert nx.is_directed_acyclic_graph(graph), '`graph` should be a DAG'

        # find nodes that do not depend on any other nodes, so that we can start with those
        # to generate a random state, which should then inform states that depend on these roots, etc.
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        roots = deque(roots)  # make it a queue; using popleft() and append() make it fifo
        
        newid = dict()  # from `graph` node id (int) to `self.dependency` node id (int)

        if not callable(numvals):
            if np.isscalar(numvals):
                numvals_f = lambda node: numvals  # ignore the `node` argument
            else:
                numvals_f = lambda node: numvals[node]  # assume `numvals` is a dict or a list
        else:
            numvals_f = numvals  # already a function that we can call to get a number of values

            assert numvals_f(roots[0]) >= 0, 'just testing if the function takes indeed one argument and it returns a plausible number of values'
        
        self.reset(use_cache_upto=use_cache_upto)  # make `self` empty (zero variables)

        time_start = time.time()

        while len(roots) > 0:
            node = roots.popleft()
            preds = list(sorted(graph.predecessors(node)))  # shorthand

            assert not node in newid, f'node {node} should not have been handled already; maybe the given `graph` is not a DAG?'

            # if not all variables that this `node` conditions on have already a pdf generated
            # for them, then just re-queue it and move on with other nodes
            if not np.all([pred in newid.keys() for pred in preds]):
                if verbose >= 2:
                    print(f'debug: \tI popped {node=} but I re-queue it because its {preds=} are not all processed already.')

                assert len(roots) > 0, 'this would for sure create an infinite loop'
                roots.append(node)
            else:
                if verbose >= 2:
                    print(f'debug: \tI popped {node=} and will now add it to the BN, with predecessors {preds}.')
                
                for succ in graph.successors(node):
                    roots.append(succ)

            nv = int(numvals_f(node))  # number of values to be used for this variable

            if len(preds) == 0:  # this is an independent (root) variable, does not condition on others
                pdf = JointProbabilityMatrix(1, nv, method_pdf)

                id = self.append_independent_variable(pdf)

                newid[node] = id
            else:  # this is a conditional variable
                cond_pdf = ConditionalProbabilityMatrix()

                if isinstance(method_cond_pdf, str):  # user passes a method name to generate cond. pdfs
                    cond_pdf.generate_random_conditional_pdf(len(preds), 1, nv, num_given_values=[self.numvalues[pred] for pred in preds], method=method_cond_pdf)

                    id = self.append_conditional_variable(cond_pdf, [newid[pred] for pred in preds], nv)
                else:
                    # TODO: make method_cond_pdf always a function so that this testing for variable type 
                    # does not have to occur inside the loop
                    if np.isscalar(method_cond_pdf):  # user wants to achieve a certain MI with the predecessor variables
                        target_mi = float(method_cond_pdf)
                    elif callable(method_cond_pdf):  # user specified a function which we can call to get the target MI
                        target_mi = float(method_cond_pdf(nv, len(preds)))
                    
                    assert target_mi >= 0.0, 'MI can only be non-negative'

                    if verbose >= 2:
                        time_start_opt = time.time()

                    optres = self.append_dependent_variable([newid[pred] for pred in preds], nv, target_mi, max_evals=max_opt_evals)
                    
                    if optres.is_completed:
                        id = len(self) - 1


                        if verbose >= 2:
                            actual_mi = self.mutual_information([newid[pred] for pred in preds], [id])
                            print(f'debug: \t\tOptimization took {time.time() - time_start_opt} seconds. {target_mi=} (upper bound: {np.log2(nv)}), {actual_mi=}, state space of conditioned variables: product of {[self.numvalues[newid[pred]] for pred in preds]}.')
                    else:
                        warnings.warn(f'append_dependent_variable\'s optimization did not complete successfully for node {id=}.')
                
                newid[node] = id
            
            if verbose >= 1:
                print(f'debug: \tappended variable {newid[node]} (originally node {node}) with {nv} values, after {time.time() - time_start} seconds.')
        
        assert len(self) == graph.number_of_nodes(), f'for each node in the given graph there should be a (cond.) pdf added in `self`: {len(self)=}, {graph.number_of_nodes()=}'


    def marginal_pmf(self, variables, verbose=0) -> 'BayesianNetwork':
        if len(variables) > 1:
            assert np.all([(variables[ix+1] > variables[ix]) for ix in range(len(variables) - 1)]), f'{variables=} should be sorted! it is still on the todo-list to support unsorted arguments (which amounts to reordering variables)'
        assert len(variables) <= len(self), f'cannot compute a marginal pmf for more variables than I have: {variables=}, {len(self)=}'
        if len(variables) > 0:
            assert max(variables) < len(self), f'at least one variable index is out of range: {variables=}, {len(self)=}'

        cond_dict = self.conditional_probabilities(variables, [], verbose=max(0,verbose-1))

        pmf = cond_dict[()]  # cond PMF will contain only one condition (empty tuple), so simply return the PMF

        if self._use_cache_upto >= len(pmf):
            self.cache_marginal(variables, pmf, deepcopy=True)

        return pmf


    def marginalize(self, variable: int) -> None:
        """Replace a conditional PMF by its marginalized version, removing its dependencies (in-edges).

        Args:
            variable (int): index of the variable to be marginalized

        Raises:
            NotImplementedError: `variable` is only supported to be a single integer index, cannot be a list etc.
        """
        if not np.isscalar(variable):
            # TODO: support also a list of variable indices in marginalize().
            raise NotImplementedError('`variable` is only supported to be a single integer index')

        if self.dependency_graph.in_degree(variable) == 0:
            return  # nothing to be done; this variable already does not depend on any other
        else:
            marginal_pmf = self.marginal_pmf([variable])  # this is a BayesianNetwork object, consisting of one pmf
            marginal_pmf = marginal_pmf.pdfs[0]  # this should be a JointProbabilityMatrix object

            assert isinstance(marginal_pmf, JointProbabilityMatrix)

            # replace the conditional pmf with this unconditional (marginalized) one
            self.pdfs[variable] = marginal_pmf

            # remove the incoming edges in the dependency graph, to reflect the new situation where
            # this variable no longer conditions on other variables
            self.dependency_graph.remove_edges_from(list(self.dependency_graph.in_edges(variable)))
    
    
    def pinning_control(self, variable: int, value: int, inplace=True) -> "BayesianNetwork":
        """Set the probability of a variable to be 1 for a specific value and 0 for all others.

        This removes any incoming causal links to this variables (keeping the marginal). Then, the
        marginal distribution is replaced by a distribution with all zeros except for one state,
        which is given by `value`.

        Args:
            variable (int): the variable whose state should be pinned (zero-indexed).
            value (int): The state that should have probability 1.

        Returns:
            BayesianNetwork: the modified BN. In case inplace=True then `self` is returned,
             which is mostly in keeping with the return type. Otherwise a copy of `self` is returned
             with the changed distribution.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        if inplace:
            self.marginalize(variable)  # this removes any incoming causal links to this variable (keeping the marginal)

            if isinstance(self.pdfs[variable], JointProbabilityMatrix):
                assert self.numvalues[variable] == self.pdfs[variable].numvalues, 'the number of values of the variable should be the same as the number of values in the JointProbabilityMatrix'

                self.pdfs[variable].joint_probabilities.joint_probabilities = ds.unit_vector(value, [self.numvalues[variable]]*self.pdfs[variable].numvariables, dtype=_type_prob)
            elif isinstance(self.pdfs[variable], "BayesianNetwork"):
                if self.pdfs[variable].numvariables == 1:
                    self.pdfs[variable].pdfs[0].pinning_control(0, value, inplace=True)  # recurse
                else:
                    raise NotImplementedError('not sure what to do here yet')
            else:
                raise NotImplementedError(f'not sure what to do with {type(self.pdfs[variable])=}')
            
            return self
        else:
            self_copy = copy.deepcopy(self)

            self_copy.pinning_control(variable, value, inplace=True)

            return self_copy
    

    def num_states(self) -> int:
        """Return the total number of possible joint states in this BN.

        To be consistent with JointProbabilityMatrix.

        Returns:
            int: _description_
        """
        return np.prod(self.numvalues)


    def entropy(self, variables: Sequence[int], base=2) -> float:
        if len(variables) > 0:
            # probs = [self.marginal_probability(variables, values) for values in self.statespace(variables)]
            # probs = np.array(probs, dtype=_type_prob)
            variables = sorted(set(variables))  # self.conditional_probabilities (called by marginal_pmf) does not work well with unsorted variables (also not duplicates)
            pmf = self.marginal_pmf(variables)  # this way potentially makes use of caching
            probs = list(pmf.joint_probabilities().values())
            # print(type(probs))
            return ss.entropy(probs, base=base)
        else:
            return 0.0  # ill-defined scenario, just return this


    def conditional_entropy(self, variables: Sequence[int], given_variables: Sequence[int], base=2,
                            verbose=0):
        # this turned out much faster than the approach in `self.conditional_entropy_explicit`
        return self.entropy(list(variables) + list(given_variables)) - self.entropy(given_variables)

    
    def conditional_entropy_explicit(self, variables: Sequence[int], given_variables: Sequence[int], base=2,
                            verbose=0):
        """The preferred function to use is conditional_entropy; this function is slower and might not be fully correct.
        """
        tot_ent = 0.0

        assert len(next(self.statespace(given_variables))) == len(given_variables)

        for given_values in self.statespace(given_variables):
            assert len(next(self.statespace(variables))) == len(variables)

            condprobs = [self.conditional_probability(variables, values, given_variables, given_values, verbose=verbose-1) 
                         for values in self.statespace(variables)]
            condprobs = np.array(condprobs, dtype=_type_prob)

            prob = self.marginal_probability(given_variables, given_values)

            if verbose > 0:
                print(f'conditional_entropy: {given_values=}:\t{condprobs=} (sum={sum(condprobs)}); prob(given_variables=given_values)={prob}')

            try:
                if not sum(condprobs) < _essential_zero_prob:
                    tot_ent += prob * ss.entropy(condprobs, base=base)
                else:
                    tot_ent += 0.0  # we define 0 log 0 = 0 as is customary (but ss.entropy does not...)
            except FloatingPointError as e:
                print(f'error: {prob=}, {condprobs=}')
                raise e
        
        return tot_ent


    def pvalue_mutual_information(self, variables_X: Sequence[int], variables_Y: Sequence[int], 
                                  num_samples: int, method='G') -> float:
        xsize = self.statespace(variables_X, only_size=True)
        ysize = self.statespace(variables_Y, only_size=True)
        crosstab = np.zeros((xsize, ysize), dtype=_type_prob)

        if method == 'G':
            for statex_ix, x in enumerate(self.statespace(variables_X)):
                for statey_ix, y in enumerate(self.statespace(variables_Y)):
                    # TODO: when `variables_X` and `variables_Y` are 'far away' from the root (so depend on a chain of variables),
                    # then this loop might become slow. Then it would probably be more efficient to first marginalize
                    # or something like that.
                    crosstab[statex_ix, statey_ix] = self.marginal_probability(list(variables_X) + list(variables_Y), x + y)
                
            _, pval, _, _ = ss.chi2_contingency(crosstab * num_samples, lambda_='log-likelihood')
        else:
            raise NotImplementedError(f'unrecognized {method=}')

        return pval


    def mutual_information(self, variables_X: Sequence[int], variables_Y: Sequence[int], base=2):
        ent = self.entropy(variables_X, base=base)
        condent = self.conditional_entropy(variables_X, variables_Y, base=base)
        mi = ent - condent

        assert mi >= -_mi_error_tol, f'{mi=} should be non-negative: {ent=}, {condent=}\njointprobs(X): TODO'  # TODO

        return mi
    

    def conditional_mutual_information(self, variables_X: Sequence[int], variables_Y: Sequence[int], 
                                       given_variables: Sequence[int], base=2):
        if len(given_variables) == 0:
            return self.mutual_information(variables_X, variables_Y, base=base)
        else:
            # # start 1:
            # for given_values in self.statespace(given_variables):
            #     prob_gv = self.marginal_probability()
            # pmf_gv = self.marginal_pmf(given_variables)
            # # start 2:
            # state_probs_gv = pmf_gv.joint_probabilities()

            # for given_values, prob_gv in state_probs_gv.items():
            #     mi = self.
            # # start 3:

            # NOTE: compute as H(X|Z) - H(X|Y,Z)
            H_X_given_Z = self.conditional_entropy(variables_X, given_variables, base=base)
            H_X_given_YZ = self.conditional_entropy(variables_X, list(variables_Y) + list(given_variables), base=base)

            return H_X_given_Z - H_X_given_YZ

    
    def wms(self, variables_X: Sequence[int], variables_Y: Sequence[int], base=2, normalized=False) -> float:
        total_mi = self.mutual_information(variables_X, variables_Y, base=base)
        indiv_mis = [self.mutual_information([xi], variables_Y, base=base) for xi in variables_X]

        assert min(indiv_mis) >= -_mi_error_tol, f'looking for bug: {indiv_mis=}'

        if not normalized:
            return total_mi - sum(indiv_mis)
        else:
            wms = total_mi - sum(indiv_mis)

            if wms >= 0.0:
                wms_norm = wms / total_mi
            else:
                # here we imagine the case where all X's are completely redundant, in which case `total_mi` will equal
                # the maximum of the individual MIs. Then, the extent to which WMS goes into the negative is that 
                # `total_mi` and this maximum indiv. MI cancel each other out in the subtraction above, and the other
                # indiv MIs make it negative.
                wms_norm = wms / (sum(indiv_mis) - max(indiv_mis))
            
            assert min(indiv_mis) >= 0.0, f'looking for bug: {indiv_mis=}'
            assert -1.0 <= wms_norm <= 1.0, f'{wms=}, {wms_norm=}, {indiv_mis=}, {total_mi=}'

            return wms_norm


    def total_correlation(self, variables_X: Sequence[int], base=2) -> float:
        indiv_ents = [self.entropy([xi], base=base) for xi in variables_X]
        total_ent = self.entropy(variables_X, base=base)

        return sum(indiv_ents) - total_ent
    

    def dual_total_correlation(self, variables_X: Sequence[int], base=2) -> float:
        total_ent = self.entropy(variables_X, base=base)
        cond_ents = [self.conditional_entropy([xi], [xj for xj in variables_X if xj != xi], base=base) for xi in variables_X]

        return total_ent - sum(cond_ents)
    

    def o_information(self, variables_X: Sequence[int], base=2) -> float:
        return self.total_correlation(variables_X, base=base) - self.dual_total_correlation(variables_X, base=base)


    def synergistic_information(self, variables_X: Sequence[int], variables_Y: Sequence[int], base=2, 
                                srvs_out=None, syninfos_out=None, max_evals=200, max_num_srvs=np.inf, rel_tol_srv=0.05,
                                return_also_pvalue=False, pvalue_method='G', pvalue_combine_method='fisher', pvalue_num_samples=None,
                                method_srv='full', method_correction='conservative', direct_srv_args={}, verbose=0) -> float:
        assert not method_srv == 'direct' or max_num_srvs == 1, f'the direct SRV calculation does not support multiple SRVs yet (which should be agnostic to each other)'

        if len(variables_X) < 2:
            warnings.warn(f'Cannot compute synergy of only {len(variables_X)} variables; must be at least 2. Will return 0.0.')

            return 0.0
        else:
            variables_X = sorted(variables_X)

        if return_also_pvalue:
            assert not pvalue_num_samples is None, 'if you want to compute p-values then you should specify `pvalue_num_samples`'
            assert pvalue_num_samples > 0

        # make a smaller pmf that only contains X and Y so that the optimization (appending SRVs) is faster
        smaller_pmf = self.marginal_pmf(list(variables_X) + list(variables_Y), verbose=max(0, verbose-1))

        smaller_X_ixs = range(len(variables_X))
        smaller_Y_ixs = range(len(variables_X), len(variables_X) + len(variables_Y))

        orig_len_smaller_pmf = len(smaller_pmf)

        syn_info = list()  # will store MI values with the SRVs
        if return_also_pvalue:
            pvals = list()
        H_Xi = sorted([smaller_pmf.entropy([Xix]) for Xix in variables_X], reverse=True)
        syninfo_upperbound = sum(H_Xi[:-1])  # rough (not necessarily tight) estimate of what the upper bound on syn. info is (assuming complete independence between the X variables)

        _debug_optres_list = list()

        # for-loop only serves to prevent infinite loop if SRV appending would always work for some reason
        for srv_ix in range(min(int((len(variables_X) * (len(variables_X) - 1)) / 2 + 1), max_num_srvs)):
            if __debug__:
                _bn_len_before_append_syn = len(smaller_pmf)

            if method_srv == 'full':
                optres = smaller_pmf.append_synergistic_variable(smaller_X_ixs, 
                                                                 agnostic_about=range(orig_len_smaller_pmf, orig_len_smaller_pmf + srv_ix), 
                                                                 max_evals=max_evals, rel_tol=rel_tol_srv)
                
                _debug_optres_list.append(optres)

                success = optres.is_completed
            elif method_srv == 'direct':
                direct_srv = smaller_pmf.append_direct_synergistic_variable(smaller_X_ixs, **direct_srv_args)

                success = (not direct_srv is None)
            else:
                raise NotImplementedError(f'unknown {method_srv=}')

            if not success:
                if verbose >= 1:
                    print(f'synergistic_information: appending an SRV failed (optimization not successful)')
                
                assert _bn_len_before_append_syn == len(smaller_pmf), 'the failed SRV should not have been appended (or I should remove it again here)'
                
                break  # done appending SRVs
            else:
                si = smaller_pmf.mutual_information([len(smaller_pmf) - 1], smaller_Y_ixs, base=base)

                if method_correction == 'conservative':
                    correction_mis = [smaller_pmf.mutual_information([Xi], smaller_Y_ixs, base=base) for Xi in smaller_X_ixs]

                    correction = sum(correction_mis)

                    # assert correction < si, f'wow this is really a poor SRV! sure it is not a bug somewhere? {correction_mis=}, {si=}, {smaller_Y_ixs=}, {smaller_X_ixs=}'
                elif method_correction == 'optimistic':
                    correction = 0.0
                else:
                    raise NotImplementedError(f'unknown {method_correction=}')

                syn_info.append(max(0., si - correction))

                # assert correction < si, f'wow this is really a poor SRV! sure it is not a bug somewhere?'

                if return_also_pvalue:
                    pval = self.pvalue_mutual_information([len(smaller_pmf) - 1], smaller_Y_ixs, method=pvalue_method, 
                                                          num_samples=pvalue_num_samples, base=base)
                    pvals.append(float(pval))

                if verbose >= 1:
                    print(f'synergistic_information: successfully appended an SRV (I(SRV_i:Y)={syn_info[-1]}' + \
                           (f', {pval=}' if return_also_pvalue else '') \
                            + ') aiming at a total of {syninfo_upperbound}).')

                if sum(syn_info) > syninfo_upperbound * (1.0 - rel_tol_srv):
                    if verbose >= 1:
                        print(f'synergistic_information: will not try to append another SRV because I already reached more than {1.0 - rel_tol_srv} fraction of the maximum possible synergistic information.')

                        break
        
        if __debug__ and method_srv == 'full':
            num_successful_opts = sum([optres.is_completed for optres in _debug_optres_list])

            assert len(syn_info) == num_successful_opts, 'missing synergistic information quantities?'
        
            if return_also_pvalue:
                assert len(syn_info) == len(pvals), 'each syn. info should have a pvalue associated to it'

        num_srvs = len(smaller_pmf) - orig_len_smaller_pmf

        if num_srvs == 0:
            syn_info = 0.0
            if return_also_pvalue:
                pval = 1.0
        elif len(syn_info) == 0:  # in this case there were SRVs found, but the optimization was not successful for any of them (maybe no convergence, or violating a tolerance)
            syn_info = 0.0
            if return_also_pvalue:
                pval = 1.0
        else:
            # store the SRVs as conditional PMFs so that they can be appended to any BN if the 
            # caller wants that
            srv_ixs = range(orig_len_smaller_pmf, orig_len_smaller_pmf + num_srvs)

            if not srvs_out is None:  # does the caller also want to receive the conditional PMFs for the SRVs?
                srvs = [smaller_pmf.conditional_probabilities([srv_ix], smaller_X_ixs, verbose=verbose-1)
                        for srv_ix in srv_ixs]
                
                srvs_out.extend(srvs)  # use a list mutation function to change the caller's list object
            
                if __debug__:
                    for cond_pmf in srvs:
                        assert cond_pmf.num_given_variables() == len(smaller_X_ixs), \
                            'I put this test because I was not sure if `conditional_probabilities` actually returns' + \
                            ' always a conditional PMF conditioned on all variables that I give as `given_variables`, ' + \
                            'or sometimes a subset in case there are variables in `given_variables` that the SRV does not' + \
                            ' depend on. So now, `conditional_probabilities` must change, e.g., have an option to return ' + \
                            'which variable indices it still depends on and which not.'
            
            if not syninfos_out is None:
                syninfos_out.extend(syn_info)
            
            if return_also_pvalue:
                pval = ss.combine_pvalues(pvals, method=pvalue_combine_method)

            assert min(syn_info) >= -_mi_error_tol, f'MI cannot be negative: {syn_info=}'
            
            syn_info = sum(syn_info)  # return amount of synergistic information
        
        if not return_also_pvalue:
            return syn_info
        else:
            return syn_info, pval

    
    def statespace(self, variables: Sequence[int] = None, flattened=True, only_size=False):
        """Return an iterator of all possible states of this PMF.

        The returned iterator will iterate over sequences of values in 'ascending order', e.g., like
        "[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]" for two variables with self.numvalues=[2,3].

        Args:
            variables (int, optional): If a list is given then it specifies the indices into self.pdf for which
            the statespace is returned. If a single integer is given, then it is interpreted as range(variables). Defaults to -1.
            flattened (bool, optional): Whether to return the state space in a flat array. Defaults to True.
            only_size (bool, optional): Whether to return the number of states, rather than the states themselves.

        Returns:
            _type_: iterator over tuples of states (one per variable) 
        """
        statespaces = []

        if variables is None:
            variables = range(len(self))
        
        assert hasattr(variables, '__contains__'), f'\'variables\' should be an integer or a list/set/tuple/array (something acting as a set), but I got: {variables=}'

        # TODO: I would like to simplify this function by not allowing a scalar argument anymore (variables).
        # First here an assertion to see if it is even used, and then change those cases.
        assert not np.isscalar(variables), 'refactor: testing if this functionality is used and where'

        if np.isscalar(variables):
            if variables == -1:
                variables = len(self)
            elif variables == 0:
                return tuple()  # return empty tuple
            else:
                assert variables <= len(self)

            start_ix_values = 0
            for ix, pdf in enumerate(self.pdfs):
                numvars_in_pdf = len(pdf)

                if start_ix_values + numvars_in_pdf <= variables:
                    statespaces.append(pdf.statespace())  # all variables of this pdf are needed
                elif start_ix_values < variables:
                    statespaces.append(pdf.statespace(variables - start_ix_values))  # only the first so many variables
                else:
                    assert False, 'strange, previous loop should have done a break or the loop not started'
            
                start_ix_values += numvars_in_pdf

                if start_ix_values >= variables:
                    if start_ix_values > variables:
                        warnings.warn('\'variables\' argument to statepace() cannot be reached by an integer number of the first N pdfs. Returning the statespace including additional variables that the last needed pdf has in addition.')

                    break
        else:
            if len(variables) > 0:
                assert np.all(np.less(variables, len(self))), 'should all be within range'

            # note: this dict is used to support the case where variables is not sorted, in which
            # case the below for-loop would append statespaces in the wrong order
            statespaces_dict = dict()  # from variable index to a statespace (iterator of allowed values)

            # note: variables is expected to be a list of indices into self.pdfs
            # note: this loop is so complicated because it tried to support having PMFs in self.pdfs
            # which consist of more than one variable, so then the indices in variables are actually
            # not directly indices into the list self.pdfs. (I wonder if this multi-PMF setting actually
            # is useful and will be ever used...)
            start_ix_values = 0
            for pdf in self.pdfs:
                numvars_in_pdf = len(pdf)

                # which variable indices of this pdf are asked for?
                relevant_variables_pdf = [vix for vix in range(numvars_in_pdf) if vix + start_ix_values in variables]

                # if any of the variable indices covered by this pdf is in 'variables' then
                # add (some of its) state space
                if len(relevant_variables_pdf) > 0:
                    for vix in relevant_variables_pdf:
                        statespaces_dict[vix + start_ix_values] = pdf.statespace([vix], only_size=only_size)
                    # statespaces.append(pdf.statespace(relevant_variables_pdf))

                start_ix_values += numvars_in_pdf
            
            for vix in variables:
                statespaces.append(statespaces_dict[vix])  # add statespaces in the order asked for

        if flattened:
            if only_size:
                return int(np.product(statespaces))
            else:
                return (tuple(flatten(s)) for s in itertools.product(*statespaces))  # a generator
        else:
            if only_size:
                return int(np.product(statespaces))
            else:
                return itertools.product(*statespaces)
    
    def __len__(self):
        return self.numvariables


### UNIT TESTING:


def test_append_and_marginalize_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf.append_variables(4)

    pdf_old = pdf.marginalize_distribution(list(range(3)))

    assert pdf_copy == pdf_old, 'adding and then removing variables should result in the same joint pdf'

    # old_params = pdf_copy.matrix2params_incremental()
    #
    # np.testing.assert_array_almost_equal(pdf.matrix2params_incremental()[:len(old_params)], old_params)


def test_reorder_test():
    pdf = JointProbabilityMatrix(4, 3)

    pdf_original = pdf.copy()

    pdf.reorder_variables([3,2,1,0])
    if __debug__:
        np.testing.assert_almost_equal(pdf.entropy(), pdf_original.entropy())
        assert pdf != pdf_original

    pdf.reorder_variables([3,2,1,0,0,1,2,3])
    if __debug__:
        assert len(pdf) == 2 * len(pdf_original)
        np.testing.assert_almost_equal(pdf.entropy(), pdf_original.entropy())
        # the first 4 and second 4 variables should be identical so MI should be maximum
        np.testing.assert_almost_equal(pdf.mutual_information([0,1,2,3], [4,5,6,7]), pdf_original.entropy())

        assert pdf.marginalize_distribution([0,1,2,3]) == pdf_original


def test_params2matrix_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf_copy.params2matrix(pdf.matrix2params())

    assert pdf_copy == pdf, 'computing parameter values from joint pdf and using those to construct a 2nd joint pdf ' \
                            'should result in two equal pdfs.'


def test_vector2matrix_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf_copy.vector2matrix(pdf.matrix2vector())

    assert pdf_copy == pdf, 'computing vector from joint pdf and using that to construct a 2nd joint pdf ' \
                            'should result in two equal pdfs.'


def test_conditional_pdf_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_marginal_1 = pdf.marginalize_distribution([1])

    assert pdf_marginal_1.numvariables == 1

    pdf_cond_23_given_0 = pdf.conditional_probability_distribution([1], [0])
    pdf_cond_23_given_1 = pdf.conditional_probability_distribution([1], [1])

    assert pdf_cond_23_given_0.numvariables == 2
    assert pdf_cond_23_given_1.numvariables == 2

    prob_000_joint = pdf([0,0,0])
    prob_000_cond = pdf_marginal_1([0]) * pdf_cond_23_given_0([0,0])

    np.testing.assert_almost_equal(prob_000_cond, prob_000_joint)

    pdf_conds_23 = pdf.conditional_probability_distributions([1])

    assert pdf_conds_23[(0,)] == pdf_cond_23_given_0
    assert pdf_conds_23[(1,)] == pdf_cond_23_given_1


def test_append_using_transitions_table_and_marginalize_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    lists_of_possible_given_values = [list(range(pdf.numvalues)) for _ in range(pdf.numvariables)]

    state_transitions = [list(existing_vars_values) + list([int(np.mod(np.sum(existing_vars_values), pdf.numvalues))])
                         for existing_vars_values in itertools.product(*lists_of_possible_given_values)]

    pdf.append_variables_using_state_transitions_table(state_transitions)

    assert not hasattr(state_transitions, '__call__'), 'append_variables_using_state_transitions_table should not ' \
                                                       'replace the caller\'s variables'

    assert pdf.numvariables == 4, 'one variable should be added'

    pdf_old = pdf.marginalize_distribution(list(range(pdf_copy.numvariables)))

    assert pdf_copy == pdf_old, 'adding and then removing variables should result in the same joint pdf'
    
    pdf_copy.append_variables_using_state_transitions_table(
        state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))])

    assert pdf_copy == pdf, 'should be two equivalent ways of appending a deterministic variable'


def test_synergistic_variables_test_with_subjects_and_agnostics(num_subject_vars=2):
    num_agnostic_vars = 1

    pdf = JointProbabilityMatrix(num_subject_vars + num_agnostic_vars, 2)

    pdf_orig = pdf.copy()

    assert num_subject_vars > 0

    subject_variables = np.random.choice(list(range(len(pdf))), num_subject_vars, replace=False)
    agnostic_variables = np.setdiff1d(list(range(len(pdf))), subject_variables)

    pdf.append_synergistic_variables(1, subject_variables=subject_variables)

    assert pdf.marginalize_distribution(list(range(len(pdf_orig)))) == pdf_orig, \
        'appending synergistic variables changed the pdf'
    
    synergistic_variables = list(range(len(pdf_orig), len(pdf)))
    
    tol_rel_err = 0.2

    # ideally, the former is max and the latter is zero
    assert pdf.mutual_information(subject_variables, synergistic_variables) \
           > sum([pdf.mutual_information([sv], synergistic_variables) for sv in subject_variables]) \
           or pdf.entropy(synergistic_variables) < 0.01
    
    assert pdf.mutual_information(synergistic_variables, agnostic_variables) / pdf.entropy(synergistic_variables) \
           < tol_rel_err


def test_synergistic_variables_test_with_subjects(num_subject_vars=2):
    num_other_vars = 1

    pdf = JointProbabilityMatrix(num_subject_vars + num_other_vars, 2)

    pdf_orig = pdf.copy()

    assert num_subject_vars > 0

    subject_variables = np.random.choice(list(range(len(pdf))), num_subject_vars, replace=False)

    pdf.append_synergistic_variables(1, subject_variables=subject_variables)

    assert pdf.marginalize_distribution(list(range(len(pdf_orig)))) == pdf_orig, \
        'appending synergistic variables changed the pdf'

    # ideally, the former is max and the latter is zero
    assert pdf.mutual_information(subject_variables, list(range(len(pdf_orig), len(pdf)))) \
           > sum([pdf.mutual_information([sv], list(range(len(pdf_orig), len(pdf)))) for sv in subject_variables])


def test_dpi_and_ordering_bn_markov_chain(numvalues=3):
    bn2 = BayesianNetwork(0)

    bn2.append_independent_variable('uniform', numvalues)
    bn2.append_dependent_variable([len(bn2)-1], numvalues, 0.2)
    bn2.append_dependent_variable([len(bn2)-1], numvalues, 0.2)
    bn2.append_dependent_variable([len(bn2)-1], numvalues, 0.2)

    assert bn2.mutual_information([0], [1]) >= bn2.mutual_information([0], [2]), 'dpi violated'
    assert bn2.mutual_information([0], [2]) >= bn2.mutual_information([0], [3]), 'dpi violated'

    assert bn2.mutual_information([0], [2]) <= bn2.mutual_information([0, 1], [2]), 'partial ordering violated'
    assert bn2.mutual_information([0], [3]) <= bn2.mutual_information([0, 1], [3]), 'partial ordering violated'
    assert bn2.mutual_information([0, 1], [3]) <= bn2.mutual_information([0, 1, 2], [3]), 'partial ordering violated'

    assert len(bn2.all_upstream_nodes([0, 1])) == 0
    assert len(bn2.all_upstream_nodes([0, 1], [3])) == 0
    assert len(bn2.all_upstream_nodes([3], [1])) == 1


# todo: make another such function but now use the subject_variables option of append_synerg*
def test_synergistic_variables_test(numvars=2):
    pdf = JointProbabilityMatrix(numvars, 2)

    pdf_syn = pdf.copy()

    assert pdf_syn == pdf

    initial_guess_summed_modulo = False

    pdf_syn.append_synergistic_variables(1, initial_guess_summed_modulo=initial_guess_summed_modulo, verbose=False)

    assert pdf_syn.numvariables == pdf.numvariables + 1

    pdf_old = pdf_syn.marginalize_distribution(list(range(pdf.numvariables)))

    # trying to figure out why I hit the assertion "pdf == pdf_old"
    np.testing.assert_almost_equal(pdf_old.joint_probabilities.sum(), 1.0), 'all probabilities should sum to 1.0'
    np.testing.assert_almost_equal(pdf.joint_probabilities.sum(), 1.0), 'all probabilities should sum to 1.0'

    np.testing.assert_array_almost_equal(pdf.joint_probabilities, pdf_old.joint_probabilities)
    assert pdf == pdf_old, 'adding and then removing variables should result in the same joint pdf'

    parameters_before = pdf.matrix2params_incremental()

    pdf_add_random = pdf.copy()
    pdf_add_random.append_variables(1)

    np.testing.assert_array_almost_equal(pdf_add_random.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)
    np.testing.assert_array_almost_equal(pdf_syn.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)

    # note: this assert is in principle probabilistic, because who knows what optimization procedure is used and
    # how much it potentially sucks. So see if you hit this more than once, if you hit it at all.
    assert pdf_add_random.synergistic_information_naive(list(range(pdf.numvariables, pdf_add_random.numvariables)),
                                                                list(range(pdf.numvariables))) <= \
           pdf_syn.synergistic_information_naive(list(range(pdf.numvariables, pdf_add_random.numvariables)),
                                                         list(range(pdf.numvariables))), 'surely the optimization procedure' \
                                                                                   ' in append_synergistic_variables ' \
                                                                                   'yields a better syn. info. than ' \
                                                                                   'an appended variable with simply ' \
                                                                                   'random interaction parameters?!'

    np.testing.assert_array_almost_equal(pdf_add_random.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)
    np.testing.assert_array_almost_equal(pdf_syn.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)

    syninfo = pdf_syn.synergistic_information_naive(list(range(pdf.numvariables, pdf_add_random.numvariables)),
                                                    list(range(pdf.numvariables)))

    condents = [pdf.conditional_entropy([varix]) for varix in range(len(pdf))]

    assert syninfo <= min(condents), 'this is a derived maximum in Quax 2015, synergy paper, right?'


def test_orthogonalization_test_null_hypothesis(num_subject_vars=2, num_ortho_vars=1, num_para_vars=1, numvals=2,
                                               verbose=True, num_repeats=5, tol_rel_error=0.05):
    """
    This is similar in spirit to run_orthogonalization_test, except now the variables to orthogonalize (X) is already
    known to be completely decomposable into two parts (X1, X2) which are orthogonal and parallel, resp. (See also
    the description of append_orthogonalized_variables().) So in this case I am sure that
    append_orthogonalized_variables should find very good solutions.
    :param tol_rel_error: A float in [0, 1) and preferably close to 0.0.
    :param num_subject_vars:
    :param num_ortho_vars:
    :param num_para_vars:
    :param numvals:
    :param verbose:
    :param num_repeats:
    """
    pdf = JointProbabilityMatrix(num_subject_vars, numvals)

    # note: I add the 'null hypothesis' parallel and orthogonal parts in reversed order compared to what
    # append_orthogonalized_variables returns, otherwise I have to insert a reordering or implement
    # append_redundant_variables for subsets of variables (not too hard but anyway)

    pdf.append_redundant_variables(num_para_vars)

    # the just added variables should be completely redundant
    np.testing.assert_almost_equal(pdf.mutual_information(list(range(num_subject_vars)),
                                                          list(range(num_subject_vars, num_subject_vars + num_para_vars))),
                                   pdf.entropy(list(range(num_subject_vars, num_subject_vars + num_para_vars))))

    pdf.append_independent_variables(JointProbabilityMatrix(num_ortho_vars, pdf.numvalues))

    # the just added variables should be completely independent of all previous
    np.testing.assert_almost_equal(pdf.mutual_information(list(range(num_subject_vars, num_subject_vars + num_para_vars)),
                                                          list(range(num_subject_vars + num_para_vars,
                                                                num_subject_vars + num_para_vars + num_ortho_vars))),
                                   0.0)

    vars_to_orthogonalize = list(range(num_subject_vars, num_subject_vars + num_para_vars + num_ortho_vars))

    pdf.append_orthogonalized_variables(vars_to_orthogonalize, num_ortho_vars, num_para_vars,
                                        num_repeats=num_repeats)

    assert len(pdf) == num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars + num_para_vars

    result_ortho_vars = list(range(num_subject_vars + num_ortho_vars + num_para_vars,
                              num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars))
    result_para_vars = list(range(num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars,
                             num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars + num_para_vars))
    subject_vars = list(range(num_subject_vars))

    if pdf.entropy(result_ortho_vars) != 0.0:
        try:
            assert pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars) \
                   <= tol_rel_error
        except AssertionError as e:

            print('debug: pdf.mutual_information(subject_vars, result_ortho_vars) =', \
                pdf.mutual_information(subject_vars, result_ortho_vars))
            print('debug: pdf.entropy(result_ortho_vars) =', pdf.entropy(result_ortho_vars))
            print('debug: pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars) =', \
                pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars))
            print('debug: (ideal of previous quantity: 0.0)')
            print('debug: tol_rel_error =', tol_rel_error)

            raise AssertionError(e)

    if pdf.entropy(result_para_vars) != 0.0:
        assert pdf.mutual_information(subject_vars, result_para_vars) \
               / pdf.entropy(result_para_vars) >= 1.0 - tol_rel_error

    if pdf.entropy(result_ortho_vars) != 0.0:
        assert pdf.mutual_information(result_para_vars, result_ortho_vars) \
               / pdf.entropy(result_ortho_vars) <= tol_rel_error

    if pdf.entropy(result_para_vars) != 0.0:
        assert pdf.mutual_information(result_para_vars, result_ortho_vars) \
               / pdf.entropy(result_para_vars) <= tol_rel_error

    if pdf.entropy(vars_to_orthogonalize) != 0.0:
        try:
            assert pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars)) \
                   / pdf.entropy(vars_to_orthogonalize) >= 1.0 - tol_rel_error, \
                'not all entropy of X is accounted for in {X1, X2}, which is of course the purpose of decomposition.'
        except AssertionError as e:
            print('debug: pdf.mutual_information(vars_to_orthogonalize, ' \
                  'list(result_para_vars) + list(result_ortho_vars)) =', \
                pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars)))
            print('debug: pdf.entropy(vars_to_orthogonalize) =', pdf.entropy(vars_to_orthogonalize))
            print('debug: pdf.mutual_information(vars_to_orthogonalize, ' \
                  'list(result_para_vars) + list(result_ortho_vars))' \
                   '/ pdf.entropy(vars_to_orthogonalize) =', \
                pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars)) \
                   / pdf.entropy(vars_to_orthogonalize))
            print('debug: tol_rel_error =', tol_rel_error)


def test_orthogonalization_test(num_subject_vars=2, num_orthogonalized_vars=1, numvals=3, verbose=True, num_repeats=1):
    """

    :param num_subject_vars:
    :param num_orthogonalized_vars:
    :param numvals:
    :param verbose:
    :param num_repeats:
    :raise AssertionError:
    """

    # note: in the code I assume that the number of added orthogonal variables equals num_orthogonalized_vars
    # note: in the code I assume that the number of added parallel variables equals num_orthogonalized_vars

    # todo: I am checking if it should be generally expected at all that any given X for any (Y, X) can be decomposed
    # into X1, X2 (orthogonal and parallel, resp.). Maybe in some cases it is not possible, due to the discrete nature
    # of the variables.

    pdf = JointProbabilityMatrix(num_subject_vars + num_orthogonalized_vars, numvals)

    pdf_original = pdf.copy()

    subject_vars = list(range(num_subject_vars))
    # these are the variables that will be 'orthogonalized', i.e., naming this X then the below appending procedure
    # will find two variable sets {X1,X2}=X where X1 is 'orthogonal' to original_vars (MI=0) and X2 is 'parallel'
    vars_to_orthogonalize = list(range(num_subject_vars, num_subject_vars + num_orthogonalized_vars))

    pdf.append_orthogonalized_variables(vars_to_orthogonalize, num_orthogonalized_vars, num_orthogonalized_vars,
                                        num_repeats=num_repeats)

    if verbose > 0:
        print('debug: computed first orthogonalization')

    assert len(pdf) == len(pdf_original) + num_orthogonalized_vars * 2

    # note: implicitly, the number of variables added for the orthogonal part is <num_orthogonalized_vars> here
    ortho_vars = list(range(num_subject_vars + num_orthogonalized_vars,
                       num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars))
    # note: implicitly, the number of variables added for the parallel part is <num_orthogonalized_vars> here
    para_vars = list(range(num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars,
                      num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars + num_orthogonalized_vars))

    assert para_vars[-1] == len(pdf) - 1, 'not all variables accounted for?'

    '''
    these should be high:
    pdf.mutual_information(original_vars, para_vars)
    pdf.mutual_information(vars_to_orthogonalize, ortho_vars)

    these should be low:
    pdf.mutual_information(original_vars, ortho_vars)
    pdf.mutual_information(para_vars, ortho_vars)
    '''

    tol_rel_err = 0.2  # used for all kinds of tests

    # some test of ordering.
    # Here: more % of the parallel variables' entropy should be correlated with subject_variables
    # than the same % for the orthogonal variables. This should be a pretty weak requirement.
    assert pdf.mutual_information(subject_vars, para_vars) / pdf.entropy(para_vars) \
           > pdf.mutual_information(subject_vars, ortho_vars) / pdf.entropy(ortho_vars), \
        '1: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    assert pdf.mutual_information(vars_to_orthogonalize, ortho_vars) > pdf.mutual_information(para_vars, ortho_vars), \
        '2: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    assert pdf.mutual_information(vars_to_orthogonalize, para_vars) > pdf.mutual_information(para_vars, ortho_vars), \
        '3: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    # at least more than <tol*>% of the entropy of para_vars is not 'wrong' (correlated with ortho_vars)
    assert pdf.mutual_information(para_vars, ortho_vars) < tol_rel_err * pdf.entropy(para_vars)
    # at least more than <tol*>% of the entropy of ortho_vars is not 'wrong' (correlated with para_vars)
    assert pdf.mutual_information(para_vars, ortho_vars) < tol_rel_err * pdf.entropy(ortho_vars)

    # todo: print some numbers to get a feeling of what (verbose) kind of accuracy I get, or let
    # append_orthogonalized_variables return those (e.g. in a dict?)?

    if verbose > 0:
        print('debug: computed first bunch of MI checks')

    ### test if the total entropy of ortho_vars + para_vars is close enough to the entropy of vars_to_orthogonalize

    # note: cannot directly use pdf.entropy(ortho_vars + para_vars) because I did not yet include a cost term
    # for the total entropy ('efficiency') of the ortho and para vars (so far entropy_cost_factor=0.0)
    entropy_ortho_and_para = pdf.mutual_information(vars_to_orthogonalize, ortho_vars + para_vars)
    entropy_vars_to_ortho = pdf.entropy(vars_to_orthogonalize)

    if verbose > 0:
        print('debug: computed few more entropy things')

    if entropy_vars_to_ortho != 0.0:
        assert abs(entropy_vars_to_ortho - entropy_ortho_and_para) / entropy_vars_to_ortho <= tol_rel_err, \
            'the total entropy of the ortho and para vars is too high/low: ' + str(entropy_ortho_and_para) + ' versus' \
            + ' entropy_vars_to_ortho=' + str(entropy_vars_to_ortho) + ' (rel. err. tol. = ' + str(tol_rel_err) + ')'

    ideal_para_entropy = pdf_original.mutual_information(vars_to_orthogonalize, subject_vars)

    assert pdf.entropy(para_vars) >= (1.0 - tol_rel_err) * ideal_para_entropy
    try:
        assert pdf.mutual_information(vars_to_orthogonalize, para_vars) >= (1.0 - tol_rel_err) * ideal_para_entropy
    except AssertionError as e:
        print('debug: pdf.mutual_information(vars_to_orthogonalize, para_vars) =', \
            pdf.mutual_information(vars_to_orthogonalize, para_vars))
        print('debug: pdf.mutual_information(subject_vars, para_vars) =', \
            pdf.mutual_information(subject_vars, para_vars))
        print('debug: ideal_para_entropy =', ideal_para_entropy)
        print('debug: tol_rel_err =', tol_rel_err)

        raise AssertionError(e)

    if verbose > 0:
        print('debug: ...and more...')

    ideal_ortho_entropy = pdf_original.conditional_entropy(vars_to_orthogonalize)

    # H(X1) should be close to H(X|Y), i.e., the entropy in X which does not share information with Y
    if not pdf.entropy(ortho_vars) == 0.0:
        assert abs(pdf.entropy(ortho_vars) - ideal_ortho_entropy) / pdf.entropy(ortho_vars) <= tol_rel_err

    # I(X1:X) should be (almost) equal to H(X1), i.e., all entropy of the orthogonal X1 should be from X, nowhere
    # else
    if not pdf.mutual_information(vars_to_orthogonalize, ortho_vars) == 0.0:
        assert abs(pdf.mutual_information(vars_to_orthogonalize, ortho_vars) - ideal_ortho_entropy) \
               / pdf.mutual_information(vars_to_orthogonalize, ortho_vars) <= tol_rel_err, \
        'pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    if verbose > 0:
        print('debug: done!')



def test_append_conditional_pdf_test():
    pdf_joint = JointProbabilityMatrix(4, 3)

    pdf_12 = pdf_joint.marginalize_distribution([0, 1])
    pdf_34_cond_12 = pdf_joint.conditional_probability_distributions([0, 1])

    pdf_merged = pdf_12.copy()
    pdf_merged.append_variables_using_conditional_distributions(pdf_34_cond_12)

    assert pdf_merged == pdf_joint

    ### add a single variable conditioned on only the second existing variable

    rand_partial_cond_pdf = {(val,): JointProbabilityMatrix(1, pdf_joint.numvalues)
                             for val in range(pdf_joint.numvalues)}

    pdf_orig = pdf_joint.copy()

    pdf_joint.append_variables_using_conditional_distributions(rand_partial_cond_pdf, [1])

    assert pdf_orig == pdf_joint.marginalize_distribution(list(range(len(pdf_orig))))

    assert pdf_joint.mutual_information([0], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'
    assert pdf_joint.mutual_information([1], [len(pdf_joint) - 1]) > 0.0, 'should not be 0, it is conditioned on'
    assert pdf_joint.mutual_information([2], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'
    assert pdf_joint.mutual_information([3], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'


def test_append_conditional_entropy_test():
    pdf_joint = JointProbabilityMatrix(4, 3)

    assert pdf_joint.conditional_entropy([1,2]) >= pdf_joint.conditional_entropy([1])
    assert pdf_joint.conditional_entropy([1,0]) >= pdf_joint.conditional_entropy([1])

    assert pdf_joint.conditional_entropy([0,2]) <= pdf_joint.entropy([0,2])

    assert pdf_joint.entropy([]) == 0, 'H(<empty-set>)=0 ... right? Yes think so'

    np.testing.assert_almost_equal(pdf_joint.conditional_entropy([1,2], [1,2]), 0.0)
    np.testing.assert_almost_equal(pdf_joint.entropy([0]) + pdf_joint.conditional_entropy([1,2,3], [0]),
                                   pdf_joint.entropy([0,1,2,3]))
    np.testing.assert_almost_equal(pdf_joint.conditional_entropy([0,1,2,3]), pdf_joint.entropy())


def test_marginalization_single_variable():
    numvalues = 3

    bn = BayesianNetwork()

    pdf0 = JointProbabilityMatrix(1, numvalues, 'uniform')
    bn.append_independent_variable(pdf0)

    cond_pdf = ConditionalProbabilityMatrix()
    cond_pdf.generate_random_conditional_pdf(1, 1, numvalues)
    bn.append_conditional_variable(cond_pdf, [0])

    cond_pdf2 = ConditionalProbabilityMatrix()
    cond_pdf2.generate_random_conditional_pdf(2, 1, numvalues)
    bn.append_conditional_variable(cond_pdf2, [0, 1])

    cond_pdf3 = ConditionalProbabilityMatrix()
    cond_pdf3.generate_random_conditional_pdf(1, 1, numvalues)
    bn.append_conditional_variable(cond_pdf3, [2])

    assert isinstance(bn.pdfs[2], ConditionalProbabilityMatrix)
    assert bn.dependency_graph.in_degree(2) == 2
    out_degree_before = bn.dependency_graph.out_degree(2)
    
    bn.marginalize(2)

    assert isinstance(bn.pdfs[2], JointProbabilityMatrix), 'should not be conditional anymore'
    assert bn.dependency_graph.in_degree(2) == 0, 'should no longer depend on any other variable'
    assert out_degree_before == bn.dependency_graph.out_degree(2), 'should be unchanged'


def test_params2matrix_incremental_test(numvars=3):
    pdf1 = JointProbabilityMatrix(numvars, 3)
    pdf2 = JointProbabilityMatrix(numvars, 3)

    params1 = pdf1.matrix2params_incremental(return_flattened=True)
    tree1 = pdf1.matrix2params_incremental(return_flattened=False)
    tree11 = pdf1.imbalanced_tree_from_scalars(params1, pdf1.numvalues)
    params1_from_tree1 = pdf1.scalars_up_to_level(tree1)
    params1_from_tree11 = pdf1.scalars_up_to_level(tree11)

    np.testing.assert_array_almost_equal(params1, params1_from_tree11)  # more a test of tree conversion itself
    np.testing.assert_array_almost_equal(params1, params1_from_tree1)

    pdf2.params2matrix_incremental(params1)

    params2 = pdf2.matrix2params_incremental()

    assert pdf1 == pdf2, 'computing parameter values from joint pdf and using those to construct a 2nd joint pdf ' \
                         'should result in two equal pdfs.\nparams1 = ' + str(params1) + '\nparms2 = ' + str(params2)

    pdf2.params2matrix_incremental(pdf2.matrix2params_incremental())

    assert pdf1 == pdf2, 'computing parameter values from joint pdf and using those to reconstruct the joint pdf ' \
                         'should result in two equal pdfs.'

    ### TEST the incrementality of the parameters

    pdf_marginal = pdf1.marginalize_distribution([0])
    params_marginal = pdf_marginal.matrix2params_incremental()
    np.testing.assert_array_almost_equal(params_marginal, pdf1.matrix2params_incremental()[:len(params_marginal)])

    pdf_marginal = pdf1.marginalize_distribution([0, 1])
    params_marginal = pdf_marginal.matrix2params_incremental()
    try:
        np.testing.assert_array_almost_equal(flatten(params_marginal),
                                             flatten(pdf1.matrix2params_incremental()[:len(params_marginal)]))
    except AssertionError as e:
        print('---------------------')
        print('debug: params_marginal =                 ', np.round(params_marginal, decimals=4))
        print('debug: pdf1.matrix2params_incremental() =', np.round(pdf1.matrix2params_incremental(), 4))
        print('---------------------')
        print('debug: params_marginal =                 ', \
            pdf_marginal.matrix2params_incremental(return_flattened=False))
        print('debug: pdf1.matrix2params_incremental() =', \
            pdf1.matrix2params_incremental(return_flattened=False))
        print('---------------------')

        raise AssertionError(e)

    if numvars >= 3:
        pdf_marginal = pdf1.marginalize_distribution([0, 1, 2])
        params_marginal = pdf_marginal.matrix2params_incremental()
        np.testing.assert_array_almost_equal(flatten(params_marginal),
                                             flatten(pdf1.matrix2params_incremental()[:len(params_marginal)]))


def test_scalars_to_tree_test():
    pdf = JointProbabilityMatrix(4, 3)

    list_of_scalars = pdf.matrix2params()  # does not matter what sequence of numbers, as long as the length is correct

    tree = pdf.imbalanced_tree_from_scalars(list_of_scalars, pdf.numvalues)
    np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), list_of_scalars)
    # np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), pdf.matrix2params_incremental())

    # another tree
    tree = pdf.matrix2params_incremental(return_flattened=False)

    # assert not np.isscalar(tree[-1]), 'you changed the matrix2params_incremental to return flat lists, which is good ' \
    #                                   'but then you should change this test, set an argument like flatten=False?'

    list_of_scalars2 = pdf.scalars_up_to_level(tree)

    tree2 = pdf.imbalanced_tree_from_scalars(list_of_scalars2, pdf.numvalues)

    np.testing.assert_array_almost_equal(flatten(tree), flatten(tree2))
    np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), pdf.scalars_up_to_level(tree2))


def test_generate_single_instance_bn():
    """Generate a single sample; train a BN on this single sample; and verify that then all samples are equal.
    """
    bn = BayesianNetwork()

    bn.infer_random_bn_on_dag(5, 2)

    sample = bn.generate_sample()

    bn_patient = BayesianNetwork()
    bn_patient.infer_bn_on_dag(pd.DataFrame([sample], columns=list(range(len(bn)))), bn.dependency_graph)

    all_same_samples = bn_patient.generate_samples(5)

    assert np.all([np.equal(same_sample, sample).all() for same_sample in all_same_samples]), 'all samples from a BN that was trained on only one sample should be the same and equal to that sample'

    bn_patient.infer_bn_on_dag([sample], bn.dependency_graph)  # second way of doing this

    bn_patient.generate_samples(5)

    assert np.all([np.equal(same_sample, sample).all() for same_sample in all_same_samples]), 'all samples from a BN that was trained on only one sample should be the same and equal to that sample'



def run_all_tests(verbose=False, all_inclusive=False):
    test_append_and_marginalize_test()
    if verbose > 0:
        print('note: test run_append_and_marginalize_test successful.')

    test_params2matrix_test()
    if verbose > 0:
        print('note: test run_params2matrix_test successful.')

    test_vector2matrix_test()
    if verbose > 0:
        print('note: test run_vector2matrix_test successful.')

    test_conditional_pdf_test()
    if verbose > 0:
        print('note: test run_conditional_pdf_test successful.')

    test_append_using_transitions_table_and_marginalize_test()
    if verbose > 0:
        print('note: test run_append_using_transitions_table_and_marginalize_test successful.')

    test_append_conditional_pdf_test()
    if verbose > 0:
        print('note: test run_append_conditional_pdf_test successful.')

    test_scalars_to_tree_test()
    if verbose > 0:
        print('note: test run_scalars_to_tree_test successful.')

    test_params2matrix_incremental_test()
    if verbose > 0:
        print('note: test run_params2matrix_incremental_test successful.')

    test_synergistic_variables_test()
    if verbose > 0:
        print('note: test run_synergistic_variables_test successful.')

    test_synergistic_variables_test_with_subjects()
    if verbose > 0:
        print('note: test run_synergistic_variables_test_with_subjects successful.')

    test_synergistic_variables_test_with_subjects_and_agnostics()
    if verbose > 0:
        print('note: test run_synergistic_variables_test_with_subjects_and_agnostics successful.')

    test_append_conditional_entropy_test()
    if verbose > 0:
        print('note: test run_append_conditional_entropy_test successful.')

    test_reorder_test()
    if verbose > 0:
        print('note: test run_reorder_test successful.')

    if all_inclusive:
        test_orthogonalization_test(verbose=verbose)
        if verbose > 0:
            print('note: test run_orthogonalization_test successful.')

    test_orthogonalization_test_null_hypothesis()
    if verbose > 0:
        print('note: test run_orthogonalization_test_null_hypothesis successful.')

    if verbose > 0:
        print('note: finished. all tests successful.')


# def sum_modulo(values, modulo):  # deprecated, can be removed?
#     """
#     An example function which can be passed as the state_transitions argument to the
#     append_variables_using_state_transitions_table function of JointProbabilityMatrix.
#     :rtype : int
#     :param values: list of values which should be in integer range [0, modulo)
#     :param modulo: value self.numvalues usually, e.g. 2 for binary values
#     :return: for binary variables it is the XOR, for others it is summed modulo of integers.
#     """
#     return int(np.mod(np.sum(values), modulo))


### START of testing functions which try to determine if the synergy variables implementation behaves as expected,
### and for which I would like to see the results e.g. to plot in the paper


def test_susceptibilities(num_vars_X, num_vars_Y, num_values, num_samples=50, synergistic=True):

    resp = TestSynergyInRandomPdfs()

    resp.susceptibility_new_local_list = []
    resp.susceptibility_new_global_list = []

    time_before = time.time()

    for sample in range(num_samples):
        if not synergistic:
            pdf = JointProbabilityMatrix(num_vars_X + num_vars_Y, num_values)
        else:
            pdf = JointProbabilityMatrix(num_vars_X, num_values)
            pdf.append_synergistic_variables(num_vars_Y)

        num_X1 = int(num_vars_X / 2)

        print('note: computing old susceptibilities... t=' + str(time.time() - time_before))

        resp.susceptibilities_local_list.append(pdf.susceptibilities_local(num_vars_Y))
        resp.susceptibility_global_list.append(pdf.susceptibility_non_local(list(range(num_vars_X, len(pdf))), list(range(num_X1)),
                                                                            list(range(num_X1, num_vars_X))))

        print('note: computing new susceptibilities... t=' + str(time.time() - time_before))

        resp.susceptibility_new_local_list.append(pdf.susceptibility(list(range(num_vars_X, len(pdf))), only_non_local=False))
        resp.susceptibility_new_global_list.append(pdf.susceptibility(list(range(num_vars_X, len(pdf))), only_non_local=True))

        resp.pdf_XY_list.append(pdf.copy())
        resp.total_mi_list.append(pdf.mutual_information(list(range(num_vars_X)), list(range(num_vars_X, len(pdf)))))
        indiv_mis = [pdf.mutual_information([xi], list(range(num_vars_X, len(pdf)))) for xi in range(num_vars_X)]
        resp.indiv_mi_list_list.append(indiv_mis)

        print('note: finished loop sample=' + str(sample+1) + ' (of ' + str(num_samples) + ') t=' \
              + str(time.time() - time_before))
        print('note: susceptibilities:', resp.susceptibilities_local_list[-1], ', ', resp.susceptibility_global_list, \
            ', ', resp.susceptibility_new_local_list[-1], ', ', resp.susceptibility_new_global_list[-1])

    return resp


# returned by test_upper_bound_single_srv_entropy()
class TestUpperBoundSingleSRVEntropyResult(object):

    def __init__(self):
        self.num_subject_variables = None
        self.num_synergistic_variables = None
        self.num_values = None
        self.theoretical_upper_bounds = []
        self.entropies_srv = []  # actually lower bounds (same as entropies_lowerbound_srv), namely I(X:SRV) - sum_i I(X_i:SRV)
        self.pdfs_with_srv = []
        self.rel_errors_srv = []  # sum_indiv_mis / total_mi
        self.entropies_lowerbound_srv = []  # I(X:SRV) - sum_i I(X_i:SRV)
        self.entropies_upperbound_srv = []  # I(X:SRV) - max_i I(X_i:SRV)


    # parameters used to produce the results
    num_subject_variables = None
    num_synergistic_variables = None
    num_values = None

    # list of floats, each i'th number should ideally be >= to the i'th element of entropies_srv. These are the
    # theoretically predicted upper bounds of any SRV from the section "Consequential properties" in the synergy
    # paper
    theoretical_upper_bounds = []

    # list of floats, each i'th number is the estimated maximum entropy of a single SRV
    entropies_srv = []  # actually lower bounds (same as entropies_lowerbound_srv), namely I(X:SRV) - sum_i I(X_i:SRV)
    pdfs_with_srv = []
    rel_errors_srv = []  # sum_indiv_mis / total_mi

    entropies_lowerbound_srv = []  # I(X:SRV) - sum_i I(X_i:SRV)
    entropies_upperbound_srv = []  # I(X:SRV) - max_i I(X_i:SRV)

    def __str__(self):
        if len(self.entropies_srv) > 0:
            return '[H(S(x))=' + str(np.mean(self.entropies_srv)) \
                   + '=' + str(np.mean(self.entropies_srv)/np.mean(self.theoretical_upper_bounds)) + 'H_syn(X), +/- ' \
                   + str(np.mean(self.rel_errors_srv)*100.0) + '%]'
        else:
            return '[H(S(x))=nan' \
                   + '=(nan)H_syn(X), +/- ' \
                   + 'nan' + '%]'


def test_upper_bound_single_srv_entropy(num_subject_variables=2, num_synergistic_variables=2, num_values=2,
                                        num_samples=10, tol_rel_err=0.05, verbose=True, num_repeats_per_sample=5):
    """
    Measure the entropy a single SRV and compare it to the theoretical upper bound derived in the synergy paper, along
    with the relative error of the synergy estimation.

    Note: instead of the entropy of the SRV I use the MI of the SRV with the subject variables, because
    the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
    the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
    (subject) variable.

    :param num_subject_variables: number of X_i variables to compute an SRV for
    :param num_synergistic_variables:
    :param num_values:
    :param tol_rel_err:
    :param num_samples: number of entropy values will be returned, one for each randomly generated pdf.
    :param verbose:
    :param num_repeats_per_sample: number of optimizations to perform for each sample, trying to get the best result.
    :return: object with a list of estimated SRV entropies and a list of theoretical upper bounds of this same entropy,
    each one for a different, randomly generated PDF.
    :rtype: TestUpperBoundSingleSRVEntropyResult
    """

    result = TestUpperBoundSingleSRVEntropyResult()  # object to hold results

    # parameters used to produce the results
    result.num_subject_variables = num_subject_variables
    result.num_synergistic_variables = num_synergistic_variables
    result.num_values = num_values

    # shorthands
    synergistic_variables = list(range(num_subject_variables, num_subject_variables + num_synergistic_variables))
    subject_variables = list(range(num_subject_variables))

    time_before = time.time()

    # generate samples
    for trial in range(num_samples):
        pdf = JointProbabilityMatrix(num_subject_variables, numvalues=num_values)

        theoretical_upper_bound = pdf.entropy() - max([pdf.entropy([si]) for si in subject_variables])

        pdf.append_synergistic_variables(num_synergistic_variables, num_repeats=num_repeats_per_sample,
                                         verbose=verbose)

        # prevent double computations:
        indiv_mis = [pdf.mutual_information(synergistic_variables, [si]) for si in subject_variables]
        sum_indiv_mis = sum(indiv_mis)
        total_mi = pdf.mutual_information(synergistic_variables, subject_variables)

        # an error measure of the optimization procedure of finding synergistic variables, potentially it can be quite
        # bad, this one is normalized in [0,1]
        rel_error_srv = (sum_indiv_mis - max(indiv_mis)) / total_mi

        if rel_error_srv <= tol_rel_err:
            # note: instead of the entropy o the SRV I use the MI of the SRV with the subject variables, because
            # the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
            # the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
            # (subject) variable.
            # note: subtracting the sum_indiv_mi is done because this part of the entropy of the SRV is not
            # synergistic, so it would overestimate the synergistic entropy. But it is also an estimate, only valid
            # in case the non-synergistic MIs I_indiv(syns : subs) indeed factorizes into the sum, occurring e.g.
            # if the individual subject variables are independent, or if the MI with each subject variable is
            # non-redundant with the MIs with other subject variables.
            # entropy_srv = pdf.entropy(synergistic_variables)

            result.entropies_lowerbound_srv.append(total_mi - sum_indiv_mis)
            result.entropies_upperbound_srv.append(total_mi - max(indiv_mis))
            # take the best estimate for synergistic entropy to be the middle-way between lb and ub
            entropy_srv = (result.entropies_upperbound_srv[-1] + result.entropies_lowerbound_srv[-1]) / 2.0
            result.theoretical_upper_bounds.append(theoretical_upper_bound)
            result.entropies_srv.append(entropy_srv)
            result.pdfs_with_srv.append(pdf)
            result.rel_errors_srv.append(rel_error_srv)

            if verbose > 0:
                print('note: added a new sample, entropy_srv =', entropy_srv, ' and theoretical_upper_bound =', \
                    theoretical_upper_bound, ', and rel_error_srv =', rel_error_srv)
        else:
            if verbose > 0:
                print('note: trial #' + str(trial) + ' will be discarded because the relative error of the SRV found' \
                      + ' is ' + str(rel_error_srv) + ' which exceeds ' + str(tol_rel_err) \
                      + ' (' + str(time.time() - time_before) + 's elapsed)')

    return result


class TestUpperBoundManySRVEntropyResult(object):

    # three-level dictionary, num_subject_variables_list --> num_synergistic_variables_list --> num_values_list,
    # and values are TestUpperBoundSingleSRVEntropyResult objects
    single_res_dic = dict()


    def plot_rel_err_and_frac_success(self, num_samples=1):

        figs = []

        for num_sub in self.single_res_dic.keys():
            for num_syn in self.single_res_dic[num_sub].keys():
                fig = plt.figure()

                num_vals_list = list(self.single_res_dic[num_sub][num_syn].keys())
                num_successes = [len(self.single_res_dic[num_sub][num_syn][n].rel_errors_srv) for n in num_vals_list]

                plt.boxplot([self.single_res_dic[num_sub][num_syn][n].rel_errors_srv for n in num_vals_list],
                            labels=list(map(str, num_vals_list)))
                # plt.ylim([0,0.15])
                # plt.xlim([0.5, 4.5])
                plt.xlabel('Number of values per variable')
                plt.ylabel('Relative error of H(S)')

                ax2 = plt.twinx()

                ax2.set_ylabel('Fraction of successful SRVs')
                ax2.plot((1,2,3,4), [s / num_samples for s in num_successes], '-o')
                ax2.set_ylim([0.80, 1.02])

                plt.show()

                figs.append(fig)

        return figs


    def plot_efficiency(self):
        """

        Warning: for num_subject_variables > 2 and num_synergistic_variables < num_subject_variables - 1 then
        the synergistic entropy does not fit in the SRV anyway, so efficiency of 1.0 would be impossible.
        :return: list of figure handles
        """
        figs = []

        for num_sub in self.single_res_dic.keys():
            for num_syn in self.single_res_dic[num_sub].keys():
                fig = plt.figure()

                num_vals_list = list(self.single_res_dic[num_sub][num_syn].keys())

                # list of lists
                effs = [np.divide(self.single_res_dic[num_sub][num_syn][n].entropies_srv,
                                  self.single_res_dic[num_sub][num_syn][n].theoretical_upper_bounds) for n in (2,3,4,5)]

                plt.boxplot(effs, labels=num_vals_list)
                plt.ylabel('H(S) / H_syn(X)')
                plt.xlabel('Number of values per variable')
                plt.show()

                figs.append(fig)

        return figs


def test_upper_bound_single_srv_entropy_many(num_subject_variables_list=list([2,3]),
                                             num_synergistic_variables_list=list([1,2]),
                                             num_values_list=list([2,3]),
                                             num_samples=10, tol_rel_err=0.05, verbose=True, num_repeats_per_sample=5):

    results_dict = dict()

    time_before = time.time()

    total_num_its = len(num_subject_variables_list) * len(num_synergistic_variables_list) * len(num_values_list)
    num_finished_its = 0

    for num_subject_variables in num_subject_variables_list:
        results_dict[num_subject_variables] = dict()

        for num_synergistic_variables in num_synergistic_variables_list:
            results_dict[num_subject_variables][num_synergistic_variables] = dict()

            for num_values in num_values_list:
                resobj = test_upper_bound_single_srv_entropy(num_subject_variables=num_subject_variables,
                                                             num_synergistic_variables=num_synergistic_variables,
                                                             num_values=num_values,
                                                             num_samples=num_samples,
                                                             tol_rel_err=tol_rel_err,
                                                             verbose=int(verbose)-1,
                                                             num_repeats_per_sample=num_repeats_per_sample)

                results_dict[num_subject_variables][num_synergistic_variables][num_values] = copy.deepcopy(resobj)

                num_finished_its += 1

                if verbose > 0:
                    print('note: finished', num_finished_its, '/', total_num_its, 'iterations after', \
                        time.time() - time_before, 'seconds. Result for (nx,ns,nv)=' \
                        + str((num_subject_variables, num_synergistic_variables, num_values)) + ': ', resobj)

                del resobj  # looking for some inadvertent use of pointer

    return results_dict


class TestSynergyInRandomPdfs(object):

    def __init__(self):
        self.syn_info_list = []
        self.total_mi_list = []
        self.indiv_mi_list_list = []  # list of list
        self.susceptibility_global_list = []
        self.susceptibilities_local_list = []  # list of list
        self.pdf_XY_list = []

    syn_info_list = []
    total_mi_list = []
    indiv_mi_list_list = []  # list of list
    susceptibility_global_list = []
    susceptibilities_local_list = []  # list of list
    pdf_XY_list = []


def test_synergy_in_random_pdfs(num_variables_X, num_variables_Y, num_values,
                                num_samples=10, tolerance_nonsyn_mi=0.05, verbose=True, minimize_method=None,
                                perturbation_size=0.1, num_repeats_per_srv_append=3):

    """


    :param minimize_method: the default is chosen if None, which is good, but not necessarily fast. One other, faster
    option I found was "SLSQP", but I have not quantified rigorously how much better/worse in terms of accuracy it is.
    :param num_variables_X:
    :param num_variables_Y:
    :param num_values:
    :param num_samples:
    :param tolerance_nonsyn_mi:
    :param verbose:
    :rtype: TestSynergyInRandomPdfs
    """

    result = TestSynergyInRandomPdfs()

    time_before = time.time()

    try:
        for i in range(num_samples):
            pdf = JointProbabilityMatrix(num_variables_X + num_variables_Y, num_values)

            result.pdf_XY_list.append(pdf)

            result.syn_info_list.append(pdf.synergistic_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                                    list(range(num_variables_X)),
                                                                    tol_nonsyn_mi_frac=tolerance_nonsyn_mi,
                                                                    verbose=bool(int(verbose)-1),
                                                                    minimize_method=minimize_method,
                                                                    num_repeats_per_srv_append=num_repeats_per_srv_append))

            result.total_mi_list.append(pdf.mutual_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                          list(range(num_variables_X))))

            indiv_mi_list = [pdf.mutual_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                          [xid]) for xid in range(num_variables_X)]

            result.indiv_mi_list_list.append(indiv_mi_list)

            result.susceptibility_global_list.append(pdf.susceptibility_global(num_variables_Y, ntrials=50,
                                                                               perturbation_size=perturbation_size))

            result.susceptibilities_local_list.append(pdf.susceptibilities_local(num_variables_Y, ntrials=50,
                                                                                 perturbation_size=perturbation_size))

            if verbose > 0:
                print('note: finished sample', i, 'of', num_samples, ', syn. info. =', result.syn_info_list[-1], 'after', \
                    time.time() - time_before, 'seconds')
    except KeyboardInterrupt as e:
        min_len = len(result.susceptibilities_local_list)  # last thing to append to in above loop, so min. length

        result.syn_info_list = result.syn_info_list[:min_len]
        result.total_mi_list = result.total_mi_list[:min_len]
        result.indiv_mi_list_list = result.indiv_mi_list_list[:min_len]
        result.susceptibility_global_list = result.susceptibility_global_list[:min_len]

        if verbose > 0:
            print('note: keyboard interrupt. Will stop the loop and return the', min_len, 'result I have so far.', \
                'Took me', time.time() - time_before, 'seconds.')

    return result


# returned by test_accuracy_orthogonalization()
class TestAccuracyOrthogonalizationResult(object):
    # parameters used to produce the results
    num_subject_variables = None
    num_orthogonalized_variables = None
    num_values = None

    # list of floats, each i'th number is the estimated entropy of the <num_subject_variables> variables
    actual_entropies_subject_variables = []
    actual_entropies_orthogonalized_variables = []
    actual_entropies_parallel_variables = []
    actual_entropies_orthogonal_variables = []

    # list of JointProbabilityMatrix objects
    joint_pdfs = []
    # list of floats, each i'th number is the relative error of the orthogonalization, in range 0..1
    rel_errors = []


def test_accuracy_orthogonalization(num_subject_variables=2, num_orthogonalized_variables=2, num_values=2,
                                        num_samples=10, verbose=True, num_repeats=5):
    """


    Note: instead of the entropy o the SRV I use the MI of the SRV with the subject variables, because
    the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
    the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
    (subject) variable.

    :param num_subject_variables:
    :param num_synergistic_variables:
    :param num_values:
    :param tol_rel_err:
    :return: object with a list of estimated SRV entropies and a list of theoretical upper bounds of this same entropy,
    each one for a different, randomly generated PDF.
    :rtype: TestUpperBoundSingleSRVEntropyResult
    """

    # todo: also include the option of known null hypothesis, to see what is the error if it is already known that 0.0
    # error is in fact possible

    # take these to be equal to num_orthogonalized_variables just to be sure
    num_orthogonal_variables = num_orthogonalized_variables
    num_parallel_variables = num_orthogonalized_variables

    result = TestAccuracyOrthogonalizationResult()  # object to hold results

    # parameters used to produce the results
    result.num_subject_variables = num_subject_variables
    result.num_synergistic_variables = num_orthogonalized_variables
    result.num_values = num_values

    # shorthands for variable index ranges for the different roles of variables
    orthogonalized_variables = list(range(num_subject_variables, num_subject_variables + num_orthogonalized_variables))
    subject_variables = list(range(num_subject_variables))
    orthogonal_variables = list(range(num_subject_variables + num_orthogonalized_variables,
                                 num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables))
    parallel_variables = list(range(num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables,
                               num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables
                               + num_parallel_variables))

    time_before = time.time()

    # generate samples
    for trial in range(num_samples):
        pdf = JointProbabilityMatrix(num_subject_variables + num_orthogonalized_variables, numvalues=num_values)

        pdf.append_orthogonalized_variables(orthogonalized_variables, num_orthogonal_variables, num_parallel_variables,
                                            verbose=verbose, num_repeats=num_repeats)

        assert len(pdf) == num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables \
                           + num_parallel_variables

        result.joint_pdfs.append(pdf)

        result.actual_entropies_subject_variables.append(pdf.entropy(subject_variables))
        result.actual_entropies_orthogonalized_variables.append(pdf.entropy(orthogonalized_variables))
        result.actual_entropies_orthogonal_variables.append(pdf.entropy(orthogonal_variables))
        result.actual_entropies_parallel_variables.append(pdf.entropy(parallel_variables))

        rel_error_1 = (pdf.mutual_information(orthogonalized_variables, subject_variables) - 0) \
                      / result.actual_entropies_subject_variables[-1]
        rel_error_2 = (pdf.mutual_information(orthogonalized_variables, subject_variables)
                      - pdf.mutual_information(parallel_variables, subject_variables)) \
                      / pdf.mutual_information(orthogonalized_variables, subject_variables)
        rel_error_3 = (result.actual_entropies_orthogonalized_variables[-1]
                      - pdf.mutual_information(orthogonal_variables + parallel_variables, orthogonalized_variables)) \
                      / result.actual_entropies_orthogonalized_variables[-1]

        # note: each relative error term is intended to be in range [0, 1]

        rel_error = rel_error_1 + rel_error_2 + rel_error_3

        # max_rel_error = result.actual_entropies_subject_variables[-1] \
        #                 + pdf.mutual_information(orthogonalized_variables, subject_variables) \
        #                 + result.actual_entropies_orthogonalized_variables[-1]
        max_rel_error = 3.0

        rel_error /= max_rel_error

        result.rel_errors.append(rel_error)

        print('note: finished trial #' + str(trial) + ' after', time.time() - time_before, 'seconds. rel_error=', \
            rel_error, '(' + str((rel_error_1, rel_error_2, rel_error_3)) + ')')

    return result


# todo: make a test function which iteratively appends a set of SRVs until no more can be found (some cut-off MI),
# then I am interested in the statistics of the number of SRVs that are found, as well as the statistics of whether
# the SRVs are statistically independent or not, and whether they have synergy about each other (if agnostic_about
# passed to append_syn* is a list of lists) and whether indeed every two SRVs necessarily will have info about an
# individual Y_i. In the end, if the assumption of perfect orthogonal decomposition of any RVs too strict, or is
# is often not needed at all? Because it would only be needed in case there are at least two SRVs, the combination of
# which (combined RV) has more synergy about Y than the two individually summed, but also the combined RV has
# info about individual Y_i so it is not an SRV by itself, so the synergy contains both synergistic and individual info.

def test_num_srvs(num_subject_variables=2, num_synergistic_variables=2, num_values=2, num_samples=10,
                  verbose=True, num_repeats=5):
    assert False


def test_impact_perturbation(num_variables_X=2, num_variables_Y=1, num_values=5,
                                num_samples=20, tolerance_nonsyn_mi=0.05, verbose=True, minimize_method=None,
                                perturbation_size=0.1, num_repeats_per_srv_append=3):

    """


    :param minimize_method: the default is chosen if None, which is good, but not necessarily fast. One other, faster
    option I found was "SLSQP", but I have not quantified rigorously how much better/worse in terms of accuracy it is.
    :param num_variables_X:
    :param num_variables_Y:
    :param num_values:
    :param num_samples:
    :param tolerance_nonsyn_mi:
    :param verbose:
    :rtype: TestSynergyInRandomPdfs
    """

    result = TestSynergyInRandomPdfs()

    time_before = time.time()

    try:
        for i in range(num_samples):
            pdf = JointProbabilityMatrix(num_variables_X, num_values)
            pdf.append_synergistic_variables(num_variables_Y, num_repeats=num_repeats_per_srv_append)

            pdf_Y_cond_X = pdf.conditional_probability_distributions(list(range(len(num_variables_X))))

            result.pdf_XY_list.append(pdf)

            result.syn_info_list.append(pdf.synergistic_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                                    list(range(num_variables_X)),
                                                                    tol_nonsyn_mi_frac=tolerance_nonsyn_mi,
                                                                    verbose=bool(int(verbose)-1),
                                                                    minimize_method=minimize_method,
                                                                    num_repeats_per_srv_append=num_repeats_per_srv_append))

            result.total_mi_list.append(pdf.mutual_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                          list(range(num_variables_X))))

            indiv_mi_list = [pdf.mutual_information(list(range(num_variables_X, num_variables_X + num_variables_Y)),
                                                          [xid]) for xid in range(num_variables_X)]

            result.indiv_mi_list_list.append(indiv_mi_list)

            result.susceptibility_global_list.append(pdf.susceptibility_global(num_variables_Y, ntrials=50,
                                                                               perturbation_size=perturbation_size))

            result.susceptibilities_local_list.append(pdf.susceptibilities_local(num_variables_Y, ntrials=50,
                                                                                 perturbation_size=perturbation_size))

            if verbose > 0:
                print('note: finished sample', i, 'of', num_samples, ', syn. info. =', result.syn_info_list[-1], 'after', \
                    time.time() - time_before, 'seconds')
    except KeyboardInterrupt as e:
        min_len = len(result.susceptibilities_local_list)  # last thing to append to in above loop, so min. length

        result.syn_info_list = result.syn_info_list[:min_len]
        result.total_mi_list = result.total_mi_list[:min_len]
        result.indiv_mi_list_list = result.indiv_mi_list_list[:min_len]
        result.susceptibility_global_list = result.susceptibility_global_list[:min_len]

        if verbose > 0:
            print('note: keyboard interrupt. Will stop the loop and return the', min_len, 'result I have so far.', \
                'Took me', time.time() - time_before, 'seconds.')

    return result

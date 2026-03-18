# from dit import Distribution
# from .info_metrics import mutual_information, mutual_information_joint

from metrics.synergy import pid_synergy_Imin


import numpy as np
from pgmpy.estimators import StructureScore


from math import log


class BICSynergy(StructureScore):
    def __init__(self, data, synergy_weight=600.0, pid=True, **kwargs):
        super(BICSynergy, self).__init__(data, **kwargs)
        self.synergy_weight = synergy_weight
        self.pid = pid

    def synergy_score(self, variable, parents):
        if len(parents) != 2:
            return 0  # Only compute synergy for triplets

        # print(variable, parents)
        # print(pid_synergy_Imin(source1=self.data[parents[0]], source2=self.data[parents[1]], 
        #                             target=self.data[variable]))
        # Extract relevant data and remove missing values
        if self.pid:
            return pid_synergy_Imin(source1=self.data[parents[0]], source2=self.data[parents[1]],
                                    target=self.data[variable])
        else:
            return mutual_information_joint(
                self.data[parents[0]], self.data[parents[1]], self.data[variable]
            )



    def local_score(self, variable, parents):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents]) if parents else 1

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)


        synergy_bonus = self.synergy_score(variable, parents)
        if synergy_bonus is None:
            synergy_bonus = 0
        return score + synergy_bonus * self.synergy_weight
# ea_matrix.py

from __future__ import annotations

import random
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

import multiprocessing as mp

from pgmpy.estimators import HillClimbSearch
from pgmpy.base import DAG as _pgmpy_DAG

from joblib import Parallel, delayed

# your custom score
from scoring.bicSynergy import BICSynergy

try:
    from pgmpy.estimators import (
        BIC, K2, BDeu, BDs, AIC,
        LogLikelihoodGauss, BICGauss, AICGauss,
        LogLikelihoodCondGauss, BICCondGauss, AICCondGauss,
    )
except ImportError:
    BIC = K2 = BDeu = BDs = AIC = None  # type: ignore
    LogLikelihoodGauss = BICGauss = AICGauss = None  # type: ignore
    LogLikelihoodCondGauss = BICCondGauss = AICCondGauss = None  # type: ignore
    _pgmpy_DAG = None  # type: ignore

def score_A_worker(args):
        scorer, A = args
        return scorer(A)

class GeneticBNSearchMatrix:
    """Same GA, but individuals are adjacency matrices (uint8)."""

    _PGMPY_SCORES = {
        "k2": K2,
        "bic": BIC,
        "aic": AIC,
        "bdeu": BDeu,
        "bds": BDs,
        "ll-g": LogLikelihoodGauss,
        "aic-g": AICGauss,
        "bic-g": BICGauss,
        "ll-cg": LogLikelihoodCondGauss,
        "aic-cg": AICCondGauss,
        "bic-cg": BICCondGauss,
        "bic-synergy": BICSynergy,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        population_size: int = 30,
        generations: int | None = 100,
        crossover_proportion: float = 0.2,
        crossover_edges: int = 2,
        mutation_prob: float = 1.0,
        elitism_proportion: float = 0.2,
        connect_disconnected: bool = False,
        max_parents: int | 2 = 2,
        score_fn: Union[str, Callable[[nx.DiGraph, pd.DataFrame], float]] = "bic",
        n_jobs: int = 1,
        random_state: int | None = None,
        hillclimb_seed: bool = False,
        show_progress: bool = False,
        crossover_method: str = "edge_swap",
        *,
        synergy_triplets: Sequence[Tuple[str, str, str]] | None = None,
        informed_ratio: float = 0.8,
        final_greedy: bool = False,
    ) -> None:
        self.data = data.copy()
        self.variables: List[str] = list(self.data.columns)
        self.m = len(self.variables)
        self.var2i = {v: i for i, v in enumerate(self.variables)}

        self.s = population_size
        self.J = generations
        self.p_C = crossover_proportion
        self.c_N = crossover_edges
        self.p_M = mutation_prob
        self.p_E = elitism_proportion
        self.k_tourn = 5
        self.max_parents = max_parents
        self.n_jobs = n_jobs
        self.hillclimb_seed = hillclimb_seed
        self.connect_disconnected = connect_disconnected
        self.show_progress = show_progress
        self.crossover_method = crossover_method
        self.final_greedy = final_greedy

        self.synergy_triplets = list(synergy_triplets) if synergy_triplets else []
        self.informed_ratio = float(np.clip(informed_ratio, 0.0, 1.0))

        self._rng = random.Random(random_state)
        self._np_rng = np.random.default_rng(random_state)

        if isinstance(score_fn, str):
            cls = self._PGMPY_SCORES.get(score_fn.lower())
            if cls is None:
                raise ValueError(f"Unknown score_fn '{score_fn}'.")
            self.scorer = cls(data=self.data)  # type: ignore
        elif callable(score_fn):
            self.scorer = score_fn  # type: ignore
        else:
            raise TypeError("score_fn must be a string or callable")

        self.population: List[np.ndarray] = []   # list of (m,m) uint8 adjacency matrices
        self.fitness: List[float] = []
        self.best_A: np.ndarray | None = None
        self.best_score: float = -np.inf
        self.bic_curve: list[float] = []

    # -------------------- conversions --------------------
    def _A_to_nx(self, A: np.ndarray) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.variables)
        ii, jj = np.where(A != 0)
        g.add_edges_from((self.variables[i], self.variables[j]) for i, j in zip(ii.tolist(), jj.tolist()))
        return g

    def _nx_to_A(self, g: nx.DiGraph) -> np.ndarray:
        A = np.zeros((self.m, self.m), dtype=np.uint8)
        for u, v in g.edges():
            A[self.var2i[u], self.var2i[v]] = 1
        np.fill_diagonal(A, 0)
        return A

    # -------------------- seeding --------------------
    def _hillclimb_seed_A(self) -> np.ndarray:
        hc = HillClimbSearch(self.data)
        estimated = hc.estimate(scoring_method="bic-d", show_progress=False)
        return self._nx_to_A(nx.DiGraph(estimated.edges()))

    def _initialize_population(self) -> None:
        n_inf = int(self.informed_ratio * self.s)

        mats: List[np.ndarray] = []
        if self.hillclimb_seed and n_inf > 0:
            synergy_n = n_inf // 2
            hill_n = n_inf - synergy_n
            for _ in range(hill_n):
                try:
                    mats.append(self._hillclimb_seed_A())
                except Exception:
                    mats.append(self._random_dag_A())
            if self.synergy_triplets:
                mats.extend(self._informed_dag_A() for _ in range(synergy_n))
        else:
            if self.synergy_triplets:
                mats.extend(self._informed_dag_A() for _ in range(n_inf))

        while len(mats) < self.s:
            mats.append(self._random_dag_A())

        self._rng.shuffle(mats)
        self.population = mats[: self.s]

    def _informed_dag_A(self) -> np.ndarray:
        A = np.zeros((self.m, self.m), dtype=np.uint8)
        # keep your “(a,b)->c” edges
        for a, b, c in self.synergy_triplets:
            if a in self.var2i and b in self.var2i and c in self.var2i:
                A[self.var2i[a], self.var2i[c]] = 1
                A[self.var2i[b], self.var2i[c]] = 1
        return self._repair_A(A)

    def _random_dag_A(self) -> np.ndarray:
        A = np.zeros((self.m, self.m), dtype=np.uint8)
        order = self._np_rng.permutation(self.m)
        edge_p = min(2.0 / self.m, 0.3)
        # add edges forward in order (acyclic by construction)
        for ii in range(self.m):
            u = order[ii]
            # sample some v's ahead
            for jj in range(ii + 1, self.m):
                if self._rng.random() < edge_p:
                    v = order[jj]
                    A[u, v] = 1
        return self._repair_A(A)

    # -------------------- fast cycle handling --------------------
    def _find_back_edge(self, A: np.ndarray) -> tuple[int, int] | None:
        """Returns (u,v) where u->v is a back-edge found by DFS, else None."""
        n = A.shape[0]
        color = np.zeros(n, dtype=np.int8)  # 0=unseen, 1=visiting, 2=done

        for start in range(n):
            if color[start] != 0:
                continue
            stack: list[int] = [start]
            it_stack: list[object] = [iter(np.flatnonzero(A[start]))]
            color[start] = 1

            while stack:
                u = stack[-1]
                it = it_stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    color[u] = 2
                    stack.pop()
                    it_stack.pop()
                    continue

                if color[v] == 0:
                    color[v] = 1
                    stack.append(int(v))
                    it_stack.append(iter(np.flatnonzero(A[v])))
                elif color[v] == 1:
                    return int(u), int(v)

        return None

    def _enforce_max_parents(self, A: np.ndarray) -> None:
        if self.max_parents is None:
            return
        for j in range(self.m):
            parents = np.flatnonzero(A[:, j])
            if parents.size > self.max_parents:
                drop = self._np_rng.choice(parents, size=(parents.size - self.max_parents), replace=False)
                A[drop, j] = 0

    def _weak_components(self, U: np.ndarray) -> list[np.ndarray]:
        n = U.shape[0]
        seen = np.zeros(n, dtype=bool)
        comps: list[np.ndarray] = []
        for s in range(n):
            if seen[s]:
                continue
            q = [s]
            seen[s] = True
            comp = [s]
            while q:
                u = q.pop()
                nbrs = np.flatnonzero(U[u])
                for v in nbrs.tolist():
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
                        comp.append(v)
            comps.append(np.array(comp, dtype=int))
        return comps

    def _connect_components(self, A: np.ndarray) -> None:
        U = ((A | A.T) != 0).astype(np.uint8)
        comps = self._weak_components(U)
        if len(comps) <= 1:
            return

        reps = [int(c[0]) for c in comps]
        for i in range(len(reps) - 1):
            u = reps[i]
            v = reps[i + 1]

            # try random direction, but don’t introduce a cycle
            if self._rng.random() < 0.5:
                cand = [(u, v), (v, u)]
            else:
                cand = [(v, u), (u, v)]

            added = False
            for a, b in cand:
                if a == b or A[a, b] == 1:
                    continue
                A[a, b] = 1
                if self._find_back_edge(A) is None:
                    added = True
                    break
                A[a, b] = 0

            if not added:
                # give up quietly (rare)
                pass

    def _repair_A(self, A: np.ndarray) -> np.ndarray:
        A = A.copy().astype(np.uint8, copy=False)
        np.fill_diagonal(A, 0)

        # 1) break cycles quickly by removing a found back-edge repeatedly
        while True:
            e = self._find_back_edge(A)
            if e is None:
                break
            A[e[0], e[1]] = 0

        # 2) cap indegree
        self._enforce_max_parents(A)

        # 3) optional connectivity
        if self.connect_disconnected:
            self._connect_components(A)
            # re-break cycles after connecting
            while True:
                e = self._find_back_edge(A)
                if e is None:
                    break
                A[e[0], e[1]] = 0
            self._enforce_max_parents(A)

        return A

    # -------------------- operators (matrix) --------------------
    def edge_swap_crossover_A(self, A1: np.ndarray, A2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c1, c2 = A1.copy(), A2.copy()
        e1 = np.argwhere(c1 != 0)
        e2 = np.argwhere(c2 != 0)
        c = min(self.c_N, e1.shape[0], e2.shape[0])
        if c == 0:
            return c1, c2

        idx1 = self._np_rng.choice(e1.shape[0], size=c, replace=False)
        idx2 = self._np_rng.choice(e2.shape[0], size=c, replace=False)

        swap1 = e1[idx1]
        swap2 = e2[idx2]

        c1[swap1[:, 0], swap1[:, 1]] = 0
        c2[swap2[:, 0], swap2[:, 1]] = 0
        c1[swap2[:, 0], swap2[:, 1]] = 1
        c2[swap1[:, 0], swap1[:, 1]] = 1

        return self._repair_A(c1), self._repair_A(c2)

    def parametric_uniform_crossover_A(self, A1: np.ndarray, A2: np.ndarray, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        # Only where they differ, decide to swap with probability p (vectorized)
        c1, c2 = A1.copy(), A2.copy()
        D = (c1 ^ c2)  # 1 where different
        if D.any():
            R = (self._np_rng.random(D.shape) < p).astype(np.uint8)
            M = (D & R).astype(np.uint8)
            c1 ^= M
            c2 ^= M
        np.fill_diagonal(c1, 0)
        np.fill_diagonal(c2, 0)
        return self._repair_A(c1), self._repair_A(c2)

    def _mutate_A(self, A: np.ndarray) -> np.ndarray:
        A = A.copy()
        op = self._rng.choice(["add", "delete", "reverse"])

        if op == "add":
            for _ in range(20):  # try a few random pairs
                u = self._rng.randrange(self.m)
                v = self._rng.randrange(self.m)
                if u == v or A[u, v] == 1:
                    continue
                A[u, v] = 1
                if self._find_back_edge(A) is None:
                    self._enforce_max_parents(A)
                    return A
                A[u, v] = 0
            return A

        if op == "delete":
            edges = np.argwhere(A != 0)
            if edges.size:
                k = self._rng.randrange(edges.shape[0])
                u, v = edges[k]
                A[int(u), int(v)] = 0
            return A

        # reverse
        edges = np.argwhere(A != 0)
        if edges.size:
            k = self._rng.randrange(edges.shape[0])
            u, v = map(int, edges[k])
            if A[v, u] == 0:
                A[u, v] = 0
                A[v, u] = 1
                if self._find_back_edge(A) is None:
                    self._enforce_max_parents(A)
                    return A
                # revert
                A[v, u] = 0
                A[u, v] = 1
        return A

    # -------------------- scoring --------------------
    def _score_A(self, A: np.ndarray) -> float:
        if hasattr(self.scorer, "local_score"):
            # build parent lists first (cheap)
            parent_lists = []
            for j, node in enumerate(self.variables):
                parents_idx = np.flatnonzero(A[:, j])
                parents = [self.variables[i] for i in parents_idx.tolist()]
                parent_lists.append((node, parents))

            scores = [self.scorer.local_score(node, parents) for node, parents in parent_lists]

            return float(np.sum(scores))

        if callable(self.scorer):
            return float(self.scorer(self._A_to_nx(A), self.data))

        if _pgmpy_DAG is not None and hasattr(self.scorer, "score"):
            dag_pg = _pgmpy_DAG()
            dag_pg.add_nodes_from(self.variables)
            ii, jj = np.where(A != 0)
            dag_pg.add_edges_from((self.variables[i], self.variables[j]) for i, j in zip(ii.tolist(), jj.tolist()))
            return float(self.scorer.score(dag_pg))

        raise RuntimeError("Scorer misconfigured")

    # -------------------- GA loop --------------------
    # def _evaluate_population(self) -> None:
    #     self.fitness = [self._score_A(A) for A in self.population]
    # top-level (module scope)
    

    def _evaluate_population(self) -> None:
        self.fitness = [self._score_A(A) for A in self.population]

    # # Pool mapping
    # def _evaluate_population(self) -> None:
    #     ctx = mp.get_context("spawn")
    #     scorer = self._score_A  # or a lightweight callable that doesn't capture huge state
    #     # Split the population into 10 chunks
    #     chunks = np.array_split(self.population, 10)
    #     with ctx.Pool(processes=getattr(self, "n_jobs", mp.cpu_count())) as pool:
    #         results = pool.map(
    #         lambda chunk: [score_A_worker((scorer, A)) for A in chunk],
    #         chunks
    #         )
    #     # Flatten results
    #     self.fitness = [score for sublist in results for score in sublist]

    def _update_best(self) -> None:
        idx = int(np.argmax(self.fitness))
        if self.fitness[idx] > self.best_score:
            self.best_score = float(self.fitness[idx])
            self.best_A = self.population[idx].copy()

    def _tournament_select_idx(self) -> int:
        pop_size = len(self.population)
        k = min(self.k_tourn, max(1, pop_size))
        idxs = self._rng.sample(range(pop_size), k)
        return max(idxs, key=lambda i: self.fitness[i])

    def run(self) -> tuple[nx.DiGraph, float]:
        self._initialize_population()
        self._evaluate_population()
        self._update_best()

        gen = 0
        patience = 5
        no_improve = 0
        eps = 0.0

        def one_generation() -> None:
            nonlocal gen
            gen += 1
            self.bic_curve.append(self.best_score)

            n_E = int(self.p_E * self.s)
            elite_idx = np.argsort(self.fitness)[-n_E:][::-1] if n_E else []
            new_pop: List[np.ndarray] = [self.population[i].copy() for i in elite_idx]

            mating = [self.population[self._tournament_select_idx()].copy() for _ in range(self.s - n_E)]
            self._rng.shuffle(mating)

            offspring: List[np.ndarray] = []
            n_pairs = int((self.p_C * len(mating)) // 2)
            for i in range(n_pairs):
                p1 = mating[2 * i]
                p2 = mating[2 * i + 1]
                if self.crossover_method == "edge_swap":
                    c1, c2 = self.edge_swap_crossover_A(p1, p2)
                else:
                    c1, c2 = self.parametric_uniform_crossover_A(p1, p2, p=0.5)
                offspring.extend([c1, c2])

            while len(offspring) < (self.s - n_E):
                offspring.append(mating[self._rng.randrange(len(mating))].copy())

            # mutation + repair
            for i in range(len(offspring)):
                if self._rng.random() < self.p_M:
                    offspring[i] = self._repair_A(self._mutate_A(offspring[i]))
                else:
                    offspring[i] = self._repair_A(offspring[i])

            new_pop.extend(offspring[: self.s - n_E])
            self.population = new_pop
            self._evaluate_population()

        if self.J is not None:
            for _ in range(self.J):
                prev = self.best_score
                one_generation()
                self._update_best()
                if self.show_progress:
                    print(f"Generation {gen}/{self.J}: Best score = {self.best_score:.2f}")
            assert self.best_A is not None
        else:
            if self.show_progress:
                print(f"Running with early stopping (patience={patience})...")
            while no_improve < patience:
                prev = self.best_score
                one_generation()
                self._update_best()
                improved = (self.best_score > prev + eps)
                no_improve = 0 if improved else (no_improve + 1)
                if self.show_progress:
                    print(f"Generation {gen}: Best score = {self.best_score:.2f} (no_improve={no_improve}/{patience})")
            assert self.best_A is not None

        best_graph = self._A_to_nx(self.best_A)
        best_score = float(self.best_score)
        if self.final_greedy:
            best_graph = self._final_greedy_refine(best_graph)
            best_score = self.scorer.local_score(best_graph, self.data) if callable(self.scorer) else float(self.scorer.score(best_graph))
            self.bic_curve.append(best_score)
        return best_graph, self.bic_curve
    
    
    def _final_greedy_refine(self, g_best: nx.DiGraph) -> nx.DiGraph:
        """
        Run a last greedy hill-climb starting from GA's best graph.
        Uses pgmpy HillClimbSearch when self.scorer is a pgmpy StructureScore.
        """
        # Convert nx -> pgmpy DAG
        start = _pgmpy_DAG()
        start.add_nodes_from(self.variables)
        start.add_edges_from(list(g_best.edges()))

        est = HillClimbSearch(self.data)

        # HillClimbSearch supports start_dag + max_indegree etc. :contentReference[oaicite:0]{index=0}
        greedy_epsilon: float = 1e-4,
        
        refined = est.estimate(
            scoring_method=self.scorer,      # your BIC/BDeu/BICSynergy instance
            start_dag=start,
            max_indegree=self.max_parents,
            epsilon=greedy_epsilon,

            show_progress=self.show_progress,
        )

        g_ref = nx.DiGraph()
        g_ref.add_nodes_from(self.variables)
        g_ref.add_edges_from(refined.edges())
        return g_ref



def genetic_bn_search_matrix(
    data: pd.DataFrame,
    population_size: int = 100,
    generations: int | None = 200,
    random_state: int | None = None,
    **kwargs,
) -> tuple[nx.DiGraph, float]:
    return GeneticBNSearchMatrix(
        data,
        population_size=population_size,
        generations=generations,
        random_state=random_state,
        **kwargs,
    ).run()


if __name__ == "__main__":
    # simple test
    import pandas as pd

    subset = "jpmf_data"
    d_idx = 350

    df = pd.read_csv(f"data/datasets//{subset}/Data_Graph_{d_idx}.csv")

    start_time = pd.Timestamp.now()
    graph, score = genetic_bn_search_matrix(
        data=df,
        population_size=20,
        generations=100,
        show_progress=True,
        final_greedy=True,
        n_jobs=1,
    )
    
    print("Best score:", score)
    print("Edges:", graph.edges())
    print("Elapsed time:", pd.Timestamp.now() - start_time)


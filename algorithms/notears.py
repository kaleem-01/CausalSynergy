import numpy as np
import networkx as nx
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

# assumes these already exist in your codebase
# from your_module import dag_to_cpdag, BIC


def notears_linear(
    X,
    lambda1=0.1,
    loss_type="logistic",
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.3,
):
    """
    Solve
        min_W L(W; X) + lambda1 * ||W||_1
        s.t. h(W) = 0
    using the linear NOTEARS augmented Lagrangian formulation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_variables)
        Data matrix.
    lambda1 : float
        L1 sparsity penalty.
    loss_type : {"l2", "logistic", "poisson"}
        Loss used by NOTEARS. For binary data, use "logistic".
    max_iter : int
        Maximum number of dual ascent steps.
    h_tol : float
        Stop when acyclicity constraint h(W) <= h_tol.
    rho_max : float
        Maximum penalty parameter.
    w_threshold : float
        Final absolute threshold for pruning small edge weights.

    Returns
    -------
    W_est : np.ndarray of shape (d, d)
        Estimated weighted adjacency matrix.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    def _loss(W):
        """Loss and gradient."""
        M = X @ W

        if loss_type == "l2":
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R

        elif loss_type == "logistic":
            # assumes X is binary {0,1}
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)

        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)

        else:
            raise ValueError(f"unknown loss_type: {loss_type}")

        return loss, G_loss

    def _h(W):
        """
        Acyclicity function h(W) = tr(exp(W ⊙ W)) - d
        and its gradient.
        """
        E = slin.expm(W * W)
        h_val = np.trace(E) - d
        G_h = E.T * W * 2.0
        return h_val, G_h

    def _adj(w):
        """Convert doubled variables back to d x d matrix."""
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    def _func(w):
        """Augmented Lagrangian objective and gradient."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h_val, G_h = _h(W)

        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h_val + alpha) * G_h

        # doubled variables for L1 with box constraints
        g_obj = np.concatenate(
            [G_smooth + lambda1, -G_smooth + lambda1],
            axis=None,
        )
        return obj, g_obj

    # Center data for l2 loss, following official implementation
    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)

    # doubled variables: w_pos, w_neg
    w_est = np.zeros(2 * d * d, dtype=float)
    rho, alpha, h_val = 1.0, 0.0, np.inf

    # no self-loops
    bnds = [
        (0, 0) if i == j else (0, None)
        for _ in range(2)
        for i in range(d)
        for j in range(d)
    ]

    for _ in range(max_iter):
        w_new, h_new = None, None

        while rho < rho_max:
            sol = sopt.minimize(
                _func,
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds,
            )
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))

            if h_new > 0.25 * h_val:
                rho *= 10.0
            else:
                break

        w_est, h_val = w_new, h_new
        alpha += rho * h_val

        if h_val <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0.0
    return W_est


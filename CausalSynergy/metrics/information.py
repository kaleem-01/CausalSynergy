import numpy as np
from typing import Iterable
import pandas as pd


def entropy(x: Iterable[int] | np.ndarray | pd.Series) -> float:
    x_arr = np.asarray(x)
    if x_arr.dtype.kind not in {"i", "u"}:
        x_arr, _ = pd.factorize(x_arr, sort=False)
    probs = np.bincount(x_arr) / x_arr.size
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _entropy_1d(x):
    return entropy(np.bincount(x))


def joint_entropy(arrs):
    """
    Joint entropy of k ≥ 2 integer arrays of equal length.
    Every joint state is encoded as a unique integer key by mixed radix
    numbering, so we need only one bincount() call.
    """
    A = np.vstack(arrs).T                          # shape (n_samples, k)
    bases = [int(A[:, i].max()) + 1 for i in range(A.shape[1])]
    weights = np.cumprod([1] + bases[:-1])
    keys = (A * weights).sum(1)
    return entropy(np.bincount(keys))

def mutual_information_joint(s1, s2, t):
    """I((S1,S2);T)."""
    return joint_entropy([s1, s2]) + _entropy_1d(t) - joint_entropy([s1, s2, t])


def mutual_information(x, y):
    return _entropy_1d(x) + _entropy_1d(y) - joint_entropy([x, y])



# def joint_entropy(*vars_: Iterable[int]) -> float:
#     codes, _ = pd.factorize(list(zip(*vars_)), sort=False)
#     return entropy(codes)


def _interaction_information(x, y, z) -> float:
    Hx, Hy, Hz = entropy(x), entropy(y), entropy(z)
    return (
        Hx
        + Hy
        + Hz
        - joint_entropy(x, y)
        - joint_entropy(x, z)
        - joint_entropy(y, z)
        + joint_entropy(x, y, z)
    )

# # ---------------------------------------------------------------------
# # public user‑facing function
# # ---------------------------------------------------------------------

# # def mutual_information(x: Iterable[int], y: Iterable[int], z: Iterable[int]) -> float:
# #     """
# #     Compute mutual information (interaction information) between three discrete variables.

# #     I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
# #     """
# #     x_arr = np.asarray(x)
# #     y_arr = np.asarray(y)
# #     z_arr = np.asarray(z)
# #     if x_arr.dtype.kind not in {"i", "u"}:
# #         x_arr, _ = pd.factorize(x_arr, sort=False)
# #     if y_arr.dtype.kind not in {"i", "u"}:
# #         y_arr, _ = pd.factorize(y_arr, sort=False)
# #     if z_arr.dtype.kind not in {"i", "u"}:
# #         z_arr, _ = pd.factorize(z_arr, sort=False)
# #     Hx = entropy(x_arr)
# #     Hy = entropy(y_arr)
# #     Hz = entropy(z_arr)
# #     Hxy = joint_entropy(x_arr, y_arr)
# #     Hxz = joint_entropy(x_arr, z_arr)
# #     Hyz = joint_entropy(y_arr, z_arr)
# #     Hxyz = joint_entropy(x_arr, y_arr, z_arr)
# #     return Hx + Hy + Hz - Hxy - Hxz - Hyz + Hxyz


# def joint_entropy(*vars_: Iterable[int]) -> float:
#     codes, _ = pd.factorize(list(zip(*vars_)), sort=False)
#     return entropy(codes)


# # def _interaction_information(x, y, z) -> float:
# #     Hx, Hy, Hz = _entropy(x), _entropy(y), _entropy(z)
# #     return (
# #         Hx
# #         + Hy
# #         + Hz
# #         - _joint_entropy(x, y)
# #         - _joint_entropy(x, z)
# #         - _joint_entropy(y, z)
# #         + _joint_entropy(x, y, z)
# #     )

# # ---------------------------------------------------------------------
# # public user‑facing function
# # ---------------------------------------------------------------------

# def mutual_information(x: Iterable[int], y: Iterable[int], z: Iterable[int]) -> float:
#     """
#     Compute mutual information (interaction information) between three discrete variables.

#     I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
#     """
#     x_arr = np.asarray(x)
#     y_arr = np.asarray(y)
#     z_arr = np.asarray(z)
#     if x_arr.dtype.kind not in {"i", "u"}:
#         x_arr, _ = pd.factorize(x_arr, sort=False)
#     if y_arr.dtype.kind not in {"i", "u"}:
#         y_arr, _ = pd.factorize(y_arr, sort=False)
#     if z_arr.dtype.kind not in {"i", "u"}:
#         z_arr, _ = pd.factorize(z_arr, sort=False)
#     Hx = entropy(x_arr)
#     Hy = entropy(y_arr)
#     Hz = entropy(z_arr)
#     Hxy = joint_entropy(x_arr, y_arr)
#     Hxz = joint_entropy(x_arr, z_arr)
#     Hyz = joint_entropy(y_arr, z_arr)
#     Hxyz = joint_entropy(x_arr, y_arr, z_arr)
#     return Hx + Hy + Hz - Hxy - Hxz - Hyz + Hxyz




def find_inflection_point(arr):
    arr = np.asarray(arr)
    # First and second discrete derivatives
    first_deriv = np.diff(arr)
    second_deriv = np.diff(first_deriv)

    # Inflection point = index where 2nd derivative is minimum (sharpest curvature)
    inflection_index = np.argmin(second_deriv) + 1  # +1 due to second derivative shift

    return inflection_index
import numpy as np


def compute_q_values(r, p, max_steps, discount, convergence_threshold=0.05):
    """Compute Q values of given MDP(r, p, discount) by Value Iteration.
    """
    n_states, n_actions = r.shape
    q = np.zeros_like(r)
    v = np.zeros(r.shape[0])

    for i in range(max_steps):
        v_old = v.copy()
        for s in range(n_states):
            q[s] = r[s] + discount * p[s].dot(v)
            v[s] = q[s].max()
        diff = np.abs(v_old - v).mean()
        if diff < convergence_threshold:
            break
    return q, v


def compute_q_values_opt(
    r, p, r_bonus, p_bonus, max_steps, discount, convergence_threshold=0.05
):
    n_states, n_actions = r.shape
    q = np.zeros_like(r)
    v = np.zeros(r.shape[0])

    for i in range(max_steps):
        v_old = v.copy()
        for s in range(n_states):
            q[s] = r[s] + r_bonus[s] + discount * p[s].dot(v) + p_bonus[s] * i
            v[s] = q[s].max()
        diff = np.abs(v_old - v).mean()
        if diff < convergence_threshold:
            break
    return q, v

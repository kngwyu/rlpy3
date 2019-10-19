import numpy as np


def compute_q_values(r, p, max_steps, discount, convergence_threshold=0.05):
    n_states, n_actions = r.shape
    q = np.zeros_like(r)
    v = np.zeros(r.shape[0])

    for i in range(max_steps):
        v_old = v.copy()
        for s in range(n_states):
            for a in range(n_actions):
                q[s, a] = r[s, a] + discount * np.dot(p[s, a], v)
            v[s] = q[s].max()
        diff = np.abs(v_old - v).mean()
        if diff < convergence_threshold:
            break
    return q

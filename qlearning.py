"""Q-Learning functions"""

import numpy as np
import settings as s


def init_q_factor(action_number):
    """

    :param action_number: number of actions
    :return: initialized Q-table
    """
    q = np.zeros([s.NUM_STATES, action_number])

    for i in range(1, action_number):
        q[0, i] = np.inf
    for i in range(0, action_number - 1):
        q[-1, i] = np.inf
    for i in range(1, s.NUM_STATES):
        q[i, 0] = np.inf
    for i in range(0, s.NUM_STATES - 1):
        q[i, -1] = np.inf
    return q  # initialized Q-table


def update_q_factor(q, state, action, diff, visits, step, ps):
    """

    :param q: Q-table to learn and update
    :param state: current sate
    :param action: current action
    :param diff: difference calculated to update Q factor
    :param visits: visit-time table
    :param step: current learning step
    :param ps: flag for PS employment
    :return: updated Q-table
    """
    if ps == 1:  # update Q-table with profit-sharing
        q[state, action] += diff / pow(visits[state, action], s.Q_FACTOR_STEP) * pow(s.LAMBDA, (s.EPOCH_LEARN - step))
    else:  # update Q-table naturally without profit-sharing
        q[state, action] += diff / pow(visits[state, action], s.Q_FACTOR_STEP)
    return q  # updated Q-table

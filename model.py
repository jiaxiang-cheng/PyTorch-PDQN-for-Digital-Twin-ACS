"""models of automated conveyor system with production station as agent"""

import numpy as np
import math
import settings as s


def cost_matrix(policy):
    """

    :param policy: policy strategy
    :return: cost matrix w.r.t. current policy strategy
    """
    costf = np.zeros([s.NUM_STATES, s.NUM_STATES])
    ave_sojourn_T = np.zeros([s.NUM_STATES, 1])

    for i in range(1, s.NUM_STATES - 1):
        replace = 0
        for j in range(s.ERLANG_ORDER):
            replace = replace + (s.ERLANG_ORDER - j) * pow(s.ERLANG_RATE, j - 1) \
                      * pow(policy[i], j) / math.factorial(j)
        ave_sojourn_T[i] = policy[i] + math.exp(-s.ERLANG_RATE * policy[i]) * replace

    exp_sojourn_T = np.zeros(s.NUM_STATES)
    for i in range(ave_sojourn_T.size):
        exp_sojourn_T[i] = math.exp(-s.ALPHA * ave_sojourn_T[i])

    ave_serve_T = s.ERLANG_ORDER / s.ERLANG_RATE
    exp_serve_T = math.exp(- s.ALPHA * ave_serve_T)

    if s.ALPHA == 0:
        dis_sojourn_T, dis_serve_T, dis_waste_T = ave_sojourn_T, ave_serve_T, ave_sojourn_T - ave_serve_T
    else:
        dis_sojourn_T = (1 - exp_sojourn_T) / s.ALPHA
        dis_serve_T = (1 - exp_serve_T) / s.ALPHA
        dis_waste_T = (exp_serve_T - exp_sojourn_T) / s.ALPHA

    costf[0, 1] = s.K1 * (s.NUM_STATES - 2) + s.K2 + s.K4 * exp_serve_T / dis_serve_T
    costf[s.NUM_STATES - 1, s.NUM_STATES - 2] = s.K3

    for i in range(1, s.NUM_STATES - 1):
        costf[i, i + 1] = s.K1 * (s.NUM_STATES - i) + (s.K2 * dis_serve_T + s.K3 * dis_waste_T[i]
                                                       + s.K4 * exp_serve_T + s.K5 * policy[i]) / dis_sojourn_T[i]
        if policy[i] == 0:
            x = 0
        else:
            dis_sojourn_TL = 1 / s.ARRIVAL_RATE - policy[i] * math.exp(- s.ARRIVAL_RATE * policy[i]) / (
                    1 - math.exp(- s.ARRIVAL_RATE * policy[i]))
            if s.ALPHA != 0:
                dis_sojourn_TL = (1 - math.exp(- s.ALPHA * dis_sojourn_TL)) / s.ALPHA
            x = policy[i] / dis_sojourn_TL
        costf[i - 1, i - 2] = s.K1 * (s.NUM_STATES - i + 1) + s.K3 + s.K5 * x

    return costf


def hQalphaV(policy):
    """docstring for hQalphaV"""

    hAlpha = np.zeros(s.NUM_STATES)
    if s.ALPHA == 0:
        hAlpha[0] = s.ERLANG_ORDER / s.ERLANG_RATE
    else:
        hAlpha[0] = (1 - pow(s.ERLANG_RATE, s.ERLANG_ORDER) / pow(s.ALPHA + s.ERLANG_RATE, s.ERLANG_ORDER)) / s.ALPHA
    hAlpha[-1] = 1 / (s.ALPHA + s.ARRIVAL_RATE)

    qalpha = np.zeros([s.NUM_STATES, s.NUM_STATES])
    qalpha[0, 1] = pow(s.ERLANG_RATE, s.ERLANG_ORDER) / pow(s.ALPHA + s.ERLANG_RATE, s.ERLANG_ORDER)
    qalpha[-1, -2] = s.ARRIVAL_RATE / (s.ALPHA + s.ARRIVAL_RATE)

    for i in range(1, s.NUM_STATES - 1):
        sumAlpha0 = 0
        sumAlpha = 0
        for j in range(s.ERLANG_ORDER):
            sumAlpha0 += pow(s.ERLANG_RATE, s.ERLANG_ORDER) * pow(policy[i], j) \
                         / pow(s.ERLANG_RATE, s.ERLANG_ORDER - j) / math.factorial(j)
            sumAlpha += pow(s.ERLANG_RATE, s.ERLANG_ORDER) * pow(policy[i], j) \
                        / pow(s.ALPHA + s.ERLANG_RATE, s.ERLANG_ORDER - j) / math.factorial(j)
        if s.ALPHA == 0:
            replace = 0
            for j in range(s.ERLANG_ORDER):
                replace += (s.ERLANG_ORDER - j) * pow(s.ERLANG_RATE, j - 1) * pow(policy[i], j) / math.factorial(j)
        else:
            replace = (sumAlpha0 - sumAlpha) / s.ALPHA

        hAlpha[i] = (1 - math.exp(- (s.ALPHA + s.ARRIVAL_RATE) * policy[i])) / (s.ALPHA + s.ARRIVAL_RATE) + math.exp(
            - (s.ALPHA + s.ARRIVAL_RATE + s.ERLANG_RATE) * policy[i]) * replace
        qalpha[i, i - 1] = s.ARRIVAL_RATE * (1 - math.exp(- (s.ALPHA + s.ARRIVAL_RATE)
                                                          * policy[i])) / (s.ALPHA + s.ARRIVAL_RATE)
        qalpha[i, i + 1] = math.exp(-(s.ALPHA + s.ARRIVAL_RATE) * policy[i]) * (
                1 - math.exp(- s.ERLANG_RATE * policy[i]) * sumAlpha0) + math.exp(
            -(s.ALPHA + s.ARRIVAL_RATE + s.ERLANG_RATE) * policy[i]) * sumAlpha

    return hAlpha, qalpha


def average_delay(embedP, hAlpha, policy):
    """docstring for average_delay"""

    state_delay = np.zeros([s.NUM_STATES, 1])
    state_delay[0, 0] = 0
    state_delay[-1, 0] = 1 / s.ARRIVAL_RATE

    if s.ALPHA != 0:
        for i in range(1, s.NUM_STATES - 1):
            replace = 0
            for j in range(s.ERLANG_ORDER):
                replace += (s.ERLANG_ORDER - j) * pow(s.ERLANG_RATE, j - 1) * pow(policy[i], j) / math.factorial(j)
            hAlpha[i] = (1 - math.exp(- s.ARRIVAL_RATE * policy[i])) / s.ARRIVAL_RATE + math.exp(
                - (s.ARRIVAL_RATE + s.ERLANG_RATE) * policy[i]) * replace

    for i in range(1, s.NUM_STATES - 1):
        state_delay[i] = hAlpha[i] - s.ERLANG_ORDER * math.exp(- s.ARRIVAL_RATE * policy[i]) / s.ERLANG_RATE

    b = np.zeros([1, s.NUM_STATES + 1])
    b[0, -1] = 1
    a = np.concatenate((embedP - s.EYE, s.ONES), axis=1)
    stable_prob = np.matmul(b, np.linalg.pinv(a))
    delay_time = np.matmul(stable_prob, state_delay)

    return delay_time


def equivalent_markov(policy):  # equivalent Markov process
    """docstring for equivalent_markov"""

    costf = cost_matrix(policy)

    embedP = np.zeros([s.NUM_STATES, s.NUM_STATES])
    embedP[0, 1] = 1
    embedP[s.NUM_STATES - 1, s.NUM_STATES - 2] = 1

    for i in range(1, s.NUM_STATES - 1):
        embedP[i, i + 1] = math.exp(- s.ARRIVAL_RATE * policy[i])
        embedP[i, i - 1] = 1 - math.exp(- s.ARRIVAL_RATE * policy[i])

    hAlpha, qalpha = hQalphaV(policy)

    uni_parameter = 1 / min(hAlpha)
    h_alpha = np.diag(hAlpha)
    inv_h_alpha = np.linalg.inv(h_alpha)

    Aalpha = s.ALPHA * s.EYE - np.matmul(inv_h_alpha, s.EYE - qalpha)

    if s.ALPHA != 0:
        p_alpha = np.matmul(inv_h_alpha, embedP - qalpha) / s.ALPHA
    else:
        averaQt = np.zeros([s.NUM_STATES, s.NUM_STATES])
        averaQt[0, 1] = h_alpha[0]
        averaQt[-1, -2] = h_alpha[-1]
        for i in range(1, s.NUM_STATES - 1):
            averaQt[i, i - 1] = 1 / s.ARRIVAL_RATE - (1 / s.ARRIVAL_RATE + policy[i]) * math.exp(
                - s.ARRIVAL_RATE * policy[i])
            averaQt[i, i + 1] = h_alpha[i] - averaQt[i, i - 1]
        p_alpha = np.matmul(inv_h_alpha, averaQt)

    f_alpha = np.zeros([s.NUM_STATES, s.NUM_STATES])
    for i in range(s.NUM_STATES):
        for j in range(s.NUM_STATES):
            f_alpha[i, j] = p_alpha[i, j] * costf[i, j]

    falpha = np.matmul(f_alpha, s.ONES)
    delay_T = average_delay(embedP, hAlpha, policy)

    return falpha, Aalpha, delay_T, uni_parameter


def stable_potential(falpha, Aalpha, uni_parameter):
    """docstring for stable_potential"""

    b = np.zeros([1, s.NUM_STATES + 1])
    b[0, -1] = 1
    a = np.concatenate((Aalpha, s.ONES), axis=1)

    stable_prob = np.matmul(b, np.linalg.pinv(a))
    potential = np.matmul(np.linalg.inv(s.ALPHA * s.EYE - Aalpha + uni_parameter
                                        * np.matmul(s.ONES, stable_prob)), falpha)
    return stable_prob, potential


def erlang_value(bound, prob):
    """docstring for erlang_value"""

    sum = 0
    for k in range(s.ERLANG_ORDER):
        sum += (pow(s.ERLANG_RATE, k) * pow(bound, k)) / math.factorial(k)
    time = 1 - prob - math.exp(- s.ERLANG_RATE * bound) * sum
    return time


def serve_erlang(prob):
    """

    :param prob: simulated probability value
    :return: simulated service time
    """
    service_T = s.ERLANG_ORDER / s.ERLANG_RATE
    inf_bound = 0
    sup_bound = 10 * service_T

    error = 1

    inf_T = erlang_value(inf_bound, prob)
    sup_T = erlang_value(sup_bound, prob)

    if inf_T == 0:
        serve_T = inf_bound
        error = 0

    while inf_T * sup_T > 0:
        inf_bound = sup_bound
        inf_T = sup_T
        sup_bound *= 2
        sup_T = erlang_value(sup_bound, prob)

    if sup_T == 0:
        serve_T = sup_bound
        error = 0

    while error > s.EPSILON_1:
        mid_bound = (inf_bound + sup_bound) / 2
        mid_T = erlang_value(mid_bound, prob)

        if mid_T == 0:
            serve_T = mid_bound
            error = 0
            break
        elif mid_T * inf_T < 0:
            sup_bound = mid_bound
        else:
            inf_bound = mid_bound
        error = sup_bound - inf_bound

    if error != 0:
        serve_T = (inf_bound + sup_bound) / 2

    return serve_T


def state_transition(current_state, current_action):
    """

    :param current_state: current state
    :param current_action: current action selected
    :return: production status, sojourn time, service time, and next state to transit
    """
    arrive_T = - math.log(1 - np.random.rand(1)) / s.ARRIVAL_RATE  # simulate the time for new unit to arrive

    if current_action >= arrive_T:  # arriving unit within inspection range, waiting for arrival

        flag = 0  # no processing, waiting
        next_state = current_state - 1  # adding arriving unit to reserve area

        serve_T = 0  # no service, only waiting for arriving unit
        sojourn_T = arrive_T  # sojourn time equal to the time for unit to arrive

    else:

        flag = 1  # no waiting, processing
        next_state = current_state + 1  # spare one place after processing one unit

        serve_T = serve_erlang(np.random.rand(1))  # simulate the service time for processing the unit
        sojourn_T = max(current_action, serve_T)

    return flag, sojourn_T, serve_T, next_state

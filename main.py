"""main script for simulating production process"""

import argparse
import os

from dqn import *
import qlearning as ql
from model import *
from visualize import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, dest="model")
parser.add_argument("--ps", type=int, dest="ps")
args = parser.parse_args()

MODEL = args.model if args.model else 1
PS = args.ps if args.ps else 1

"""

MODEL SELECTION:

MODEL 1 + PS 1 = DEEP Q-NETWORKS (DQN) + PROFIT-SHARING (PS)
MODEL 1 + PS 2 = DEEP Q-NETWORKS (DQN)
MODEL 2 + PS 1 = Q-LEARNING (QL) + PROFIT-SHARING (PS)
MODEL 2 + PS 2 = Q-LEARNING (QL)

"""

PATH = os.path.abspath(os.getcwd())


def simulation():
    """
    simulation of automated conveyor system with production station
    with range-inspection control optimized with Deep Q-Networks (DQN) and Profit-Sharing (PS)
    """
    # initialize action set
    action_set = np.zeros(int((s.MAX_INSPECT - s.MIN_INSPECT) / s.DELTA) + 3)
    x, i = s.MIN_INSPECT, 1
    while x <= s.MAX_INSPECT:
        action_set[i] = x
        x += s.DELTA
        i += 1
    action_set[-1] = np.inf
    action_number = len(action_set)

    # initialize current state
    current_state = math.floor(np.random.rand(1) * s.NUM_STATES)

    # initialize action index
    if current_state == 0:
        action_index = 0
    elif current_state == s.NUM_STATES - 1:
        action_index = action_number - 1

    if current_state != 0 and current_state != s.NUM_STATES - 1:
        action_index = action_number - 2

    # initialize policy set
    greedy_policy = np.zeros(s.NUM_STATES)
    greedy_policy[-1] = np.inf
    for i in range(1, s.NUM_STATES - 1):
        greedy_policy[i] = s.MAX_INSPECT

    visit_times = np.zeros([s.NUM_STATES, action_number])

    # initialization for simulation
    falpha, Aalpha, delay_T, uni_parameter = equivalent_markov(greedy_policy)
    stable_prob, potential = stable_potential(falpha, Aalpha, uni_parameter)
    last_value = falpha + np.matmul(Aalpha, potential)
    dis_value = last_value
    # ave_vector = np.matmul(stable_prob, falpha)
    # ave_estimate = ave_vector.tolist()
    each_transit_cost, each_transit_time, total_reward = (0 for i in range(3))

    # initialize DQN model if selected
    dqn = DQN() if MODEL == 1 else None
    # initialize Q-table if Q-learning selected
    q_factor = ql.init_q_factor(action_number) if MODEL == 2 else None

    for out_step in range(s.EPOCH):
        epsilon = s.EPSILON_1 if MODEL == 1 else s.EPSILON_2

        for inner_step in range(s.EPOCH_LEARN):

            visit_times[current_state, action_index] += 1
            current_action = greedy_policy[current_state]

            inspect_cost = 0 if current_state == s.NUM_STATES - 1 else s.K5 * current_action

            flag, sojourn_T, service_T, next_state = state_transition(current_state, current_action)
            each_transit_time = s.DISCOUNT * each_transit_time + (sojourn_T - each_transit_time) / pow(
                out_step * s.EPOCH_LEARN + (inner_step + 1), s.Q_AVE_STEP)
            end_sojourn_T = math.exp(- s.ALPHA * sojourn_T)
            end_serve_T = math.exp(- s.ALPHA * service_T)

            if s.ALPHA == 0:
                dis_T, dis_serve_T, dis_wait_T = sojourn_T, service_T, sojourn_T - service_T
            else:
                dis_T, dis_serve_T = (1 - end_sojourn_T) / s.ALPHA, (1 - end_serve_T) / s.ALPHA
                dis_wait_T = (end_serve_T - end_sojourn_T) / s.ALPHA

            if flag == 0:  # no processing, waiting
                cost_real = (s.K1 * (s.NUM_STATES - current_state) + s.K3) * sojourn_T + inspect_cost
                cost_purt = (s.K1 * (s.NUM_STATES - current_state) + s.K3) * dis_T + inspect_cost
            else:  # no waiting, processing
                cost_real = s.K1 * (s.NUM_STATES - current_state - 1) * sojourn_T + s.K2 * service_T + s.K3 * (
                        sojourn_T - service_T) + s.K4 + inspect_cost
                cost_purt = s.K1 * (s.NUM_STATES - current_state - 1) * dis_T + s.K2 * dis_serve_T + s.K3 * dis_wait_T \
                            + s.K4 * end_serve_T + inspect_cost

            each_transit_cost = s.DISCOUNT * each_transit_cost + (cost_real - each_transit_cost) / (
                pow(out_step * s.EPOCH_LEARN + (inner_step + 1), s.Q_AVE_STEP))

            ave_q_cost = each_transit_cost / each_transit_time
            # ave_estimate.append(ave_q_cost)
            cost_dis = cost_purt - ave_q_cost * dis_T

            if MODEL == 1:
                reward = - cost_dis
                dqn.store_transition(current_state, action_index, reward, next_state)
                if dqn.memory_counter >= s.MEMORY_CAPACITY:
                    dqn.learn(s.EPOCH_LEARN, inner_step, PS)
            else:
                difference = cost_dis + end_sojourn_T * min(q_factor[next_state, :]) \
                             - q_factor[current_state, action_index]
                q_factor = ql.update_q_factor(q_factor, current_state, action_index, difference,
                                              visit_times, inner_step, PS)
            current_state = next_state  # transit to next state

            if current_state == 0:
                action_index = 0
            elif current_state == s.NUM_STATES - 1:
                action_index = action_number - 1
            else:
                if MODEL == 1:
                    action_index = int(dqn.choose_action(current_state, epsilon))
                    if action_set[action_index] <= 1:
                        greedy_policy[current_state] = action_set[action_index]
                    else:
                        greedy_policy[current_state] = 1
                else:
                    if np.random.rand(1) < epsilon:
                        action_index = int(np.floor(np.random.rand(1) * (action_number - 2)) + 1)
                    else:
                        # minimal_q_value = np.min(q_factor[current_state, :])
                        action_index = np.argmin(q_factor[current_state, :])
                    greedy_policy[current_state] = action_set[action_index]

        # store the policy learned from the iterations
        optimal_policy = greedy_policy

        if MODEL != 1:
            for i in range(1, s.NUM_STATES - 1):
                # minimal_q_value_temp = np.min(q_factor[i, :])
                action_index_temp = np.argmin(q_factor[i, :])
                optimal_policy[i] = action_set[action_index_temp]

        falpha, Aalpha, delay_T, uni_parameter = equivalent_markov(optimal_policy)
        stable_prob, potential = stable_potential(falpha, Aalpha, uni_parameter)

        last_value = falpha + np.matmul(Aalpha, potential)
        dis_value = np.concatenate((dis_value, last_value), axis=1)
        total_reward += - np.ndarray.item(last_value[0])
        # new_ave_cost = np.matmul(stable_prob, falpha)
        # ave_vector = np.concatenate((ave_vector, new_ave_cost))
        print("epoch: {} , the epoch reward is {}".format(out_step, round(- np.ndarray.item(last_value[0]), 2)))

    # result = np.asarray(dis_value)
    print("total reward:", total_reward)

    return dis_value, total_reward


if __name__ == "__main__":
    rewards, total_reward = simulation()
    visualize(MODEL, PS, rewards, total_reward, PATH)

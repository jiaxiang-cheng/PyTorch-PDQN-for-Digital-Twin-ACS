"""Result Visualization"""

from datetime import datetime
import matplotlib.pyplot as plt


def visualize(model, ps, curve, reward, path):
    """

    :param model: index of selected model
    :param ps: flag of PS employment
    :param curve: discount cost curve
    :param reward: total reward accumulated
    :param path: path for saving plot
    """
    plt.figure(figsize=(10, 6))
    plt.axvline(x=len(curve[0, :]), c='r', linestyle='--')

    if model == 1:
        if ps == 1:
            plt.plot(curve[0, :], label='Cost Curve with Deep Q-Network + PS with Total Reward = {}'
                     .format(round(reward, 3)))
        else:
            plt.plot(curve[0, :], label='Cost Curve with Deep Q-Network with Total Reward = {}'
                     .format(round(reward, 3)))
    else:
        if ps == 1:
            plt.plot(curve[0, :], label='Cost Curve with Q-learning + PS with Total Reward = {}'
                     .format(round(reward, 3)))
        else:
            plt.plot(curve[0, :], label='Cost Curve with Q-learning with Total Reward = {}'
                     .format(round(reward, 3)))

    plt.title('DT-ACS Simulation')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    if model == 1:
        if ps == 1:
            plt.savefig(path + '/_trials/{} DT-ACS (DQN + PS) {}.png'.format(round(reward, 1), datetime.now()))
        else:
            plt.savefig(path + '/_trials/{} DT-ACS (DQN) {}.png'.format(round(reward, 1), datetime.now()))
    else:
        if ps == 1:
            plt.savefig(path + '/_trials/{} DT-ACS (QL + PS) {}.png'.format(round(reward, 1), datetime.now()))
        else:
            plt.savefig(path + '/_trials/{} DT-ACS (QL) {}.png'.format(round(reward, 1), datetime.now()))

    plt.show()

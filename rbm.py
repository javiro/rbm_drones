from __future__ import print_function

from matplotlib import pyplot as plt
import scipy.special
import numpy as np
import pandas as pd
import itertools


class RbmImpl:
    '''
    This class implements Restricted Boltzman Machines
    '''

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.verbose = True
        np_rng = np.random.RandomState(3412)

        self.weights = np.asarray(np_rng.uniform(
                    low=-4 * np.sqrt(6. / (num_hidden + num_visible)),
                    high=4 * np.sqrt(6. / (num_hidden + num_visible)),
                    size=(num_visible, num_hidden)))

        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def forward(self, v, weights):
        v = np.insert(v, 0, 1)
        pos_hid_activations = np.dot(v, weights)
        pos_hid_probs = self.sigmoid(pos_hid_activations)
        pos_hid_states = pos_hid_probs[1:] > np.random.rand(self.num_hidden)
        return pos_hid_states * 1

    def backward(self, h, weights):
        h = np.insert(h, 0, 1)
        neg_vis_activations = np.dot(h, weights.T)
        neg_vis_probs = self.sigmoid(neg_vis_activations)
        neg_vis_states = neg_vis_probs[1:] > np.random.rand(self.num_visible)
        return neg_vis_states * 1

    def rbm_sampling(self, data, n_samples):
        return (self.backward(self.forward(v, self.weights), self.weights)
                for v in data[np.random.choice(len(data), n_samples)])

    def train_rbm(self, data, max_epochs = 2000, learning_rate = 0.08):

        num_examples = data.shape[0]
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            pos_hid_activations = np.dot(data, self.weights)
            pos_hid_probs = self.sigmoid(pos_hid_activations)
            pos_hid_probs[:, 0] = 1
            pos_hid_states = pos_hid_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hid_probs)

            neg_vis_activations = np.dot(pos_hid_states, self.weights.T)
            neg_vis_probs = self.sigmoid(neg_vis_activations)
            neg_vis_probs[:, 0] = 1

            neg_hid_activations = np.dot(neg_vis_probs, self.weights)
            neg_hid_probs = self.sigmoid(neg_hid_activations)
            neg_associations = np.dot(neg_vis_probs.T, neg_hid_probs)

            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
            error = np.sum((data - neg_vis_probs) ** 2)
            if self.verbose:
                print('Epoch %s: Error is: %s', (epoch, error))

    def sigmoid(self, val):
        return 1.0 / (1 + np.exp(-val))


class plotEvolution(RbmImpl):

    def __init__(self, num_visible, num_hidden, n_of_agents, data):
        super().__init__(num_visible, num_hidden)
        self.n_of_agents = n_of_agents
        self.data = data
        self.count_of_states = self.get_initial_count_of_states()

    def get_initial_count_of_states(self):
        num_of_states = int(scipy.special.binom(self.n_of_agents + 1, 1))
        dist_of_states = np.zeros(num_of_states)
        random_initial_condition = [200, 0]
        dist_of_states[random_initial_condition[1]] = 1
        return dist_of_states

    def plot_mean_dynamics(self):
        count_of_states = self.get_initial_count_of_states()
        s = list(self.rbm_sampling(self.data, 1))[0]
        mean_dynamic = []
        for strategies in np.reshape(s, v.shape):
            distribution = np.histogram(strategies, bins=list(range(len(set(s)) + 1)))[0]
            count_of_states[distribution[1]] += 1
            x = range(len(count_of_states))
            integral_f_dx = sum(count_of_states * np.diff(range(len(count_of_states) + 1)))
            f_bar = count_of_states / integral_f_dx
            dx = np.diff(range(len(count_of_states) + 1))
            expectation = sum(x * f_bar * dx)
            mean_dynamic.append(expectation)
        plt.plot(mean_dynamic)
        plt.show()


def read_evolutions(name, length):
    data = []
    for i in range(length):
        v = pd.read_csv('{}{}'.format(name, i), header=None)
        data.append(list(itertools.chain.from_iterable(v.values)))
    return np.array(data)


if __name__ == '__main__':

    # training_data = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0], [1,1,1,0,0,0],
    #                 [0,0,1,1,1,0], [0,0,1,1,0,0], [0,0,1,1,1,0]])
    # rbmInstance.train_rbm(data = training_data, max_epochs = 5000)
    #
    # rbmInstance.rbm_sampling(training_data)

    training_data = read_evolutions('../drones/microstates', 10)
    num_visible = len(training_data[0])
    num_hidden = 1000
    rbmInstance = RbmImpl(num_visible, num_hidden)
    rbmInstance.train_rbm(data=training_data, max_epochs=10)
    s = list(rbmInstance.rbm_sampling(training_data, 1))[0]
    v = pd.read_csv('../drones/microstates1', header=None)

    plt.plot(np.reshape(s, v.shape).mean(axis=1))
    plt.plot(v.mean(axis=1))
    plt.show()

    p = plotEvolution(num_visible, num_hidden, n_of_agents=200, data=training_data)
    p.plot_mean_dynamics()

    print('The weights obtained after training are:')
    print(rbmInstance.weights)


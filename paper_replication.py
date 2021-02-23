import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from collections import Counter
from copy import deepcopy
from itertools import product
from scipy.stats import entropy
from tqdm.notebook import tqdm

from ca_funcs import get_network_entropies, make_ca, make_glider, make_table_walk
from train_ca import initialize_model
from utils import all_combinations

M = 2
D = (3, 3)
ALL_INPUTS = all_combinations(M, D)


def sample_CAs(rng=None):
    rng = rng or np.random.default_rng(0)
    inputs = ALL_INPUTS
    outputs = make_table_walk(len(ALL_INPUTS), rng=rng)
    for o in outputs:
        yield make_ca(inputs, o)


def generate_CA_train_data(ca, height=10, width=10, n_samples=500, rng=None, noise=0.0):
    rng = rng or np.random.default_rng(0)
    X_train = torch.from_numpy(rng.choice([0, 1], (n_samples, height, width), p=[.5, .5])).float()
    Y_train = ca(X_train).float()
    foo = Y_train.detach().clone()
    flat_Y_train = Y_train.view(-1)
    flat_Y_indices = rng.choice(range(Y_train.numel()), size=int(Y_train.numel() * noise), replace=False)
    flat_Y_train[flat_Y_indices] = 1 - flat_Y_train[flat_Y_indices]
    return X_train, Y_train


def ca_entropy(ca):
    inputs = torch.from_numpy(ALL_INPUTS)
    outputs = ca(inputs)
    output_counts = np.array(list(Counter(tuple(torch.reshape(o, [-1]).numpy()) for o in outputs).values()))
    output_ps = output_counts / len(inputs)
    return entropy(output_ps, base=2)


def train(training_epochs, ca, layer_dims=None, rng=None, train_noise=0.0):
    layer_dims = layer_dims or [100] + [100] * 11  # neighborhood conv + mlpconv layers
    rng = rng or np.random.default_rng(0)

    input_dims = [10, 10]

    learning_rate = 1e-4

    loss = torch.nn.MSELoss()

    samples = 500
    batch_size = 10
    num_batches = samples // batch_size

    def make_model(seed=0):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        model = initialize_model(input_dims, layer_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #         display(model)

        if torch.cuda.is_available():
            model.cuda()

        return model, optimizer

    def learn_CA(model, optimizer):
        losses = []
        X_train, Y_train = generate_CA_train_data(ca, *input_dims, n_samples=samples, rng=rng, noise=train_noise)
        if torch.cuda.is_available():
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

        for _ in tqdm(range(training_epochs), leave=False):
            batch_losses = []
            for i in range(num_batches):
                X_batch = X_train[i * batch_size: (i + 1) * batch_size]
                Y_batch = Y_train[i * batch_size: (i + 1) * batch_size]

                optimizer.zero_grad()
                Y_pred = model(X_batch)
                l = loss(Y_batch, Y_pred)
                l.backward()
                optimizer.step()
                batch_losses.append(l.item())
            losses.append(np.mean(batch_losses))
        return losses

    model, optimizer = make_model()
    losses = learn_CA(model, optimizer)

    return model, optimizer, losses


def calculate_entropies(model, layer_dims):
    def get_activations(x_input):
        activations = []
        for m in model.children():
            x_input = m(x_input)
            activations.append(x_input)
        return activations[1:-3:2]

    X_test = np.pad(all_combinations(2, (3, 3)), [(0, 0), (3, 4), (3, 4)], 'wrap')
    X_test = torch.from_numpy(X_test).float()

    if torch.cuda.is_available():
        X_test = X_test.cuda()

    res = [activation.cpu().detach().numpy() for activation in get_activations(X_test)]
    layer_activations = np.array(res)
    # Layer activations are floats, but to calculate entropy,
    # we want to map activations to binary values,
    # 1 if a given activation is >0, and 0 otherwise.
    binary_activations = np.digitize(layer_activations, [0], right=True)
    binary_activations = binary_activations.transpose(0, 1, -2, -1, 2) \
        .reshape(len(layer_dims), np.product(X_test.shape), layer_dims[0])
    return get_network_entropies(binary_activations)


def prune_model_and_test(model, optimizer, Pruner, config):
    model_copy = deepcopy(model)
    pruner = Pruner(model_copy, config, optimizer=optimizer)
    pruner.compress()
    return model_copy


def l2(Y):
    return np.sum(np.power(Y, 2))


def test_model(model, ca):
    x = make_glider(10)
    X_test = torch.from_numpy(x.reshape(1, 10, 10)).float()
    Y_test = ca(X_test).float()

    if torch.cuda.is_available():
        X_test = X_test.cuda()
    Y_pred = model(X_test)

    if torch.cuda.is_available():
        Y_pred = Y_pred.cpu()

    Y_test = Y_test.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_diff = Y_test - Y_pred
    return l2(Y_diff)


def display_test(model, ca):
    x = make_glider(10)
    X_test = torch.from_numpy(x.reshape(1, 10, 10)).float()
    Y_test = ca(X_test).float()

    if torch.cuda.is_available():
        X_test = X_test.cuda()
    Y_pred = model(X_test)

    if torch.cuda.is_available():
        X_test = X_test.cpu()
        Y_pred = Y_pred.cpu()

    X_test = X_test.detach().numpy()
    Y_test = Y_test.detach().numpy()
    Y_pred = Y_pred.detach().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(141)
    plt.imshow(X_test[0])
    plt.axis('off')
    plt.title("Input")

    plt.subplot(142)
    plt.imshow(Y_test[0])
    plt.axis('off')
    plt.title("Expected Output")

    plt.subplot(143)
    plt.imshow(Y_pred[0])
    plt.axis('off')
    plt.title("Observed Output")

    plt.subplot(144)
    plt.imshow((Y_pred[0] - Y_test[0]) ** 2)
    plt.axis('off')
    plt.title("Normalised Diff")

    print('max loss:', ((Y_pred[0] - Y_test[0]) ** 2).max())


def lmc_complexity(P, N):
    P = np.array(P)
    P = P / P.sum()
    H = entropy(P, base=2)

    if N <= np.finfo(P.dtype).max:
        uniform_ps = np.full(len(P), 1.0 / N)
        D = np.sum((P - uniform_ps) ** 2) + (N - len(P)) * (1.0 / N) ** 2
    else:
        # assuming N >> len(P) >= 1 >= P, so that len(P)/N, 1/N and P/N are negligible
        D = np.sum(P ** 2)

    return H * D


def ca_lmc(ca, m=M, d=D):
    inputs = torch.from_numpy(ALL_INPUTS)
    outputs = ca(inputs)
    output_counts = np.array(list(Counter(tuple(torch.reshape(o, [-1]).numpy()) for o in outputs).values()))
    output_ps = output_counts / len(inputs)
    return lmc_complexity(output_ps, m ** np.product(d))


def get_network_lmcs(layers_samples_neurons):
    neuron_lmcs_by_layer = []
    layer_lmcs = []

    layer_count = layers_samples_neurons.shape[0]
    neuron_count = layers_samples_neurons.shape[2]

    for l in layers_samples_neurons:
        neuron_ps = l.mean(axis=0)
        neuron_lmcs_by_layer.append(np.array([lmc_complexity([p, 1 - p], 2) for p in neuron_ps]))

        layer_patterns = (tuple(sample) for sample in l)
        layer_pattern_counts = list(Counter(layer_patterns).values())
        layer_lmcs.append(lmc_complexity(layer_pattern_counts, 2 ** neuron_count))

    network_patterns = (tuple(sample.ravel()) for sample in layers_samples_neurons.swapaxes(0, 1))
    network_pattern_counts = list(Counter(network_patterns).values())
    network_lmc = lmc_complexity(network_pattern_counts, 2 ** (layer_count * neuron_count))

    return network_lmc, layer_lmcs, neuron_lmcs_by_layer


def model_lmc(model, layer_dims):
    def get_activations(x_input):
        activations = []
        for m in model.children():
            x_input = m(x_input)
            activations.append(x_input)
        return activations[1:-3:2]

    X_test = np.pad(all_combinations(2, (3, 3)), [(0, 0), (3, 4), (3, 4)], 'wrap')
    X_test = torch.from_numpy(X_test).float()

    if torch.cuda.is_available():
        X_test = X_test.cuda()

    res = [activation.cpu().detach().numpy() for activation in get_activations(X_test)]
    layer_activations = np.array(res)
    # Layer activations are floats, but to calculate lmc complexity,
    # we want to map activations to binary values,
    # 1 if a given activation is >0, and 0 otherwise.
    binary_activations = np.digitize(layer_activations, [0], right=True)
    binary_activations = binary_activations.transpose(0, 1, -2, -1, 2) \
        .reshape(len(layer_dims), np.product(X_test.shape), layer_dims[0])
    return get_network_lmcs(binary_activations)


def model_stats(ca_id, seed, train_noise, ca, model, layer_dims, losses):
    model_entropy, layer_entropies, neuron_entropies = calculate_entropies(model, layer_dims)
    neuron_entropies = np.array(neuron_entropies).mean(axis=-1)

    model_lmc_, layer_lmcs, neuron_lmcs = model_lmc(model, layer_dims)
    neuron_lmcs = np.array(neuron_lmcs).mean(axis=-1)

    return [
        (ca_id, seed, train_noise, 'ca_entropy', np.nan, ca_entropy(ca)),
        (ca_id, seed, train_noise, 'ca_lmc', np.nan, ca_lmc(ca)),
        (ca_id, seed, train_noise, 'model_entropy', np.nan, model_entropy),
        *((ca_id, seed, train_noise, 'layer_entropy', i, e) for i, e in enumerate(layer_entropies)),
        *((ca_id, seed, train_noise, 'neuron_entropy', i, e) for i, e in enumerate(neuron_entropies)),
        (ca_id, seed, train_noise, 'model_lmc', np.nan, model_lmc_),
        *((ca_id, seed, train_noise, 'layer_lmc', i, lmc) for i, lmc in enumerate(layer_lmcs)),
        *((ca_id, seed, train_noise, 'neuron_lmc', i, lmc) for i, lmc in enumerate(neuron_lmcs)),
        (ca_id, seed, train_noise, 'losses', np.nan, losses),
    ]


def collect_stats(ixs, noises, seed, training_epochs):
    layer_dims = [100] + [100] * 11  # neighborhood conv + mlpconv layers

    rng = np.random.default_rng(seed)

    cas = np.array(list(sample_CAs(rng=rng)))[ixs]
    for (i, ca), n in tqdm(list(product(zip(ixs, cas), noises))):
        model, optimizer, losses = train(
            training_epochs,
            ca,
            layer_dims,
            rng,
            train_noise=n
        )

        stats = model_stats(i, seed, n, ca, model, layer_dims, losses)
        stats = pd.DataFrame(stats, columns=['ca_id', 'seed', 'noise', 'type', 'layer', 'value'])
        stats.to_csv(f'stats/stats_{i}_{seed}_{n}.csv')

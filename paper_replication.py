# %% md

# Paper replication

# %%

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %%

from utils import all_combinations

M = 2
D = 3 * 3
ALL_INPUTS = all_combinations(M, D)

# %%

## Define CAs and training data

# %%

from ca_funcs import make_table_walk, make_ca


def sample_CAs(seed=None):
    if seed:
        np.random.seed(seed)

    inputs = ALL_INPUTS
    outputs = make_table_walk(len(ALL_INPUTS))
    for o in outputs:
        yield make_ca(inputs, o)


def generate_CA_train_data(ca, height=10, width=10, n_samples=500):
    X_train = torch.from_numpy(np.random.choice([0, 1], (n_samples, height, width), p=[.5, .5])).float()
    Y_train = ca(X_train).float()
    return X_train, Y_train


# %%

from ca_funcs import make_glider
from IPython.display import clear_output

np.random.seed(0)

# for i, ca in enumerate(sample_CAs()):
#     X_test = torch.from_numpy(make_glider(10).reshape(1, 10, 10))
#     Y_test = ca(X_test)
#
#     plt.figure(figsize=(12, 4))
#     plt.suptitle(i)
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(X_test[0])
#     plt.axis('off')
#     plt.title("Input")
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(Y_test[0])
#     plt.axis('off')
#     plt.title("Output")
#
#     plt.show()
#     plt.close()
#     clear_output(wait=True)

# %%

## Find entropy of the training CA

# %%

from collections import Counter
from utils import shannon_entropy


def ca_entropy(ca):
    inputs = torch.from_numpy(ALL_INPUTS).float()
    outputs = ca(inputs).float()
    output_counts = np.array(list(Counter(tuple(torch.reshape(o, [-1]).numpy()) for o in outputs).values()))
    output_ps = output_counts / len(inputs)
    return shannon_entropy(output_ps)

#
# # %%
#
# entropies = [ca_entropy(a) for a in tqdm(sample_CAs(seed=0))]
# plt.plot(entropies)

# %%

## Define the model

# %%

from train_ca import initialize_model

seed = 0
print('seed =', seed)

np.random.seed(seed)
torch.random.manual_seed(seed)

num_classes = 2
samples = 500
input_dims = [10, 10]
layer_dims = [100] + [100] * 11  # neighborhood conv + mlpconv layers
batch_size = 10
num_batches = samples / batch_size
learning_rate = 1e-4
training_epochs = 10  # 1500
display_step = int(training_epochs / 10)
loss = torch.nn.MSELoss()

model = initialize_model(input_dims, layer_dims)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# %%

## Define the learning loop

# %%

losses = []


def learn_CA(ca, model):
    for epoch in tqdm(range(training_epochs)):
        X_train, Y_train = generate_CA_train_data(ca, *input_dims, n_samples=samples)

        optimizer.zero_grad()
        Y_pred = model(X_train)
        l = loss(Y_train, Y_pred)
        l.backward()
        optimizer.step()
        losses.append(l.item())


# %%

## Train the model

# %%

ca = list(sample_CAs(seed=0))[250]
learn_CA(ca, model)

# %%

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.plot(losses)
# plt.subplot(122)
# plt.plot(losses)
# plt.loglog()

# %%

from ca_funcs import make_glider

# x = np.random.choice([0, 1], size=100)
x = make_glider(10)
X_test = torch.from_numpy(x.reshape(1, 10, 10)).float()
Y_test = ca(X_test).float()
Y_pred = model(X_test)

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
plt.title("Diff")

# %%

np.max(np.abs(Y_pred[0] - Y_test[0]))

# %%

## Find model entropies

# %%

from ca_funcs import get_network_entropies


def get_activations(x_input):
    activations = [x_input]
    for m in model.modules():
        x_input = m(x_input)
        activations.append(x_input)
    return activations[5:len(activations) - 1]


X_test = torch.from_numpy(np.random.choice([0, 1], (500, 10, 10))).float()
# X_test = torch.unsqueeze(X_test, dim=1)

layer_activations = np.array([activation.detach().numpy() for activation in get_activations(X_test)])
binary_activations = np.digitize(layer_activations, [0], right=True)
entropies = get_network_entropies(binary_activations)

# %%

ca_entropy(ca)

# %%

entropies

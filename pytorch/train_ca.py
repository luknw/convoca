from collections import OrderedDict

import torch
import random


def periodic_padding(imbatch, padding=1):
    """
    Create a periodic padding (wrap) around an image batch, to emulate
    periodic boundary conditions. Padding occurs along the middle two axes
    """
    pad_u = imbatch[:, -padding:, :]
    pad_b = imbatch[:, :padding, :]

    partial_image = torch.cat([pad_u, imbatch, pad_b], dim=1)

    pad_l = partial_image[..., :, -padding:]
    pad_r = partial_image[..., :, :padding]

    padded_imbatch = torch.cat([pad_l, partial_image, pad_r], dim=2)

    return padded_imbatch


class Wraparound2D(torch.nn.Module):
    """
    Apply periodic boundary conditions on an image by padding 
    along the axes
    padding : int or tuple, the amount to wrap around    
    """

    def __init__(self, padding=2):
        super().__init__()
        self.padding = padding

    def forward(self, inputs):
        return torch.unsqueeze(periodic_padding(inputs, self.padding), dim=1)


def initialize_model(shape, layer_dims, nhood=1, totalistic=False,
                     nhood_type="moore", bc="periodic"):
    """
    Given a domain size and layer specification, initialize a model that assigns
    each pixel a class
    shape : the horizontal and vertical dimensions of the CA image
    layer_dims : list of number of hidden units per layer
    num_classes : int, the number of output classes for the automaton
    totalistic : bool, whether to assume that the CA is radially symmetric, making
        it outer totalistic
    nhood_type : string, default "moore". The type of neighborhood to use for the 
        CA. Currently, the only other option, "Neumann," only works when "totalistic"
        is set to True
    bc : string, whether to use "periodic" or "constant" (zero padded) boundary conditions
    """
    diameter = 2 * nhood + 1
    model = []

    if bc == "periodic":
        model.append(('Wraparound2D', Wraparound2D(padding=nhood)))

    if totalistic:
        model.append(('SymmetricConvolution', SymmetricConvolution(nhood, n_type=nhood_type, bc=bc)))
    else:
        model.append(('Conv2d_0', torch.nn.Conv2d(in_channels=1, out_channels=layer_dims[0], kernel_size=(diameter, diameter))))
        model.append(('ReLU_0', torch.nn.ReLU()))
        model.append(('Lambda_0', Lambda(lambda x: torch.transpose(torch.transpose(x, 1, 2), 2, 3))))

    for i in range(1, len(layer_dims)):
        model.append((f'Dense_{i}', torch.nn.Linear(layer_dims[i], layer_dims[i])))
        model.append((f'ReLU_{i}', torch.nn.ReLU()))

    model.append(('Lambda_1', Lambda(lambda x: torch.sum(x, dim=-1))))
    return torch.nn.Sequential(OrderedDict(model))


def logit_to_pred(logits, shape=None):
    """
    Given logits in the form of a network output, convert them to 
    images
    """

    labels = torch.argmax(torch.nn.functional.softmax(logits), dim=-1)
    if shape:
        labels = torch.reshape(labels, shape)
    return labels


def augment_data(x, y, n=None):
    """
    Generate an augmented training dataset with random reflections
    and 90 degree rotations
    x, y : Image sets of shape (Samples, Width, Height, Channels) 
        training images and next images
    n : number of training examples
    """
    n_data = x.shape[0]

    if not n:
        n = n_data
    x_out, y_out = list(), list()

    for i in range(n):
        r = random.randint(0, n_data)
        x_r, y_r = x[r], y[r]

        if random.random() < 0.5:
            x_r = torch.fliplr(x_r)
            y_r = torch.fliplr(y_r)
        if random.random() < 0.5:
            x_r = torch.flipud(x_r)
            y_r = torch.flipud(y_r)

        num_rots = random.randint(0, 4)
        x_r = torch.rot90(x_r, k=num_rots)
        y_r = torch.rot90(y_r, k=num_rots)

        x_out.append(x_r), y_out.append(y_r)
    return torch.stack(x_out), torch.stack(y_out)


def make_square_filters(rad):
    """
    rad : the pixel radius for the filters
    """
    m = 2 * rad + 1
    square_filters = torch.stack([
        torch.nn.functional.pad(torch.ones([i, i]),
                                [[int((m - i) / 2), int((m - i) / 2)],
                                 [int((m - i) / 2), int((m - i) / 2)]])
        for i in range(1, m + 1, 2)])
    square_filters = [square_filters[0]] + [item for item in square_filters[1:] - square_filters[:-1]]
    square_filters = torch.unsqueeze(torch.stack(square_filters), dim=1)

    return square_filters


def make_circular_filters(rad):
    """
    rad : the pixel radius for the filters
    """

    m = 2 * rad + 1

    qq = torch.range(start=0, end=m) - int((m - 1) / 2)
    pp = torch.sqrt((qq[..., None] ** 2 + qq[None, ...] ** 2).type('pytorch.float32'))

    val_range = (torch.range(start=0, end=((m + 1) / 2)).type('pytorch.float32'))
    circ_filters = make_square_filters(rad) * val_range[..., None, None, None]
    rr = circ_filters * (1 / pp)[None, ..., None]
    rr = torch.where(torch.isnan(rr), torch.zeros_like(rr), rr)
    return torch.stack([make_square_filters(rad)[0]] + [item for item in rr][1:])


class SymmetricConvolution(torch.nn.Module):
    """
    A non-trainable convolutional layer that extracts the 
    summed values in the neighborhood of each pixel. No activation
    is applied because this feature extractor does not change during training
    parametrized by the radius
    r : int, the max neighborhood size
    nhood_type : "moore" (default) uses the Moore neighborhood, while "neumann"
        uses the generalized von Neumann neighborhood, which is similar 
        to a circle at large neighborhood radii
    bc : "periodic" or "constant"
    TODO : implement the "hard" von Neumann neighborhood
    """

    def __init__(self, r, nhood_type="moore", bc="periodic", **kwargs):
        super().__init__()

        self.r = r

        if nhood_type == "moore":
            filters = make_square_filters(r)
        elif nhood_type == "neumann":
            filters = make_circular_filters(r)
        else:
            raise Exception("Neighborhood specification not recognized.")
        self.filters = torch.squeeze(torch.transpose(filters, 0, 1))[..., None, :]

    def forward(self, inputs):
        return torch.nn.functional.conv2d(inputs, self.filters)


class Lambda(torch.nn.Module):
    """An easy way to create a pytorch layer for a simple `func`."""

    def __init__(self, func):
        """create a layer that simply calls `func` with `x`"""
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

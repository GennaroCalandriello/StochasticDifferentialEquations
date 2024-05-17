import torch
from torch import einsum, roll, Tensor
import numpy as np


class GCMConv(torch.nn.Module):
    """
        A module that combines gauge covariant convolutions and gauge covariant bilinear layers
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True):
        super(GCMConv, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements

        # initialize weights
        w_in_size = self.n_in
        t_w_size = self.n_in * (self.kernel_size - 1) * len(self.dims)
        w_out_size = self.n_out

        # double channels for complex conjugation
        w_in_size = 2 * w_in_size
        t_w_size = 2 * t_w_size

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        variance = 1.0 / (w_in_size * t_w_size)
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))

        # gather all terms along lattice axes up to kernel_size
        transported_terms = []
        for axis in range(len(self.dims)):
            w_transport = w.clone()
            for d in range(1, self.kernel_size):
                # get transported terms
                for step in range(self.dilation + 1):
                    w_transport = transport(u, w_transport, axis=axis, orientation=1, dims=self.dims)
                # and add to list
                transported_terms.append(w_transport)

        # combine terms into a single tensor
        t_w = torch.cat(transported_terms, dim=2)

        # enlarge tensors by complex conjugates
        w_c, t_w_c = cconj(w), cconj(t_w)
        w = repack_x(w, w_c)
        t_w = repack_x(t_w, t_w_c)

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w = repack_x(w, unit_tensor)
            t_w = repack_x(t_w, unit_tensor)

        # perform multiplication and apply weights
        w = complex_einsum('bxvij, bxwjk -> bxvwik', w, t_w)
        w = einsum('uvw, bxvwijc -> bxuijc', self.weight, w)

        return repack_x(u, w)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims


class GCMConvSymmetric(torch.nn.Module):
    """
        A module that combines gauge covariant convolutions and gauge covariant bilinear layers
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True):
        super(GCMConvSymmetric, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements

        # initialize weights
        w_in_size = self.n_in
        t_w_size = self.n_in * (2 * (self.kernel_size - 1) * len(self.dims) + 1)
        w_out_size = self.n_out

        # double channels for complex conjugation
        w_in_size = 2 * w_in_size
        t_w_size = 2 * t_w_size

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        variance = 1.0 / (w_in_size * t_w_size)
        self.weight = torch.nn.Parameter(data=Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.nc)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack((self.unit_matrix_re, self.unit_matrix_im), dim=-1)

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))

        transported_terms = [w.clone()]
        # gather all terms along lattice axes up to kernel_size
        for orientation in [+1, -1]:
            for axis in range(len(self.dims)):
                w_transport = w.clone()
                for d in range(1, self.kernel_size):
                    # get transported terms
                    for step in range(self.dilation + 1):
                        w_transport = transport(u, w_transport, axis=axis, orientation=orientation, dims=self.dims)
                    # and add to list
                    transported_terms.append(w_transport)

        # combine terms into a single tensor
        t_w = torch.cat(transported_terms, dim=2)

        # enlarge tensors by complex conjugates
        w_c, t_w_c = cconj(w), cconj(t_w)
        w = repack_x(w, w_c)
        t_w = repack_x(t_w, t_w_c)
        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w = repack_x(w, unit_tensor)
            t_w = repack_x(t_w, unit_tensor)

        # perform multiplication and apply weights
        w = complex_einsum('bxvij, bxwjk -> bxvwik', w, t_w)
        w = einsum('uvw, bxvwijc -> bxuijc', self.weight, w)

        return repack_x(u, w)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims


class GCTrace(torch.nn.Module):
    """
        Computes the trace of Wilson loops.
        This renders the output gauge-invariant.
    """
    def __init__(self, dims):
        self.update_dims(dims)
        super(GCTrace, self).__init__()

    def forward(self, x):
        u, w = unpack_x(x, len(self.dims))
        tr = einsum('bxwiic -> bxwc', w)

        return tr

    def update_dims(self, dims):
        self.dims = dims


"""
    Various helper functions
"""


def unpack_x(x, n_dims):
    u = x[:, :, 0:n_dims]
    w = x[:, :, n_dims:]
    return u, w


def repack_x(u, w):
    x = torch.cat((u, w), dim=2)
    return x


def cconj(w):
    w = w.transpose(dim0=3, dim1=4).clone()
    w[:, :, :, :, :, 1] = -w[:, :, :, :, :, 1]
    return w


def transport(u, w, axis, orientation, dims):
    ws = shift(w, axis, -orientation, dims)

    # select links of appropriate axis
    ua = u.select(dim=2, index=axis)

    if orientation > 0:
        wt = complex_einsum('bxij,bxwjk -> bxwik', ua, ws)
        wt = complex_einsum('bxwij,bxkj -> bxwik', wt, ua, conj_b=-1)

    else:
        # apply shift
        ua = shift(ua, axis, +1, dims)
        wt = complex_einsum('bxji,bxwjk -> bxwik', ua, ws, conj_a=-1)
        wt = complex_einsum('bxwij,bxjk -> bxwik', wt, ua)

    return wt


def shift(a, axis, orientation, dims):
    # tensor layout size: [B, N^D, ..., N_C, N_C, 2]
    # add +1 to axis to skip batch dimension

    # old and new shapes for torch.roll
    o_s = tuple(a.shape)
    n_s = (o_s[0], *dims, *tuple(a.shape[2:]))

    a_shift = roll(a.view(*n_s), orientation, axis + 1).view(*o_s)

    return a_shift


def complex_bmm(a, b, nc=2, conj_a=1, conj_b=1):
    a, b = a.view(-1, nc, nc, 2), b.view(-1, nc, nc, 2)
    shape = a.shape[:-4]
    aR, aI = a.select(-1, 0), a.select(-1, 1)
    bR, bI = b.select(-1, 0), b.select(-1, 1)

    if conj_a == -1:
        aI = - torch.transpose(aI, dim0=1, dim1=2)
        aR = torch.transpose(aR, dim0=1, dim1=2)

    if conj_b == -1:
        bI = - torch.transpose(bI, dim0=1, dim1=2)
        bR = torch.transpose(bR, dim0=1, dim1=2)

    c = torch.zeros_like(a, dtype=a.device)
    c[:, :, :, 0] = aR.bmm(bR) - aI.bmm(bI)
    c[:, :, :, 1] = aR.bmm(bI) + aI.bmm(bR)
    c = c.view((*shape, nc, nc, 2))
    return c


def complex_einsum(pattern, a, b, conj_a=1, conj_b=1):
    aR, aI = a.select(-1, 0), conj_a * a.select(-1, 1)
    bR, bI = b.select(-1, 0), conj_b * b.select(-1, 1)

    cR = einsum(pattern, aR, bR) - einsum(pattern, aI, bI)
    cI = einsum(pattern, aR, bI) + einsum(pattern, aI, bR)
    c = torch.stack((cR, cI), dim=-1)
    return c


def update_kernel_size(kernel_size):
    new_kernel_size = kernel_size
    if type(kernel_size) == int:
        new_kernel_size = [kernel_size, kernel_size]
    elif type(kernel_size) == list:
        if len(kernel_size) == 1:
            new_kernel_size = [kernel_size[0], kernel_size[0]]
    return new_kernel_size


class CConv2d(torch.nn.Module):
    """
        A wrapper for Conv2d which correctly implements circular padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(CConv2d, self).__init__()

        # Conv2d without padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

        # padding size
        self.padding = []

        # Check and fix size of kernel_size
        kernel_size = update_kernel_size(kernel_size)

        # for some reason `padding` is in reverse order compared to kernel_size (see also pytorch repository)
        for k in reversed(kernel_size):
            if k % 2 == 0:
                # even kernel
                self.padding.append((k - 1) // 2)
                self.padding.append(k // 2)
            else:
                # odd kernel)
                self.padding.append(k // 2)
                self.padding.append(k // 2)

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, self.padding, mode='circular')
        x_conv = self.conv(x_pad)
        return x_conv

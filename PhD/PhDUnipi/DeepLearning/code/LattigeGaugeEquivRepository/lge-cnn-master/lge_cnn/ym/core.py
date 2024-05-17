import numpy as np
from lge_cnn.ym.numba_target import use_cuda, myjit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

if use_cuda:
    import numba.cuda as cuda
import lge_cnn.ym.su as su
import lge_cnn.ym.lattice as l
import math
import numba

pi = np.pi


class Simulation:
    def __init__(self, dims, beta):
        self.beta = beta
        self.dims = np.array(dims, dtype=np.int)
        self.acc = np.append(np.cumprod(self.dims[::-1])[::-1], 1)
        self.n_dims = len(dims)
        self.sites = np.int(np.prod(dims))
        self.u = np.zeros((self.sites, self.n_dims, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.u[:, :, 0] = 1.0
        self.u_swap = np.zeros((self.sites, self.n_dims, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        self.d_u = self.u
        self.d_u_swap = self.u_swap
        self.d_dims = self.dims
        self.d_acc = self.acc

        # polakov loops (traced)
        self.p = np.zeros(self.sites // dims[0], dtype=su.GROUP_TYPE_REAL)
        self.d_p = self.p

        # wilson loops
        self.n_w = self.n_dims * (self.n_dims - 1) // 2
        self.w = np.zeros((self.sites, self.n_w, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE_REAL)
        self.d_w = self.w

        # larger wilson loops (traced)
        self.wl = np.zeros((self.sites, self.n_w), dtype=su.GROUP_TYPE_REAL)
        self.d_wl = self.wl

        self.threads_per_block = 256
        self.blocks = math.ceil(self.sites / self.threads_per_block)
        self.rng = create_xoroshiro128p_states(self.threads_per_block * self.blocks, seed=np.random.randint(1000000))
        self.copy_to_device()

    def copy_to_device(self):
        self.d_u = cuda.to_device(self.u)
        self.d_u_swap = cuda.to_device(self.u_swap)
        self.d_dims = cuda.to_device(self.dims)
        self.d_acc = cuda.to_device(self.acc)
        self.d_w = cuda.to_device(self.w)

    def copy_to_host(self):
        self.d_u.copy_to_host(self.u)
        self.d_u_swap.copy_to_host(self.u_swap)
        self.d_dims.copy_to_host(self.dims)
        self.d_acc.copy_to_host(self.acc)
        self.d_w.copy_to_host(self.w)

    def apply_to_swap(self):
        copy_u_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.d_u_swap, self.d_dims)

    def swap(self):
        self.d_u, self.d_u_swap = self.d_u_swap, self.d_u

    def init(self, steps=10, use_flips=True):
        init_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.rng, steps, self.d_dims, use_flips)

    def metropolis(self, steps, updates_per_link=10, amplitude=0.5):
        for i in range(steps):
            for d in range(len(self.dims)):
                for cell_type in [0, 1]:
                    metropolis_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.rng, self.beta,
                                                                           cell_type, d, updates_per_link, amplitude, self.d_dims,
                                                                           self.d_acc)
                    numba.cuda.synchronize()

    def cooling(self, steps):
        for i in range(steps):
            for d in range(len(self.dims)):
                for cell_type in [0, 1]:
                    cooling_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, cell_type, d, self.d_dims, self.d_acc)
                    numba.cuda.synchronize()

    def polyakov(self):
        trP = np.zeros((self.sites // self.dims[0]), dtype=su.GROUP_TYPE)
        d_trP = cuda.to_device(trP)
        polyakov_kernel[self.blocks, self.threads_per_block](self.sites // self.dims[0], self.d_u, d_trP,
                                                             self.d_dims, self.d_acc)
        d_trP.copy_to_host(trP)
        return trP

    def normalize(self):
        normalize_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.d_dims)

    def wilson(self):
        wilson_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, self.d_w, self.d_dims, self.d_acc)

    def wilson_large(self, mu, nu, Lmu, Lnu):
        trW = np.zeros((self.sites), dtype=su.GROUP_TYPE_COMPLEX)
        d_trW = cuda.to_device(trW)
        wilson_large_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_trW, mu, nu, Lmu, Lnu, self.d_dims, self.d_acc)
        d_trW.copy_to_host(trW)
        return trW

    def topological_charge(self, mode=0):
        q = np.zeros((self.sites), dtype=su.GROUP_TYPE_REAL)
        d_q = cuda.to_device(q)
        topological_charge_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_q, self.d_dims, self.d_acc, mode)
        d_q.copy_to_host(q)
        return q

    def get_config(self):
        out_w = np.zeros((self.sites, self.n_w, su.NC, su.NC), dtype=su.GROUP_TYPE_COMPLEX)
        d_out_w = cuda.to_device(out_w)
        to_matrix_kernel[self.blocks, self.threads_per_block](self.sites, self.n_w, self.d_w, d_out_w)
        d_out_w.copy_to_host(out_w)

        out_u = np.zeros((self.sites, self.n_dims, su.NC, su.NC), dtype=su.GROUP_TYPE_COMPLEX)
        d_out_u = cuda.to_device(out_u)
        to_matrix_kernel[self.blocks, self.threads_per_block](self.sites, self.n_dims, self.d_u, d_out_u)
        d_out_u.copy_to_host(out_u)

        return out_u, out_w

    def load_config(self, u_matrices):
        d_u_matrices = cuda.to_device(u_matrices)
        to_repr_kernel[self.blocks, self.threads_per_block](self.sites, self.n_dims, d_u_matrices, self.d_u)

    def random_gauge_transform(self, steps=10, amplitude=1.0):
        v = np.zeros((self.sites, su.GROUP_ELEMENTS), dtype=su.GROUP_TYPE)
        d_v = cuda.to_device(v)
        random_gauge_transformation_kernel[self.blocks, self.threads_per_block](self.sites, self.rng, steps, amplitude, d_v)
        gauge_transform_kernel[self.blocks, self.threads_per_block](self.sites, self.d_u, d_v, self.d_dims, self.d_acc)

@cuda.jit
def copy_u_kernel(n, u, u_swap, dims):
    xi = cuda.grid(1)
    if xi < n:
        for d in range(len(dims)):
            for i in range(su.GROUP_ELEMENTS):
                u_swap[xi, d, i] = u[xi, d, i]

@cuda.jit
def init_kernel(n, u, rng, steps, dims, use_flips=True):
    """
    Initialization routine which randomizes all gauge links (warm start). If steps = 0, then all links are set to
    unit matrices.

    :param n:       number of lattice sizes
    :param u:       gauge link array
    :param rng:     random number states
    :param steps:   randomization steps per link
    :param dims:    lattice size array
    :param use_flips:   option to flip initial links from 1 to -1
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        X_vals = numba.cuda.local.array(su.ALGEBRA_ELEMENTS, numba.float32)
        for d in range(len(dims)):
            su.store(u[xi, d], su.unit())

            if use_flips:
                r = xoroshiro128p_uniform_float32(rng, xi)
                if r < 0.5:
                    su.store(u[xi, d], su.mul_s(u[xi, d], -1))

            for i in range(steps):
                for j in range(su.ALGEBRA_ELEMENTS):
                    X_vals[j] = xoroshiro128p_normal_float32(rng, xi)
                X = su.mexp(su.get_algebra_element(X_vals))
                su.store(u[xi, d], su.mul(X, u[xi, d]))


@cuda.jit
def normalize_kernel(n, u, dims):
    """
    Normalizes all gauge links, i.e. repairs unitarity and determinant.

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param dims:    lattice size array
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        for d in range(len(dims)):
            su.normalize(u[xi, d])


@cuda.jit
def metropolis_kernel(n, u, rng, beta, checkerboard_mode, d, updates, amplitude, dims, acc):
    """
    Performs a single metropolis sweep over the lattice in a checkerboard pattern.

    :param n:                   number of lattice sites
    :param u:                   gauge link array
    :param rng:                 random number states
    :param beta:                coupling constant
    :param checkerboard_mode:   checkerboard pattern: 'white' or 'black' cells
    :param updates:             number of consecutive updates per link
    :param dims:                lattice size array
    :param acc:                 cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        X_components = numba.cuda.local.array(su.ALGEBRA_ELEMENTS, numba.float32)
        if checkerboard(xi, dims) == checkerboard_mode:
            staples = l.staple_sum(xi, d, u, dims, acc)

            # compute previous plaquette sum
            P0 = su.mul(u[xi, d], staples)

            for k in range(updates):
                # generate updated link
                for j in range(su.ALGEBRA_ELEMENTS):
                    X_components[j] = amplitude * xoroshiro128p_normal_float32(rng, xi)
                X = su.mexp(su.get_algebra_element(X_components))
                new_u = su.mul(X, u[xi, d])

                # compute updated plaquette sum
                P1 = su.mul(new_u, staples)

                # compute action difference
                delta_S = - beta / su.NC * (su.tr(P1).real - su.tr(P0).real)

                # propose update
                r = xoroshiro128p_uniform_float32(rng, xi)
                if r <= math.exp(- delta_S):
                    su.store(u[xi, d], new_u)
                    P0 = P1


@cuda.jit
def cooling_kernel(n, u, checkerboard_mode, d, dims, acc):
    xi = cuda.grid(1)
    if xi < n:
        if checkerboard(xi, dims) == checkerboard_mode:
            staples = l.staple_sum(xi, d, u, dims, acc)
            staples = su.dagger(staples)
            det_staples = su.det(staples)
            new_u = su.mul_s(staples, 1.0 / det_staples)
            su.store(u[xi, d], new_u)

@myjit
def checkerboard(xi, dims):
    """
    Tests whether a lattice site is 'white' or 'black'

    :param xi:      lattice index
    :param dims:    lattice size array
    :return:        0 or 1 depending on 'white' or 'black'
    """
    res = 0
    x0 = xi
    for d in range(len(dims) - 1, -1, -1):
        cur_pos = x0 % dims[d]
        res += cur_pos
        x0 -= cur_pos
        x0 /= dims[d]
    return res % 2


@cuda.jit
def polyakov_kernel(n, u, p, dims, acc):
    """
    Computes the Polyakov loop trace for each spatial lattice site.

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param p:       output array for polyakov loop trace
    :param dims:    lattice size array
    :param acc:     cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        # xi only iterates through spatial coordinates
        x0 = xi
        P = su.load(u[x0, 0])
        for t in range(dims[0]-1):
            x0 = l.shift(x0, 0, +1, dims, acc)
            P = su.mul(P, u[x0, 0])
        p[xi] = su.tr(P).real / su.NC


@cuda.jit
def wilson_kernel(n, u, w, dims, acc):
    """
    Computes all 1x1 Wilson loops on the lattice

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param w:       output array for wilson loops
    :param dims:    lattice size array
    :param acc:     cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        w_i = 0
        for i in range(len(dims)):
            for j in range(i):
                su.store(w[xi, w_i], l.plaq(u, xi, i, j, 1, 1, dims, acc))
                w_i += 1


@cuda.jit
def wilson_large_kernel(n, u, w, mu, nu, Lmu, Lnu, dims, acc):
    """
    Computes all Lmu x Lnu Wilson loops on the lattice

    :param n:       number of lattice sites
    :param u:       gauge link array
    :param w:       output array for wilson loops
    :param dims:    lattice size array
    :param acc:     cumulative product of lattice sizes
    :return:
    """
    xi = cuda.grid(1)
    if xi < n:
        x0 = xi

        x1 = x0
        U = su.unit()
        for x in range(Lmu):
            U = su.mul(U, u[x1, mu])
            x1 = l.shift(x1, mu, +1, dims, acc)

        x2 = x1
        for y in range(Lnu):
            U = su.mul(U, u[x2, nu])
            x2 = l.shift(x2, nu, +1, dims, acc)

        x3 = x2
        for x in range(Lmu):
            x3 = l.shift(x3, mu, -1, dims, acc)
            U = su.mul(U, su.dagger(u[x3, mu]))

        x4 = x3
        for y in range(Lnu):
            x4 = l.shift(x4, nu, -1, dims, acc)
            U = su.mul(U, su.dagger(u[x4, nu]))

        w[xi] = su.tr(U) / su.NC


@cuda.jit
def to_matrix_kernel(n, num_components, arr, out):
    """
    Converts an array 'arr' of parametrized SU(N_c) matrices to
    complex N_c x N_c matrices (fundamental representation).
    """
    x = cuda.grid(1)
    if x < n:
        for c in range(num_components):
            a = arr[x, c]
            c_matrix = su.to_matrix(a)
            for i in range(su.NC):
                for j in range(su.NC):
                    out[x, c, i, j] = c_matrix[i * su.NC + j]


@cuda.jit
def to_repr_kernel(n, num_components, arr, out):
    """
    Converts an array 'arr' of complex N_c x N_c matrices (fundamental
    representation, flattened) to parametrized SU(N_c) matrices.
    """
    x = cuda.grid(1)
    if x < n:
        for c in range(num_components):
            matrix = arr[x, c]
            repr = su.to_repr(matrix)
            su.store(out[x, c], repr)


@cuda.jit
def random_gauge_transformation_kernel(n, rng, steps, amplitude, v):
    """
    Generates a random gauge transformation and stores it in 'v'
    """
    xi = cuda.grid(1)
    if xi < n:
        X_vals = numba.cuda.local.array(su.ALGEBRA_ELEMENTS, numba.float32)
        su.store(v[xi], su.unit())
        for i in range(steps):
            for j in range(su.ALGEBRA_ELEMENTS):
                X_vals[j] = amplitude * xoroshiro128p_normal_float32(rng, xi)
            X = su.mexp(su.get_algebra_element(X_vals))
            su.store(v[xi], su.mul(X, v[xi]))


@cuda.jit
def gauge_transform_kernel(n, u, v, dims, acc):
    """
    Applies the gauge transformation 'v' to the links 'u'
    """
    xi = cuda.grid(1)
    if xi < n:
        for i in range(len(dims)):
            xs = l.shift(xi, i, +1, dims, acc)
            new_u = su.mul(v[xi], u[xi, i])
            new_u = su.mul(new_u, su.dagger(v[xs]))
            su.store(u[xi, i], new_u)

@myjit
def lc_2(a, b):
    """
    Levi-Civita in 2D
    """
    if a == b:
        return 0
    elif a < b:
        return -1
    else:
        return +1

@myjit
def lc_4(a, b, c, d):
    """
    Levi-Civita in 4D
    """
    return lc_2(b, a) * lc_2(c, b) * lc_2(c, a) * lc_2(d, c) * lc_2(d, b) * lc_2(d, a)

@cuda.jit
def topological_charge_kernel(n, u, q, dims, acc, mode):
    """
    Computes the topological charge density q

    mode 0: plaquette discretization
    mode 1: clover discretization
    """
    xi = cuda.grid(1)
    if xi < n:
        local_q = 0.0j
        D = len(dims)
        for a in range(D):
            for b in range(D):
                for c in range(D):
                    for d in range(D):
                        levi_civita = lc_4(a, b, c, d)
                        if levi_civita != 0:
                            # compute C_ab and C_cd
                            if mode == 0:
                                tr_c_abcd = l.c_plaq_abcd(u, xi, a, b, c, d, dims, acc)
                            elif mode == 1:
                                tr_c_abcd = l.c_clov_abcd(u, xi, a, b, c, d, dims, acc)
                            elif mode == 2:
                                tr_c_abcd = l.c_plaq2_abcd(u, xi, a, b, c, d, dims, acc)
                            else:
                                return None

                            local_q += levi_civita * tr_c_abcd
        #q[xi] = local_q.real / (32.0 * pi ** 2) # old normalization
        q[xi] = local_q.real


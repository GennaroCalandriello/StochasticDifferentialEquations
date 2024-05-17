import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su2"
os.environ["PRECISION"] = "single"

import sys
sys.path.append('..')

import argparse
import numpy as np
import h5py
from tqdm import tqdm

from lge_cnn.ym.core import Simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path(s) to output file(s)
    parser.add_argument("--paths", type=str, nargs='+', required=True)

    # lattice and monte carlo options
    parser.add_argument("--nums", "-n", type=int, nargs='+', required=True)
    parser.add_argument("--sweeps", "-s", type=int, required=True)
    parser.add_argument("--warmup", "-w", type=int, required=True)
    parser.add_argument("--beta_min", "-bmin", type=float, required=True)
    parser.add_argument("--beta_max", "-bmax", type=float, required=True)
    parser.add_argument("--beta_steps", "-bs", type=int, required=True)
    parser.add_argument("--dims", "-d", type=int, nargs='+', required=True)

    # optional labels and observables
    parser.add_argument("--loops", "-l", type=int, nargs='+')
    parser.add_argument("--loop_axes", "-la", type=int, nargs='+')
    parser.add_argument("--polyakov", "-p", action="store_true")
    parser.add_argument("--charge_plaq", "-qp", action="store_true")
    parser.add_argument("--charge_clov", "-qc", action="store_true")

    # parse arguments
    args = parser.parse_args()

    # iterate through paths and sample numbers

    pbar0 = tqdm(total=len(args.paths), position=0)

    for path, num in zip(args.paths, args.nums):
        # set progress description
        pbar0.set_description("Current file: {}".format(path))

        # run monte carlo simulation
        with h5py.File(path, "w") as f:
            # beta range
            betas = np.linspace(args.beta_min, args.beta_max, args.beta_steps)

            # create required attributes and datasets in the hdf5 file
            num_betas = len(betas)
            total_samples = num_betas * num
            D = len(args.dims)
            W = D * (D - 1) // 2
            NC = 2
            u_shape = (np.prod(args.dims), D, NC, NC)
            w_shape = (np.prod(args.dims), W, NC, NC)
            f.create_dataset('beta', (total_samples, ), dtype='float32')
            f.create_dataset('u', (total_samples, *u_shape), dtype='complex64')
            f.create_dataset('w', (total_samples, *w_shape), dtype='complex64')
            f.create_dataset('dims', data=np.array(args.dims))


            if args.loops is not None:
                for size in args.loops:
                    f.create_dataset('trW_{}'.format(size), (total_samples, np.prod(args.dims)), dtype='complex64')

            if args.polyakov:
                f.create_dataset('trP'.format(size), (total_samples, np.prod(args.dims) // args.dims[0]),
                                 dtype='float32')

            if args.charge_plaq:
                f.create_dataset('QP'.format(size), (total_samples, np.prod(args.dims)), dtype='float32')

            if args.charge_clov:
                f.create_dataset('QC'.format(size), (total_samples, np.prod(args.dims)), dtype='float32')

            pbar1 = tqdm(total = total_samples, position=1)
            for beta_index, beta in enumerate(betas):
                # set progress description
                pbar1.set_description("Beta = {:3.2f}".format(beta))

                s = Simulation(args.dims, beta)
                s.init(steps=5, use_flips=True)
                s.metropolis(args.warmup)

                for i in range(num):
                    # determine sample index
                    sample_index = beta_index * num + i

                    # perform monte carlo sweeps
                    s.metropolis(args.sweeps)

                    # save coupling constant
                    f['beta'][sample_index] = beta

                    # compute 1x1 wilson loops and generate field configuration in terms of complex matrixes
                    s.wilson()
                    u, w = s.get_config()
                    f['u'][sample_index] = u
                    f['w'][sample_index] = w

                    # compute observables
                    if args.loops is not None:
                        mu, nu = args.loop_axes
                        for size in args.loops:
                            trW = s.wilson_large(mu, nu, size, size)
                            f['trW_{}'.format(size)][sample_index] = trW

                    if args.polyakov:
                        trP = s.polyakov()
                        f['trP'][sample_index] = trP

                    if args.charge_plaq:
                        qp = s.topological_charge(mode=0)
                        f['QP'][sample_index] = qp

                    if args.charge_clov:
                        qc = s.topological_charge(mode=1)
                        f['QC'][sample_index] = qc

                    # advance progress bar
                    pbar1.update(1)

            # close progress bar
            pbar1.close()

        # advance progress bar
        pbar0.update(1)

    # close progress bar
    pbar0.close()
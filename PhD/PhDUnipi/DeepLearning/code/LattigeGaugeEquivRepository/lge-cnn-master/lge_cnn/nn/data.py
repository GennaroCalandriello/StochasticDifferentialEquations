from torch.utils.data import Dataset
import torch
from torch import Tensor
import h5py
import numpy as np


class YMDatasetHDF5(Dataset):
    def __init__(self, filename, mode_in='uw', mode_out='trW_2', use_idx=True):
        self.filename = filename
        self.f = None
        with h5py.File(self.filename, 'r') as file:
            self.num_samples = file['u'].shape[0]
            self.dims = np.array(file['dims'])

        self.beta = None
        self.mode_in, self.mode_out = mode_in, mode_out
        self.use_idx = use_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
            Multiprocessing fix for hdf5 files based on
            https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        """
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')
            self.beta = self.f['beta'][:]

        # input mode
        if self.mode_in == 'uw':
            # format for x: combine u and w including real/imag in single tensor
            # layout: (batch_dim, lattice, channels, matrix structure, real/imag)
            # channels includes D links and N_W wilson loops
            u = self.f['u'][idx]
            u = torch.stack((Tensor(u.real), Tensor(u.imag)), dim=-1)
            w = self.f['w'][idx]
            w = torch.stack((Tensor(w.real), Tensor(w.imag)), dim=-1)
            x = torch.cat((u, w), dim=1)
        else:
            # put other possible input modes here
            print("Unknown mode_in for YMDatasetHDF5.")

        # output modes
        if self.mode_out == 'trW_1':
            y = self.f['trW_1'][idx].real
        elif self.mode_out == 'trW_1x2':
            y = self.f['trW_1x2'][idx].real
        elif self.mode_out == 'trW_2':
            y = self.f['trW_2'][idx].real
        elif self.mode_out == 'trW_4':
            y = self.f['trW_4'][idx].real
        elif self.mode_out == 'trP':
            y = self.f['trP'][idx]
        elif self.mode_out == 'QP':
            y = self.f['QP'][idx]
        elif self.mode_out == 'QC':
            y = self.f['QC'][idx]
        else:
            # put other possible output modes here
            print("Unknown mode_out for YMDatasetHDF5.")

        if self.use_idx:
            return x, y, idx
        else:
            return x, y

    def get_beta(self, idx):
        if self.f is None:
            self.f = h5py.File(self.filename, 'r')
        beta = self.f['beta'][:]
        return beta[idx]

    def close(self):
        if self.f is not None:
            self.f.close()

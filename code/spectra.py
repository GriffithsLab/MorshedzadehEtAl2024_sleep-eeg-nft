#!/usr/bin/env python3

# IMPORTS
import numpy as np, pandas as pd
import os, sys, time, glob
import scipy.io.savemat as savemat, scipy.io.loadmat as loadmat
import mne
from IPython.display import Markdown, display, Latex, HTML
from tqdm.notebook import tqdm,trange

# define a class to take the epochs and hypnogram, transform into fitting material, fit, store results.
class Spectra:
    def __init__(self, epochs, hypnogram, fname=None):
        self.epochs = epochs
        self.hypnogram = hypnogram
        self.epochs_inds = self.epochs.to_data_frame(index='epoch').index.unique().to_numpy()
        self.sfreq = int(self.epochs.info['sfreq'])
        # in case fname is included, override the filename in epochs.info
        if fname is None:
            self.fname = epochs.info['filename']
        

    def get_spectra(self, save=False):
        """_summary_
        """
        welchargs = dict(fmin=0.5, # arguments for Welch's method PSD calculation
                    fmax=45,
                    n_fft=4*self.sfreq,
                    n_per_seg=30*self.sfreq,
                    n_overlap=self.sfreq)
        
        # compute PSD using MNE
        spec = self.epochs.compute_psd(**welchargs).get_data(return_freqs=True)
        # create a custom DataFrame with the PSD data
        df_psd=pd.DataFrame(spec[0].reshape(-1, spec[0].shape[-1]),
                        columns=spec[1],
                        index = self.epochs_inds,
                        )
        return(df_psd)

    def save_tfs(self):
        """_summary_
        """
        # save the tfs file using scipy.io.savemat
        file = epochs.info['filename']
        tfs_fname = os.path.splitext(file)[0] + 'tfs.mat'
        tfs_file = os.path.join(os.path.dirname(file), tfs_fname)

        # create the tfs file
        colheaders = self.epochs.ch_names
        f = self.df_psd.columns.astype(float).values.reshape((1, len(self.df_psd.columns)))
        p = self.df_psd.values
        ptrap=np.trapz(p, f)
        p /= ptrap.reshape(len(ptrap), 1)

        t = np.arange(p.shape[1])+1
        nspec = np.ones_like(t, dtype=int)

        state_score = np.zeros_like(t)
        state_str = self.hypnogram

        mat = {"colheaders": colheaders,
            "s": np.double(p),
            "f": np.double(f),
            "t": np.double(t),
            "nspec": np.double(nspec),
            "state_score": state_score,
            "state_str": state_str,
            # "start_idx": 1,
        }

        # save the tfs file
        print('saving `mat` file to', tfs_file)
        savemat(file_name=tfs_file, mdict=mat, format='5')

        return(tfs_file)        
        

    def fit(self, mode='local'):
        """_summary_
        """
        # fit the model
        pass

# Main:
def main():
    return([])

# if main:
if __name__ == "__main__":
    pass
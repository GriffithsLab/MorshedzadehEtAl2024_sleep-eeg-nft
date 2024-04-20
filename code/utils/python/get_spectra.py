#!/usr/bin/env python3

# IMPORTS
import numpy as np, pandas as pd
import os, sys, time, glob
import scipy.io.savemat as savemat, scipy.io.loadmat as loadmat
import mne
from IPython.display import Markdown, display, Latex, HTML
from tqdm.notebook import tqdm,trange

# define a class to read raw data, get spectra, and save spectra
class Spectra:
    def __init__(self, fname):
        self.fname = fname
        self.raw = self.read_raw(fname)
        self.spectra = self.get_spectra()
        self.save_spectra()
    
    def read_raw(self, fname):
        raw = mne.io.read_raw_fif(fname)
        return(raw)

def read_raw(fname):
    raw = mne.io.read_raw_fif(fname)
    return(raw)

def get_epochs(raw, hypnogram, chan_choice, save=False, drop=False):
    raw.set_annotations(hypnogram)
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=30, baseline=None)
    epochs_inds = epochs.to_data_frame(index='epoch').index.unique().to_numpy()
    ext = os.path.splitext(raw.filenames[0])[1]

    if drop: # drop epochs with std > 1, in case drop is True
        epochs_clean = epochs.copy()
        epochs_clean.load_data()
        data = epochs_clean.get_data()[:,epochs.ch_names.index(chain_choice),:]
        stdev = data.std()
        bads = data.std(axis=1)>stdev
        epochs_clean.drop(bads, reason='>1 x std', verbose=verbose)
        epochs_clean_inds = epochs_clean.to_data_frame(index='epoch').index.unique().to_numpy()
        return(epochs_clean, epochs_clean_inds)
    else:
        return(epochs, epochs_inds)


# Functions
def get_spectra(epochs, save=False):
    spec = epochs.compute_psd(**welchargs).get_data(return_freqs=True)
    df_psd=pd.concat([pd.DataFrame(spec[0].reshape(-1, spec[0].shape[-1]),
                    columns=spec[1],
                    index = epochs_all_clean_inds,
                    ) for i in range(1)], keys = epochs_all_clean['raw_non'], axis=1)
    
    ### WIP: >>>
    df_psd.columns.names=['channel', 'freq']
    df_psd.index.names=['epoch']
    
    hypno_df=pd.read_csv(fs.hypnos[i], index_col=0)
    hypno_df.index.names=['epoch']
    hypno_df.columns=pd.MultiIndex.from_tuples([('','')])
    _df = pd.concat([df_psd, hypno_df], keys=['normal','stage'], axis=1)
    _df.columns.names=['preproc', 'channel', 'freq']
    _df = _df.drop(columns=[('stage','','')]).dropna(how='all', axis=0).join(_df[('stage','','')])
    ### <<<< WIP

    ### NOT DONE: >>>
    if save:
        fname=fs.sid[sno]
        tfs_file=os.path.join(GP, 'data', dataset, 'tfs', fname+f'_{chan_choice[0]}_tfs.mat')

        colheaders = chan_choice


        f = psd_df.columns.astype(float).values.reshape((1, len(psd_df.columns)))
        p = psd_df.loc[sno].values
        ptrap=np.trapz(p, f)
        p /= ptrap.reshape(len(ptrap), 1)
        print(f'{sno} - file = {fname}')

        t = np.arange(p.shape[1])+1
        nspec = np.ones_like(t, dtype=int)

        state_score = np.zeros_like(t)
        state_str = stages.loc[sno].values

        mat = {"colheaders": colheaders,
            "s": np.double(p),
            "f": np.double(f),
            "t": np.double(t),
            "nspec": np.double(nspec),
            "state_score": state_score,
            "state_str": state_str,
            # "start_idx": 1,
        } 

        # print('saving `mat` file to', tfs_file)

        savemat(file_name=tfs_file, mdict=mat, format='5')
        ### NOT DONE: >>>

    return(df_psd)

# Main:
def main():
    return([])

# if main:
if __name__ == "__main__":
    pass
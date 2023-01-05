import concurrent
import random

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from tqdm import tqdm


def fp_from_smi(smi, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, fingerprint)
    return fingerprint


def compute_fingerprint(smiles, verbose=True):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fingerprints = list(tqdm(executor.map(fp_from_smi, smiles, chunksize=1000), total=len(smiles)))
    return fingerprints

def cutoff_func_factor(cutoff=0.5):
    def cutoff_(x):
        return 1 if x > cutoff else 0
    return cutoff_


# Custom PyTorch dataset for SMILES strings and activity scores
class SMILESDataset(Dataset):
    def __init__(self, csv_file=None, df=None, cutoff=1, pre_compute=False, smiles_key='smiles', score_key='score', test=False):
        assert csv_file is not None or df is not None
        if csv_file is not None:
            df = pd.read_csv(csv_file, low_memory=False, engine='c')
        self.pre_compute = pre_compute
        self.test = test
        self.smiles_key = smiles_key
        self.score_key = score_key
        self.xdata = np.array(df[self.smiles_key])
        self.ydata = np.array(df[self.score_key], dtype=np.float32)
        self.cutoff = cutoff
        self.cutoff_value = np.percentile(self.ydata, self.cutoff)
        self.xfeaturizer = fp_from_smi
        self.ydata_cutoff = self.ydata < self.cutoff_value
        self.pos_idx = self.ydata_cutoff == 1
        self.neg_idx = self.ydata_cutoff == 0

        if pre_compute:
            self.smiles = self.xdata
            self.xdata = compute_fingerprint(self.xdata)
            self.xdata = np.stack(self.xdata, axis=0)
            self.xdata_neg_smiles = self.smiles[self.neg_idx]
            self.xdata_pos_smiles = self.smiles[self.pos_idx]

        self.xdata_pos = self.xdata[self.pos_idx]
        self.ydata_pos = self.ydata[self.pos_idx]
        self.xdata_neg = self.xdata[self.neg_idx]
        self.ydata_neg = self.ydata[self.neg_idx]
        self.ydata_cutoff_pos = self.ydata_cutoff[self.pos_idx]
        self.ydata_cutoff_neg = self.ydata_cutoff[self.neg_idx]
        self.nact = len(self.ydata_pos)
        self.ninact = len(self.ydata_neg)
        print(f'Total number of compounds in the TRAINING set: {len(self.ydata)}')
        print(f'Number of active compounds (1): {self.nact}; number of inactive compounds (0): {self.ninact}\n')

    def __len__(self):
        if not self.test:
            return min(self.nact, self.ninact)
        else:
            return len(self.ydata)

    def __getitem__(self, index, return_row_too=False):
        if not self.test:
            pos_sample_idx = index
            neg_sample_idx = random.randint(1, self.ninact // self.nact) * (self.ninact % (1+index))

            if not self.pre_compute:
                pos_smiles = self.xdata_pos[pos_sample_idx]
                pos_score = self.ydata_cutoff_pos[pos_sample_idx]
                neg_smiles = self.xdata_neg[neg_sample_idx]
                neg_score = self.ydata_cutoff_neg[neg_sample_idx]
                xfeat_pos = self.xfeaturizer(pos_smiles)
                xfeat_neg = self.xfeaturizer(neg_smiles)
            else:
                xfeat_pos = self.xdata_pos[pos_sample_idx]
                pos_score = self.ydata_cutoff_pos[pos_sample_idx]
                xfeat_neg = self.xdata_neg[neg_sample_idx]
                neg_score = self.ydata_cutoff_neg[neg_sample_idx]
                pos_smiles = self.xdata_neg_smiles[pos_sample_idx]
                neg_smiles = self.xdata_neg_smiles[neg_sample_idx]

            if not return_row_too:
                return xfeat_pos, xfeat_neg, pos_score, neg_score
            elif self.pre_compute:
                return xfeat_pos, xfeat_neg, pos_score, neg_score, pos_smiles, self.ydata_pos[pos_sample_idx], neg_smiles, self.ydata_neg[neg_sample_idx]
            else:
                return xfeat_pos, xfeat_neg, pos_score, neg_score, pos_smiles, self.ydata_pos[pos_sample_idx], neg_smiles, self.ydata_neg[neg_sample_idx]
        else:
            if not self.pre_compute:
                smiles = self.xdata[index]
                score = self.ydata_cutoff[index]
                xfeat = self.xfeaturizer(smiles)
                if not return_row_too:
                    return xfeat, score
                else:
                    return xfeat, score, smiles, self.ydata[index]
            else:
                xfeat = self.xdata[index]
                score = self.ydata_cutoff[index]
                smiles = self.smiles[index]
                if not return_row_too:
                    return xfeat, score
                else:
                    return xfeat, score, smiles, self.ydata[index]

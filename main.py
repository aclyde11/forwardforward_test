from argparse import ArgumentParser
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from data_loader import SMILESDataset
from train import train_dense, train_ff
import matplotlib.pyplot as plt
def get_args():
    device_choices = ['cpu']
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device_choices.append("mps")

    if torch.cuda.is_available():
        device_choices.append('cuda')

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='/Users/austin/Downloads/AmpC_screen_table.csv')
    parser.add_argument('--device', type=str, default='cpu', choices=device_choices)
    return parser.parse_args()

def main():
    args = get_args()
    df = pd.read_csv(args.data, low_memory=False, engine='c', nrows=1000000)
    print(f"Read in data with {len(df)} rows and headers {df.columns}")

    df_train, df_test = train_test_split(df, test_size=0.3)
    train_data = SMILESDataset(df=df_train, cutoff=10, pre_compute=True, smiles_key='smiles', score_key='dockscore')
    test_data = SMILESDataset(df=df_test, cutoff=10, test=True, pre_compute=True, smiles_key='smiles', score_key='dockscore')

    dense_model, dense_rocs,  dense_pred, dense_test = train_dense(train_data, test_data, device_str=args.device,
                input_dim=2048,
                num_iters=20,
                batch_size=32,
                lr=0.00025,
                dropout=0.05)

    ff_model, ff_rocs, ff_pred, ff_test = train_ff(train_data, test_data, device_str=args.device,
             input_dim=2048,
             num_iters=20,
             batch_size=32,
             lr=0.00025,
             dropout=0.05)


    dense_fpr, dense_tpr, dense_thresholds = roc_curve(dense_test, dense_pred)
    ff_fpr, ff_tpr, ff_thresholds = roc_curve(ff_test, ff_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(dense_rocs, lw=3, label='SimpleNet')
    ax1.plot(ff_rocs, lw=3, label='Forward-Forward')
    ax1.set_xlabel('Iteration', fontsize=20)
    ax2.set_ylabel('ROC-AUC', fontsize=20)

    ax2.plot(dense_fpr, dense_tpr, lw=3, label='SimpleNet')
    ax2.plot(ff_fpr, ff_tpr, lw=3, label="Forward-Forward")
    ax2.set_xlabel('false positive rate', fontsize=20)
    ax2.set_ylabel('true positive rate', fontsize=20)
    ax1.legend(fontsize=20)

    plt.show()

if __name__ == '__main__':
    main()
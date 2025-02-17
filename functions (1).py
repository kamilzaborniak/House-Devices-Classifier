import os
import glob
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from hmmlearn import hmm
from statistics import mean
from sklearn.model_selection import TimeSeriesSplit
import argparse
from IPython.display import display
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description = '#TODO')
    parser.add_argument('--train', help = 'File with training data', default = "house3_5devices_train.csv")
    #parser.add_argument('--test', help='source file', default="test_folder2")
    #parser.add_argument('--output', help='source file', default="results.txt")
    parser.add_argument('--test', help='File with data to test', default="test_folder")
    parser.add_argument('--output', help='File with result data', default="results.txt")
    args = parser.parse_args()

    return args.train, args.test, args.output
def DfOfCritHMM(sample, minNoComp, maxNoComp):
    sample = np.asarray(sample)
    df = pd.DataFrame({'N_components': [],
                       'AIC': [],
                       'BIC': [],
                       'LL': []})
    train_data, validation_data = sample[:12000], sample[12000:]
    for nc in range(minNoComp, maxNoComp + 1):
        models = []
        scores = []
        BestModel = None
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NC: ", nc, "\n")
        for rs in range(10):
            model = hmm.GaussianHMM(n_components=nc, random_state=rs)  # , random_state=rs)
            model.fit(train_data[:, None])
            models.append(model)
            scores.append(model.score(validation_data[:, None]))

        BestModel = models[np.argmax(scores)]
        ll = BestModel.score(validation_data[:, None])
        aic = BestModel.aic(validation_data[:, None])
        bic = BestModel.bic(validation_data[:, None])

        row = pd.DataFrame({'N_components': [nc],
                            'AIC': [aic],  # sum(list(map(lambda x, y: x*y, w, aic)))/sum(w),
                            'BIC': [bic],  # sum(list(map(lambda x, y: x*y, w, bic)))/sum(w),
                            'LL': [ll]  # sum(list(map(lambda x, y: x*y, w, ll)))/sum(w)
                            })
        df = pd.concat([df, row])
    return df  # , BestOfi

def FindBestModel(train, valid, n_components, no_rs_max=5):
    train = np.asarray(train)
    valid = np.asarray(valid)
    models = []
    scores = []
    for rs in range(no_rs_max):
        model = hmm.GaussianHMM(n_components=n_components, random_state=rs)  # , random_state=rs)
        model.fit(train[:, None])
        models.append(model)
        scores.append(model.score(valid[:, None]))
    BestModel = models[np.argmax(scores)]
    return BestModel

def TrainModelsForDevices(train, n_splits=5, no_rs_max=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    devNames = train.columns.tolist()[1:]
    no_stat = [5, 6, 9, 15, 9]
    models = []

    for dn in range(len(devNames)):
        train_col = train[devNames[dn]]
        #print(dn, "\t", devNames[dn])
        BestModel = None
        BestLL = None
        for train_index, valid_index in tscv.split(train_col):
            t_train = train_col[train_index]
            t_valid = train_col[valid_index]
            model = FindBestModel(train=t_train, valid=t_valid, n_components= no_stat[dn], no_rs_max=no_rs_max)
            score = model.score(np.asarray(t_valid)[:, None])
            if not BestLL or BestLL <= score:
                BestLL = score
                BestModel = model

        models.append(BestModel)
    return models
def ClassifierHMM(train, sample_to_class, models):
    devNames = train.columns.tolist()[1:]
    sample = np.asarray(sample_to_class)
    scores = [m.score(sample[:,None]) for m in models]
    result = devNames[np.argmax(scores)]

    return result

def LookingForBestNoComp():
    for devName in colNames[1:]:
        t = train[devName]
        print(devName)

        df = DfOfCritHMM(t, 2, 20)
        display(df)
        display(df.sort_values(by=["AIC"]).head(n=3))
        display(df.sort_values(by=["BIC"]).head(n=3))
        display(df.sort_values(by=["LL"]).tail(n=3))

        fig, ax = plt.subplots()
        ln1 = ax.plot(df["N_components"], df["AIC"], label="AIC", color="blue", marker="o")
        ln2 = ax.plot(df["N_components"], df["BIC"], label="BIC", color="green", marker="o")
        ax2 = ax.twinx()
        ln3 = ax2.plot(df["N_components"], df["LL"], label="LL", color="orange", marker="o")

        ax.legend(handles=ax.lines + ax2.lines)
        ax.set_title("Using AIC/BIC for Model Selection")
        ax.set_ylabel("Criterion Value (lower is better)")
        ax2.set_ylabel("LL (higher is better)")
        ax.set_xlabel("Number of HMM Components" + devName)
        fig.tight_layout()

        plt.show()

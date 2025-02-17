from functions import *

train, test, output = parse_arguments()

if __name__ == "__main__":

    test_files = glob.glob(os.path.join(test, "*.csv"))
    train = pd.read_csv(train)

    models = TrainModelsForDevices(train=train, n_splits=5, no_rs_max=10)
    devNames = train.columns.tolist()[1:]
    outDF = pd.DataFrame({'file': [],
                          'dev_classified': []})
    for f in test_files:
        sample = pd.read_csv(f)
        sample = sample['dev']
        cl = ClassifierHMM(train, sample, models=models)
        f_name = f[f.rfind('\\') + 1:]
        row = pd.DataFrame({'file': [f_name],
                            'dev_classified': [cl]})
        outDF = pd.concat([outDF, row])

    outDF.to_csv(output, sep=',', index=False, header=True)

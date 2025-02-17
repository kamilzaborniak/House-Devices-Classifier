from functions import *


colNames = train.columns.tolist()
#print(colNames[1])
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
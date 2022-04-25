from pywr.core import Model
from pywr.recorders import Recorder
from pywr.recorders._recorders import NodeRecorder
import pandas
import numpy as np


if __name__ == "__main__":

    m = Model.load("codegua.json")
    stats = m.run()
    print(stats)

    from matplotlib import pyplot as plt

    # print(m.recorders["catchment1_flow"].values())

    df = m.to_dataframe()
    # print(df.head())
    df.plot(subplots=True)
    plt.show()

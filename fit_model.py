import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def date_process(data):
    """
    :param data: original data
    :return: groups: dict type: key is date and the corresponding intraday volumn size
    """
    data_filter = data[data.isna().any(axis=1)]
    data_filter["reshape_time"] = pd.to_datetime(data_filter["date"] + data_filter["time"], format='%Y-%m-%d%H:%M:%S.%f')
    data_filter.index = data_filter["reshape_time"].values

    result = data_filter.resample("5T").sum()
    result = result.loc[(result!=0).any(axis=1)]

    result["time"] = result.index
    result["date"] = result["time"].dt.to_period("D")
    result = result.drop(columns = ["price", "time"])

    result.index = result["size"].values
    groups = result.groupby("date").groups

    return groups


def plot_intraday_volumn(groups, Tnum):
    """
    :param groups: the dict with day and volumn size
    :param Tnum: number of days shown in the plot
    """
    keys = list(groups.keys())

    plt.figure(0, figsize=(12, ))

    for i in np.arange(0, Tnum):
        key_ = keys[i]
        size = np.array(groups[key_])
        t = np.arange(0, 1, 1 / len(size))
        plt.plot(t, size/sum(size), label="Date"+str(i+1))

    plt.xlabel('time',  fontsize=17)
    plt.ylabel('Volumn', fontsize=17)
    plt.legend(fontsize = 12)
    plt.title("U-shape intraday volumn", fontsize = 23)
    plt.grid()
    plt.show()


def mean_std_relative_volumn(groups, Tnum):
    """
    :param groups: the dict with day and volumn size
    :param Tnum: number of days in the calculation
    :return: mean and std of relative volumn
    """
    keys = list(groups.keys())
    df_relative_volumn = pd.DataFrame()

    for i in np.arange(0, Tnum):
        key_ = keys[i]
        size = np.array(groups[key_])
        cum_size = np.cumsum(size)
        relative_size = cum_size / cum_size[-1]
        try:
            df_relative_volumn[str(key_)] = relative_size
        except:
            print("not match in date:" + str(key_))

    result = pd.DataFrame()
    result["mean"] = df_relative_volumn.mean(axis=1)
    result["std"] = df_relative_volumn.std(axis=1)
    result["time"] = np.arange(0, 1, 1/len(result)) + 1/len(result)

    return result







if __name__ == '__main__':

    trades0700_path = "/Users/huangzz/Desktop/capstone/project_code2/trades0700.csv"
    data = pd.read_csv(trades0700_path, sep=',')

    groups = date_process(data)
    keys = list(groups.keys())
    keys_num = len(keys)

    plot_intraday_volumn(groups, 4)

    mean_std = mean_std_relative_volumn(groups, 70)

    plt.figure(0, figsize=(10, 5))

    plt.subplot(121)
    plt.plot(mean_std.time.values, mean_std["mean"].values, label="60 day sample")
    plt.yscale('linear')
    plt.title('mean of relative volumn curve')
    plt.xlabel('time')
    plt.ylabel('Relative Volumn')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(mean_std.time.values, mean_std["std"].values, label="60 day sample")
    plt.yscale('linear')
    plt.title('std of relative volumn curve')
    plt.xlabel('time')
    plt.ylabel('Relative Volumn')
    plt.legend()
    plt.grid()
    plt.show()


















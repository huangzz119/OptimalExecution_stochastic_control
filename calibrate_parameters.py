import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def mean_func(x):
    """
    :param x: time interval
    :return: the mean of gamma bridge
    """
    return (a*(tf.pow(x, 3)) + b*(x*x) + (1-a-b)*x) / 1

def var_func(x):
    """
    :param x: time interval
    :return: the variance of gamma bridge
    """
    def alpha(x):
        return m * (a * (tf.pow(x, 3)) + b * (x * x) + (1 - a - b) * x)
    def beta(x):
        return m - alpha(x)
    var = (( alpha(x) * beta(x) )/( (alpha(x)+beta(x))**2 * (alpha(x)+beta(x)+1) ))
    return var

def model_mean_calculate(x, p, q):
    """
     :param x: time intervar
     :param a: model parameter
     :param b: model parameter
     :return: mean of model
     """
    return (p * (pow(x, 3)) + q * (x * x) + (1 - p - q) * x) / 1

def model_var_calculate(x, p, q, n):
    """
    :param x: time intervar
    :param a: model parameter
    :param b: model parameter
    :param m: model parameter
    :return: variance of model
    """
    def alpha_(x):
        return n * (p * (pow(x, 3)) + q * (x * x) + (1 - p - q) * x)
    def beta_(x):
        return n - alpha_(x)
    return ( (alpha_(x)*beta_(x)) / ((alpha_(x)+beta_(x))**2 * (alpha_(x)+beta_(x)+1) ))


def plot_result(t, real_mean, model_mean_value, real_var, model_var_value):

    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.plot(t, real_mean, label="60 day sample")
    plt.plot(t, model_mean_value, label="calibrate mean")
    plt.yscale('linear')
    plt.title('Mean of relative volumn curve')
    plt.xlabel('Time')
    plt.ylabel('Relative Volumn')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(t, real_var, label="60 day sample")
    plt.plot(t, model_var_value, label="calibrate variance")
    plt.yscale('linear')
    plt.title('Variance of relative volumn curve')
    plt.xlabel('Time')
    plt.ylabel('Relative Volumn')
    plt.legend()
    plt.grid()
    plt.savefig("mean_var_result.png")
    plt.show()


def plot_error(errors_set):
    plt.figure(0, figsize=(10, 8))
    plt.plot([np.mean(errors_set[i - 2:i]) for i in range(len(errors_set))])
    plt.xlabel('Epoch')
    plt.ylabel('Error value')
    plt.savefig("errors.png")
    plt.show()

if __name__ == '__main__':

    EVresults_path = "/home/zhuangby/project_code2/statistic_result.csv"
    EVresults = pd.read_csv(EVresults_path, sep=',')

    # x and y are placeholders for our training data
    t = tf.placeholder("float")
    sample_mean = tf.placeholder("float")
    sample_var = tf.placeholder("float")

    a = tf.Variable(2, name='a', dtype=tf.float32)
    b = tf.Variable(2, name='b', dtype=tf.float32)
    m = tf.Variable(100, name='m', dtype=tf.float32)

    model_mean = mean_func(t)
    model_var = var_func(t)
    model_error = tf.reduce_mean(tf.square(sample_mean - model_mean)+tf.square(sample_var - model_var) )
    train_optimize_mean = tf.train.AdamOptimizer(0.01).minimize(model_error)

    model = tf.global_variables_initializer()
    errors_set = []

    with tf.Session() as session:
        session.run(model)

        for step in range(10000):
            randomize = np.arange(len(EVresults))
            np.random.shuffle(randomize)

            fit_time = EVresults["time"].values[randomize]
            fit_mean = EVresults["mean"].values[randomize]
            fit_var = EVresults["var"].values[randomize]

            _, error_value = session.run([train_optimize_mean, model_error], feed_dict={ t:fit_time, sample_mean: fit_mean, sample_var: fit_var})
            errors_set.append(error_value)
            print("Epoch {0}, error = {1}".format(step, error_value))

        ahat, bhat, mhat = session.run([a, b, m])
        print("Predicted parameters: {a:.3f}, {b:.3f}, {m:.3f}".format(a=ahat, b=bhat, m=mhat))


    model_mean_value = model_mean_calculate(EVresults["time"].values, ahat, bhat)
    model_var_value = model_var_calculate(EVresults["time"].values, ahat, bhat, mhat)

    plot_error(errors_set)
    plot_result(EVresults["time"].values, EVresults["mean"].values, model_mean_value, EVresults["var"].values, model_var_value)










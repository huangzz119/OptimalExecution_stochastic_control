import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class trajectory():
    def __init__(self, gamma_bridge, t_set, init_holding, init_t, T = 1, lamb = 1, sigma = 0.01, kappa = 10e-8):

        self.T = T                     # over period [0, T]
        self.sigma = sigma             # the volatility of asset price process
        self.kappa = kappa             # the coefficient of the linear market impact model
        self.lamb = lamb               # mean-variance tradeoff parameter

        self.gamma_bridge = gamma_bridge  # the path of gamma_bridge over period [0, T]
        self.t_set = t_set                # the time interval over period [0, T], same length with gamma bridge

        self.init_holding = init_holding   # init share holdings
        self.init_t = init_t               # init time

    # wolformalpha.com
    def __a_func(self, t_value):
        """
        the function of a
        """
        temp1 = np.exp(2 * self.T * np.sqrt(self.lamb * (self.sigma ** 2) / self.kappa))
        temp2 = np.exp(2 * t_value * np.sqrt(self.lamb * (self.sigma ** 2) / self.kappa))
        ans = np.sqrt(self.kappa * self.lamb * (self.sigma**2)) * (temp1 + temp2)/(temp1 - temp2)
        return ans

    def __b_func(self, t_value):
        """
        the function of b
        """
        return -2 * self.__a_func(t_value)+ (2*self.kappa) / (self.T- t_value)

    def __c_func(self, t_value):
        """
        the function of c
        """
        return -(2*self.kappa)/(self.T-t_value)

    def frontier(self):
        """
        the frontier of share holdings
        """
        t_value = self.t_set
        return ( - self.__b_func(t_value) * self.gamma_bridge - self.__c_func(t_value) )/(2 * self.__a_func(t_value))


    def __share_holding_for_time_s(self, time_s):
        """
        the function to calculate the corresponding share holdings from time s
        :return:
        """

        def exponent_integ(lower_limit, upper_limit):
            """
            :return: the integrate value of exponential function
            """
            integ = integrate.quad(lambda x: self.__a_func(x), lower_limit, upper_limit)[0]
            # the first term is the value of integral
            # the second term is an upper bound on the error
            return np.exp(-1 / self.kappa * integ)

        def gamma_bridge_fun(x):
            """
            return the corresponding gamma bridge value for given time x
            """
            idx = np.where(self.t_set <= x)[0][-1]
            return self.gamma_bridge[idx]

            # the second term in the formula
        def second_integ(x):
            return (self.__b_func(x) * gamma_bridge_fun(x) + self.__c_func(x)) * exponent_integ(x, time_s)

        temp1 = self.init_holding * exponent_integ(self.init_t, time_s)

        temp2 =  -1/(2*self.kappa) * integrate.quad(lambda x: second_integ(x), self.init_t, time_s)[0]

        return temp1 + temp2


    def share_holdings(self):
        """
        :return: share holdings after init time
        """
        holdings_set = []

        # the smallest index after initial time
        init_t_idx = np.where(self.t_set >= self.init_t)[0][0]
        time_set = t_set[init_t_idx:-1]

        for time_s in time_set:
            # the share holdings for time s
            holdings = self.__share_holding_for_time_s( time_s)
            holdings_set.append(holdings)

        return holdings_set, time_set



class gamma_bridge_generation():
    def __init__(self,Tnum, m , T, t_set ):

        self.Tnum = Tnum   # the length of the gamma bridge
        self.m = m         # the parameter of gamma process
        self.T = T         # the time interval (0, T)

        self.t_set = t_set   # in the range of (0, T), not include the 0, but include T

    def __createbridge(self):
        """
        this function is to create the bridge for gamma process
        :param Tnum: the length of the gamma bridge
        :return: bI: the index of bridge
                 rI: the right index of gamma
                 lI: the left index of gamma
                 bS: beta random parameter
        """

        done = np.zeros(self.Tnum) # the indexes which are done already
        done[-1] = 1  # last point is mapped to first of bridge

        bI = np.zeros(self.Tnum)  # initial bridge index
        bI[0] = self.Tnum-1
        lI = np.zeros(self.Tnum)  # initial left index of gamma
        rI = np.zeros(self.Tnum)  # initial right index of gamma
        rI[0] = self.Tnum-1

        bS = np.zeros(self.Tnum+1)

        # initial condition, when the index of bridge is 0
        # the first jump is with parameter t_set[0] and T
        bS[0] = np.random.beta(self.t_set[0] * self.m, self.T * self.m)

        for i in np.arange(1, self.Tnum, 1):
            ind = np.where(done == 0)[0]    # all unset entries
            j = ind[0]                      # the smallest unset entries

            ind = np.where(done > 0)[0]
            ind = ind[ind > j]
            ridx = int(ind[0])               # right index of gamma
            rI[i] = ridx

            bidx = int(j + (ridx - j - ((ridx - j) % 2)) / 2)  # the midpoint between j and k, bridge index
            bI[i] = bidx
            done[bidx] = i  # for the next go

            ind = np.where(done > 0)[0]
            ind = np.where(ind < bidx)[0]
            if len(ind) == 0:
                lI[i] = 0
                left_time = 0
            else:
                lI[i] = ind[-1]
                left_time = self.t_set[int(lI[i])]

            # the jump with
            bS[i] = np.random.beta( (self.t_set[bidx] - left_time) * self.m, (self.t_set[ridx] - self.t_set[bidx]) * self.m)

        return lI, rI, bI, bS

    def buildpath(self):
        """
        using the bridge to build path
        :param Tnum: number of time array
        :return: path
        """
        lI, rI, bI, bS = self.__createbridge()

        path = np.zeros(self.Tnum)
        path[-1] = np.random.gamma( self.T * self.m, 1)  # set the endpoint

        for idx in np.arange(1, self.Tnum, 1):
            lidx = int(lI[idx])                          # left index
            ridx = int(rI[idx])                          # right index
            bidx = int(bI[idx])

            if lidx < self.Tnum-1 and lidx > 0:
                path[bidx] = path[lidx] + bS[bidx]*(path[ridx]-path[lidx])
            else:
                path[bidx] = bS[bidx]*path[ridx]

        return path


    def gamma_bridge(self):
        gamma_process = self.buildpath()
        gamma_bridge = gamma_process / gamma_process[-1]
        return gamma_bridge



if __name__ == '__main__':

    T = 1
    m = 2500
    lamb = 1
    sigma = 0.01
    kappa = 10e-8

    # the length of the time array
    Tnum =  500

    # not include 0
    time_interval = np.linspace(0, 1, num = Tnum, endpoint=False, retstep=True)
    t_set = (time_interval[0] + time_interval[1])

    gamma_class = gamma_bridge_generation(Tnum = Tnum, m = m, T = T, t_set = t_set)
    gamma_bridge = gamma_class.gamma_bridge()

    init_holding_set = [0.4, 0.05, 0.6, 0.1]
    init_t_set = [0.1,0.1, 0.2, 0.2]

    share_trajectory_set = []
    time_set = []

    for i in np.arange(0, len(init_holding_set)):
        init_holding = init_holding_set[i]
        init_t = init_t_set[i]
        trajectory_class = trajectory(gamma_bridge[:-1], t_set[:-1], init_holding, init_t)
        frontier_value = trajectory_class.frontier()

        share_trajectory, time_range = trajectory_class.share_holdings()
        share_trajectory_set.append(share_trajectory)
        time_set.append(time_range)

    plt.figure(0, figsize=(12, 8))
    plt.step(t_set, gamma_bridge, c = 'black', label="Gamma Bridge")
    plt.plot(t_set[:-1], frontier_value, c = 'darkblue', label="Frontier")
    for i in np.arange(0, len(share_trajectory_set)):
        plt.plot(time_set[i], share_trajectory_set[i],"-.",  label="trajectory"+str(i+1))

    plt.xlabel('time',  fontsize=17)
    plt.ylabel('share holdings', fontsize=17)
    plt.legend(fontsize = 12)
    plt.title("The Step = "+ str(Tnum), fontsize = 23)
    plt.grid()
    plt.show()











import numpy as np
import matplotlib.pyplot as plt

T = 1
m = 250
lamb = 1
sigma = 0.01
kappa = 10e-8

# the length of the time array
Tnum = 400

# not include 0
time_interval = np.linspace(0, 0.95, num=Tnum, endpoint=False, retstep=True)
t = (time_interval[0] + time_interval[1])

#init_t_index = 50
#init_holdings = 0.0003


def a_func(t):
    """
    the function of a
    :param t: array type, eg: np.array([0.1, 0.2]). The whole discrete t period
    :return: array type
    """
    temp1 = np.exp(2 * T * np.sqrt(lamb * (sigma ** 2) / kappa))
    temp2 = np.exp(2 * t * np.sqrt(lamb * (sigma ** 2) / kappa))
    ans = np.sqrt(kappa*lamb*(sigma**2)) * (temp1 + temp2)/(temp1 - temp2)
    return ans

def b_func(t):
    """
    the function of b, the type of input and output is the same as a
    """
    return -2 * a_func(t)+ (2*kappa) / (T-t)

def c_func(t):
    """
    the function of c, the type of input and output is the same as a
    """
    return -(2*kappa)/(T-t)

def frontier(t, gamma_bridge):
    """
    the frontier of share holdings
    :param gamma_bridge: array type with same length as t, eg: np.array([0.1, 0.2])
    :return: array type
    """
    return ( - b_func(t) * gamma_bridge - c_func(t) )/(2 * a_func(t))

def share_holdings(t, gamma_bridge, init_t_index, init_holdings):
    """
    the function to calculate the corresponding share holdings from init_time to T
    :param init_t_index: the initial time index
    :param init_holdings: the initial share holdings
    :return:
    """

    t_ = t[init_t_index:]

    # the first term in the formula
    a_integral = np.cumsum(a_func(t_))
    temp1 =  init_holdings * np.exp( -1/kappa * a_integral)

    #the second term in the formula
    dul_integral = [ b_func(t_[0])*gamma_bridge[0] + c_func(t_[0])]
    for i in np.arange(1, len(t_)):
        t_idx = int(t_[i])
        add_term = dul_integral[i-1]*np.exp( - 1/kappa * a_func(t_idx)) + b_func(t_idx)*gamma_bridge[t_idx]+ c_func(t_idx)
        dul_integral.append(add_term)
    temp2 = -1/(2*kappa) * np.array(dul_integral)

    ans = temp1 + temp2
    return ans


def createbridge(Tnum):
    """
    this function is to create the bridge for gamma process
    :param Tnum:
    :return: bI: the index of bridge
             rI: the right index of gamma
             lI: the left index of gamma
             bS: beta random parameter
    """

    done = np.zeros(Tnum) # the indexes which are done already
    done[-1] = 1  # last point is mapped to first of bridge

    bI = np.zeros(Tnum)  # initial bridge index
    bI[0] = Tnum-1
    lI = np.zeros(Tnum)  # initial left index of gamma
    rI = np.zeros(Tnum)  # initial right index of gamma
    rI[0] = Tnum-1

    bS = np.zeros(Tnum+1)
    bS[0] = np.random.beta(t[0] * m, T * m)

    for i in np.arange(1, Tnum, 1):
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
            leftval = 0
        else:
            lI[i] = ind[-1]
            leftval = t[int(lI[i])]

        bS[i] = np.random.beta( (t[bidx] - leftval) *m, (t[ridx] - t[bidx]) *m)

    return lI, rI, bI, bS

def buildpath(Tnum):
    """
    using the bridge to build path
    :param Tnum: number of time array
    :return: path
    """
    lI, rI, bI, bS = createbridge(Tnum)

    path = np.zeros(Tnum)
    path[-1] = np.random.gamma(T*m, 1)  # set the endpoint

    for idx in np.arange(1, Tnum, 1):
        lidx = int(lI[idx])                          # left index
        ridx = int(rI[idx])                          # right index
        bidx = int(bI[idx])

        if lidx < Tnum-1 and lidx > 0:
            path[bidx] = path[lidx] + bS[bidx]*(path[ridx]-path[lidx])
        else:
            path[bidx] = bS[bidx]*path[ridx]

    return path


gamma_process = buildpath(Tnum)
gamma_bridge = gamma_process/gamma_process[-1]
frontier_value = frontier(t, gamma_bridge)
#shares = share_holdings(t, gamma_bridge, init_t_index, init_holdings)

plt.figure(0, figsize=(12, 8))
plt.plot(t, gamma_bridge, label="Gamma Bridge")
plt.plot(t, frontier_value, label="Frontier")
#plt.plot(t[init_t_index:], shares, label="path")
plt.xlabel('time',  fontsize=15)
plt.ylabel('units', fontsize=15)
plt.legend(fontsize = 15)
plt.show()










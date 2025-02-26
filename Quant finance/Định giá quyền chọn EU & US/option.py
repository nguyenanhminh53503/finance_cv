import numpy as np
from math import exp
'''
Rate is quoted actual/360
Risk neutral probability : (u-(1+r))/(u-d) & (1+r-d)/(u-d)
At payoff: Option payoff CT = Max(0,ST-K)
At anytime: Ct = Max(0,E[exp(-rt)*(St+1-k)|Ft])
Spot price at time t: S0*(u^number_of_upmove)*(d^number_of_downmove)
'''
u = 2
d = 1/u
r = 0.05
S0 = 100 #Spot price
delta_t = 30 #Each period is 30 days
option_type = "C" #C or P
#Calculate risk neutral probability
q = (u-(1+r)*(delta_t/360))/(u-d) #up move
p = ((1+r)*(delta_t/360)-d)/(u-d) #down move

def European_option(K, T):
    CT = np.zeros(T+1)
    if(option_type == "C"):
        for i in range(0,T+1):
            CT[i] = max(S0*(u**i)*(d**(T-i)) - K,0)
    elif(option_type == "P"):
        for i in range(0,T+1):
            CT[i] = max(K-S0*(u**i)*(d**(T-i)),0)
    #Loop through t = T-1 to 0
    tmp = CT
    for t in range(T-1,-1,-1): #loop from t = T-1 to t = 0
        cur_tmp = []
        for k in range(0,t+1): #k is number of upmoves, from 0 to t
            ck = exp(-r*delta_t/360)*(q*tmp[k+1]+ p*tmp[k])
            cur_tmp.append(max(0,ck))
        tmp = cur_tmp

    return(tmp[0])
    
def American_option(K,T):
    CT = np.zeros(T+1)
    if(option_type == "C"):
        for i in range(0,T+1):
            CT[i] = max(0, S0*(u**i)*(d**(T+1-i))-K)
    elif(option_type == "P"):
        for i in range(0,T+1):
            CT[i] = max(0, K-S0*(u**i)*(d**(T+1-i)))
    tmp = CT
    for t in range(T-1,-1,-1):
        cur_tmp = []
        for k in range(0,t+1):
            spot = S0*(u**k)*(d**(t+1-k))
            if(option_type=="C"):
                ck = max(max(spot - K,0),exp(-r*delta_t/360)*(q*tmp[k+1]+ p*tmp[k]))
            elif(option_type=="P"):
                ck = max(max(k-spot,0),exp(-r*delta_t/360)*(q*tmp[k+1]+ p*tmp[k]))
            cur_tmp.append(ck)
        tmp = cur_tmp
    return tmp[0]
print(American_option(1500,5))


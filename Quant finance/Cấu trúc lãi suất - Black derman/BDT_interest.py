'''
BDT model assumption about short rate:  Ri,j = ai * EXP(bi*j)
q, 1-q: Risk-neutral probability
market_term[0]: market yield for 1-yr term
market_term[n+1]: market yield for nth-term
Note: the BDT short rate lattice starts with 1st term rate
Elementary price: a security that only pays 1$ at time i, state j, and 0 at every other times and states. Priced using the forward equation
At expiry: The price of swap equals E0[(fixed - floatT)/(1 + rT)^T]
At other time: The price of 
'''
from scipy import optimize
from math import exp, comb
import numpy as np
#market_term = [0.017,	0.02,	0.023,	0.026,	0.029,	0.031,	0.033,	0.035,	0.037,	0.039,	0.039,	0.04,	0.041,	0.042,	0.043,	0.043,	0.043,	0.042,	0.042,	0.042]
market_term = [0.017,	0.02,	0.023,	0.026,	0.029,	0.031]
q = 0.5 #risk-neutral proba
b = 0.01
no_term = len(market_term)
'''
def sample(params):
    x,y = params
    return x**2 -2*x + y**4 - 2*y**2
minimum = op.fmin(sample, x0 = np.zeros(2)) #can skip the args completely
print(minimum)
'''
def error_func(a):
    term_structure = []
    elementary = []
    z = []

    #first rate equals market rate
    term_structure.append([market_term[0]])
    for i in range(1, no_term):
        cur_term = []
        for j in range(0,i+1):
            #term i+1, state j
            cur_term.append(a[i]*exp(b*j))
        term_structure.append(cur_term)
    

    #Calculate the elementary price. Note that elementary term has a time 0 price
    elementary.append([1]) #0th term elementary
    #Price the lattice using forward equation
    for i in range(1,no_term):
        cur_term_elemen = []
        for j in range(0,i+1):
            if(j == 0):
                cur_term_elemen.append(q*elementary[i-1][j]/(1+term_structure[i-1][j]))
            elif(j >= i):
                cur_term_elemen.append(q*elementary[i-1][j-1]/(1+term_structure[i-1][j-1]))
            else:
                cur_term_elemen.append(q*elementary[i-1][j]/(1+term_structure[i-1][j])+q*elementary[i-1][j-1]/(1+term_structure[i-1][j-1]))
        elementary.append(cur_term_elemen)
    #calculate zero coupon price
    for i in elementary:
        z.append(sum(i))

    #calculate the implied_spot
    implied_spot = []
    implied_spot.append(market_term[0])
    for i in range(1,no_term):
        cur_implied = (1/z[i])**(1/i)-1
        implied_spot.append(cur_implied)

    #return the squared error
    square_error = 0
    for i in range(0,no_term):
        square_error = square_error + (implied_spot[i]-market_term[i])**2
    return square_error

def modelled_rate(param):
    rate = []
    rate.append([market_term[0]])
    for i in range(1, no_term):
        cur_term = []
        for j in range(0,i+1):
            #term i+1, state j
            cur_term.append((param[i]*exp(b*j)).item())
        rate.append(cur_term)
    return rate

#initialize a
param = []
for i in range(0,no_term):
    param.append(0.01)

#calibrate model
final_param = optimize.fmin(error_func, param)
calibrated_term_structure = modelled_rate(final_param)

'''
#Calculate the expected term structure. Realize that the expected short rate is not equal to the implied spot rate
expected_term_structure = []
for i in range(0,no_term):
    #Use binomial pmf to compute expected value; number of moves = i+1
    expected_ith_term = 0
    for j in range(0,i+1):
        expected_ith_term = expected_ith_term + comb(i,j)*(q**j)*(q**(i-j))*calibrated_term_structure[i][j]
    expected_term_structure.append(expected_ith_term)
'''


#Suppose we want to price a swap with fixed rate = 3%, TTM = 3, nominal value = 1$ - Long pays fixed
fixed_rate = 0.03
T = 3
#Price terminal value
def price_option(fixed_rate,T):
    last_value = []
    for i in range(0,T+1):
        last_value.append(calibrated_term_structure[T][i]-fixed_rate)

    for i in range(T-1,-1,-1):
        #time i
        current_value = []
        for j in range(0, i+1):
            #calculate the expected payoff plus expected option value, discounted
            float_rate = calibrated_term_structure[i][j]
            payoff = fixed_rate - float_rate
            expected_val = q*last_value[j]+last_value[j+1]
            current_value.append((payoff+expected_val)/(1+float_rate))
        last_value = current_value
    return(last_value[0])

print(price_option(fixed_rate,T))        




# from cmath import inf
import numpy as np
import pandas as pd
import CNGARCH as cg
# import matplotlib.pyplot as plt
# import ewma
# import TDA_STUFF as TD
# import time
# from copy import deepcopy
# import multiprocessing as mp

def main():


    # Get a time series of prices
    TS = pd.read_csv(f"./hd_QQQ.csv")
    # Calculate daily log-returns
    TS['lret'] = np.log1p(TS.close.pct_change())
    # Drop NAN or whatever
    TS = TS.dropna()
    # Keep only the the last 3200 days if time series if longer
    if len(TS)>3020:
        TS = TS.iloc[-3020:]

    # Divide the time series in 75% to estimate, and 25% to test
    n = len(TS['lret'])
    n_is1 = int(np.round(n*0.75,0))
    Rin1 = np.array(TS['lret'].iloc[0:n_is1])
    Rout = np.array(TS['lret'].iloc[n_is1+1:])

    
    GARCH = cg.sgarch11([0.1, 0.02, 0.95, 0.01], Rin1)
    NGARCH = cg.ngarch11([0.1, 0.02, 0.95, 0.01, 0.1], Rin1)
    CNGARCH = cg.ngarch22([0.1, 0.02, 0.65, 0.01, 0.1, 0.99, 0.01, 0.1], Rin1)

    bd11   = ((0,None), (0.001,0.06), (0.8,1), (0,0.5), (-5,+5))
    bd22   = ((0,None), (0.001,0.06), (0.5,1), (0,0.1), (-5,+5), (0.9,1), (0,0.5), (-5,+5))


    NGARCH.OptimizationBounds=bd11
    CNGARCH.OptimizationBounds=bd22

    GARCH.estimate()
    GARCH.filter()
    GARCH.forecast(kdays=20)
    
    NGARCH.estimate()
    NGARCH.filter()
    NGARCH.forecast(kdays=20)

    CNGARCH.estimate()
    CNGARCH.filter()
    CNGARCH.forecast(kdays=20)
    


#### __name__ MAIN()
if __name__ == '__main__':
    main()

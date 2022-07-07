"""
    Non-Affine GARCH models specified with multiple components

    All coded by some guy

    based on  
    
    Christoffersen, Peter, Christian Dorion, Kris Jacobs, and Yintian Wang. "Volatility components, affine restrictions, and nonnormal innovations."
     Journal of Business & Economic Statistics 28, no. 4 (2010): 483-502.
    https://www.tandfonline.com/doi/pdf/10.1198/jbes.2009.06122?casa_token=kmZvCbSJSCIAAAAA:GVoSrmEIeI_R763L2Plx_aRwjok1vmhASgrMTRKpRsEoR83iMXqoWgN58vmXegUo2T-423mKCHd-

"""

# TODO : GENERAL beef up all the __str__

from numba import jit

from math import pi
import warnings

import numpy as np
# from numpy import log
# from numpy import divide
# from numpy import multiply
from numpy import sqrt
from numpy import linalg as LA
from scipy.optimize import basinhopping, minimize
import multiprocessing as mp
from copy import deepcopy
import time


# TODO include estimation time series, and filter time series.
# TODO include a simulation method, to be hooked in the LSMC block.
class gmodel:
    def __init__(self, x, R=np.zeros((1,))) -> None:
        self._x = x
        self._R = R
        self._PenK = 999999.9
        self.success = False
        self.loglikelihood = 0.0
        self.scipyresult = None
        self.day_per_year = 250
        self.vpath = np.zeros((1,), dtype=float)
        self.qpath = np.zeros((1,), dtype=float)
        self.ppath = np.zeros((1,), dtype=float)
        self.npath = np.zeros((1,), dtype=float)
        self.mpath = np.zeros((1,), dtype=float)
        self._optimizeoptions = {'ftol':1e-12, 'gtol': 1e-12, 'disp': False, 'eps': 1e-10, 'iprint':1, 'maxcor':30} 
        self._debug = False
        self._bounds = None
        self._estimationtime = -1.0

    def __str__(self) -> str:
        return "General GARCH Model\n"

    def set_theta(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

    @property
    def R(self):
        return self._R

    @property
    def OptimizationOptions(self):
        return self._optimizeoptions

    @property
    def OptimizationBounds(self):
        return self._bounds

    @R.setter
    def R(self, value):
        # TODO make sure the format of the vector is appropriate
        self._R = value.flatten()

    @OptimizationOptions.setter
    def OptimizationOptions(self, value):
        self._optimizeoptions = value

    @OptimizationBounds.setter
    def OptimizationBounds(self, value):
        self._bounds = value

    def _pen(self,theta_check):
        if theta_check<0:
            penalty = 0
        else:
            penalty = self._PenK*theta_check**3
        return penalty

    def _penalty_constraints(self):
        pass

    def filter(self, output="variance", debug=False):
        pass

    def forecast(self, kdays:int)->np.ndarray:
        pass

    def estimate(self, optimizer='minimize', nbhopping=10):
        # optimizer='minimize' or 'basinhopping'
        if optimizer=='minimize':
            args = (self)
            try:
                theta_hat = minimize(_optimize, self.x, args, method='L-BFGS-B', options=self._optimizeoptions, bounds=self._bounds)            
            except ValueError:
                theta_hat = minimize(_optimize, self.x, args, method='L-BFGS-B', options=self._optimizeoptions)
                
            self.scipyresult = theta_hat
            if theta_hat['success']:
                self.set_theta(theta_hat['x'])
                self.loglikelihood = theta_hat['fun']
                self.success=True
            else:
                self.success=False

        elif optimizer=='basinhopping':
            args = {'method': 'L-BFGS-B', 'args': self}
            theta_hat = basinhopping(_optimize, self.x, niter=nbhopping, T=0.20, stepsize=0.005, minimizer_kwargs=args, seed=1)
            # TODO make sure the optimization was successful
            self.set_theta(theta_hat['x'])
            self.success=True
            self.loglikelihood = theta_hat['fun']

        elif optimizer=='tf_sgd':
            # TODO finish coding this estimation method
            done_tf = 1


    def fullestimate(self, nbhopping=10):
        self.estimate()
        if self.success:
            self.filter()
            return True
        # else
        self.estimate(optimizer='basinhopping', nbhopping=nbhopping)
        if self.success:
            self.filter()
            return True
        else:
            self.filter()
            return False


    def parallel(self, thetas, Ncores=4, estpool=None):
        nbest = np.size(thetas,0)
        thetaout = np.ones_like(thetas)
        timers   = np.zeros((nbest,), dtype=float)
        worked = [None]*nbest
        alltests = [None]*nbest
        for iest in range(nbest):
            alltests[iest] = deepcopy(self)
            alltests[iest].set_theta(thetas[iest,:])
            # alltests[iest].fullestimate()

        # output = alltests
        if estpool is None:
            estpool = mp.Pool(Ncores)

        output = estpool.map(_paralelle, alltests)

        # process_pool.close()

        
        best = np.zeros((nbest,))
        for iest in range(nbest):
            best[iest] = output[iest].loglikelihood
            thetaout[iest,:] = output[iest].x
            timers[iest] = output[iest]._estimationtime
            worked[iest] = output[iest].success


        bestindex = np.nanargmin(best)
        self.set_theta(output[bestindex].x)
        self.success = output[bestindex].success
        self.filter()

        # doneparalelle = 1



@jit
def _numbafiltergarch11(_R, vpath, W, Z, _lambda, omega, _beta, _alpha, N):
    for t in range(1,N):
        penalty = False
        vpath[t] = omega + _beta*vpath[t-1] + _alpha*vpath[t-1]*(Z[t-1])**2
        if (vpath[t]<0 or vpath[t]>1):
            LL = 9999.0*(omega+_beta+_alpha)**2
            penalty = True
            break

        W[t] = (_R[t] - _lambda*sqrt(vpath[t]) + 0.5*vpath[t])
        Z[t] = W[t] / np.sqrt(vpath[t])

    if not penalty:
        objective = -0.5 *( np.log(2*pi) + np.log(vpath) + np.divide(np.multiply(W,W),vpath) )
        LL = -np.sum(objective)

    return LL, vpath

@jit(nopython=True)
def _numbafiltersgarch11(_R, vpath, W, Z, _la, _p1, _a1, N):
    penalty = False
    for t in range(1,N):
        vpath[t] = (vpath[0]) + _p1*(vpath[t-1] - vpath[0]) + _a1*vpath[t-1]*(Z[t-1]*Z[t-1] - 1)
        if (vpath[t]<=1E-6 or vpath[t]>1):
            LL = 9999.0*(_p1+_a1)**2
            penalty = True
            break

        W[t] = (_R[t] - _la*sqrt(vpath[t]) + 0.5*vpath[t])
        Z[t] = W[t] / sqrt(vpath[t])

    if not penalty:
        objective = -0.5 *( np.log(2*pi) + np.log(vpath) + np.divide(np.multiply(W,W),vpath) )
        LL = -np.sum(objective)

    return LL, vpath


@jit(nopython=True)
def _numbafilterngarch11(_R, vpath, W, Z, _la, _p1, _a1, _g1, N):
    penalty = False
    for t in range(1,N):
        vpath[t] = (vpath[0]) + _p1*(vpath[t-1] - vpath[0]) + _a1*vpath[t-1]*(Z[t-1]*Z[t-1] - 1 - 2*_g1*Z[t-1])
        if (vpath[t]<=1E-6 or vpath[t]>1):
            LL = 9999.0*(_p1+_a1+_g1)**2
            penalty = True
            break

        W[t] = (_R[t] - _la*sqrt(vpath[t]) + 0.5*vpath[t])
        Z[t] = W[t] / sqrt(vpath[t])

    if not penalty:
        objective = -0.5 *( np.log(2*pi) + np.log(vpath) + np.divide(np.multiply(W,W),vpath) )
        LL = -np.sum(objective)

    return LL, vpath

@jit(nopython=True)
def _numbafilterngarch22(_R, vpath, qpath, W, Z, _la, _p1, _a1, _g1, _p2, _a2, _g2, N):
    penalty = False
    for t in range(1,N):
        qpath[t] = (vpath[0])   + _p2*(qpath[t-1] - vpath[0])   + _a2*vpath[t-1]*(Z[t-1]*Z[t-1] - 1 - 2*_g2*Z[t-1])
        vpath[t] = (qpath[t])   + _p1*(vpath[t-1] - qpath[t-1]) + _a1*vpath[t-1]*(Z[t-1]*Z[t-1] - 1 - 2*_g1*Z[t-1])

        if (vpath[t]<=1E-6 or qpath[t]<=1E-6 or vpath[t]>1 or qpath[t]>1):
            LL = 9999.0*(_p1+_a1+_g1+_p2+_a2+_g2)**2
            penalty = True
            break

        W[t] = (_R[t] - _la*sqrt(vpath[t]) + 0.5*vpath[t])
        Z[t] = W[t] / sqrt(vpath[t])

    if not penalty:
        objective = -0.5 *( np.log(2*pi) + np.log(vpath) + np.divide(np.multiply(W,W),vpath) )
        LL = -np.sum(objective)

    return LL, vpath, qpath



class sgarch11(gmodel):
    # https://www.proquest.com/openview/c3f3b3ee3fcca1e754816685c8bb56cc/1.pdf/advanced
    def __init__(self, x='[lambda, sigma, persistense, alpha]', R=np.zeros((1, )), targetK=False) -> None:
        super().__init__(x, R=R)
        self._targetK = targetK
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        if not self._targetK:
            self._a1 = x[3]
            # self._g1 = x[4]
        else:
            print('This is not implemented for this specification')
            self._a1 = x[3]
            # self._g1 = x[4]

    def __str__(self) -> str:
        smodel = "NGARCH(1,1)"
        if not self._targetK:
            smodel = smodel + "\n"
        else:
            smodel = smodel + f" with kurtosis targetting at {self._targetK}\n"
        sparam = f"[lambda, sigma, persistense,  alpha] = [{self._la}, {self._sg}, {self._p1}, {self._a1}]\n"
        return super().__str__() + smodel + sparam

    @property
    def glabel(self):
        if not self._targetK:
            return 'NGARCH(1,1)'
        else:
            return f"NGARCH-KTE-{self._targetK}"

    def set_theta(self, x):
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        if not self._targetK:
            self._a1 = x[3]
            # self._g1 = x[4]
        else:
            print('This is not implemented for this specification')
            self._a1 = x[3]
            # self._g1 = x[4]
        return super().set_theta(x)


    @property
    def persistenceP(self):
        return self._p1


    @property
    def persistenceQ(self):
        return self._p1 - self._a1  + self._a1 * (1 + ( self._la)*( self._la))


    def _penalty_constraints(self):
        # unc.vol. positive
        penalty = self._pen(-self._sg)
        # Q Persistence < 1
        penalty = penalty + self._pen(self.persistenceQ-1)
        # alpha > 0
        penalty = penalty + self._pen(-self._a1)
        penalty = penalty + self._pen(self._sg-1) # That means a long term annual volatility of 1,581% ...

        return penalty


    def forecast(self, kdays:int)->np.ndarray:
        if not self.vpath[-1]>0:
            self.filter()
        
        self.vforecast = np.ones((kdays,),dtype=float)*self.vpath[-1]
        variance = self._sg*self._sg
        for ik in range(1,kdays):
            self.vforecast[ik] = variance + self._p1*(self.vforecast[ik-1] - variance) 

    def filter(self, output="variance", debug=False):
        penalty = self._penalty_constraints()
        # if debug:
        #     print(f"Penalty = {penalty}")
        if penalty>0:
            if output=='estimate':
                return penalty
            # else:
            #     warnings.warn('The set of parameters creates a penalty. Filtering might be bad')
        
        N = len(self._R)
        vpath, Z = np.ones((N,), dtype=float), np.ones((N,), dtype=float)
        # W, Z     = np.zeros((N,), dtype=float), np.zeros((N,), dtype=float)
        W     = np.ones((N,), dtype=float)

        vpath[0] = self._sg*self._sg
        if vpath[0]<1E-6:
            if output=='estimate':
                return 9999*LA.norm(self.x)
            # else:
                # warnings.warn('The filtering found a negative variance. Filtering might be bad')

        W[0] = self._R[0] - self._la*sqrt(vpath[0]) + 0.5*vpath[0]
        Z[0] = W[0] / sqrt(vpath[0]) 

        LL, vpath = _numbafiltersgarch11(self._R, vpath, W, Z, self._la, self._p1, self._a1, N)

        self.vpath = vpath
        self.loglikelihood = LL
        return LL


class ngarch11(gmodel):
    # https://www.proquest.com/openview/c3f3b3ee3fcca1e754816685c8bb56cc/1.pdf/advanced
    def __init__(self, x='[lambda, sigma, persistense, alpha, gamma]', R=np.zeros((1, )), targetK=False) -> None:
        super().__init__(x, R=R)
        self._targetK = targetK
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        if not self._targetK:
            self._a1 = x[3]
            self._g1 = x[4]
        else:
            print('This is not implemented for this specification')
            self._a1 = x[3]
            self._g1 = x[4]

    def __str__(self) -> str:
        smodel = "NGARCH(1,1)"
        if not self._targetK:
            smodel = smodel + "\n"
        else:
            smodel = smodel + f" with kurtosis targetting at {self._targetK}\n"
        sparam = f"[lambda, sigma, persistense,  alpha,  gamma] = [{self._la}, {self._sg}, {self._p1}, {self._a1}, {self._g1}]\n"
        return super().__str__() + smodel + sparam

    @property
    def glabel(self):
        if not self._targetK:
            return 'NGARCH(1,1)'
        else:
            return f"NGARCH-KTE-{self._targetK}"

    def set_theta(self, x):
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        if not self._targetK:
            self._a1 = x[3]
            self._g1 = x[4]
        else:
            print('This is not implemented for this specification')
            self._a1 = x[3]
            self._g1 = x[4]
        return super().set_theta(x)


    @property
    def persistenceP(self):
        return self._p1


    @property
    def persistenceQ(self):
        return self._p1 - self._a1 * (1 + self._g1*self._g1) + self._a1 * (1 + (self._g1 + self._la)*(self._g1 + self._la))


    def _penalty_constraints(self):
        # unc.vol. positive
        penalty = self._pen(-self._sg)
        # Q Persistence < 1
        penalty = penalty + self._pen(self.persistenceQ-1)
        # alpha > 0
        penalty = penalty + self._pen(-self._a1)
        penalty = penalty + self._pen(self._sg-1) # That means a long term annual volatility of 1,581% ...

        return penalty


    def forecast(self, kdays:int)->np.ndarray:
        if not self.vpath[-1]>0:
            self.filter()
        
        self.vforecast = np.ones((kdays,),dtype=float)*self.vpath[-1]
        variance = self._sg*self._sg
        for ik in range(1,kdays):
            self.vforecast[ik] = variance + self._p1*(self.vforecast[ik-1] - variance) 

    def filter(self, output="variance", debug=False):
        penalty = self._penalty_constraints()
        # if debug:
        #     print(f"Penalty = {penalty}")
        if penalty>0:
            if output=='estimate':
                return penalty
            # else:
            #     warnings.warn('The set of parameters creates a penalty. Filtering might be bad')
        
        N = len(self._R)
        vpath, Z = np.ones((N,), dtype=float), np.ones((N,), dtype=float)
        # W, Z     = np.zeros((N,), dtype=float), np.zeros((N,), dtype=float)
        W     = np.ones((N,), dtype=float)

        vpath[0] = self._sg*self._sg
        if vpath[0]<1E-6:
            if output=='estimate':
                return 9999*LA.norm(self.x)
            # else:
                # warnings.warn('The filtering found a negative variance. Filtering might be bad')

        W[0] = self._R[0] - self._la*sqrt(vpath[0]) + 0.5*vpath[0]
        Z[0] = W[0] / sqrt(vpath[0]) 

        LL, vpath = _numbafilterngarch11(self._R, vpath, W, Z, self._la, self._p1, self._a1, self._g1, N)

        self.vpath = vpath
        self.loglikelihood = LL
        return LL


class ngarch22(gmodel):
    # https://www.proquest.com/openview/c3f3b3ee3fcca1e754816685c8bb56cc/1.pdf/advanced
    def __init__(self, x='[lambda, sigma, persistense, alpha, gamma, rho, alph2, gamm2]', R=np.zeros((1, )), Qpers=False) -> None:
        super().__init__(x, R=R)
        self._Qpers = Qpers
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        self._a1 = x[3]
        self._g1 = x[4]
        if not self._Qpers:
            self._p2 = x[5]
            self._a2 = x[6]
            self._g2 = x[7]
        else:
            self._a2 = x[5]
            self._g2 = x[6]
            self._p2 = 1 - self._a2 * (1 + (self._g2 + self._la)*(self._g2 + self._la)) + self._a2 * (1 + self._g2*self._g2)

    def __str__(self) -> str:
        smodel = "NGARCH(1,1)"
        if not self._Qpers:
            smodel = smodel + "\n"
        else:
            smodel = smodel + f" with full Qpers\n"
        sparam = f"[lambda, sigma, persistense, alpha, gamma, rho, alph2, gamm2] = ["\
            f"{self._la}, {self._sg}, {self._p1}, {self._a1}, {self._g1}, "\
                f"{self._p2}, {self._a2}, {self._g2}]\n"
        return super().__str__() + smodel + sparam

    @property
    def glabel(self):
        if not self._Qpers:
            return 'NGARCH(2,2)'
        else:
            return f"NGARCH(2,2) with full Q persistence"

    def set_theta(self, x):
        self._la = x[0]
        self._sg = x[1]
        self._p1 = x[2]
        self._a1 = x[3]
        self._g1 = x[4]
        if not self._Qpers:
            self._p2 = x[5]
            self._a2 = x[6]
            self._g2 = x[7]
        else:
            self._a2 = x[5]
            self._g2 = x[6]
            self._p2 = 1 - self._a2 * (1 + (self._g2 + self._la)*(self._g2 + self._la)) + self._a2 * (1 + self._g2*self._g2)
        return super().set_theta(x)


    @property
    def persistenceP(self):
        return self._p1


    @property
    def persistenceQ(self):
        return self._p1 - self._a1 * (1 + self._g1*self._g1) + self._a1 * (1 + (self._g1 + self._la)*(self._g1 + self._la))


    def _penalty_constraints(self):
        # unc.vol. positive
        penalty = self._pen(-self._sg+1E-10)
        # Q Persistence < 1
        penalty = penalty + self._pen(self.persistenceQ-1)
        # alpha > 0
        penalty = penalty + self._pen(-self._a1)
        penalty = penalty + self._pen(-self._a2)
        penalty = penalty + self._pen(-self._p1)
        penalty = penalty + self._pen(-self._p2)
        penalty = penalty + self._pen(self._a1-1)
        penalty = penalty + self._pen(self._a2-1)
        penalty = penalty + self._pen(self._p2-1)
        penalty = penalty + self._pen(self._sg-1) # That means a long term annual volatility of 1,581% ...
    
        return penalty


    def forecast(self, kdays:int)->np.ndarray:
        if not self.vpath[-1]>0:
            self.filter()
        
        self.vforecast = np.ones((kdays,),dtype=float)*self.vpath[-1]
        self.qforecast = np.ones((kdays,),dtype=float)*self.qpath[-1]
        
        var = self._sg*self._sg

        for t in range(1,kdays):
            self.qforecast[t] = var                + self._p2*(self.qforecast[t-1] - var)         
            self.vforecast[t] = self.qforecast[t]  + self._p1*(self.vforecast[t-1] - self.qforecast[t])   
            if (self.vforecast[t]<1E-6):
                self.vforecast[t]=1E-6
            if (self.qforecast[t]<1E-6):
                self.qforecast[t]=1E-6

    def filter(self, output="variance", debug=False):
        np.seterr(all='raise')
        N = len(self._R)
        penalty = self._penalty_constraints()
        if debug:
            print(f"Penalty = {penalty}")
        if penalty>0:
            if output=='estimate':
                return penalty
            else:
                warnings.warn('The set of parameters creates a penalty. Filtering might be bad')
                # vpath = np.zeros((N,), dtype=float)
                # qpath = np.zeros((N,), dtype=float)
                self.qpath = np.zeros((N,), dtype=float)
                self.vpath = np.zeros((N,), dtype=float)
                self.success = False
                return
        
        vpath = np.zeros((N,), dtype=float)
        qpath = np.zeros((N,), dtype=float)
        W     = np.zeros((N,), dtype=float)
        Z     = np.zeros((N,), dtype=float)

        vpath[0] = self._sg*self._sg
        if vpath[0]<1E-6:
            if output=='estimate':
                return 9999*LA.norm(self.x)
            else:
                warnings.warn('The filtering found a negative variance. Filtering might be bad')
                self.qpath = np.zeros((N,), dtype=float)
                self.vpath = np.zeros((N,), dtype=float)
                self.success = False
                return
        qpath[0] = vpath[0]


        W[0] = self._R[0] - self._la*sqrt(vpath[0]) + 0.5*vpath[0]
        Z[0] = W[0] / sqrt(vpath[0]) 

        LL, vpath, qpath = _numbafilterngarch22(self._R, vpath, qpath, W, Z, self._la, self._p1, self._a1, self._g1, self._p2, self._a2, self._g2, N)
        
        self.qpath = qpath
        self.vpath = vpath
        self.loglikelihood = LL
        return LL




def _optimize(x, thisset:gmodel):
    thisset.set_theta(x)
    LL = thisset.filter(output="estimate")
    if thisset._debug:
        print(f"{LL}  {thisset.x}")
        
    return LL


def _paralelle(thisset:gmodel):

    start = time.perf_counter()
    thisset.estimate()
    thisset._estimationtime = time.perf_counter() - start
    return thisset












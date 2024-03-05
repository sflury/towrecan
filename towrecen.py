from numpy.linalg import det, eig, inv, pinv
from numpy.random import rand, randn, seed
from numpy import argsort, array, linspace, log10, min, max, nanmean, ones, real, sign, sqrt
from scipy.optimize import fminbound, newton
import matplotlib.pyplot as plt
plt.ioff()

class towrecen(object):
    '''
    Name:
        towrecen

    Purpose:
        Calculate the dispersion of a bivariate data set

    Arguments:
        :x (*np.ndarray*): independent/predictor variable as 1xN array
        :y (*np.ndarray*): dependent/response variable as 1xN array

    Keyword Arguments:
        :fun (*function*): function describing the relation betwen `x` and `y`,
            default is a simple linear relation y=m*x+b
        :xerr (*np.ndarray*): (optional) error in independent/predictor variable
        :yerr (*np.ndarray*): (optional) error in dependent/response variable

    Attributes:
        :pc (*np.ndarray*): Each of the principal components of `x` and `y`
        :pd (*np.ndarray*): Orthogonal distance according to PCA (model
            independent, but boils down to a linear relation sans intercept)
        :x0 (*np.ndarray*): Location along `fun` which minimizes the distance
            between `x`, `y` and `x0`, `fun(x0)`
        :od (*np.ndarray*): Orthogonal distances of `x` and `y` from `x0` and
            `fun(x0)`

    Methods:
        :pplot: plots the orthogonal distance between the data and `fun` versus
            the PCA dispersion axis (second principal component)

    '''
    def __init__(self,x,y,fun=None,xerr=None,yerr=None):
        self.x  = array(x)
        self.y  = array(y)
        self.xe = array(xerr)
        self.ye = array(yerr)
        self.PCA(fun=fun)
        if fun == None:
            X = array([x,ones(len(self.x))]).T
            Y = self.y
            if det(X.T @ X) != 0 :
                B = inv( X.T @ X ) @ (X.T @ Y)
            else:
                B = pinv( X.T @ X ) @ (X.T @ Y)
            self.fun = lambda x0 : B @ array([x0,ones(len(x0))]).T.T \
                    if hasattr(x0,'__len__') else B @ array([x0,1]).T.T
        else:
            self.fun = fun
        self.OrthoDist()

    def PCA(self,fun = None):
        # design matrix is simply x and y data
        Z = array([self.x,self.y]).T
        # find and subtract data means
        if fun == None:
            Zbar = nanmean(Z,axis=0)
        else:
            Zbar = array([nanmean(self.x),nanmean(fun(self.x))])
        Z -= Zbar
        # spectral decomposition to get eigenvalues and eigenvectors
        lam,e = eig(Z.T @ Z)
        # take only real values of eigenvalues
        lam = real(lam)
        # sort eigenvectors and eigenvalues on the amount of dispersion
        e   = e[argsort(lam)]
        lam = lam[argsort(lam)]
        # store all principal components
        self.pc = array([e[i].T @ Z.T for i in range(len(lam))])
        self.pd = -self.pc[1]

    def OrthoDist(self,verbose=False):
        # function defining orthogonal distance between data and model
        dfun = lambda x0,xi,yi: sqrt((self.fun(x0)-yi)**2+(x0-xi)**2)
        # range in root-finding based on data
        xmin = (min((self.x-self.x.mean())/self.x.std())//1-1)*self.x.std()+self.x.mean()
        xmax = (max((self.x-self.x.mean())/self.x.std())//1+1.5)*self.x.std()+self.x.mean()
        # root-finding to determine where along fun(x) the distance is minimized
        self.x0 = array(list(map(
                    lambda xi,yi: fminbound(dfun,xmin,xmax,args=(xi,yi)),\
                    self.x,self.y)))
        self.y0 = self.fun(self.x0)
        # orthogonal distance for all data
        self.od = dfun(self.x0,self.x,self.y)*sign(self.y-self.y0)
        if verbose:
            print(f'\n    root-finding limits\n'+32*'-'+f'\nx_min : {xmin:.3f}, x_max : {xmax:.3f}')

    def pprint(self):
        head = '   PCA '+len(f'{self.pd[0]: >6.3f}')*' '+'Ortho   '
        print(len(head)*'-'+'\n'+head+'\n'+len(head)*'-')
        for i in range(len(self.x)):
            print(f'  {self.pd[i]: >6.3f}    {self.od[i]: >6.3f}  ')

    def pplot(self,verbose=False):

        X = array([-self.pc[1]]).T
        Y = self.od
        if det(X.T @ X) != 0:
            m = inv( X.T @ X ) @ (X.T @ Y)
        else:
            m = pinv( X.T @ X ) @ (X.T @ Y)
        if verbose:
            print(f'\nOrthoDist to PCA dist ratio : {m[0]:.3f}\n')

        tmp = linspace(1.1*min([self.od,self.pd]),1.1*max([self.od,self.pd]),11)

        fig,ax = plt.subplots(2,2)
        ax[1,0].plot(tmp,tmp,'--',color='#001e52')
        ax[1,0].plot(tmp,m*tmp,'-.',color='#64008f')
        ax[1,0].scatter(self.pd,self.od,c='#fdca40')
        ax[1,0].set_xlabel('PCA')
        ax[1,0].set_ylabel('Euclid')
        ax[1,0].set_xlim(1.1*min(self.pd),1.1*max(self.pd))
        ax[1,0].set_ylim(1.1*min(self.od),1.1*max(self.od))
        ax[1,1].hist(self.od,histtype='stepfilled',facecolor='none',edgecolor='k')
        ax[1,1].set_xlim(1.1*min(self.od),1.1*max(self.od))
        ax[1,1].set_yticks([])
        ax[1,1].set_xlabel('Euclid')
        ax[0,0].hist(self.pd,histtype='stepfilled',facecolor='none',edgecolor='k')
        ax[0,0].set_xlim(1.1*min(self.pd),1.1*max(self.pd))
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticks([])
        ax[0,1].axis('off')
        plt.show()

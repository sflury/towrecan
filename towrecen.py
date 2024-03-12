from numpy.linalg import det, eig, inv, pinv
from numpy.random import rand, randn, seed
from numpy import any, argsort, array, diag, diff, linspace, log10, min, max, \
                  nanmean, ones, real, sign, sqrt, quantile, zeros
from scipy.optimize import fminbound, newton
import matplotlib.pyplot as plt
plt.ioff()

class towrecen(object):
    '''
    Name:
        towrecen

    Purpose:
        Quantify the dispersion of a bivariate data set

    Arguments:
        :x (*np.ndarray*): independent/predictor variable as 1xN array
        :y (*np.ndarray*): dependent/response variable as 1xN array

    Keyword Arguments:
        :fun (*function*): function describing the relation betwen `x` and `y`,
            default is a simple linear relation y=m*x+b computed by Demming
            regression on `x` and `y`
        :xerr (*np.ndarray*): (optional) error in independent/predictor variable
        :yerr (*np.ndarray*): (optional) error in dependent/response variable

    Attributes:
        :pc (*np.ndarray*): Each of the principal components of `x` and `y`
        :pd (*np.ndarray*): Orthogonal distance according to PCA (model
            independent, but boils down to a linear relation sans intercept)
        :pdErr (*np.ndarray*): 2xN array for the lower, upper uncertainties in
            the PCA orthogonal distances
        :x0 (*np.ndarray*): Location along `fun` which minimizes the distance
            between `x`, `y` and `x0`, `fun(x0)`
        :od (*np.ndarray*): Euclidean distances of `x` and `y` from `x0` and
            `fun(x0)`
        :odErr (*np.ndarray*): 2xN array for the lower, upper uncertainties in
            the Euclidean orthogonal distances
        :DemmingX (*np.ndarray*): "true" values of `x` predicted by Demming
            regression, accounting for scatter in `x` and `y`
        :DemmingY (*np.ndarray*): "true" values of `y` predicted by Demming
            regression, accounting for scatter in `x` and `y`

    Methods:
        :Demming: predicts a dependent variable from a given independent
            variable after training on `x` and `y`
        :plotDistCompare: plots the orthogonal distance between the data and
            `fun` versus the PCA dispersion axis (second principal component),
            along with a Demming regression on the computed distances
        :plotDistPCA: plots `x` and `y` with the total least squares obtained
            from the eigenvectors used to determine the principal components;
            divergent color-coding by PCA dispersion distance
        :plotDistOrtho: plots `x` and `y` with the user-provided function or,
            if not provided, the Demming regression on `x` and `y`;
            divergent color-coding by Euclidean orthogonal distance
    '''
    def __init__(self,x,y,fun=None,xerr=None,yerr=None,nSamp=None):
        # store data
        self.x  = array(x)
        self.y  = array(y)
        self.xe = array(xerr)
        self.ye = array(yerr)
        # check sampling for uncertainties
        if nSamp == None:
            self.nSamp = 1000
        else:
            self.nSamp = nSamp
        # set up design matrix
        # design matrix is simply x and y data
        self.Z = array([self.x,self.y]).T
        # find and subtract data means
        self.Zbar = nanmean(self.Z,axis=0)
        self.Z -= self.Zbar
        # call regression functions
        self.RegressLeastSquares()
        self.RegressDemming()
        # calculate PCA distances
        self.PCA()
        if fun == None:
            self.usrfun = False
            self.fun = self.Demming
        else:
            self.usrfun = True
            self.fun = fun
        # calculate orthogonal Euclidean distances
        self.OrthoDist()
        # compare the two distances
        self.DistCompare()

    def PCA(self):
        '''
        Calculate principal components using spectral decomposition. Second
        component will always be the dispersion axis. First component will
        always be the general "trend" in the data.
        '''
        # spectral decomposition to get eigenvalues and eigenvectors
        lam,e = eig(self.Z.T @ self.Z)
        # take only real values of eigenvalues
        lam = real(lam)
        # sort eigenvectors and eigenvalues on the amount of dispersion
        self.e   = e[argsort(lam)]
        self.lam = lam[argsort(lam)]
        # store all principal components
        self.pc = array([self.e[i].T @ self.Z.T for i in range(len(self.lam))])
        self.pd = -self.pc[1]
        # total least squares
        self.SlopeTLS = (e[:-1,0]/e[-1,0])[0]
        if sign(self.SlopeTLS) != sign(self.SlopeDemming):
            self.SlopeTLS *= -1
        self.InterTLS = self.Zbar[1]-self.SlopeTLS*self.Zbar[0]
        # uncertainties, if provided
        if any(self.xe==None) and not any(self.ye==None):
            xSamp = array(self.nSamp*[self.x])
            ySamp = randn(self.nSamp,len(self.y))*self.ye+self.y
            pdSamp = array(list(map(lambda xSi,ySi:
                        -self.e[1].T @ array([xSi,ySi]),xSamp,ySamp)))
            self.pdErr = diff(quantile(pdSamp,[0.1587,0.5,0.8413],axis=0),axis=0)
        elif not any(self.xe==None) and not any(self.ye==None):
            xSamp = randn(self.nSamp,len(self.x))*self.xe+self.x
            ySamp = randn(self.nSamp,len(self.y))*self.ye+self.y
            pdSamp = array(list(map(lambda xSi,ySi:
                        -self.e[1].T @ array([xSi,ySi]),xSamp,ySamp)))
            self.pdErr = diff(quantile(pdSamp,[0.1587,0.5,0.8413],axis=0),axis=0)
        else:
            self.pdErr = zeros((2,len(self.pd)))


    def OrthoDist(self,verbose=False):
        '''
        Calculate Euclidean orthogonal distances between data and function
        '''
        # function defining orthogonal distance between data and model
        dfun = lambda x0,xi,yi: sqrt((self.fun(x0)-yi)**2+(x0-xi)**2)
        # range in root-finding based on data
        xmin = 0.8*(min((self.x-self.x.mean())/self.x.std())//1-1)*self.x.std()+self.x.mean()
        xmax = 1.2*(max((self.x-self.x.mean())/self.x.std()+1)//1)*self.x.std()+self.x.mean()
        # root-finding to determine where along fun(x) the distance is minimized
        self.x0 = array(list(map(
                    lambda xi,yi: fminbound(dfun,xmin,xmax,args=(xi,yi)),\
                    self.x,self.y)))
        self.y0 = self.fun(self.x0)
        # orthogonal distance for all data
        self.od = dfun(self.x0,self.x,self.y)*sign(self.y-self.y0)
        if verbose:
            print(f'\n    root-finding limits\n'+32*'-'+f'\nx_min : {xmin:.3f}, x_max : {xmax:.3f}')
        # uncertainties, if provided
        if any(self.xe==None) and not any(self.ye==None):
            xSamp = array(self.nSamp*[self.x])
            ySamp = randn(self.nSamp,len(self.y))*self.ye+self.y
            x0Samp = array(list(map(lambda xSi,ySi: list(map(
                        lambda xi,yi: fminbound(dfun,xmin,xmax,args=(xi,yi)),\
                        xSi,ySi)),xSamp,ySamp)))
            y0Samp = self.fun(x0Samp)
            odSamp = dfun(x0Samp,xSamp,ySamp)*sign(ySamp-y0Samp)
            self.odErr = diff(quantile(odSamp,[0.1587,0.5,0.8413],axis=0),axis=0)
        elif not any(self.xe==None) and not any(self.ye==None):
            xSamp = randn(self.nSamp,len(self.x))*self.xe+self.x
            ySamp = randn(self.nSamp,len(self.y))*self.ye+self.y
            x0Samp = array(list(map(lambda xSi,ySi: list(map(
                        lambda xi,yi: fminbound(dfun,xmin,xmax,args=(xi,yi)),\
                        xSi,ySi)),xSamp,ySamp)))
            y0Samp = self.fun(x0Samp)
            odSamp = dfun(x0Samp,xSamp,ySamp)*sign(ySamp-y0Samp)
            self.odErr = diff(quantile(odSamp,[0.1587,0.5,0.8413],axis=0),axis=0)
        else:
            self.odErr = zeros((2,len(self.od)))

    def RegressLeastSquares(self):
        '''
        Perform linear least squares regression. If uncertainties in y provided,
        the least squares regression weights the fit by the inverse variance.
        '''
        X = array([self.x,ones(len(self.x))]).T
        Y = self.y
        if not any(self.ye==None):
            W = diag(self.ye**-2)
        else:
            W = diag(ones(len(self.x)))
        if det(X.T @ X) != 0 :
            Bw = inv( X.T @ W @ X ) @ (X.T @ W @ Y)
        else:
            Bw = pinv( X.T @ W @ X ) @ (X.T @ W @ Y)
        self.SlopeWLS = Bw[0]
        self.InterWLS = Bw[1]

    def RegressDemming(self):
        '''
        Perform linear Demming regression
        '''
        sxx = nanmean((self.x-self.Zbar[0])**2)
        syy = nanmean((self.y-self.Zbar[1])**2)
        sxy = nanmean((self.x-self.Zbar[0])*(self.y-self.Zbar[1]))
        self.SlopeDemming =  syy-sxx+sqrt( (syy-sxx)**2 + 4*sxy**2 )
        self.SlopeDemming *= (2*sxy)**-1
        self.InterDemming = self.Zbar[1]-self.SlopeDemming*self.Zbar[0]
        self.DemmingY = self.InterDemming+self.SlopeDemming*self.x
        self.DemmingX = self.x + (self.y-self.DemmingY)*\
                        self.SlopeDemming/(self.SlopeDemming**2+1)
        self.Demming = lambda x0: self.SlopeDemming*x0+self.InterDemming

    def DistCompare(self):
        Zbar = nanmean(array([self.pd,self.od]).T,axis=0)
        sxx = nanmean((self.pd-Zbar[0])**2)
        syy = nanmean((self.od-Zbar[1])**2)
        sxy = nanmean((self.pd-Zbar[0])*(self.od-Zbar[1]))
        self.SDC =  syy-sxx+sqrt( (syy-sxx)**2 + 4*sxy**2 )
        self.SDC *= (2*sxy)**-1
        self.IDC = Zbar[1]-self.SDC*Zbar[0]
        self.DCY = self.IDC+self.SDC*self.pd
        self.DCX = self.pd + (self.od-self.DCY)*self.SDC/(self.SDC**2+1)
        self.DemmingCompare = lambda x0: self.SDC*x0+self.IDC
        #print(f'Ratio of Orthogonal Distance to PCA Distance : {self.SDC:.3f}')

    def pprint(self):
        head = '   PCA '+len(f'{self.pd[0]: >6.3f}')*' '+'Ortho   '
        print(len(head)*'-'+'\n'+head+'\n'+len(head)*'-')
        for i in range(len(self.x)):
            print(f'  {self.pd[i]: >6.3f}    {self.od[i]: >6.3f}  ')

    def printRegress(self):
        head = '    METHOD    SLOPE    INTER    '
        print(len(head)*'-'+'\n'+head+'\n'+len(head)*'-')
        method = ['Wgt LstSq','Tot LstSq','Demming']
        slopes = [self.SlopeWLS,self.SlopeTLS,self.SlopeDemming]
        inters = [self.InterWLS,self.InterTLS,self.InterDemming]
        for l,m,b in zip(method,slopes,inters):
            print(f'{l: >10s}    {m:.3f}    {b:.3f}')

    def plotDistCompare(self,verbose=False):

        tmp = linspace(1.1*min([self.od,self.pd]),1.1*max([self.od,self.pd]),11)
        fig,ax = plt.subplots(2,2,figsize=(6,5.5))
        h1, = ax[1,0].plot(tmp,tmp,'--',color='#001e52',label='1:1 agreement')
        h2, = ax[1,0].plot(tmp,self.DemmingCompare(tmp),color='#64008f',label='Demming')
        h3  = ax[1,0].errorbar(self.pd,self.od,xerr=self.pdErr,yerr=self.odErr,\
                        ls='none',color='k',lw=1,fmt='o',\
                        markerfacecolor='#fdca40',label='distances')
        #ax[1,0].scatter(self.pd,self.od,c='#fdca40',zorder=1)
        ax[1,0].set_xlabel('PCA')
        ax[1,0].set_ylabel('Ortho Euclid')
        ax[1,0].set_xlim(1.1*min([self.pd,self.od]),1.1*max([self.pd,self.od]))
        ax[1,0].set_ylim(1.1*min([self.pd,self.od]),1.1*max([self.pd,self.od]))
        ax[1,1].hist(self.od,histtype='stepfilled',facecolor='none',edgecolor='k')
        ax[1,1].set_xlim(1.1*min([self.pd,self.od]),1.1*max([self.pd,self.od]))
        ax[1,1].set_yticks([])
        ax[1,1].set_xlabel('Ortho Euclid')
        ax[0,0].hist(self.pd,histtype='stepfilled',facecolor='none',edgecolor='k')
        ax[0,0].set_xlim(1.1*min([self.pd,self.od]),1.1*max([self.pd,self.od]))
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticks([])
        ax[0,1].axis('off')
        ax[0,1].legend(handles=[h1,h2,h3])
        plt.show()

    def plotRegress(self):

        plt.figure(figsize=(6,5))
        kwargs = dict(ls='none',lw=1,color='k')
        if any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,yerr=self.ye,**kwargs)
        elif not any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,xerr=self.xe,yerr=self.ye,**kwargs)

        tmp = linspace(min([self.x.min(),self.x0.min()]),\
                       max([self.x.max(),self.x0.max()]),101)

        plt.plot(tmp,self.Demming(tmp),color='#cb1f1f',label='Demming')
        plt.plot(tmp,self.SlopeWLS*tmp+self.InterWLS,color='#fdca40',label='Weighted LstSq')
        plt.plot(tmp,self.SlopeTLS*tmp+self.InterTLS,color='#2c755f',label='Total LstSq')
        if self.usrfun:
            plt.plot(tmp,self.fun(tmp),color="#64008f",label='user func')
        #plt.scatter(self.DemmingX, self.DemmingY,zorder=5,marker='s',s=5,c='#219fff',label=r'Demming data')
        plt.scatter(self.x, self.y,c='#d97b08',zorder=6,label='data')
        plt.legend()
        plt.show()

    def plotDistOrtho(self):

        tmp = linspace(min([self.x.min(),self.x0.min()]),\
                       max([self.x.max(),self.x0.max()]),101)
        plt.plot(tmp,self.fun(tmp),color='#64008f')

        kwargs = dict(ls='none',lw=1,color='k')
        if any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,yerr=self.ye,**kwargs)
        elif not any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,xerr=self.xe,yerr=self.ye,**kwargs)

        for i in range(len(self.x)):
        	plt.plot([self.x[i],self.x0[i]],[self.y[i],self.fun(self.x0[i])],color='0.6',zorder=1)
        plt.scatter(self.x, self.y, c=self.od/self.od.std(), \
                    cmap='seismic',vmin=-3,vmax=3,ec='k',lw=1,zorder=2)

        plt.show()

    def plotDistPCA(self):

        tmp = linspace(min([self.x.min(),self.x0.min()]),\
                       max([self.x.max(),self.x0.max()]),101)
        plt.plot(tmp,self.SlopeTLS*tmp+self.InterTLS)
        for i in range(len(self.x)):
            x0 = (self.y[i]-self.InterTLS)/self.SlopeTLS
            #(self.y[i]-self.Zbar[1])/self.SlopeTLS + self.Zbar[0]
            y0 = self.SlopeTLS*self.x[i]+self.InterTLS
            plt.plot([self.x[i],(x0+self.x[i])/2],[self.y[i],(y0+self.y[i])/2],color='0.6',zorder=1)
        kwargs = dict(ls='none',lw=1,color='k')
        if any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,yerr=self.ye,**kwargs)
        elif not any(self.xe==None) and not any(self.ye==None):
            plt.errorbar(self.x,self.y,xerr=self.xe,yerr=self.ye,**kwargs)

        plt.scatter(self.x, self.y, c=self.pd/self.pd.std(), \
                    cmap='seismic',vmin=-3,vmax=3,ec='k',lw=1,zorder=2)
        plt.show()

import sys
import numpy as np
from .utils import *
from .model_specific import *

np.random.seed(241)

def sklm(dataList, tmpl, ff):
    """Sequential Karhunen-Loeve Transform.

        Attributes
        ----------
        dataList : list
            initial/additional data
        tmpl : dict
            template params
        ff : float
            forgetting factor

        Return
        ----------
        U : ndarray
            new basis of size (N,d+n)
        D : ndarray
            new singular values of size (d+n,1)
        mu : ndarray
            new mean of size (N,1)
        m : ndarray
            new number of data
    """
    
    U0 = tmpl['basis']      # old basis
    D0 = tmpl['eigval']     # old singular values
    mu0 = tmpl['mean']      # old sample mean
    n0 = tmpl['nsamples']   # number of previous data

    N = dataList[0].size
    m = len(dataList)
    data = np.array(dataList).T
    U = np.zeros((N, m), dtype=np.float32)
    D = np.zeros(m, dtype=np.float32)
    mu = np.zeros(N, dtype=np.float32)

    if U0.shape[1] == 0: # first eigenbasis calculation

        mu = np.mean(data, axis = 1, dtype = np.float32)
        zeroMeanData = np.zeros(data.shape, dtype=np.float32)
        for i in range(m):
            zeroMeanData[:, i] = data[:, i] - mu
        U, D, _ = np.linalg.svd(zeroMeanData, full_matrices = False)

    else: # incremental update of eigenbasis

        mu1 = np.mean(data, axis = 1, dtype = np.float32)
        zeroMeanData = np.zeros(data.shape, dtype=np.float32)
        for i in range(m):
            zeroMeanData[:, i] = data[:, i] - mu1

        # Compute new mean Ic = (fn/(fn+m))Ia + (m/(fn+m))Ib
        acoeff = (ff*n0) / ((ff*n0) + m)
        bcoeff = m / ((ff*n0) + m)
        mu = acoeff*mu0 + bcoeff*mu1

        # Compute B{^} = [ (I_m+1 - Ib) ... (I_n+m - Ib) sqrt(nm/(n+m))(Ib - Ia) ]
        harmean = (m * n0) / float(m + n0)
        B = np.zeros((N, m + 1), dtype=np.float32)
        B[:, 0:m] = zeroMeanData
        B[:, m:m+1] = np.expand_dims( np.sqrt(harmean) * (mu1 - mu0), axis = 1 )
        m = m + ff*n0

        Bproj = (U0.T @ B)
        Bdiff = B - (U0 @ Bproj)
        Borth, _ = np.linalg.qr(Bdiff)
        Q = np.hstack( (U0, Borth) )

        # Compute R
        t1 = np.diag(D0 * ff)
        t2 = Bproj
        t3 = np.zeros((B.shape[1], D0.size))
        t4 = (Borth.T @ Bdiff)
        R = np.zeros((t1.shape[0] + t3.shape[0], t1.shape[1] + t2.shape[1]), dtype = np.float32)
        R[0:t1.shape[0], 0:t1.shape[1]] = t1
        R[0:t2.shape[0], t1.shape[1]:t1.shape[1] + t2.shape[1]] = t2
        R[t1.shape[0]:t1.shape[0] + t3.shape[0], 0:t3.shape[1]] = t3
        R[t1.shape[0]:t1.shape[0] + t4.shape[0], t3.shape[1]:t3.shape[1] + t4.shape[1]] = t4

        # Compute the SVD of R
        U, D, _ = np.linalg.svd(R, full_matrices = False)
        cutoff = cv.norm(D) * 0.001
        keep = (D >= cutoff)
        D = D[keep]
        U = np.matmul(Q, U[:, keep])

    return U, D, mu, m


class IncrementalTracker():
    """Incremental robust self-learning algorithm for visual tracking.

        Attributes
        ----------
        affsig : ndarray
            stdevs of dynamic process
        nparticles : int
            number of particles
        condenssig : float
            stdev of observation likelihood
        forgetting : float
            forgetting factor for PCA
        batchsize : int
            size of frames after which do PCA update
        tmplShape : tuple
            size of object window for PCA
        maxbasis : int
            number of eigenvectors for PCA
        errfunc : str
            error function for minimizing the effect of noisy pixels
    """

    def __init__(self, affsig, nparticles=100, condenssig=0.75, forgetting=0.95, batchsize=5, tmplShape=(32, 32), maxbasis=16, errfunc='L2'):
        self.nparticles = nparticles
        self.condenssig = condenssig
        self.forgetting = forgetting
        self.batchsize = batchsize
        self.tmplShape = tmplShape
        self.tmplDim = self.tmplShape[0]*self.tmplShape[1]
        self.maxbasis = maxbasis
        self.errfunc = errfunc

        # dummyData = []
        # for i in range(5):
        #     dummyData.append(np.ones(9, dtype=np.float32) * i)
        # dummyTmpl = {
        #     'mean'      : np.zeros((self.tmplDim,), dtype=np.float32),     # sample mean of the images (1024,)
        #     'basis'     : np.zeros((self.tmplDim, 0), dtype=np.float32),   # eigenbasis (1024, MAXBASIS)
        #     'eigval'    : np.array([]),                                     # eigenvalues (MAXBASIS,)
        #     'nsamples'  : 0,                                                # effective number of data
        # }
        # U,D,mu,m = sklm(dummyData, dummyTmpl, self.forgetting)
        # dummyTmpl['basis'] = U
        # dummyTmpl['eigval'] = D
        # dummyTmpl['mean'] = mu
        # dummyTmpl['nsamples'] = m

        # for i in range(5):
        #     dummyData[i] += (i+5)
        # U,D,mu,m = sklm(dummyData, dummyTmpl, self.forgetting)
        # sys.exit()

        self.affsig = affsig
        self.dof = self.affsig.size # degrees of freedom
        if self.dof < 4:
            sys.exit('ValueError: dof must be greater or equal 4')
        if self.dof != affsig.size:
            sys.exit('ValueError: dof and affsig size must be the same')

        self.trackerInitialized = False
        self.param = {}
        self.tmpl = {
            'mean'      : np.zeros((self.tmplDim,), dtype=np.float32),      # sample mean of the images (1024,)
            'basis'     : np.zeros((self.tmplDim, 0), dtype=np.float32),    # eigenbasis (1024, MAXBASIS)
            'eigval'    : np.array([]),                                     # eigenvalues (MAXBASIS,)
            'nsamples'  : 0,                                                # effective number of data
            'reseig'    : 0
        }
        self.wimgs = []

        # Auxilary for optimization
        self.diff = np.zeros((self.tmplDim, self.nparticles), dtype = np.float32)
        self.param['conf'] = np.full(self.nparticles, 1./self.nparticles, dtype = np.float32)

    def init(self, gray, initialBox):
        """Initialize tracker."""

        if self.trackerInitialized:
            return

        if initialBox.size < 4:
            sys.exit("init: Given empty box")
        
        # Make initial state parameters
        param0 = np.zeros(self.dof, dtype = np.float32)
        param0[0] = initialBox[0] # x center
        param0[1] = initialBox[1] # y center
        param0[2] = initialBox[2] / self.tmplShape[0] # scale
        param0[3] = initialBox[3] / initialBox[2] # aspect ratio
        if self.dof > 4:
            param0[4] = initialBox[4] if initialBox.size > 4 else 0.0 # rotation angle
        if self.dof > 5:
            param0[5] = 0.0 # skew (shear) angle
        
        # Set initial tracker parameters
        mean2d = warpimg(gray, param0, self.tmplShape)
        self.tmpl['mean'] = mean2d.flatten('C').copy()
        self.param['est'] = param0.copy()
        self.param['wimg'] = mean2d.copy()
        self.trackerInitialized = True

    def track(self, gray, initialBox = None):
        """Track object location on given frame."""

        # Initialize tracker when the first object box is given
        if not self.trackerInitialized and initialBox is not None:
            self.init(gray, initialBox)

        # Do the condensation magic and find the most likely location
        self.estimateWarpCondensation(gray)
        
        # Do incremental update when we accumulate enough data
        if len(self.wimgs) >= self.batchsize:
            if 'coef' in self.param:
                UUTDiff = (self.tmpl['basis'] @ self.param['coef'])
                recon = (UUTDiff.T + self.tmpl['mean']).T

                basis, eigval, mean, nsamples = sklm(self.wimgs, self.tmpl, self.forgetting)
                self.tmpl['basis'] = basis
                self.tmpl['eigval'] = eigval
                self.tmpl['mean'] = mean
                self.tmpl['nsamples'] = nsamples

                recon = (recon.T - self.tmpl['mean']).T
                self.param['coef'] = (self.tmpl['basis'].T @ recon)
            else:
                basis, eigval, mean, nsamples = sklm(self.wimgs, self.tmpl, self.forgetting)
                self.tmpl['basis'] = basis
                self.tmpl['eigval'] = eigval
                self.tmpl['mean'] = mean
                self.tmpl['nsamples'] = nsamples
            self.wimgs = []

            nCurrentEigenvectors = self.tmpl['basis'].shape[1]
            if nCurrentEigenvectors > self.maxbasis:
                self.tmpl['reseig'] = self.forgetting * self.tmpl['reseig'] + np.sum( np.power(self.tmpl['eigval'][self.maxbasis:self.tmpl['eigval'].size], 2) )
                self.tmpl['basis'] = self.tmpl['basis'][:, 0:self.maxbasis]
                self.tmpl['eigval'] = self.tmpl['eigval'][0:self.maxbasis]
                if 'coef' in self.param:
                    self.param['coef'] = self.param['coef'][0:self.maxbasis, :]

        return self.param['est']

    def estimateWarpCondensation(self, gray):
        """CONDENSATION affine warp estimator. It looks for the most likely particle"""

        # Propagate density
        if 'param' not in self.param: # the first iteration. Just tile initial template
            self.param['param'] = np.tile(self.param['est'], (self.nparticles, 1))
        else:
            cumconf = self.param['conf'].cumsum(axis = 0)
            cumconf = np.expand_dims(cumconf, axis = 1)
            uniformNN = np.tile(np.random.random((1, self.nparticles)), (self.nparticles,1))
            cumconfNN = np.tile(cumconf, (1, self.nparticles))
            cdfIds = np.sum(uniformNN > cumconfNN, axis = 0).astype(np.int16)
            self.param['param'] = self.param['param'][cdfIds, :]

        # Apply dynamical model
        self.param['param'] = DynamicalProcess(self.param['param'], self.affsig)
        
        # Apply observation model

        # Retrieve image patches It predicated by Xt
        _wimgs = warpimgs(gray, self.param['param'], self.tmplShape)
        wimgsFlatten = np.reshape(_wimgs, (self.tmplDim, self.nparticles))
        for i in range(self.nparticles):
            self.diff[:, i] = wimgsFlatten[:, i] - self.tmpl['mean']

        # Compute likelihood under the observation model for each patch
        coefdiff = 0
        nCurrentEigenvectors = self.tmpl['basis'].shape[1]
        if nCurrentEigenvectors > 0:

            # Compute (I - mu) - UU.T(I - mu)
            UTDiff = (self.tmpl['basis'].T @ self.diff)
            self.diff = self.diff - (self.tmpl['basis'] @ UTDiff)

            # if 'coef' in self.param:
            #     # coefdiff = (abs(UDiff)-abs(param.coef))*tmpl.reseig./repmat(tmpl.eigval,[1,n]);
            #     tmp = (np.abs(UDiff) - np.abs(self.param['coef'])) * self.tmpl['reseig']
            #     coefdiff = tmp / np.swapaxes(np.tile( (self.tmpl['eigval']), (self.nparticles, 1) ), 0, 1 ) # <- TODO: need to test
            # else:
            #     numer = UDiff * self.tmpl['reseig']
            #     coefdiff = np.zeros(numer.shape, dtype=np.float32)
            #     for i in range(self.nparticles):
            #         coefdiff[:, i] = numer[:, i] / self.tmpl['eigval']
            self.param['coef'] = UTDiff

        diff2 = np.power(self.diff, 2)
        prec = 1.0 / self.condenssig
        if self.errfunc == 'robust':
            rsig = 0.1
            self.param['conf'] = np.exp( np.sum(diff2 / (diff2 + rsig), axis = 0).T * -prec )
        elif self.errfunc == 'ppca':
            pass
        else:
            # Compute P_dt(I | X) = exp( -|| (I - mu) - UU.T(I - mu) ||^2 )
            self.param['conf'] = np.exp( np.sum(diff2, axis = 0).T * -prec )
        
        # Store most likely particle
        self.param['conf'] = self.param['conf'] / np.sum(self.param['conf'])
        maxprob = np.max(self.param['conf'])
        maxidx = np.argmax(self.param['conf'])
        self.param['est'] = self.param['param'][maxidx, :]
        self.param['wimg'] = _wimgs[:,:,maxidx]
        self.param['err'] = np.reshape(self.diff[:,maxidx], self.tmplShape)
        self.param['recon'] = self.param['wimg'] + self.param['err']

        self.wimgs.append(self.param['wimg'].flatten('C'))

    def getParam(self):
        return self.param

    def getTemplate(self):
        return self.tmpl
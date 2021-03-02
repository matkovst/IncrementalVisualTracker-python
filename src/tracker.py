import sys
import numpy as np
from .utils import *
from .model_specific import *

# Incremental SVD algorithm
def sklm(data, tmpl, ff):
    
    U0 = tmpl['basis']
    D0 = tmpl['eigval']
    mu0 = tmpl['mean']
    n0 = tmpl['nsamples']

    N = data[0].size
    n = len(data)
    data = np.swapaxes( np.array(data), 0, 1 )
    U = np.zeros((N, n), dtype=np.float32)
    D = np.zeros(n, dtype=np.float32)
    mu = np.zeros(N, dtype=np.float32)

    if U0 is None:
        if n == 1:
            mu = data[0].copy()
            U = np.zeros(N, dtype = np.float32)
            U[0] = 1
        else:
            mu = np.mean(data, axis = 1)
            dataCentered = np.zeros(data.shape, dtype=np.float32)
            for i in range(n):
                dataCentered[:, i] = data[:, i] - mu

            U, D, V = np.linalg.svd(dataCentered, full_matrices = False)
        # if nargin >= 7
        #     keep = 1:min(K,length(D));
        #     D = D(keep);
        #     U = U(:,keep);
        # end
    else:
        mu1 = np.mean(data, axis = 1)
        dataCentered = np.zeros(data.shape, dtype=np.float32)
        for i in range(n):
            dataCentered[:, i] = data[:, i] - mu1

        # Compute Ic = (fn/(fn+m))Ia + (m/(fn+m))Ib
        Ic = ((ff*n0)*mu0 + n*mu1)/(n+ff*n0)

        # Compute B{^} = [ (I_m+1 - Ib) ... (I_n+m - Ib) sqrt(nm/(n+m))(Ib - Ia) ]
        B = np.zeros((N, n + 1), dtype=np.float32)
        B[:, 0:n] = dataCentered
        B[:, n:n+1] = np.expand_dims( np.sqrt( (n * n0)/(n + n0) ) * (mu0 - mu1), axis = 1 )
        dataCentered = B.copy()
        mu = np.reshape( Ic, mu0.shape )
        # mu = reshape((ff*n0*mu0(:) + n*mu1)/(n+ff*n0), size(mu0));
        n = n + ff*n0

        data_proj = np.matmul(U0.T, dataCentered)
        data_res = dataCentered - np.matmul(U0, data_proj)
        q, dummy = np.linalg.qr(data_res)
        Q = np.hstack( (U0, q) )

        t1 = np.diag(D0) * ff
        t2 = data_proj
        t3 = np.zeros((dataCentered.shape[1], D0.size))
        t4 = np.matmul(q.T, data_res)
        R = np.zeros((t1.shape[0] + t3.shape[0], t1.shape[1] + t2.shape[1]), dtype = np.float32)
        R[0:t1.shape[0], 0:t1.shape[1]] = t1
        R[0:t2.shape[0], t1.shape[1]:t1.shape[1] + t2.shape[1]] = t2
        R[t1.shape[0]:t1.shape[0] + t3.shape[0], 0:t3.shape[1]] = t3
        R[t1.shape[0]:t1.shape[0] + t4.shape[0], t3.shape[1]:t3.shape[1] + t4.shape[1]] = t4

        U, D, V = np.linalg.svd(R, full_matrices = False)
        cutoff = np.sum(np.power(D, 2)) * 0.000001
        keep = np.power(D, 2) >= cutoff
        D = D[keep]
        U = np.matmul(Q, U[:, keep])

    return U, D, mu, n


class IncrementalTracker():

    def __init__(self, affsig, nsamples = 600, condenssig = 0.75, forgetting = 0.95, batchsize = 5, tmplShape = (32, 32), maxbasis = 16, errfunc = 'L2'):
        self.affsig = affsig
        self.nsamples = nsamples
        self.condenssig = condenssig
        self.forgetting = forgetting
        self.batchsize = batchsize
        self.tmplShape = tmplShape
        self.tmplSize = self.tmplShape[0]*self.tmplShape[1]
        self.maxbasis = maxbasis
        self.errfunc = errfunc
        self.dof = self.affsig.size # <- degrees of freedom

        self.trackerInitialized = False
        self.param = {}
        self.tmpl = {}
        self.tmpl['basis'] = None
        self.tmpl['eigval'] = np.array([])
        self.tmpl['nsamples'] = 0
        self.tmpl['reseig'] = 0
        self.wimgs = []

        # Auxilary for optimization
        self.diff = np.zeros((self.tmplSize, self.nsamples), dtype = np.float32)
        self.param['conf'] = np.full(self.nsamples, 1./self.nsamples, dtype = np.float32)

    # Initialize tracker
    def init(self, gray, initialBox):

        if self.trackerInitialized:
            return

        if initialBox.size < 4:
            sys.exit("[ERROR] Given incorrect initial box")
        
        # Parse initial state parameters
        cx = initialBox[0]
        cy = initialBox[1]
        scale = initialBox[2] / self.tmplShape[0]
        angle = initialBox[4] if initialBox.size > 4 else 0.0
        aspectRatio = initialBox[3] / initialBox[2]
        skew = 0
        param0 = np.array([cx, cy, scale, angle, aspectRatio, skew], dtype=np.float32)
        
        # Set initial tracker parameters
        self.tmpl['mean'] = warpimg(gray, param0, self.tmplShape).flatten('C')
        self.param['est'] = param0
        self.param['wimg'] = np.reshape( self.tmpl['mean'], self.tmplShape )
        self.trackerInitialized = True

    # Track object location on frame
    def track(self, gray, initialBox = None):

        # Initialize tracker when first object box is given
        if not self.trackerInitialized and initialBox is not None:
            self.init(gray, initialBox)

        self.estimateWarpCondensation(gray)
        self.wimgs.append(self.param['wimg'].flatten('C'))
        
        # Do incremental update when we accumulate enough data
        if len(self.wimgs) >= self.batchsize:
            if 'coef' in self.param:
                ncoef = self.param['coef'].shape[1]
                recon = np.tile(np.expand_dims(self.tmpl['mean'], axis=1), (1, self.nsamples)) + \
                    np.matmul( self.tmpl['basis'], self.param['coef'] )
                basis, eigval, mean, nsamples = sklm(self.wimgs, self.tmpl, self.forgetting)
                self.tmpl['basis'] = basis
                self.tmpl['eigval'] = eigval
                self.tmpl['mean'] = mean
                self.tmpl['nsamples'] = nsamples

                tmp = np.zeros((self.tmplSize, ncoef), dtype=np.float32)
                for i in range(ncoef):
                    tmp[:, i] = recon[:, i] - self.tmpl['mean']
                self.param['coef'] = np.matmul( self.tmpl['basis'].T, tmp )
            else:
                basis, eigval, mean, nsamples = sklm(self.wimgs, self.tmpl, self.forgetting)
                self.tmpl['basis'] = basis
                self.tmpl['eigval'] = eigval
                self.tmpl['mean'] = mean
                self.tmpl['nsamples'] = nsamples
            self.wimgs = []

            if self.tmpl['basis'].shape[1] > self.maxbasis:
                self.tmpl['reseig'] = self.forgetting * self.tmpl['reseig'] + np.sum( np.power(self.tmpl['eigval'][self.maxbasis:self.tmpl['eigval'].size], 2) )
                self.tmpl['basis'] = self.tmpl['basis'][:, 0:self.maxbasis]
                self.tmpl['eigval'] = self.tmpl['eigval'][0:self.maxbasis]
                if 'coef' in self.param:
                    self.param['coef'] = self.param['coef'][0:self.maxbasis, :]

        return self.param['est']

    # CONDENSATION affine warp estimator
    def estimateWarpCondensation(self, gray):

        if 'param' not in self.param: # <- first iteration
            self.param['param'] = np.tile(self.param['est'], (self.nsamples, 1))
        else:
            # Propagate density
            cumconf = self.param['conf'].cumsum(axis = 0)
            cumconf = np.expand_dims(cumconf, axis = 1)
            idx = np.sum(np.tile(np.random.random((1, self.nsamples)), (self.nsamples,1)) > np.tile(cumconf, (1, self.nsamples)), axis = 0)
            idx = np.floor(idx).astype(np.int16)
            self.param['param'] = self.param['param'][idx, :]

        # Dynamical model
        for i in range(self.nsamples):
                self.param['param'][i, :] = DynamicalProcess(self.param['param'][i, :], self.affsig)
        
        # Observation model
        _wimgs = warpimgs(gray, self.param['param'], self.tmplShape)
        wimgsFlatten = np.reshape(_wimgs, (self.tmplSize, self.nsamples))
        for i in range(self.nsamples):
            self.diff[:, i] = wimgsFlatten[:, i] - self.tmpl['mean']

        coefdiff = 0

        if self.tmpl['basis'] is not None:
            # Compute UU.T(I - mu)
            coef = np.matmul(self.tmpl['basis'].T, self.diff)
            self.diff = self.diff - np.matmul(self.tmpl['basis'], coef)
            if 'coef' in self.param:
                # coefdiff = (abs(coef)-abs(param.coef))*tmpl.reseig./repmat(tmpl.eigval,[1,n]);
                tmp = (np.abs(coef) - np.abs(self.param['coef'])) * self.tmpl['reseig']
                coefdiff = tmp / np.swapaxes(np.tile( (self.tmpl['eigval']), (self.nsamples, 1) ), 0, 1 ) # <- TODO: need to test
            else:
                numer = coef * self.tmpl['reseig']
                coefdiff = np.zeros(numer.shape, dtype=np.float32)
                for i in range(self.nsamples):
                    coefdiff[:, i] = numer[:, i] / self.tmpl['eigval']
            self.param['coef'] = coef

        if self.errfunc == 'robust':
            pass
        elif self.errfunc == 'ppca':
            pass
        else:
            # Compute P_dt(I | X) = exp( -|| (I - mu) - UU.T(I - mu) ||^2 )
            self.param['conf'] = np.exp( -np.sum(np.power(self.diff, 2), axis = 0) / self.condenssig ).transpose()
        
        # Store most likely particle
        self.param['conf'] = self.param['conf'] / np.sum(self.param['conf'])
        maxprob = np.max(self.param['conf'])
        maxidx = np.argmax(self.param['conf'])
        self.param['est'] = self.param['param'][maxidx, :]
        self.param['wimg'] = _wimgs[:,:,maxidx]
        self.param['err'] = np.reshape(self.diff[:,maxidx], self.tmplShape)
        self.param['recon'] = self.param['wimg'] + self.param['err']
        return self.param

    def getParam(self):
        return self.param

    def getTemplate(self):
        return self.tmpl
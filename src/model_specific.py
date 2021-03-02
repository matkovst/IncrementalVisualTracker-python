import numpy as np
np.random.seed(0) # <- for testing only

# -------------------- Model hyperparameters --------------------

# david_indoor.avi
INITIAL_BOX = None
TMPLSIZE = 32
NSAMPLES = 600
MAXBASIS = 16
BATCHSIZE = 5
CONDENSSIG = 0.75
FORGETTING = 0.99
INITIAL_BOX = np.array([160, 106, 62, 78, -0.02])
affsig = np.array([5, 5, .01, .02, .002, .001])

# # rocks.mp4
# TMPLSIZE = 32
# NSAMPLES = 10
# MAXBASIS = 16
# BATCHSIZE = 55
# CONDENSSIG = 0.75
# FORGETTING = 0.99
# affsig = np.array([5, 5, .01, .01, .02, .001])


# -------------------- Model dynamics --------------------

def BrownianMotion(X, Noise):
    return np.random.normal(X, Noise)

def DynamicalProcess(X, Noise):
    return BrownianMotion(X, Noise)
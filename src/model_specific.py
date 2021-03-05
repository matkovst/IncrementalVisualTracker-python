import numpy as np
np.random.seed(0) # <- for testing only

# -------------------- Model hyperparameters --------------------

# # david_indoor.avi
# INITIAL_BOX = None
# TMPLSIZE = 32
# NSAMPLES = 600
# MAXBASIS = 16
# BATCHSIZE = 5
# CONDENSSIG = 0.75
# FORGETTING = 0.99
# INITIAL_BOX = np.array([160, 106, 62, 78, -0.02])
# RESIZE_RATE = 1.0
# affsig = np.array([5, 5, .01, .02, .002, .001])

INITIAL_BOX = None
TMPLSIZE = 32
NSAMPLES = 50
MAXBASIS = 16
BATCHSIZE = 5
CONDENSSIG = 0.75
FORGETTING = 0.5
RESIZE_RATE = 0.1
affsig = np.array([5, 5, .01, .02, .002, .001])


# -------------------- Model dynamics --------------------

def BrownianMotion(X, Noise):
    return np.random.normal(X, Noise)

def DynamicalProcess(X, Noise):
    return BrownianMotion(X, Noise)
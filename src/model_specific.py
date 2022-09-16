import numpy as np
# np.random.seed(0) # <- for testing only

# -------------------- Model parameters --------------------

# NPARTICLES. The number of particles used in the condensation
# algorithm/particle filter.  Increasing this will likely improve the
# results, but make the tracker slower.
NPARTICLES = 400

# CONDENSSIG. The standard deviation of the observation likelihood.
CONDENSSIG = 0.75

# FORGETTING. The forgetting factor, as described in the original paper. When
# doing the incremental update, 1 means remember all past data, and 0
# means remeber none of it.
FORGETTING = 0.95

# BATCHSIZE. How often to update the eigenbasis. We've used this
# value (update every 5th frame) fairly consistently, so it most
# likely won't need to be changed.  A smaller batchsize means more
# frequent updates, making it quicker to model changes in appearance,
# but also a little more prone to drift, and require more computation.
BATCHSIZE = 5

# AFFSIG. These are the standard deviations of the dynamics distribution, 
# that is how much we expect the target object might move from one frame to the next. 
# The meaning of each number is as follows:
#    AFFSIG(0) = x translation (pixels)
#    AFFSIG(1) = y translation (pixels)
#    AFFSIG(2) = scale
#    AFFSIG(3) = aspect ratio
#    AFFSIG(4) = rotation angle (radians)
#    AFFSIG(5) = skew angle (radians)
AFFSIG = np.array([10, 10, .05, .002], dtype=np.float32)

# TMPLSIZE. The resolution at which the tracking window is
# sampled, 32-by-32 pixels by default.  If your initial
# object window is very large you may need to increase this.
TMPLSIZE = 32

# MAXBASIS. The number of basis vectors to keep in the learned
# apperance model.
MAXBASIS = 16

# RESIZE_RATE. Of input frames.
RESIZE_RATE = 0.4

# INITIAL_BOX. Location of the target object on the first frame.
# None if not neccessary, ndarray otherwise
#INITIAL_BOX = np.array([0.802083333, 0.113333334, 0.09375, 0.166666667, -0.02], dtype=np.float32)
#INITIAL_BOX = np.array([0.734375, 0.5659722222222222, 0.140625, 0.22916666666666666, -0.02], dtype=np.float32)
INITIAL_BOX = None

# My custom overriden parameters used here for convenience
# INITIAL_BOX = None
# TMPLSIZE = 32
# NPARTICLES = 100
# MAXBASIS = 16
# BATCHSIZE = 5
# CONDENSSIG = 0.75
# FORGETTING = 0.99
# RESIZE_RATE = 0.25
# DOF = 4
# AFFSIG = np.array([4, 4, .1, .02])


# -------------------- Model dynamics --------------------

def BrownianMotion(X, Noise):
    return np.random.normal(X, Noise) # NOTE: force float64

# The dynamical process represents change of state between frames.
# Emplace new logic here if needed
def DynamicalProcess(X, Noise):
    return BrownianMotion(X, Noise)
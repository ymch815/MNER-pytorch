import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset partitioning functions for train/test/validation splits
def generate_dataset_logical_indices(train_fraction, cv_fraction, nsamp, njack=1):
    """Generate boolean masks for train/validation/test splits using sklearn.
    
    This function maintains backward compatibility while using sklearn's more robust
    train_test_split internally for better reproducibility and edge case handling.
    
    Args:
        train_fraction: fraction of data for training
        cv_fraction: fraction of data for validation/cross-validation  
        nsamp: total number of samples
        njack: number of jackknife splits (for compatibility)
        
    Returns:
        tuple: (trainset, cvset, testset, nshift) - boolean masks and shift size
    """
    # Create dummy data indices for splitting
    indices = np.arange(nsamp)
    
    # Use sklearn for robust splitting with fixed random state for reproducibility
    if train_fraction + cv_fraction == 1.0:
        # Two-way split: train and validation only
        train_indices, cv_indices = train_test_split(
            indices, test_size=cv_fraction, random_state=42, shuffle=False
        )
        test_indices = np.array([])
    else:
        # Three-way split: train, validation, and test
        # First split: separate test set
        train_cv_indices, test_indices = train_test_split(
            indices, test_size=1.0-(train_fraction+cv_fraction), random_state=42, shuffle=False
        )
        # Second split: separate train and validation from remaining data
        if len(train_cv_indices) > 0:
            train_indices, cv_indices = train_test_split(
                train_cv_indices, 
                test_size=cv_fraction/(train_fraction+cv_fraction), 
                random_state=42, shuffle=False
            )
        else:
            train_indices = train_cv_indices
            cv_indices = np.array([])
    
    # Convert to boolean masks (maintaining original interface)
    trainset = np.zeros(nsamp, dtype=bool)
    cvset = np.zeros(nsamp, dtype=bool)  
    testset = np.zeros(nsamp, dtype=bool)
    
    if len(train_indices) > 0:
        trainset[train_indices] = True
    if len(cv_indices) > 0:
        cvset[cv_indices] = True
    if len(test_indices) > 0:
        testset[test_indices] = True
    
    # Maintain compatibility with jackknife parameter
    nshift = int(nsamp/njack)
    
    return trainset, cvset, testset, nshift


def roll_dataset_logical_indices(trainset, cvset, testset, nshift, djack):

    trainset = np.roll(trainset, djack*nshift)
    cvset = np.roll(cvset, djack*nshift)
    testset = np.roll(testset, djack*nshift)

    return trainset, cvset, testset


def convert_dataset_logical_indices_to_array_indices(trainset, cvset, testset):
    """Convert boolean masks to integer indices using efficient numpy operations.
    
    Uses np.flatnonzero for better performance compared to np.where.
    
    Args:
        trainset: boolean mask for training samples
        cvset: boolean mask for validation samples  
        testset: boolean mask for test samples
        
    Returns:
        tuple: (trainInd, cvInd, testInd) - integer index arrays
    """
    trainInd = np.flatnonzero(trainset)
    cvInd = np.flatnonzero(cvset)
    testInd = np.flatnonzero(testset)

    return trainInd, cvInd, testInd


def zscore_features(feat):
    """Apply z-score normalization to features using sklearn's StandardScaler.
    
    This provides better numerical stability and handles edge cases like 
    zero-variance features more robustly than the manual implementation.
    
    Args:
        feat: feature matrix of shape (n_samples, n_features)
        
    Returns:
        tuple: (normalized_features, mean, std) - maintains original interface
    """
    scaler = StandardScaler()
    feat_normalized = scaler.fit_transform(feat)
    
    # Extract statistics to maintain original interface
    feat_mean = scaler.mean_
    feat_std = scaler.scale_  # StandardScaler stores std as scale_
    
    return feat_normalized, feat_mean, feat_std

# MNEr weight conversions
def weights_to_vec(a=None, h=None, U=None, V=None, Q=None, **kwargs):
    # transform weight matrices into a weight vector
    
    if a is None:
        return np.array([])
    assert a.size == 1

    if h is not None:
        ndim = h.size
    if U is not None:
        if "csigns" not in kwargs:
            assert V is not None
        else:
            csigns = kwargs.get("csigns")
            V = np.dot(U, np.diag(csigns.ravel()))
        if h is not None:
            assert U.shape[0] == ndim
        else:
            ndim = U.shape[0]
        rank = U.shape[1]
        assert (V.shape[0] == ndim) and (V.shape[1] == rank)
        Q = np.concatenate([U, V], axis=0)
    elif Q is not None:
        if h is not None:
            assert Q.shape[0] == 2*ndim
        else:
            ndim = Q.shape[0]/2
        rank = Q.shape[1]            
    else:
        Q = None

    if (h is not None) and (Q is not None):
        return np.concatenate([a.reshape((1,)), h.reshape((h.size,)), Q.T.reshape((Q.size,))])
    elif (h is not None):
        return np.concatenate([a.reshape((1,)), h.reshape((h.size,))])
    elif (Q is not None):
        return np.concatenate([a.reshape((1,)), Q.T.reshape((Q.size,))])
    else:
        return a.reshape((1,))


def vec_to_weights(x, ndim, rank, **kwargs):
    # transform weight vector to weight matrices
    
    if x.size == 1:
        a = np.copy(x).reshape((1,))
        h = None
        U = None
        V = None
    elif x.size == (1+ndim):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        U = None
        V = None
    elif x.size == (1+ndim+ndim*rank):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        U = np.copy(x[1+ndim:1+(1+rank)*ndim].reshape((rank, ndim)).T)
        if "csigns" not in kwargs:
            V = None
        else:
            csigns = kwargs.get("csigns")
            V = np.dot(U, np.diag(csigns.ravel())) 
    elif x.size == (1+ndim+2*ndim*rank):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        Q = np.copy(x[1+ndim:1+(1+2*rank)*ndim].reshape((rank, 2*ndim)).T)
        U = Q[:ndim,:]
        V = Q[ndim:,:]
    else:
        assert x.size == 0
        a = None
        h = None
        U = None
        V = None

    return a, h, U, V

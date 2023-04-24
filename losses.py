import numpy as np
def mse(gt, pred):
    """
    Compute the mse between two numpy arrays
    Parameters
    ----------
    gt : numpy array
        Ground truth
    pred : numpy array
        Prediction
    Returns
    -------
    mse : float
        Mean squared error
    """

    return np.mean((gt - pred)**2)



def psnr(gt, pred):
    """
    Compute the psnr between two numpy arrays
    Parameters
    ----------
    gt : numpy array
        Ground truth
    pred : numpy array
        Prediction
    Returns
    -------
    psnr : float
        Peak signal to noise ratio
    """
    
    mse = np.mean((gt - pred)**2)
    return 10 * np.log10((1)**2 / mse)
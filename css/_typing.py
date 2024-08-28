import numpy as np
import scipy.stats as ss

ScipyRVType = type[ss._distn_infrastructure.rv_continuous]

Array1DFloat = np.ndarray[tuple[int], np.dtype[np.floating]]
Array2DFloat = np.ndarray[tuple[int, int], np.dtype[np.floating]]
FloatOrIntDtype = np.dtype[np.integer] | np.dtype[np.floating]
Array1DFloatOrInt = np.ndarray[tuple[int], FloatOrIntDtype]
Array2DFloatOrInt = np.ndarray[tuple[int, int], FloatOrIntDtype]
Array3DFloatOrInt = np.ndarray[tuple[int, int, int], FloatOrIntDtype]

PeerDims2DArray = Array2DFloatOrInt
PeerDimsSets3DArray = Array3DFloatOrInt
DistParams2DArray = Array2DFloat

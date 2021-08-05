from typing import List, Tuple, Optional

import numpy as np
from tensorflow import convert_to_tensor
from skimage.filters import threshold_otsu
from scipy.ndimage.measurements import label

from src.types import Image

SIZE_RATIO = .1


def get_threshold(image: Image) -> float:
    np_image = image.image.numpy()
    return threshold_otsu(np_image)


def remove_noise(image: Image,
                 size_ratio: Optional[float] = SIZE_RATIO) -> Image:
    if len(image.image.shape) not in [3, 4]:
        return image
    if size_ratio is None:
        size_ratio = SIZE_RATIO
    shape_image = image.image.shape
    np_image = np.squeeze(image.image.numpy())
    otsu = get_threshold(image)
    binary_image = np_image > otsu

    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(binary_image, structure)

    for component in range(ncomponents):
        vecs = np.where(labeled == component + 1)
        size_comp = labeled[vecs]
        if size_comp.shape[0] / image.dim() < size_ratio:
            to_remove = list(zip(*vecs))
            high_thresh = .75 * image.image.dtype.max
            np_image = pad_vec(np_image,
                               to_remove=to_remove,
                               low=0,
                               high=high_thresh)
    np_image = np.reshape(np_image, shape_image)
    image.image = convert_to_tensor(np_image)
    return image


def pad_vec(vector: np.ndarray,
            to_remove: List[Tuple[int, int]],
            low: int,
            high: int,
            radius: Optional[int] = 20) -> np.ndarray:
    for idx in to_remove:
        slices, surrs = surrounding(vector, idx, radius=radius, fill=None)
        surrs = surrs[(surrs > low) & (surrs < high)]
        vector[idx] = np.mean(surrs)
        vector[tuple(slices)] = np.mean(surrs)
    return vector


def surrounding(x, idx, radius=1, fill=0):
    """ 
	Gets surrounding elements from a numpy array 

	Parameters: 
	x (ndarray of rank N): Input array
	idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
		the entire axis is returned.
	radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis, 
		the entire axis is returned.
	fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
		If value is None, only the surrounding indices that are within the original array are returned.

	Returns: 
	ndarray: The surrounding elements at the specified index
	"""

    assert len(idx) == len(x.shape)

    if np.isscalar(radius):
        radius = tuple([radius for i in range(len(x.shape))])

    slices = []
    paddings = []
    for axis in range(len(x.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, x.shape[axis]))
            paddings.append((0, 0))
            continue

        r = radius[axis]
        l = idx[axis] - r
        r = idx[axis] + r

        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < x.shape[axis] else r - x.shape[axis] + 1

        slices.append(slice(max(0, l), min(x.shape[axis], r + 1)))
        paddings.append((pl, pr))

    if fill is None: return slices, x[tuple(slices)]
    return slices, np.pad(x[tuple(slices)],
                          paddings,
                          'constant',
                          constant_values=fill)

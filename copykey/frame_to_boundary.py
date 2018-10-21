import cv2
import numpy as np

def get_edges(image, low=65, high=165):
    edges = cv2.Canny(image, low, high)
    return cv2.bitwise_not(edges)

def spike_removal_filter(array, threshold, window_size):
    """
    Filter out spikes from a masked one-dimensional array.

    Let `window` be the window around a point `x`. If median(window) is masked or abs(median(window) - x) > threshold, then x is added to the mask.
    This removes local outliers (spikes) as well as values that are surrounded by already masked values.

    Parameters
    ---------
    array: one-dimensional np.ma.masked_array
    threshold: the threshold value, in pixels. Any values that deviate from the median of their window by more than the threshold will be removed.
    window_size: the size of the window arround each value.
    """
    result = np.ma.copy(array)
    for i, elem in enumerate(array):
        if elem is np.ma.masked:
            continue
        start = max(0, i - (window_size // 2))
        end = min(len(array), i + (window_size // 2) + 1)
        dist = abs(elem - np.median(array[start:end]))
        if dist is np.ma.masked or dist > threshold:
            result[i] = np.ma.masked
    return result


def get_boundary_raytracing(binary_image, filter_threshold, filter_window_size):
    """
    Gets the boundary of the key from a binary image using vertical raytracing.

    Parameters
    ---------
    binary_image: 2D array of 0 and 255 values, with zeros where the key should be
    filter_threshold: threshold value to use for spike_removal_filter
    filter_window_size: window size to use for spike_removal_filter
    """
    boolean_image = binary_image == 0
    empty_mask = np.count_nonzero(boolean_image,
                                  axis=0) == 0  # mask for vertical columns that don't have any black pixels

    # flattened into one entry for each column
    maxes = np.ma.masked_array(np.argmax(boolean_image, axis=0), mask=empty_mask)
    mins = np.ma.masked_array(len(boolean_image) - np.argmax(boolean_image[::-1], axis=0) - 1, mask=empty_mask)

    filtered_maxes = spike_removal_filter(maxes, filter_threshold, filter_window_size)
    filtered_mins = spike_removal_filter(mins, filter_threshold, filter_window_size)

    max_xvals, = np.nonzero(~filtered_maxes.mask)
    max_yvals = filtered_maxes[~filtered_maxes.mask].data

    min_xvals, = np.nonzero(~filtered_mins.mask)
    min_yvals = filtered_mins[~filtered_mins.mask].data

    # concatenate maxes from left to right with mins from right to left
    boundary = (np.concatenate([max_xvals, min_xvals[::-1]]), np.concatenate([max_yvals, min_yvals[::-1]]))
    return np.transpose(boundary)

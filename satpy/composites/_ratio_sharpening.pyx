cimport cython
cimport numpy as np
from cython cimport floating
from libc.math cimport NAN, isnan

import numpy as np


def ratio_sharpened_rgb(
        np.ndarray[floating, ndim=2] red,
        np.ndarray[floating, ndim=2] green,
        np.ndarray[floating, ndim=2] blue,
        high_res=None,
        int low_resolution_index=0):
    print(red.dtype, green.dtype, blue.dtype, high_res.dtype if high_res is not None else "None")
    cdef np.ndarray[floating, ndim=3] rgb_data = np.empty((3, red.shape[0], red.shape[1]), dtype=red.dtype)
    cdef floating[:, :, ::1] rgb_view = rgb_data
    cdef floating[:, ::1] red_view = red
    cdef floating[:, ::1] green_view = green
    cdef floating[:, ::1] blue_view = blue
    cdef np.ndarray[floating, ndim=2] hres
    cdef floating[:, ::1] hres_view

    if high_res is not None:
        hres = high_res.astype(red.dtype)
        hres_view = hres
        with nogil:
            _ratio_sharpened_rgb(
                red_view,
                green_view,
                blue_view,
                hres_view,
                low_resolution_index,
                rgb_view,
            )
    with nogil:
        _ratio_unsharpened_rgb(
            red_view,
            green_view,
            blue_view,
            rgb_view,
        )

    return rgb_data


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _ratio_sharpened_rgb(
        floating[:, ::1] red,
        floating[:, ::1] green,
        floating[:, ::1] blue,
        floating[:, ::1] high_res,
        int low_resolution_index,
        floating[:, :, ::1] rgb_out
) nogil:
    cdef Py_ssize_t row_idx, col_idx
    cdef floating ratio
    cdef floating red_val, green_val, blue_val, hres_val
    cdef floating this_nan = <floating>NAN
    for row_idx in range(rgb_out.shape[1]):
        for col_idx in range(rgb_out.shape[2]):
            red_val = red[row_idx, col_idx]
            green_val = green[row_idx, col_idx]
            blue_val = blue[row_idx, col_idx]
            hres_val = high_res[row_idx, col_idx]
            if low_resolution_index == 0:
                ratio = hres_val / red_val
            elif low_resolution_index == 1:
                ratio = hres_val / green_val
            else:
                ratio = hres_val / blue_val
            ratio = _correct_ratio(ratio)

            red_val = hres_val if low_resolution_index == 0 else red_val * ratio
            green_val = hres_val if low_resolution_index == 1 else green_val * ratio
            blue_val = hres_val if low_resolution_index == 2 else blue_val * ratio

            if isnan(red_val) or isnan(green_val) or isnan(blue_val):
                red_val = this_nan
                green_val = this_nan
                blue_val = this_nan
            rgb_out[0, row_idx, col_idx] = red_val
            rgb_out[1, row_idx, col_idx] = green_val
            rgb_out[2, row_idx, col_idx] = blue_val


cdef inline floating _correct_ratio(floating ratio) nogil:
    if isnan(ratio) or ratio < 0.0:
        ratio = 1.0
    if ratio > 1.5:
        ratio = 1.5
    return ratio


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _ratio_unsharpened_rgb(
        floating[:, ::1] red,
        floating[:, ::1] green,
        floating[:, ::1] blue,
        floating[:, :, ::1] rgb_out
) nogil:
    cdef Py_ssize_t row_idx, col_idx
    cdef floating ratio
    cdef floating red_val, green_val, blue_val
    cdef floating this_nan = <floating>NAN
    for row_idx in range(rgb_out.shape[1]):
        for col_idx in range(rgb_out.shape[2]):
            red_val = red[row_idx, col_idx]
            green_val = green[row_idx, col_idx]
            blue_val = blue[row_idx, col_idx]

            if isnan(red_val) or isnan(green_val) or isnan(blue_val):
                red_val = this_nan
                green_val = this_nan
                blue_val = this_nan
            rgb_out[0, row_idx, col_idx] = red_val
            rgb_out[1, row_idx, col_idx] = green_val
            rgb_out[2, row_idx, col_idx] = blue_val

# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc, free
from cython.parallel import parallel

import numpy as np
cimport numpy as np

cdef extern from "voronoi_occupation_c.h":
    void c_voronoi_occupation_3d(
        int* out, const double* lattice, size_t lattice_size,
        const double* points, size_t num_points) nogil
    
    void c_voronoi_index_3d(
        int* out, const double* lattice, size_t lattice_size, 
        const double* points, size_t num_points) nogil
    
    void c_voronoi_occupation(
        int* out, const double* lattice, size_t lattice_size,
        const double* points, size_t num_points, size_t ndim) nogil
    
    void c_voronoi_index(
        int* out, const double* lattice, size_t lattice_size, 
        const double* points, size_t num_points, size_t ndim) nogil

def check_lattice_shape(lattice):
    if len(lattice.shape) != 2:
        raise ValueError("lattice must be a 2D array.")

def check_points_shape(points, lattice):
    if len(points.shape) != 2:
        raise ValueError("points must be a 2D array.")
    if points.shape[1] != lattice.shape[1]:
        raise ValueError("lattice and points must have same last dimension.")

def voronoi_occupation(lattice, points):
    check_lattice_shape(lattice)
    check_points_shape(points, lattice)
    
    assert lattice.shape[1] == points.shape[1]
    
    cdef int Nl = lattice.shape[0]
    cdef int Np = points.shape[0]
    cdef int Ndim = lattice.shape[1]
    
    if not lattice.flags['C_CONTIGUOUS']:
        lattice = np.ascontiguousarray(lattice)
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    
    cdef double[::1] lattice_view = lattice.flatten()
    cdef double[::1] points_view = points.flatten()
    
    cdef np.ndarray[int, ndim=1] out = np.zeros(Nl, dtype=np.int32)
    if (Ndim == 3):
        with nogil:
            c_voronoi_occupation_3d(
                    &out[0], &lattice_view[0], Nl, &points_view[0], Np)
    else:
        with nogil:
            c_voronoi_occupation(
                    &out[0], &lattice_view[0], Nl, &points_view[0], Np, 
                    Ndim)
    
    return out

def voronoi_index(lattice, points):
    check_lattice_shape(lattice)
    
    cdef int nd_points = len(points.shape)
    cdef int Ndim = points.shape[nd_points - 1]
    assert lattice.shape[1] == Ndim
    
    cdef int Nl = lattice.shape[0]
    cdef int Np = points.size/Ndim
    
    if not lattice.flags['C_CONTIGUOUS']:
        lattice = np.ascontiguousarray(lattice)
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    
    cdef double[::1] lattice_view = lattice.flatten()
    cdef double[::1] points_view = points.flatten()
    
    cdef np.ndarray[int, ndim=1] out = np.zeros(Np, dtype=np.int32)
    
    if (Ndim == 3):
        with nogil:
            c_voronoi_occupation_3d(
                    &out[0], &lattice_view[0], Nl, &points_view[0], Np)
    else:
        with nogil:
            c_voronoi_occupation(
                    &out[0], &lattice_view[0], Nl, &points_view[0], Np, 
                    Ndim)
    
    return np.reshape(out, points.shape[:nd_points - 1])
    

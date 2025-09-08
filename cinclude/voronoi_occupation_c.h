#ifndef VORONOI_OCCUPATION_C_H
#define VORONOI_OCCUPATION_C_H

#include <stddef.h>

void c_voronoi_occupation(
    int* out, const double* lattice, size_t lattice_size,
    const double* points, size_t num_points, size_t ndim);

void c_voronoi_index(
    int* out, const double* lattice, size_t lattice_size, 
    const double* points, size_t num_points, size_t ndim);

void c_voronoi_occupation_3d(
    int* out, const double* lattice, size_t lattice_size,
    const double* points, size_t num_points);

void c_voronoi_index_3d(
    int* out, const double* lattice, size_t lattice_size, 
    const double* points, size_t num_points);

#endif
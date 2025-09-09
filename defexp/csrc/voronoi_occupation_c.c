#include "voronoi_occupation_c.h"

#include <math.h>
#include <string.h>

static inline int closest_index(
    const double* point, size_t ndim, const double* lattice,
    size_t lattice_size)
{
    double mindist2 = INFINITY;
    size_t minindex = 0;
    for (size_t j = 0; j < lattice_size; ++j)
    {
        double dist2 = 0;
        for (size_t k = 0; k < ndim && dist2 < mindist2; ++k)
        {
            double disp = lattice[ndim*j + k] - point[k];
            dist2 += disp*disp;
        }
        if (dist2 < mindist2)
        {
            mindist2 = dist2;
            minindex = j;
        }
    }
    return (int) minindex;
}


static inline int closest_index_3d(
    const double point[3], const double* lattice,
    size_t lattice_size)
{
    double mindist2 = INFINITY;
    size_t minindex = 0;
    for (size_t j = 0; j < lattice_size; ++j)
    {
        const double* lattice_point = &lattice[3*j];
        const double sep[3] = {
                lattice_point[0] - point[0],
                lattice_point[1] - point[1],
                lattice_point[2] - point[2]
        };
        const double dist2 = sep[0]*sep[0] + sep[1]*sep[1] + sep[2]*sep[2];
        if (dist2 < mindist2)
        {
            mindist2 = dist2;
            minindex = j;
        }
    }
    return (int) minindex;
}

void c_voronoi_occupation(
    int* restrict out, const double* lattice, size_t lattice_size,
    const double* points, size_t num_points, size_t ndim)
{
    memset(out, 0, lattice_size*sizeof(int));
    for (size_t i = 0; i < num_points; ++i)
        out[closest_index(&points[ndim*i], ndim, lattice, lattice_size)]++;
}

void c_voronoi_index(
    int* restrict out, const double* lattice, size_t lattice_size, 
    const double* points, size_t num_points, size_t ndim)
{
    for (size_t i = 0; i < num_points; ++i)
    {
        out[i] = closest_index(&points[ndim*i], ndim, lattice, lattice_size);
    }
}

void c_voronoi_occupation_3d(
    int* restrict out, const double* lattice, size_t lattice_size,
    const double* points, size_t num_points)
{
    memset(out, 0, lattice_size*sizeof(int));
    for (size_t i = 0; i < num_points; ++i)
        out[closest_index_3d(&points[3*i], lattice, lattice_size)]++;
}

void c_voronoi_index_3d(
    int* restrict out, const double* lattice, size_t lattice_size, 
    const double* points, size_t num_points)
{
    for (size_t i = 0; i < num_points; ++i)
    {
        out[i] = closest_index_3d(&points[3*i], lattice, lattice_size);
    }
}
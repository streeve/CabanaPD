/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

#include <CabanaPD_Constants.hpp>

// Generate a unidirectional fiber-reinforced composite geometry.
void fiberReinforcedCompositeExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================

    // FIXME: We need to modify CabanaPD_Inputs.hpp, so that we can use arrays
    // for density and elastic_modulus in the JSON file.
    //        We use "density_temp" and "elastic_modulus_temp" as placeholders.

    // Matrix material
    double rho0_m = inputs["density_temp"][0];
    double E_m = inputs["elastic_modulus_temp"][0];
    double nu_m = 1.0 / 3.0;
    double K_m = E_m / ( 3.0 * ( 1.0 - 2.0 * nu_m ) );
    double G0_m = inputs["fracture_energy"][0];
    // double G_m = E_m / ( 2.0 * ( 1.0 + nu_m ) ); // Only for LPS.

    // Fiber material
    double rho0_f = inputs["density_temp"][1];
    double E_f = inputs["elastic_modulus_temp"][1];
    double nu_f = 1.0 / 3.0;
    double K_f = E_f / ( 3.0 * ( 1.0 - 2.0 * nu_f ) );
    double G0_f = inputs["fracture_energy"][1];
    // double G_f = E_f / ( 2.0 * ( 1.0 + nu_f ) ); // Only for LPS.

    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    // Matrix material
    using model_matrix = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_matrix force_model_matrix( delta, K_m, G0_m );

    // Fiber material
    using model_fiber = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_fiber force_model_fiber( delta, K_f, G0_f );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles = CabanaPD::createParticles<memory_space, model_fiber>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================

    std::array<double, 3> system_size = inputs["system_size"];

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto type = particles->sliceType();

    // Fiber-reinforced composite geometry parameters
    double Vf = inputs["fiber_volume_fraction"];
    double Df = inputs["fiber_diameter"];
    std::vector<double> stacking_sequence = inputs["stacking_sequence"];
    // std::vector<double> stacking_vector = inputs["stacking_sequence"];
    // Kokkos::View<double*, memory_space> stacking_sequence(
    // stacking_vector.data() );

    // Fiber radius
    double Rf = 0.5 * Df;

    // System sizes
    double Lx = system_size[0];
    double Ly = system_size[1];
    double Lz = system_size[2];

    // Number of plies
    auto Nplies = stacking_sequence.size();
    // Ply thickness (in z-direction)
    double dzply = Lz / Nplies;

    // Single-fiber volume (assume a 0° fiber orientation)
    double Vfs = CabanaPD::pi * Rf * Rf * Lx;
    // Domain volume
    double Vd = Lx * Ly * Lz;
    // Total fiber volume
    double Vftotal = Vf * Vd;
    // Total number of fibers
    int Nf = std::floor( Vftotal / Vfs );
    // Cross section corresponding to a single fiber in the YZ-plane
    // (assume all plies have 0° fiber orientation)
    double Af = Ly * Lz / Nf;
    // Number of fibers in y-direction (assume Af is a square area)
    int Nfy = std::round( Ly / std::sqrt( Af ) );
    // Ensure Nfy is even.
    if ( Nfy % 2 == 1 )
        Nfy = Nfy + 1;

    // Number of fibers in z-direction
    int Nfz = std::round( Nf / Nfy );
    // Ensure number of fibers in z-direction within each ply is even.
    int nfz = std::round( Nfz / Nplies );
    if ( nfz % 2 == 0 )
    {
        Nfz = nfz * Nplies;
    }
    else
    {
        Nfz = ( nfz + 1 ) * Nplies;
    };

    // Fiber grid spacings (assume all plies have 0° fiber orientation)
    double dyf = Ly / Nfy;
    double dzf = Lz / Nfz;

    // Domain center x- and y-coordinates
    double Xc = 0.5 * ( low_corner[0] + high_corner[0] );
    double Yc = 0.5 * ( low_corner[1] + high_corner[1] );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Particle position
        double xi = x( pid, 0 );
        double yi = x( pid, 1 );
        double zi = x( pid, 2 );

        // Find ply number of particle (counting from 0).
        int nply = Kokkos::floor( ( zi - low_corner[2] ) / dzply );

        // Ply fiber orientation (in radians)
        double theta = stacking_sequence[nply] * CabanaPD::pi / 180;
        // double theta = stacking_sequence(nply) * CabanaPD::pi / 180;

        // Translate then rotate (clockwise) y-coordinate of particle in
        // XY-plane.
        double yinew = -Kokkos::sin( theta ) * ( xi - Xc ) +
                       Kokkos::cos( theta ) * ( yi - Yc );

        // Find center of ply in z-direction (first ply has nply = 0).
        double Zply_bot = low_corner[2] + nply * dzply;
        double Zply_top = Zply_bot + dzply;
        double Zcply = 0.5 * ( Zply_bot + Zply_top );

        // Translate point in z-direction.
        double zinew = zi - Zcply;

        // Find nearest fiber grid center point in YZ plane.
        double Iyf = Kokkos::floor( yinew / dyf );
        double Izf = Kokkos::floor( zinew / dzf );
        double YI = 0.5 * dyf + dyf * Iyf;
        double ZI = 0.5 * dzf + dzf * Izf;

        // Check if point belongs to fiber
        if ( ( yinew - YI ) * ( yinew - YI ) + ( zinew - ZI ) * ( zinew - ZI ) <
             Rf * Rf + 1e-8 )
        {
            // Material type: 1 = fiber (default is 0 = matrix)
            type( pid ) = 1;
            // Density (fiber)
            rho( pid ) = rho0_f;
        }
        else
        {
            // Density (matrix)
            rho( pid ) = rho0_m;
        }
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        *particles, CabanaPD::AverageTag{}, force_model_matrix,
        force_model_fiber );
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, models );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init();
    cabana_pd->run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fiberReinforcedCompositeExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

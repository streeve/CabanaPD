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

// Generate a unidirectional fiber-reinforced composite geometry
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
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 1.0 / 3.0;
    double K = E / ( 3.0 * ( 1.0 - 2.0 * nu ) );
    double G0 = inputs["fracture_energy"];
    // double G = E / ( 2.0 * ( 1.0 + nu ) ); // Only for LPS.
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
    using model_type1 = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_type1 force_model1( delta, K, G0 );
    using model_type2 = CabanaPD::ForceModel<CabanaPD::LinearPMB>;
    model_type2 force_model2( delta, K / 10.0, G0 / 10.0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles = CabanaPD::createParticles<memory_space, model_type2>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================

    std::array<double, 3> system_size = inputs["system_size"];

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto type = particles->sliceType();

    // Fiber-reinforced composite geometry parameters
    int Nfx = inputs["number_of_fibers"][0];
    int Nfy = inputs["number_of_fibers"][1];
    double Rf = inputs["fiber_radius"];

    // Fiber grid spacings
    double dxf = system_size[0] / Nfx;
    double dyf = system_size[1] / Nfy;

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // x- and y-coordinates of particle
        double xi = x( pid, 0 );
        double yi = x( pid, 1 );

        // Find nearest fiber grid center point
        double Ixf = floor( ( xi - low_corner[0] ) / dxf );
        double Iyf = floor( ( yi - low_corner[1] ) / dyf );
        double XI = low_corner[0] + 0.5 * dxf + dxf * Ixf;
        double YI = low_corner[1] + 0.5 * dyf + dyf * Iyf;

        if ( ( xi - XI ) * ( xi - XI ) + ( yi - YI ) * ( yi - YI ) <
             Rf * Rf + 1e-10 )
            type( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        *particles, CabanaPD::AverageTag{}, force_model1, force_model2 );
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

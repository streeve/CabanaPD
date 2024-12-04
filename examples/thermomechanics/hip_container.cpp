/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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

// Simulate a HIP container
void HIPContainerExample( const std::string filename )
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
    double K = inputs["bulk_modulus"];
    // double G = inputs["shear_modulus"]; // Only for LPS.
    double sc = inputs["critical_stretch"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // For PMB or LPS with influence_type == 1
    double G0 = 9 * K * delta * ( sc * sc ) / 5;
    // For LPS with influence_type == 0 (default)
    // double G0 = 15 * K * delta * ( sc * sc ) / 8;

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
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );
    // using model_type =
    //      CabanaPD::ForceModel<CabanaPD::LPS, CabanaPD::Fracture>;
    // model_type force_model( delta, K, G, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double d_out = inputs["outer_cylinder_diameter"];
    double d_in = inputs["inner_cylinder_diameter"];
    double Rout = 0.5 * d_out;
    double Rin = 0.5 * d_in;
    double Wall_th = inputs["wall_thickness"];

    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center );

        // Convert domain block into hollow cylinder
        if ( rsq > Rout * Rout || rsq < ( Rin - Wall_th ) * ( Rin - Wall_th ) )
        {
            return false;
        }
        // Leave remaining bottom wall particles and remove particles in between
        // inner and outer cylinder
        else if ( x[2] > low_corner[2] + Wall_th && rsq > Rin * Rin &&
                  rsq < ( Rout - Wall_th ) * ( Rout - Wall_th ) )
        {
            return false;
        }
        else
        {
            return true;
        }
    };
    particles->createParticles( exec_space(), init_op );

    auto rho = particles->sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, force_model );

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

    HIPContainerExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

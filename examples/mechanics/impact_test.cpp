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

// Simulate impact of a ball on a disk
void ballDiskImpactExample( const std::string filename )
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
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_type force_model( delta, K, G0 );
    // using model_type =
    //      CabanaPD::ForceModel<CabanaPD::LPS, CabanaPD::Fracture>;
    // model_type force_model( delta, K, G, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double disk_radius = 5.0;
    double disk_thickness = 1.0;
    double disk_x_center = 0.0;
    double disk_y_center = 0.0;
    double disk_z_center = 0.0;

    double ball_radius = 1.0;
    double ball_x_center = 0.0;
    double ball_y_center = 0.0;
    double ball_z_center = 0.6 + ball_radius;

    // Create particles only inside ball and inside disk
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Check if particle is inside ball
        double ball_rsq = ( x[0] - ball_x_center ) * ( x[0] - ball_x_center ) +
                          ( x[1] - ball_y_center ) * ( x[1] - ball_y_center ) +
                          ( x[2] - ball_z_center ) * ( x[2] - ball_z_center );

        double disk_rsq = ( x[0] - disk_x_center ) * ( x[0] - disk_x_center ) +
                          ( x[1] - disk_y_center ) * ( x[1] - disk_y_center );

        if ( ball_rsq < ball_radius * ball_radius ||
             disk_rsq < disk_radius * disk_radius &&
                 std::abs( x[2] ) < 0.5 * disk_thickness )
            return true;
        return false;
    };

    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width, init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    // auto f = particles->sliceForce();

    auto dx = particles->dx;

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Check if particle is inside ball
        double ball_rsq =
            ( x( pid, 0 ) - ball_x_center ) * ( x( pid, 0 ) - ball_x_center ) +
            ( x( pid, 1 ) - ball_y_center ) * ( x( pid, 1 ) - ball_y_center ) +
            ( x( pid, 2 ) - ball_z_center ) * ( x( pid, 2 ) - ball_z_center );

        // Velocity
        if ( ball_rsq < ball_radius * ball_radius )
            // if ( x( pid, 2 ) > 1.0 )
            v( pid, 2 ) = -500.0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //  Simulation run with contact physics
    // ====================================================
    if ( inputs["use_contact"] )
    {
        double r_c = inputs["contact_horizon_factor"];
        r_c *= dx[0];
        CabanaPD::NormalRepulsionModel contact_model( delta, r_c, K );

        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, contact_model );
        cabana_pd->init();
        cabana_pd->run();
    }
    // ====================================================
    //  Simulation run without contact
    // ====================================================
    else
    {
        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model );
        cabana_pd->init();
        cabana_pd->run();
    }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    ballDiskImpactExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

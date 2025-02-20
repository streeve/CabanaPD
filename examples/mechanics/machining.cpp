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

// Simulate pseudo-2d plane strain machining.
void machiningExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
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
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    std::array<double, 3> system_size = inputs["system_size"];
    double height = system_size[0];
    double width = system_size[1];
    double thickness = system_size[2];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureIndependent;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    auto geom_op = KOKKOS_LAMBDA( const int p, const double x[3] )
    {
        // Check if particle is inside either.
        if ( x[0] < width / 4.0 && x[1] > height / 4.0 )
            return true;
        return false;
    };

    CabanaPD::Region work_piece( geom_op );
    CabanaPD::Region<CabanaPD::RectangularPrism> tool(
        width / 2.0, hight_corner[0], low_corner[1], height / 2.0, -thickness,
        thickness );

    // Create particles only inside ball and inside disk
    auto init_op = KOKKOS_LAMBDA( const int p, const double x[3] )
    {
        // Check if particle is inside either.
        if ( work_piece.inside( x, p ) || tool.inside( x, p ) )
            return true;
        return false;
    };

    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width, init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto nofail = particles->sliceNoFail();
    auto dx = particles->dx;
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Velocity and no fail in tool.
        if ( tool.inside( x, p ) )
        {
            v( pid, 0 ) = 500.0;
            nofail( pid ) = 1;
        }
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, *particles, delta, K, G0, sigma_y );

    // ====================================================
    //  Simulation run
    // ====================================================
    double r_c = inputs["contact_horizon_factor"];
    r_c *= dx[0];
    CabanaPD::NormalRepulsionModel contact_model( delta, r_c, K );

    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, force_model, contact_model );
    cabana_pd->init();
    cabana_pd->run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    machiningExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

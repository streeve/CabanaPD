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

// Simulate crack initiation and propagation in a quenched ceramic plate.
void thermalDeformationExample( const std::string filename )
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
    //            Material and problem parameters
    // ====================================================
    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double delta = inputs["horizon"];
    double alpha = inputs["thermal_coefficient"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];
    double temp_w = inputs["water_temperature"];
    double t_ramp = inputs["pulse_ramping_time"];

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
    //                Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureDependent;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles = std::make_shared<
        CabanaPD::Particles<memory_space, model_type, thermal_type>>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { rho( pid ) = rho0; };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model =
        CabanaPD::createForceModel<model_type, CabanaPD::Fracture,
                                   thermal_type>( *particles, delta, K, alpha,
                                                  temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, force_model );

    // ====================================================
    //                   Imposed field
    // ====================================================
    auto x = particles->sliceReferencePosition();
    auto temp = particles->sliceTemperature();
    const double low_corner_y = low_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // --------------------------------------------
        //                Thermal shock
        // --------------------------------------------
        // Define a time-dependent surface temperature:
        // An inverted triangular pulse over a 2*t_ramp period
        // starting at temp0 and linearly decreasing to temp_w within t_ramp,
        // then linearly increasing back to temp0, and finally staying constant
        // at temp0
        if ( t <= t_ramp )
        {
            // Increasing pulse
            double temp_infty = temp0 - ( temp0 - temp_w ) * ( t / t_ramp );
        }
        else if ( t < 2 * t_ramp )
        {
            // Decreasing pulse
            double temp_infty =
                temp_w + ( temp0 - temp_w ) * ( t - t_ramp ) / t_ramp;
        }
        else
        {
            // Constant value
            double temp_infty = temp0;
        };

        // Plate limits
        double X0 = low_corner[0];
        double Xn = high_corner[0];
        double Y0 = low_corner[1];
        double Yn = high_corner[1];

        // Rescale x and y particle position values
        double xi = ( 2 * x( pid, 0 ) - ( X0 + Xn ) ) / ( Xn - X0 );
        double eta = ( 2 * x( pid, 1 ) - ( Y0 + Yn ) ) / ( Yn - Y0 );

        // Define profile powers in x- and y-directions
        double sx = 1 / 50;
        double sy = 1 / 10;

        // Define profiles in x- and y-direcions
        double fx = 1.0 - std::pow( std::abs( xi ), 1 / sx );
        double fy = 1.0 - std::pow( std::abs( eta ), 1 / sy );

        // Compute particle temperature
        temp( pid ) = temp_infty + ( temp0 - temp_infty ) * fx * fy;
    };
    auto body_term = CabanaPD::createBodyTerm( temp_func, false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( body_term );
    cabana_pd->run( body_term );

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output displacement along the x-axis
    // createDisplacementProfile( MPI_COMM_WORLD, num_cells[0], 0,
    //                           "xdisplacement_profile.txt", *particles );

    // Output displacement along the y-axis
    // createDisplacementProfile( MPI_COMM_WORLD, num_cells[1], 1,
    //                           "ydisplacement_profile.txt", *particles );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

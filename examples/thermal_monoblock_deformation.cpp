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

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    {
        // ====================================================
        //                  Setup Kokkos
        // ====================================================
        Kokkos::ScopeGuard scope_guard( argc, argv );

        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        // ====================================================
        //                   Read inputs
        // ====================================================
        CabanaPD::Inputs inputs( argv[1] );

        // ====================================================
        //                Material parameters
        // ====================================================
        double rho0 = inputs["density"];
        double E = inputs["elastic_modulus"];
        double nu = 0.25;
        double K = E / ( 3 * ( 1 - 2 * nu ) );
        double delta = inputs["horizon"];
        double alpha = inputs["thermal_coefficient"];
        double temp_ref = inputs["reference_temperature"];
        // Reference temperature
        // double temp0 = 0.0;

        // ====================================================
        //                  Discretization
        // ====================================================
        std::array<double, 3> low_corner = inputs["low_corner"];
        std::array<double, 3> high_corner = inputs["high_corner"];
        std::array<int, 3> num_cells = inputs["num_cells"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // ====================================================
        //                    Force model
        // ====================================================
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        // model_type force_model( delta, K, alpha );
        model_type force_model( delta, K, alpha, temp_ref );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        // model_type force_model( delta, K, G );

        // ====================================================
        //                 Particle generation
        // ====================================================
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

        // Do not create particles within given cylindrical region
        auto x = particles->sliceReferencePosition();
        double x_center = inputs["cylindrical_hole"][0];
        double y_center = inputs["cylindrical_hole"][1];
        double radius = inputs["cylindrical_hole"][2];
        auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
        {
            if ( ( ( x[0] - x_center ) * ( x[0] - x_center ) +
                   ( x[1] - y_center ) * ( x[1] - y_center ) ) <
                 radius * radius )
                return false;
            return true;
        };

        particles->createParticles( exec_space(), init_op );

        // ====================================================
        //                Boundary conditions
        // ====================================================
        auto temp = particles->sliceTemperature();
        // Reslice after updating size.
        x = particles->sliceReferencePosition();
        auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
        {
            // temp( pid ) = 5000.0 * ( x( pid, 1 ) - ( -0.014 ) ) * t;
            //  temp( pid ) = 5000.0 * ( x( pid, 1 ) - low_corner[1] ) * t;
            temp( pid ) = 2.0e+6 * ( x( pid, 1 ) - low_corner[1] ) * t;
        };
        auto body_term = CabanaPD::createBodyTerm( temp_func );

        // ====================================================
        //            Custom particle initialization
        // ====================================================
        auto rho = particles->sliceDensity();
        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // ====================================================
        //                   Simulation run
        // ====================================================
        auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
            inputs, particles, force_model, body_term );
        cabana_pd->init_force();
        cabana_pd->run();

        // ====================================================
        //                      Outputs
        // ====================================================

        // Displacement profiles
        createDisplacementProfile( MPI_COMM_WORLD, num_cells[0], 0,
                                   "displacement_x.txt", *particles );
        createDisplacementProfile( MPI_COMM_WORLD, num_cells[1], 1,
                                   "displacement_y.txt", *particles );

        createDisplacementMagnitudeProfile( MPI_COMM_WORLD, num_cells[0], 0,
                                            "displacement_magnitude_x.txt",
                                            *particles );
        createDisplacementMagnitudeProfile( MPI_COMM_WORLD, num_cells[1], 1,
                                            "displacement_magnitude_y.txt",
                                            *particles );
    }

    MPI_Finalize();
}

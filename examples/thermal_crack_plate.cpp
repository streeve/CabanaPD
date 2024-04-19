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
        double alpha = inputs["thermal_coeff"];
        double G0 = inputs["fracture_energy"];
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
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        // CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        // model_type force_model( delta, K, alpha );
        model_type force_model( delta, K, alpha, G0 );
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
        particles->createParticles( exec_space() );

        // ====================================================
        //                Boundary conditions
        // ====================================================
        CabanaPD::RegionBoundary domain1( low_corner[0], high_corner[0],
                                          low_corner[1], high_corner[1],
                                          low_corner[2], high_corner[2] );
        std::vector<CabanaPD::RegionBoundary> domain = { domain1 };

        auto temp = particles->sliceTemperature();
        auto x = particles->sliceReferencePosition();
        auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
        {
            // We need to read these from input
            double Theta0 = 300;   // oC
            double ThetaW = 20;    // oC
            double t_ramp = 0.001; // s

            double ThetaInf = Theta0;

            if ( t <= t_ramp )
            {
                ThetaInf = Theta0 - ( ( Theta0 - ThetaW ) * t / t_ramp );
            }
            else if ( t > t_ramp && t < 2 * t_ramp )
            {
                ThetaInf =
                    ThetaW + ( Theta0 - ThetaW ) * ( t - t_ramp ) / t_ramp;
            }
            else
            {
                ThetaInf = Theta0;
            }

            // We need to read these from input
            double X0 = -0.05 / 2;
            double Xn = 0.05 / 2;

            double Y0 = -0.01 / 2;
            double Yn = 0.01 / 2;

            double sx = 1 / 50;
            double sy = 1 / 10;

            temp( pid ) =
                ThetaInf +
                ( Theta0 - ThetaInf ) *
                    ( 1 - Kokkos::pow(
                              Kokkos::abs( ( 2 * x( pid, 0 ) - ( X0 + Xn ) ) /
                                           ( Xn - X0 ) ),
                              1 / sx ) ) *
                    ( 1 - Kokkos::pow(
                              Kokkos::abs( ( 2 * x( pid, 1 ) - ( Y0 + Yn ) ) /
                                           ( Yn - Y0 ) ),
                              1 / sy ) );
        };
        auto temp = CabanaPD::createBodyTerm( temp_func );

        // ====================================================
        //            Custom particle initialization
        // ====================================================
        auto rho = particles->sliceDensity();
        auto x = particles->sliceReferencePosition();
        auto u = particles->sliceDisplacement();
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        // auto temp = particles->sliceTemperature();

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // ====================================================
        //                   Simulation run
        // ====================================================
        CabanaPD::Prenotch<1> prenotch;
        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, temp, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}

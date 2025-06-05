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

// Simulate powder settling for angle of repose calculation.
void angleOfReposeExample( const std::string filename )
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
    double vol0 = inputs["volume"];
    double radius = inputs["radius"];
    double radius_extend = inputs["radius_extend"];
    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int halo_width = 1;
    double delta = inputs["horizon"];

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::HertzianModel;
    model_type contact_model( radius, radius_extend, nu, E, e );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double min_height = inputs["min_height"];
    double max_height = inputs["max_height"];
    double diameter = inputs["cylinder_diameter"];
    double cylinder_radius = 0.5 * diameter;
    double wall_thickness = inputs["wall_thickness"];
    double particle_radius = cylinder_radius - wall_thickness;
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles for container.
        double rsq = x[0] * x[0] + x[1] * x[1];
        if ( ( x[2] > min_height && rsq > particle_radius * particle_radius &&
               rsq < cylinder_radius * cylinder_radius ) )
            return true;
        if ( ( rsq < particle_radius * particle_radius &&
               x[2] > high_corner[2] - wall_thickness ) )
            return true;
        return false;
    };
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, CabanaPD::BaseOutput{}, low_corner,
        high_corner, num_cells, halo_width, Cabana::InitRandom{},
        create_container, exec_space{}, true );
    auto create = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles inside cylinder.
        double rsq = x[0] * x[0] + x[1] * x[1];
        if ( x[2] > min_height && x[2] < max_height &&
             rsq < particle_radius * particle_radius )
            return true;
        return false;
    };
    particles.createParticles( exec_space{}, Cabana::InitRandom{}, create,
                               particles.localOffset() );

    double fraction_fused_powder = inputs["fraction_fused_powder"];
    auto current_particles = particles.localOffset();
    int hybrid_particles = static_cast<int>(
        static_cast<double>( current_particles ) * fraction_fused_powder );
    Kokkos::View<double* [3], memory_space> x_v( "custom_position",
                                                 hybrid_particles );
    Kokkos::View<double*, memory_space> vol_v( "custom_volume",
                                               hybrid_particles );
    Kokkos::View<double*, memory_space> type_v( "custom_id", hybrid_particles );
    auto x = particles.sliceReferencePosition();
    auto vol = particles.sliceVolume();

    Kokkos::Random_XorShift64_Pool<exec_space> pool( 12345 );
    using random_type = Kokkos::Random_XorShift64<exec_space>;
    auto hybrid_powder = KOKKOS_LAMBDA( const int i )
    {
        // Create random fused powder.
        auto gen = pool.get_state();
        auto rand = Kokkos::rand<random_type, int>::draw(
            gen, particles.frozenOffset(), particles.localOffset() );
        pool.free_state( gen );

        for ( std::size_t d = 0; d < 3; d++ )
            x_v( i, d ) = x( rand, d );
        x_v( i, 2 ) += delta * 0.9;
        vol_v( i ) = vol( rand );
        type_v( i ) = rand;
    };
    Kokkos::RangePolicy<exec_space> policy( 0, hybrid_particles );
    Kokkos::parallel_for( "create_random", policy, hybrid_powder );
    particles.createParticles( exec_space{}, x_v, vol_v, current_particles );
    std::cout << "Original: " << current_particles << "\n";
    std::cout << "With fused: " << particles.localOffset() << "\n";

    // Set density/volumes.
    auto rho = particles.sliceDensity();
    vol = particles.sliceVolume();
    auto type = particles.sliceType();
    x = particles.sliceReferencePosition();
    auto init_functor = KOKKOS_LAMBDA( const std::size_t pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
        if ( pid > current_particles )
            type( pid ) = type_v( pid - current_particles );
        else
            type( pid ) = pid;
    };
    particles.updateParticles( exec_space{}, init_functor );

    // Artificially reduced to keep system stable.
    double K = E / ( 3 * ( 1 - 2 * nu ) ) / 1e5;
    CabanaPD::ForceModel force_model( CabanaPD::PMB{}, CabanaPD::NoFracture{},
                                      delta, K );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model, contact_model );
    solver.init();

    // ====================================================
    //                   Boundary condition
    // ====================================================
    auto f = solver.particles.sliceForce();
    auto v = solver.particles.sliceVelocity();
    auto y = solver.particles.sliceCurrentPosition();
    auto body_func = KOKKOS_LAMBDA( const int p, const double )
    {
        // Gravity force.
        f( p, 2 ) -= 9.8 * rho( p );

        // Interact with a horizontal wall.
        double rz = y( p, 2 );
        if ( rz - radius < 0.0 || rz > high_corner[2] - radius )
        {
            double vz = v( p, 2 );
            double vn = rz * vz;
            vn /= rz;

            f( p, 2 ) +=
                -contact_model.forceCoeff( rz + radius, vn, vol0, rho0 );
        }
        // Interact with cylinder until near the bottom.
        /*
        double cr = cylinder_radius * cylinder_radius;
        double xy = y( p, 0 ) * y( p, 0 ) + y( p, 1 ) * y( p, 1 );
        if ( xy > cr - radius * radius && rz > min_height )
        {
            double vx = v( p, 0 );
            double vy = v( p, 1 );
            double rx = y( p, 0 ) - cylinder_radius;
            double ry = y( p, 1 ) - cylinder_radius;
            double vn = vx * rx + vy * ry;
            double r = Kokkos::sqrt( rx * rx + ry * ry );
            vn /= r;

            auto coeff = -contact_model.forceCoeff( r, vn, vol0, rho0 );
            f( p, 0 ) += rx / r * coeff;
            f( p, 1 ) += ry / r * coeff;
        }*/
    };
    CabanaPD::BodyTerm body( body_func, solver.particles.size(), true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.run( body );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    angleOfReposeExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

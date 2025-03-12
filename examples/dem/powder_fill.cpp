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

// Base random particle initialization from a uniform distribution.
// Using CRTP to enable non-uniform sampling.
template <typename ExecutionSpace, typename Derived>
struct RandomSample
{
    using pool_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using random_type = Kokkos::Random_XorShift64<ExecutionSpace>;
    pool_type pool;
    int _num_states = 0;
    double _min = 0.0;
    double _max = 1.0;

    RandomSample( const uint64_t seed )
    {
        // Construct a random number generator (init states internally).
        pool = pool_type( seed );
    }

    RandomSample( const uint64_t seed, const double min, const double max )
        : _min( min )
        , _max( max )
    {
        // Construct a random number generator (init states internally).
        pool = pool_type( seed );
    }

    RandomSample( const uint64_t seed, const int num_states )
        : _num_states( num_states )
    {
        // Initialize a random number generator with a given number of states.
        pool.init( seed, _num_states );
    }

    RandomSample( const uint64_t seed, const int num_states, const double min,
                  const double max )
        : _min( min )
        , _max( max )
        , _num_states( num_states )
    {
        // Create a random number generator.
        pool.init( seed, _num_states );
    }

    // Sample with default/constructed min and max.
    KOKKOS_INLINE_FUNCTION double generate() const
    {
        return generate( _min, _max );
    }

    // Sample with custom min and max.
    KOKKOS_INLINE_FUNCTION double generate( const double min,
                                            const double max ) const
    {
        auto gen = pool.get_state();
        auto rand = Kokkos::rand<random_type, double>::draw( gen, min, max );
        pool.free_state( gen );
        // Modify uniform sampling with custom functor (if used).
        auto derived = static_cast<const Derived*>( this );
        return derived->modify( rand );
    }

    auto numStates() const { return _num_states; }
};

template <typename ExecutionSpace>
struct RandomSampleLogNormal
    : public RandomSample<ExecutionSpace, RandomSampleLogNormal<ExecutionSpace>>
{
    using base_type =
        RandomSample<ExecutionSpace, RandomSampleLogNormal<ExecutionSpace>>;
    using base_type::base_type;

    double mu = 0.0;
    double sigma = 0.99;
    double shift = 50.0;

    KOKKOS_INLINE_FUNCTION double erfinv( const double rand ) const
    {
        double sign = 1.0;
        if ( rand < 0 )
            sign = -1.0;

        auto rand2 = ( 1.0 - rand ) * ( 1.0 + rand );
        auto randlog = Kokkos::log( rand2 );

        auto x1 = 2.0 / ( CabanaPD::pi * 0.147 ) + 0.5 * randlog;
        auto x2 = 1.0 / ( 0.147 ) * randlog;

        return sign * Kokkos::sqrt( -x1 + Kokkos::sqrt( x1 * x1 - x2 ) );
    }

    KOKKOS_INLINE_FUNCTION double modify( const double rand ) const
    {
        return shift - 1.0 +
               Kokkos::exp( mu + Kokkos::sqrt( 2.0 * sigma * sigma ) *
                                     erfinv( 2.0 * rand - 1.0 ) );
    }
};

// Simulate powder settling.
void powderSettlingExample( const std::string filename )
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

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double diameter = inputs["cylinder_diameter"];
    double cylinder_radius = 0.5 * diameter;
    double wall_thickness = inputs["wall_thickness"];

    // Create container.
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Convert domain block into cylinder
        if ( rsq > cylinder_radius * cylinder_radius )
            return false;
        // Leave remaining bottom wall particles and remove particles inside
        // cylinder
        if ( x[2] > low_corner[2] + wall_thickness &&
             rsq < ( cylinder_radius - wall_thickness ) *
                       ( cylinder_radius - wall_thickness ) )
            return false;

        return true;
    };
    // Container particles should be frozen, never updated.
    auto particles = CabanaPD::createParticles<memory_space, CabanaPD::PMB>(
        exec_space(), low_corner, high_corner, num_cells, halo_width,
        CabanaPD::BaseOutput{}, create_container, 0, true );

    // Create powder.
    double min_height = inputs["min_height"];
    double max_height = inputs["max_height"];
    auto create_powder = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Only create particles inside cylinder.
        if ( x[2] > low_corner[2] + wall_thickness + min_height &&
             x[2] < high_corner[2] - max_height &&
             rsq < ( cylinder_radius - wall_thickness ) *
                       ( cylinder_radius - wall_thickness ) )
            return true;

        return false;
    };
    // Update domain to create fewer powder particles.
    std::array<double, 3> dx;
    for ( int d = 0; d < 3; d++ )
        dx[d] = particles->dx[d] * 2.0;
    particles->createDomain( low_corner, high_corner, dx );
    particles->createParticles( exec_space(), Cabana::InitRandom{},
                                create_powder, particles->numFrozen() );

    // Set density/volumes/radii.
    auto rho = particles->sliceDensity();
    auto vol = particles->sliceVolume();
    auto rp = particles->sliceType();
    RandomSampleLogNormal<exec_space> random( 12345678 );
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rp( pid ) = random.generate() * 1e-5; // should be microns..
        // std::cout << rp( pid ) << std::endl;
        rho( pid ) = rho0;
        vol( pid ) = rp( pid ) * rp( pid ) * rp( pid );
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    CabanaPD::HertzianModel contact_model( rp, radius, radius_extend, nu, E,
                                           e );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolver<memory_space>( inputs, particles,
                                                           contact_model );

    // ====================================================
    //                   Simulation init
    // ====================================================
    cabana_pd->init();

    // ====================================================
    //                   Boundary condition
    // ====================================================
    auto f = cabana_pd->particles->sliceForce();
    rho = cabana_pd->particles->sliceDensity();
    auto body_functor = KOKKOS_LAMBDA( const int pid, const double )
    {
        f( pid, 2 ) -= 9.8 * rho( pid );
    };
    auto gravity =
        CabanaPD::createBodyTerm( body_functor, particles->size(), true, true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->run( gravity );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    powderSettlingExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

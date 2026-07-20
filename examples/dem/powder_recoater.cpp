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
#include <limits>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

template <typename ContactType, typename DensityType, typename ForceType,
          typename VelType, typename PosType>
struct CustomBodyTerm
{
    ContactType contact_model;
    DensityType rho;
    ForceType f;
    VelType v;
    PosType y;
    double rho0;
    double vol0;
    double radius;
    std::array<double, 3> low_corner;
    std::array<double, 3> high_corner;

    // Rake parameters
    double rake_angle;   // Radians
    double rake_speed;   // Positive => moves in +x
    double rake_start_x; // x-position at t=0 for some reference point on plane
    double rake_gap;     // Vertical gap from bed to rake's lowest edge (m)
    double rake_thickness; // Not used for contact thickness, but can be used
                           // for visualization

    CustomBodyTerm( CabanaPD::Inputs inputs, ContactType c, DensityType _rho,
                    ForceType _f, VelType _v, PosType _y )
        : contact_model( c )
        , rho( _rho )
        , f( _f )
        , v( _v )
        , y( _y )
    {
        rho0 = inputs["density"];
        vol0 = inputs["volume"];
        radius = inputs["radius"];
        low_corner = inputs["low_corner"];
        high_corner = inputs["high_corner"];

        double angle_deg = inputs["rake_angle"];
        rake_angle = angle_deg * M_PI / 180.0;
        rake_speed = inputs["rake_speed"]; // m/s
        rake_start_x = inputs["rake_start_x"];
        rake_gap = inputs["rake_gap"];
        rake_thickness = inputs["rake_thickness"];
    }

    KOKKOS_FUNCTION
    void operator()( const int p, const double time )
        const // Second input should be time for the rake motion to work
    {
        // Gravity force.
        f( p, 2 ) -= 9.8 * rho( p );

        // Interact with a horizontal wall (floor).
        double rz = y( p, 2 );
        if ( rz - radius < 0.0 )
        {
            double vz = v( p, 2 );
            double vn = rz * vz;
            vn /= rz;

            v( p, 0 ) = 0.0;
            v( p, 1 ) = 0.0;
            v( p, 2 ) = 0.0;

            f( p, 2 ) +=
                -contact_model.normalForce( rz + radius, vn, vol0, rho0 );
        }

        // Interact with vertical boundary walls.
        double rx = y( p, 0 ) >= 0.0 ? high_corner[0] - y( p, 0 )
                                     : y( p, 0 ) - low_corner[0];
        double ry = y( p, 1 ) >= 0.0 ? high_corner[1] - y( p, 1 )
                                     : y( p, 1 ) - low_corner[1];

        if ( rx - radius < 0.0 )
        {
            double nx = y( p, 0 ) >= 0.0 ? -1.0 : 1.0;
            double vn = v( p, 0 ) * nx;
            f( p, 0 ) +=
                -contact_model.normalForce( rx + radius, vn, vol0, rho0 ) * nx;
        }
        if ( ry - radius < 0.0 )
        {
            double ny = y( p, 1 ) >= 0.0 ? -1.0 : 1.0;
            double vn = v( p, 1 ) * ny;
            f( p, 1 ) +=
                -contact_model.normalForce( ry + radius, vn, vol0, rho0 ) * ny;
        }

        // -------------------------
        // Moving rake interaction
        // -------------------------
        // Rake moves in +x with rake_speed
        double rake_x_now = rake_start_x + rake_speed * time;

        // Define rake Corner point P0 = (rake_x_now, 0, rake_gap)
        double px = rake_x_now;
        double py = 0.0;
        double pz = rake_gap;

        // Rake plane normal (rotate about y-axis) (points towards +x, +z)
        double nx = Kokkos::sin( rake_angle );
        double ny = 0.0;
        double nz = -Kokkos::cos( rake_angle );

        // Signed distance from particle center to plane: d = n · (y - P0)
        double dx = y( p, 0 ) - px;
        double dy = y( p, 1 ) - py;
        double dz = y( p, 2 ) - pz;
        double signed_dist = nx * dx + ny * dy + nz * dz;

        // If signed distance < radius => overlap/contact (plane considered
        // infinitely thin)
        if ( signed_dist < radius )
        {
            // Relative normal velocity: particle vel dot n - rake_velocity dot
            // n
            double vn_particle =
                v( p, 0 ) * nx + v( p, 1 ) * ny + v( p, 2 ) * nz;
            // Rake velocity vector (rake moves only in x)
            double rake_vx = rake_speed;
            double vn_rake = rake_vx * nx; // dot product
            double vn_rel = vn_particle - vn_rake;

            // Gap penetration amount (positive = amount of overlap)
            double penetration = radius - signed_dist;

            // Contact normal force magnitude (use contact model)
            double fn =
                -contact_model.normalForce( penetration, vn_rel, vol0, rho0 );

            // Apply normal force on particle
            f( p, 0 ) += fn * nx;
            f( p, 1 ) += fn * ny;
            f( p, 2 ) += fn * nz;
        }
    }
};

template <class SolverType, class ContactType>
auto createBodyTerm( const SolverType& solver, const ContactType& contact_model,
                     CabanaPD::Inputs inputs )
{
    auto rho = solver.particles.sliceDensity();
    auto f = solver.particles.sliceForce();
    auto v = solver.particles.sliceVelocity();
    auto y = solver.particles.sliceCurrentPosition();

    return CustomBodyTerm( inputs, contact_model, rho, f, v, y );
}

template <class SolverType>
void updateParticles( SolverType& solver, CabanaPD::Inputs inputs,
                      const int num_previous = 0 )
{
    double rho0 = inputs["density"];
    double vol0 = inputs["volume"];
    double vel_init = inputs["velocity_init"];
    auto rho = solver.particles.sliceDensity();
    auto vol = solver.particles.sliceVolume();
    auto vel = solver.particles.sliceVelocity();
    auto y_init = solver.particles.sliceCurrentPosition();
    auto high_corner = inputs["high_corner"];
    double high_z = high_corner[2];
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        if ( pid < num_previous )
            return false;

        rho( pid ) = rho0;
        vol( pid ) = vol0;
        vel( pid, 2 ) =
            vel_init -
            Kokkos::sqrt( 2.0 * 9.8 * ( high_z - y_init( pid, 2 ) ) );
        vel( pid, 0 ) = 0.0;
        vel( pid, 1 ) = 0.0;
        vel( pid, 2 ) = 0.0;
        return true;
    };
    using exec_space = Kokkos::DefaultExecutionSpace;
    solver.particles.update( exec_space{}, num_previous, init_functor );
}

// Simulate pile + rake example.
void rakePileExample( const std::string filename )
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
    double radius = inputs["radius"];
    double radius_extend = inputs["radius_extend"];

    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];
    double gamma = inputs["surface_adhesion"];
    double mu = inputs["coulomb_friction"];

    // ====================================================
    //                  Mesh Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int halo_width = 1;

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::HertzianJKRModel;
    model_type contact_model( radius, radius_extend, nu, E, e, gamma, mu );

    // ====================================================
    //                   Creation
    // ====================================================
    // Initial bed of particles.
    CabanaPD::Particles particles( memory_space{}, model_type{},
                                   CabanaPD::BaseOutput{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, Cabana::InitRandom{} );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, contact_model );
    // int seed = 3837485;
    // addConeParticles( solver, inputs, seed );
    updateParticles( solver, inputs );
    solver.init();

    // ====================================================
    //                   Boundary condition
    // ====================================================
    auto body_func = createBodyTerm( solver, contact_model, inputs );
    CabanaPD::BodyTerm body( body_func, solver.particles.size(), true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    for ( int step = 1; step <= solver.num_steps; ++step )
    {
        solver.runStep( step, body );
    }
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    rakePileExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

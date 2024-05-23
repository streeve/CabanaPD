#include <cmath>
#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

void powderFillExample( const std::string filename )
{
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
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    double global_mesh_ext = high_corner[0] - low_corner[0];
    auto create_functor = KOKKOS_LAMBDA( const int, const double px[3] )
    {
        auto width = global_mesh_ext / 2.0;
        auto inner_width = global_mesh_ext / 2.0;
        auto r2 = px[0] * px[0] + px[1] * px[1];
        if ( r2 > width * width || r2 < inner_width * inner_width )
            return false;

        return true;
    };

    // Create particles from mesh.
    // Does not set displacements, velocities, etc.
    auto particles = std::make_shared<
        CabanaPD::Particles<memory_space, typename model_type::base_model>>(
        exec_space(), inputs["low_corner"], inputs["high_corner"],
        inputs["num_cells"], halo_width, create_functor );
    double dx = particles->dx[0];
    double r_c = dx * 2.0;

    // Define particle initialization.
    auto v = particles->sliceVelocity();
    auto rho = particles->sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        for ( int d = 0; d < 3; d++ )
            v( pid, d ) = 0.0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    auto f = particles->sliceForce();
    auto vol = particles->sliceVolume();
    auto body_functor = KOKKOS_LAMBDA( const int pid )
    {
        f( pid, 2 ) -= 9.8 * rho( pid ) * vol( pid );
    };
    auto gravity = CabanaPD::createBodyTerm( body_functor );

    // ====================================================
    //                   Simulation run
    // ====================================================
    CabanaPD::NormalRepulsionModel contact_model( delta, r_c, K );
    auto cabana_pd = CabanaPD::createSolverContact<memory_space>(
        inputs, particles, force_model, gravity, contact_model );
    cabana_pd->init_force();
    cabana_pd->run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    powderFillExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
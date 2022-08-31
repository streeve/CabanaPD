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

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_config.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Particles.hpp>

namespace Test
{

struct LinearTag
{
};
struct QuadraticTag
{
};

//---------------------------------------------------------------------------//
// Reference particle summations.
//---------------------------------------------------------------------------//
// Note: all of these reference calculations assume uniform volume and a full
// particle neighborhood.

//---------------------------------------------------------------------------//
// Get the PMB strain energy density (at the center point).
// Simplified here because the stretch is constant.
double computeReferenceStrainEnergyDensity( LinearTag, CabanaPD::PMBModel model,
                                            const int m, const double s0,
                                            const double )
{
    double W = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                {
                    W += 0.25 * model.c * s0 * s0 * xi * vol;
                }
            }
    return W;
}

double computeReferenceStrainEnergyDensity( QuadraticTag,
                                            CabanaPD::PMBModel model,
                                            const int m, const double u11,
                                            const double x )
{
    double W = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                {
                    W += 0.25 * model.c * s * s * xi * vol;
                }
            }
    return W;
}

template <class ModelType>
double computeReferenceForceX( LinearTag, ModelType, const int, const double,
                               const double )
{
    return 0.0;
}

// Get the PMB force (at one point).
// Assumes zero y/z displacement components.
double computeReferenceForceX( QuadraticTag, CabanaPD::PMBModel model,
                               const int m, const double u11, const double x )
{
    double fx = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                    fx += model.c * s * vol * rx / r;
            }
    return fx;
}

double computeReferenceWeightedVolume( const double delta, const int m,
                                       const double vol )
{
    double dx = delta / m;
    double weighted_volume = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                if ( xi > 0.0 && xi < delta + 1e-14 )
                    weighted_volume += 1.0 / xi * xi * xi * vol;
            }
    return weighted_volume;
}

// LinearTag
double computeReferenceDilatation( const double delta, const int m,
                                   const double s0, const double vol,
                                   const double weighted_volume )
{
    double dx = delta / m;
    double theta = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < delta + 1e-14 )
                    theta +=
                        3.0 / weighted_volume * 1.0 / xi * s0 * xi * xi * vol;
            }
    return theta;
}

// QuadraticTag
// Assumes zero y/z displacement components.
double computeReferenceDilatation( const double delta, const int m,
                                   const double u11, const double vol,
                                   const double weighted_volume,
                                   const double x )
{
    double dx = delta / m;
    double theta = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < delta + 1e-14 )
                    theta +=
                        3.0 / weighted_volume * 1.0 / xi * s * xi * xi * vol;
            }
    return theta;
}

double computeReferenceNeighbors( const double delta, const int m )
{
    double dx = delta / m;
    double num_neighbors = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < delta + 1e-14 )
                    num_neighbors += 1.0;
            }
    return num_neighbors;
}

// Get the LPS strain energy density (at one point).
double computeReferenceStrainEnergyDensity( LinearTag, CabanaPD::LPSModel model,
                                            const int m, const double s0,
                                            const double )
{
    double W = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;

    auto weighted_volume =
        computeReferenceWeightedVolume( model.delta, m, vol );
    auto theta =
        computeReferenceDilatation( model.delta, m, s0, vol, weighted_volume );
    auto num_neighbors = computeReferenceNeighbors( model.delta, m );

    // Strain energy.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                {
                    W += ( 1.0 / num_neighbors ) * 0.5 * model.theta_coeff /
                             3.0 * ( theta * theta ) +
                         0.5 * ( model.s_coeff / weighted_volume ) * 1.0 / xi *
                             s0 * s0 * xi * xi * vol;
                }
            }
    return W;
}

double computeReferenceStrainEnergyDensity( QuadraticTag,
                                            CabanaPD::LPSModel model,
                                            const int m, const double u11,
                                            const double x )
{
    double W = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;

    auto weighted_volume =
        computeReferenceWeightedVolume( model.delta, m, vol );
    auto num_neighbors = computeReferenceNeighbors( model.delta, m );
    auto theta_i = computeReferenceDilatation( model.delta, m, u11, vol,
                                               weighted_volume, x );

    // Strain energy.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                double x_j = x + xi_x;
                auto theta_j = computeReferenceDilatation(
                    model.delta, m, u11, vol, weighted_volume, x_j );
                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                {
                    W += ( 1.0 / num_neighbors ) * 0.5 * model.theta_coeff /
                             3.0 * ( theta_i * theta_j ) +
                         0.5 * ( model.s_coeff / weighted_volume ) * 1.0 / xi *
                             s * s * xi * xi * vol;
                }
            }
    return W;
}

// Get the LPS strain energy density (at one point).
// Assumes zero y/z displacement components.
double computeReferenceForceX( QuadraticTag, CabanaPD::LPSModel model,
                               const int m, const double u11, const double x )
{
    double fx = 0.0;
    double dx = model.delta / m;
    double vol = dx * dx * dx;

    auto weighted_volume =
        computeReferenceWeightedVolume( model.delta, m, vol );
    auto theta_i = computeReferenceDilatation( model.delta, m, u11, vol,
                                               weighted_volume, x );
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                double x_j = x + xi_x;
                auto theta_j = computeReferenceDilatation(
                    model.delta, m, u11, vol, weighted_volume, x_j );
                if ( xi > 0.0 && xi < model.delta + 1e-14 )
                {
                    fx += ( model.theta_coeff * ( theta_i / weighted_volume +
                                                  theta_j / weighted_volume ) +
                            model.s_coeff * s *
                                ( 1.0 / weighted_volume +
                                  1.0 / weighted_volume ) ) *
                          1 / xi * xi * vol * rx / r;
                }
            }
    return fx;
}

//---------------------------------------------------------------------------//
// System creation.
//---------------------------------------------------------------------------//

auto createParticles( LinearTag, const double dx, const double s0 )
{
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    int nc = ( box_max[0] - box_min[0] ) / dx;
    std::array<int, 3> num_cells = { nc, nc, nc };

    // Create particles based on the mesh.
    using ptype = CabanaPD::Particles<TEST_MEMSPACE>;
    ptype particles( TEST_EXECSPACE{}, box_min, box_max, num_cells, 0 );

    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto v = particles.slice_v();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        for ( int d = 0; d < 3; d++ )
        {
            u( pid, d ) = s0 * x( pid, d );
            v( pid, d ) = 0.0;
        }
    };
    particles.update_particles( TEST_EXECSPACE{}, init_functor );
    return particles;
}

auto createParticles( QuadraticTag, const double dx, const double s0 )
{
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    int nc = ( box_max[0] - box_min[0] ) / dx;
    std::array<int, 3> num_cells = { nc, nc, nc };

    // Create particles based on the mesh.
    using ptype = CabanaPD::Particles<TEST_MEMSPACE>;
    ptype particles( TEST_EXECSPACE{}, box_min, box_max, num_cells, 0 );
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto v = particles.slice_v();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        for ( int d = 0; d < 3; d++ )
        {
            u( pid, d ) = 0.0;
            v( pid, d ) = 0.0;
        }
        u( pid, 0 ) = s0 * x( pid, 0 ) * x( pid, 0 );
    };
    particles.update_particles( TEST_EXECSPACE{}, init_functor );
    return particles;
}

//---------------------------------------------------------------------------//
// Check all particles.
//---------------------------------------------------------------------------//
template <class HostParticleType, class TestTag, class ModelType>
void checkResults( HostParticleType aosoa_host, double local_min[3],
                   double local_max[3], TestTag test_tag, ModelType model,
                   const int m, const double s0, const int boundary_width,
                   const double Phi )
{
    double delta = model.delta;
    double ref_Phi = 0.0;
    auto f_host = Cabana::slice<0>( aosoa_host );
    auto x_host = Cabana::slice<1>( aosoa_host );
    auto W_host = Cabana::slice<2>( aosoa_host );
    auto vol_host = Cabana::slice<3>( aosoa_host );
    auto theta_host = Cabana::slice<4>( aosoa_host );
    // Check the results: avoid the system boundary for per particle values.
    int particles_checked = 0;
    for ( std::size_t p = 0; p < aosoa_host.size(); ++p )
    {
        double x = x_host( p, 0 );
        double y = x_host( p, 1 );
        double z = x_host( p, 2 );
        if ( x > local_min[0] + delta * boundary_width &&
             x < local_max[0] - delta * boundary_width &&
             y > local_min[1] + delta * boundary_width &&
             y < local_max[1] - delta * boundary_width &&
             z > local_min[2] + delta * boundary_width &&
             z < local_max[2] - delta * boundary_width )
        {
            // These are constant for linear, but vary for quadratic.
            double ref_W = computeReferenceStrainEnergyDensity( test_tag, model,
                                                                m, s0, x );
            double ref_f = computeReferenceForceX( test_tag, model, m, s0, x );
            checkParticle( test_tag, model, s0, f_host( p, 0 ), f_host( p, 1 ),
                           f_host( p, 2 ), ref_f, W_host( p ), ref_W, x );
            particles_checked++;
        }
        checkAnalyticalDilatation( model, test_tag, s0, theta_host( p ) );

        // Check total sum of strain energy matches per particle sum.
        ref_Phi += W_host( p ) * vol_host( p );
    }

    EXPECT_NEAR( Phi, ref_Phi, 1e-5 );
    std::cout << "Particles checked: " << particles_checked << std::endl;
}

//---------------------------------------------------------------------------//
// Individual checks per particle.
//---------------------------------------------------------------------------//
template <class ModelType>
void checkParticle( LinearTag tag, ModelType model, const double s0,
                    const double fx, const double fy, const double fz,
                    const double, const double W, const double ref_W,
                    const double )
{
    EXPECT_LE( fx, 1e-13 );
    EXPECT_LE( fy, 1e-13 );
    EXPECT_LE( fz, 1e-13 );

    // Check strain energy (all should be equal for fixed stretch).
    EXPECT_FLOAT_EQ( W, ref_W );

    // Check energy with analytical value.
    checkAnalyticalStrainEnergy( tag, model, s0, W, -1 );
}

template <class ModelType>
void checkParticle( QuadraticTag tag, ModelType model, const double s0,
                    const double fx, const double fy, const double fz,
                    const double ref_f, const double W, const double ref_W,
                    const double x )
{
    // Check force in x with discretized result (reference currently incorrect).
    EXPECT_FLOAT_EQ( fx, ref_f );

    // Check force in x with analytical value.
    checkAnalyticalForce( tag, model, s0, fx );

    // Check force: other components should be zero.
    EXPECT_LE( fy, 1e-13 );
    EXPECT_LE( fz, 1e-13 );

    // Check energy. Not quite within the floating point tolerance.
    EXPECT_NEAR( W, ref_W, 1e-6 );

    // Check energy with analytical value.
    checkAnalyticalStrainEnergy( tag, model, s0, W, x );
}

void checkAnalyticalStrainEnergy( LinearTag, CabanaPD::PMBModel model,
                                  const double s0, const double W,
                                  const double )
{
    // Relatively large error for small m.
    double threshold = W * 0.15;
    double analytical_W = 9.0 / 2.0 * model.K * s0 * s0;
    EXPECT_NEAR( W, analytical_W, threshold );
}

void checkAnalyticalStrainEnergy( LinearTag, CabanaPD::LPSModel model,
                                  const double s0, const double W,
                                  const double )
{
    // LPS is exact.
    double analytical_W = 9.0 / 2.0 * model.K * s0 * s0;
    EXPECT_FLOAT_EQ( W, analytical_W );
}

void checkAnalyticalStrainEnergy( QuadraticTag, CabanaPD::PMBModel model,
                                  const double u11, const double W,
                                  const double x )
{
    double threshold = W * 0.05;
    double analytical_W =
        18.0 * model.K * u11 * u11 *
        ( 1.0 / 5.0 * x * x + model.delta * model.delta / 42.0 );
    EXPECT_NEAR( W, analytical_W, threshold );
}

void checkAnalyticalStrainEnergy( QuadraticTag, CabanaPD::LPSModel model,
                                  const double u11, const double W,
                                  const double x )
{
    double threshold = W * 0.20;
    double analytical_W =
        u11 * u11 *
        ( ( 2 * model.K + 8.0 / 3.0 * model.G ) * x * x +
          75.0 / 2.0 * model.G * model.delta * model.delta / 49.0 );
    EXPECT_NEAR( W, analytical_W, threshold );
}

void checkAnalyticalForce( QuadraticTag, CabanaPD::PMBModel model,
                           const double s0, const double fx )
{
    double threshold = fx * 0.10;
    double analytical_f = 18.0 / 5.0 * model.K * s0;
    EXPECT_NEAR( fx, analytical_f, threshold );
}

void checkAnalyticalForce( QuadraticTag, CabanaPD::LPSModel model,
                           const double s0, const double fx )
{
    double threshold = fx * 0.10;
    double analytical_f = 2.0 * ( model.K + 4.0 / 3.0 * model.G ) * s0;
    EXPECT_NEAR( fx, analytical_f, threshold );
}

void checkAnalyticalDilatation( CabanaPD::PMBModel, LinearTag, const double,
                                const double theta )
{
    EXPECT_FLOAT_EQ( 0.0, theta );
}

void checkAnalyticalDilatation( CabanaPD::LPSModel, LinearTag, const double s0,
                                const double theta )
{
    EXPECT_FLOAT_EQ( 3 * s0, theta );
}

template <class ModelType>
void checkAnalyticalDilatation( ModelType, QuadraticTag, const double,
                                const double )
{
}

//---------------------------------------------------------------------------//
// Main test function.
//---------------------------------------------------------------------------//
template <class ModelType, class TestTag>
void testForce( ModelType model, const double dx, const double m,
                const double boundary_width, const TestTag test_tag,
                const double s0 )
{
    auto particles = createParticles( test_tag, dx, s0 );

    // This needs to exactly match the mesh spacing to compare with the single
    // particle calculation.
    CabanaPD::Force<TEST_EXECSPACE, ModelType> force( true, model );

    double mesh_min[3] = { particles.ghost_mesh_lo[0],
                           particles.ghost_mesh_lo[1],
                           particles.ghost_mesh_lo[2] };
    double mesh_max[3] = { particles.ghost_mesh_hi[0],
                           particles.ghost_mesh_hi[1],
                           particles.ghost_mesh_hi[2] };
    using verlet_list =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    // Add to delta to make sure neighbors are found.
    auto x = particles.slice_x();
    verlet_list neigh_list( x, 0, particles.n_local, model.delta + 1e-14, 1.0,
                            mesh_min, mesh_max );

    auto f = particles.slice_f();
    auto W = particles.slice_W();
    auto vol = particles.slice_vol();
    auto theta = particles.slice_theta();
    auto m = particles.slice_m();
    force.initialize( particles, neigh_list, Cabana::SerialOpTag() );
    compute_force( force, particles, neigh_list, Cabana::SerialOpTag() );

    auto Phi =
        compute_energy( force, particles, neigh_list, Cabana::SerialOpTag() );

    // Make a copy of final results on the host
    std::size_t num_particle = x.size();
    using HostAoSoA = Cabana::AoSoA<
        Cabana::MemberTypes<double[3], double[3], double, double, double>,
        Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", num_particle );
    auto f_host = Cabana::slice<0>( aosoa_host );
    auto x_host = Cabana::slice<1>( aosoa_host );
    auto W_host = Cabana::slice<2>( aosoa_host );
    auto vol_host = Cabana::slice<3>( aosoa_host );
    auto theta_host = Cabana::slice<4>( aosoa_host );
    Cabana::deep_copy( f_host, f );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( W_host, W );
    Cabana::deep_copy( vol_host, vol );
    Cabana::deep_copy( theta_host, theta );

    double local_min[3] = { particles.local_mesh_lo[0],
                            particles.local_mesh_lo[1],
                            particles.local_mesh_lo[2] };
    double local_max[3] = { particles.local_mesh_hi[0],
                            particles.local_mesh_hi[1],
                            particles.local_mesh_hi[2] };

    checkResults( aosoa_host, local_min, local_max, test_tag, model, m, s0,
                  boundary_width, Phi );
}

//---------------------------------------------------------------------------//
// GTest tests.
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_force_pmb )
{
    // dx needs to be decreased for increased m: boundary particles are ignored.
    double m = 3;
    double dx = 2.0 / 11.0;
    double delta = dx * m;
    double K = 1.0;
    CabanaPD::PMBModel model( delta, K );
    testForce( model, dx, m, 1.1, LinearTag{}, 0.1 );
    testForce( model, dx, m, 1.1, QuadraticTag{}, 0.01 );
}
TEST( TEST_CATEGORY, test_force_linear_pmb )
{
    double m = 3;
    double dx = 2.0 / 11.0;
    double delta = dx * m;
    double K = 1.0;
    CabanaPD::LinearPMBModel model( delta, K );
    testForce( model, dx, m, 1.1, LinearTag{}, 0.1 );
}
TEST( TEST_CATEGORY, test_force_lps )
{
    double m = 3;
    // Need a larger system than PMB because the boundary region is larger.
    double dx = 2.0 / 15.0;
    double delta = dx * m;
    double K = 1.0;
    double G = 0.5;
    CabanaPD::LPSModel model( delta, K, G );
    testForce( model, dx, m, 2.1, LinearTag{}, 0.1 );
    testForce( model, dx, m, 2.1, QuadraticTag{}, 0.01 );
}
TEST( TEST_CATEGORY, test_force_linear_lps )
{
    double m = 3;
    double dx = 2.0 / 15.0;
    double delta = dx * m;
    double K = 1.0;
    double G = 0.5;
    CabanaPD::LinearLPSModel model( delta, K, G );
    testForce( model, dx, m, 2.1, LinearTag{}, 0.1 );
}

} // end namespace Test

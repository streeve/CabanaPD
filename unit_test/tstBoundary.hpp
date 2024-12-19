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

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_BodyTerm.hpp>
#include <CabanaPD_Boundary.hpp>
#include <CabanaPD_Particles.hpp>

namespace Test
{
template <typename ParticlesType, typename BoundaryType>
void checkBoundaryValues( const ParticlesType& particles,
                          const BoundaryType& boundary, const double expected,
                          const std::size_t start, const std::size_t end )
{
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3]>, Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", particles.referenceOffset() );
    auto f_host = Cabana::slice<0>( aosoa_host );
    auto f = particles.sliceForce();
    Cabana::deep_copy( f_host, f );

    // Check the particles were created in the right box.
    for ( std::size_t p = start; p < end; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_EQ( f_host( p, d ), expected );
        }
}

//---------------------------------------------------------------------------//
void testBoundaryCondition()
{
    /*
    using exec_space = TEST_EXECSPACE;

    double val = 11.5;
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    // Frozen or all particles first.
    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space(), box_min, box_max, num_cells, 0 );

    checkBoundaryValues( particles, particles.localOffset() );
    */
}

void testBodyTerm()
{
    using exec_space = TEST_EXECSPACE;

    double val = 7.5;
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    // Create particles.
    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space(), box_min, box_max, num_cells,
                   0 ); // , 0, true );

    // Create and apply body force.
    auto f = particles.sliceForce();
    auto func = KOKKOS_LAMBDA( const int p, const double )
    {
        for ( int d = 0; d < 3; ++d )
            f( p, d ) = val;
    };
    auto body = CabanaPD::createBodyTerm( func, true );
    body.apply( exec_space{}, particles, 0.0 );

    // Ensure we created particles.
    std::size_t expected_local = num_cells[0] * num_cells[1] * num_cells[2];
    EXPECT_EQ( particles.localOffset(), expected_local );

    // Check results for all particles.
    checkBoundaryValues( particles, body, val, 0, particles.localOffset() );
}

void testBodyTermFrozen()
{
    using exec_space = TEST_EXECSPACE;

    double val = 7.5;
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    // Create particles.
    // Frozen in bottom half.
    auto init_bottom = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( x[2] < 0.0 )
            return true;
        return false;
    };
    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space(), box_min, box_max, num_cells, 0, init_bottom, 0,
                   true );

    // Unfrozen in top half.
    auto init_top = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( x[2] > 0.0 )
            return true;
        return false;
    };
    // Create more, starting from the current number of frozen points.
    particles.createParticles( exec_space{}, init_top, particles.numFrozen() );

    // Create and apply body force.
    auto f = particles.sliceForce();
    auto func = KOKKOS_LAMBDA( const int p, const double )
    {
        for ( int d = 0; d < 3; ++d )
            f( p, d ) = val;
    };
    auto body = CabanaPD::createBodyTerm( func, true );
    body.apply( exec_space{}, particles, 0.0 );

    // Ensure we created particles.
    std::size_t expected_local = num_cells[0] * num_cells[1] * num_cells[2];
    EXPECT_EQ( particles.localOffset(), expected_local );

    // Check results. Frozen should not have been updated.
    checkBoundaryValues( particles, body, 0.0, 0, particles.frozenOffset() );
    checkBoundaryValues( particles, body, val, particles.frozenOffset(),
                         particles.localOffset() );

    // Do it again, but including frozen particles.
    Cabana::deep_copy( f, 0.0 );
    auto body_all = CabanaPD::createBodyTerm( func, true, true );
    body_all.apply( exec_space{}, particles, 0.0 );
    checkBoundaryValues( particles, body_all, val, particles.frozenOffset(),
                         particles.localOffset() );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_boundary ) { testBoundaryCondition(); }
TEST( TEST_CATEGORY, test_body )
{
    testBodyTerm();
    testBodyTermFrozen();
}

//---------------------------------------------------------------------------//

} // end namespace Test

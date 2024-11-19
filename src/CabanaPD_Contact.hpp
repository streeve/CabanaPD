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

#ifndef CONTACT_H
#define CONTACT_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
/******************************************************************************
  Contact model
******************************************************************************/
struct ContactModel
{
    double delta;
    double Rc;

    ContactModel(){};
    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta )
        , Rc( _Rc ){};
};

/* Normal repulsion */

struct NormalRepulsionModel : public ContactModel
{
    using ContactModel::delta;
    using ContactModel::Rc;

    double c;
    double K;

    NormalRepulsionModel(){};
    NormalRepulsionModel( const double delta, const double Rc, const double _K )
        : ContactModel( delta, Rc )
        , K( _K )
    {
        set_param( delta, Rc, K );
    }

    void set_param( const double _delta, const double _Rc, const double _K )
    {
        delta = _delta;
        Rc = _Rc;
        K = _K;
        // This could inherit from PMB (same c)
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const double sc, const double vol )
    {
        // Normal repulsion uses a 15 factor compared to the PMB force
        return 15.0 * c * sc * vol;
    }
};

// Forward declaration.
template <class MemorySpace, class ContactModelType>
class ContactForce;

/******************************************************************************
  Normal repulsion computation
******************************************************************************/
template <class MemorySpace>
class ContactForce<MemorySpace, NormalRepulsionModel>
    : public Force<MemorySpace, BaseForceModel>
{
  public:
    using base_type = Force<MemorySpace, BaseForceModel>;

    template <class ParticleType>
    ContactForce( const bool half_neigh, const ParticleType& particles,
                  const NormalRepulsionModel model )
        : base_type( half_neigh, model.Rc, particles.sliceCurrentPosition(),
                     particles.n_local, particles.ghost_mesh_lo,
                     particles.ghost_mesh_hi )
        , _model( model )
    {
        for ( int d = 0; d < particles.dim; d++ )
        {
            mesh_min[d] = particles.ghost_mesh_lo[d];
            mesh_max[d] = particles.ghost_mesh_hi[d];
        }
    }

    template <class ForceType, class ParticleType, class ParallelType>
    void computeForceFull( ForceType& fc, const ParticleType& particles,
                           const int n_local, ParallelType& neigh_op_tag ) const
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto y = particles.sliceCurrentPosition();

        _neigh_list.build( y, 0, n_local, model.Rc, 1.0, mesh_min, mesh_max );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Contact "stretch"
            const double sc = ( r - model.Rc ) / model.delta;

            // Normal repulsion uses a 15 factor compared to the PMB force
            const double coeff = model.forceCoeff( sc, vol( j ) );
            fcx_i = coeff * rx / r;
            fcy_i = coeff * ry / r;
            fcz_i = coeff * rz / r;

            fc( i, 0 ) += fcx_i;
            fc( i, 1 ) += fcy_i;
            fc( i, 2 ) += fcz_i;
        };

        // FIXME: using default space for now.
        using exec_space = typename MemorySpace::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::Contact::compute_full" );
    }

  protected:
    NormalRepulsionModel _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;

    double mesh_max[3];
    double mesh_min[3];
};

template <class ForceType, class ParticleType, class ParallelType>
void computeContact( const ForceType& force, ParticleType& particles,
                     const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto f = particles.sliceForce();
    auto f_a = particles.sliceForceAtomic();

    // Do NOT reset force - this is assumed to be done by the primary force
    // kernel.

    // if ( half_neigh )
    // Forces must be atomic for half list
    // compute_force_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading
    if constexpr ( std::is_same<decltype( neigh_op_tag ),
                                Cabana::TeamOpTag>::value )
        force.computeContactFull( f_a, particles, n_local, neigh_op_tag );
    else
        force.computeContactFull( f, particles, n_local, neigh_op_tag );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif

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

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef FORCE_PMB_H
#define FORCE_PMB_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>
#include <force/CabanaPD_ForceModels_PMB.hpp>

namespace CabanaPD
{
template <class ExecutionSpace, class... ModelParams>
class Force<ExecutionSpace, ForceModel<PMB, Elastic, ModelParams...>>
{
  public:
    using exec_space = ExecutionSpace;
    using model_type = ForceModel<PMB, Elastic, ModelParams...>;

  protected:
    bool _half_neigh;
    model_type _model;

  public:
    Force( const bool half_neigh, const model_type model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeWeightedVolume( ParticleType&, const NeighListType&,
                                const ParallelType )
    {
    }
    template <class ParticleType, class NeighListType, class ParallelType>
    void computeDilatation( ParticleType&, const NeighListType&,
                            const ParallelType )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, const int n_local,
                           ParallelType& neigh_op_tag ) const
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            model.thermalStretch( s, i, j );

            const double coeff = model.c * s * vol( j );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::computeFull" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              const ParticleType& particles,
                              const NeighListType& neigh_list,
                              const int n_local,
                              ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.sliceVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
            // the integrand (pairwise potential).
            double w = 0.25 * c * s * s * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForcePMB::computeEnergyFull" );

        return strain_energy;
    }
};

template <class ExecutionSpace, class... ModelParams>
class Force<ExecutionSpace, ForceModel<PMB, Fracture, ModelParams...>>
    : public Force<ExecutionSpace, ForceModel<PMB, Elastic, ModelParams...>>
{
  public:
    using exec_space = ExecutionSpace;
    using model_type = ForceModel<PMB, Fracture>;

  protected:
    using base_type =
        Force<ExecutionSpace, ForceModel<PMB, Elastic, ModelParams...>>;
    using base_type::_half_neigh;
    model_type _model;

  public:
    Force( const bool half_neigh, const model_type model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, MuView& mu,
                           const int n_local, ParallelType& ) const
    {
        auto model = _model;
        auto break_coeff = _model.bond_break_coeff;
        const auto vol = particles.sliceVolume();
        const auto nofail = particles.sliceNoFail();

        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

                model.thermalStretch( s, i, j );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( r * r >= break_coeff * xi * xi && !nofail( i ) &&
                     !nofail( j ) )
                {
                    mu( i, n ) = 0;
                }
                // Else if statement is only for performance.
                else if ( mu( i, n ) > 0 )
                {
                    const double coeff = model.c * s * vol( j );
                    double muij = mu( i, n );
                    fx_i = muij * coeff * rx / r;
                    fy_i = muij * coeff * ry / r;
                    fz_i = muij * coeff * rz / r;

                    f( i, 0 ) += fx_i;
                    f( i, 1 ) += fy_i;
                    f( i, 2 ) += fz_i;
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForcePMBDamage::computeFull", policy,
                              force_full );
    }

    template <class PosType, class WType, class DamageType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              DamageType& phi, const ParticleType& particles,
                              const NeighListType& neigh_list, MuView& mu,
                              const int n_local, ParallelType& ) const
    {
        auto c = _model.c;
        const auto vol = particles.sliceVolume();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                // 0.25 factor is due to 1/2 from outside the integral and 1/2
                // from the integrand (pairwise potential).
                double w = mu( i, n ) * 0.25 * c * s * s * xi * vol( j );
                W( i ) += w;

                phi_i += mu( i, n ) * vol( j );
                vol_H_i += vol( j );
            }
            Phi += W( i ) * vol( i );
            phi( i ) = 1 - phi_i / vol_H_i;
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_reduce( "CabanaPD::ForcePMBDamage::computeEnergyFull",
                                 policy, energy_full, strain_energy );

        return strain_energy;
    }
};

template <class ExecutionSpace, class... ModelParams>
class Force<ExecutionSpace, ForceModel<LinearPMB, Elastic, ModelParams...>>
    : public Force<ExecutionSpace, ForceModel<PMB, Elastic, ModelParams...>>
{
  public:
    using exec_space = ExecutionSpace;
    using model_type = ForceModel<LinearPMB, Elastic>;

  protected:
    using base_type = Force<ExecutionSpace, ForceModel<PMB, Elastic>>;
    using base_type::_half_neigh;
    model_type _model;

  public:
    Force( const bool half_neigh, const model_type model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           ParticleType& particles,
                           const NeighListType& neigh_list, const int n_local,
                           ParallelType& neigh_op_tag ) const
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance, displacement, and linearized stretch.
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x,
                                             xi_y, xi_z );

            model.thermalStretch( linear_s, i, j );

            const double coeff = model.c * linear_s * vol( j );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLinearPMB::computeFull" );
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double
    computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                       ParticleType& particles, const NeighListType& neigh_list,
                       const int n_local, ParallelType& neigh_op_tag ) const
    {
        auto c = _model.c;
        const auto vol = particles.sliceVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and linearized stretch.
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
            // the integrand (pairwise potential).
            double w = 0.25 * c * linear_s * linear_s * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLinearPMB::computeEnergyFull" );

        return strain_energy;
    }
};

} // namespace CabanaPD

#endif

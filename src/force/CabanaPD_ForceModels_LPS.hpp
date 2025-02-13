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

#ifndef FORCE_MODELS_LPS_H
#define FORCE_MODELS_LPS_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <>
struct ForceModel<LPS, Elastic, NoFracture> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = LPS;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;

    int influence_type;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    ForceModel( const double _delta, const double _K, const double _G,
                const int _influence = 0 )
        : base_type( _delta )
        , influence_type( _influence )
        , K( _K )
        , G( _G )
    {
        theta_coeff = 3.0 * K - 5.0 * G;
        s_coeff = 15.0 * G;
    }

    KOKKOS_INLINE_FUNCTION double influenceFunction( double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }

    KOKKOS_INLINE_FUNCTION auto weightedVolume( const double xi,
                                                const double vol ) const
    {
        return influenceFunction( xi ) * xi * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION auto dilatation( const int, const double s,
                                            const double xi, const double vol,
                                            const double m_i ) const
    {
        double theta_i = influenceFunction( xi ) * s * xi * xi * vol;
        return 3.0 * theta_i / m_i;
    }

    KOKKOS_INLINE_FUNCTION auto forceCoeff( const double s, const double xi,
                                            const double vol, const double m_i,
                                            const double m_j,
                                            const double theta_i,
                                            const double theta_j ) const
    {
        return ( theta_coeff * ( theta_i / m_i + theta_j / m_j ) +
                 s_coeff * s * ( 1.0 / m_i + 1.0 / m_j ) ) *
               influenceFunction( xi ) * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto energy( const double s, const double xi, const double vol,
                 const double m_i, const double theta_i,
                 const double num_bonds ) const
    {
        return 1.0 / num_bonds * 0.5 * theta_coeff / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff / m_i ) * influenceFunction( xi ) * s * s * xi *
                   xi * vol;
    }
};

template <>
struct ForceModel<LPS, Elastic, Fracture>
    : public ForceModel<LPS, Elastic, NoFracture>
{
    using base_type = ForceModel<LPS, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using mechanics_type = Elastic;
    using thermal_type = base_type::thermal_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel( const double _delta, const double _K, const double _G,
                const double _G0, const int _influence = 0 )
        : base_type( _delta, _K, _G, _influence )
        , G0( _G0 )
    {
        if ( influence_type == 1 )
        {
            s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / K / delta ); // 1/xi
        }
        else
        {
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / K / delta ); // 1
        }
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    // Constructor to work with plasticity.
    ForceModel( const double _delta, const double _K, const double _G,
                const double _G0, const double _s0, const int _influence = 0 )
        : base_type( _delta, _K, _G, _influence )
        , G0( _G0 )
        , s0( _s0 )
    {
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <typename MemorySpace>
struct ForceModel<LPS, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>,
      public BasePlasticity<MemorySpace>

{
    using base_type = ForceModel<LPS, Elastic, Fracture>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = typename base_type::mechanics_type;
    using thermal_type = typename base_type::thermal_type;

    using base_type::bond_break_coeff;
    using base_type::delta;
    using base_type::G0;
    using base_type::K;
    using base_type::s0;

    using base_plasticity_type::_s_p;
    double s_Y;

    // Define which base functions to use.
    using base_plasticity_type::updateBonds;
    using base_type::energy;
    using base_type::forceCoeff;

    ForceModel( const double delta, const double K, const double G,
                const double G0, const double sigma_y )
        : base_type( delta, K, G, G0,
                     // s0
                     ( 5.0 * G0 / sigma_y / delta + sigma_y / K ) / 6.0 )
        , base_plasticity_type()
        , s_Y( sigma_y / 3.0 / K )
    {
    }
};

template <typename ModelType, typename ParticleType>
auto createForceModel( ModelType, ElasticPerfectlyPlastic, ParticleType,
                       const double delta, const double K, const double G,
                       const double G0, const double sigma_y )
{
    using memory_space = typename ParticleType::memory_space;
    return ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture,
                      TemperatureIndependent, memory_space>( delta, K, G, G0,
                                                             sigma_y );
}

template <typename DensityType>
struct ForceModel<LPS, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, DynamicDensity, DensityType>
    : public ForceModel<LPS, ElasticPerfectlyPlastic, Fracture,
                        TemperatureIndependent,
                        typename DensityType::memory_space>,
      ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<LPS, ElasticPerfectlyPlastic, Fracture,
                   TemperatureIndependent, typename DensityType::memory_space>;
    using lps_base_type =
        ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = typename base_type::mechanics_type;
    using thermal_type = typename base_type::thermal_type;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;

    using base_type::_s_p;
    using base_type::s_Y;
    double coeff;

    double rho0;
    DensityType rho;

    // Define which base functions to use.
    using base_type::energy;
    using base_type::forceCoeff;

    ForceModel( const double delta, const double K, const double G0,
                const double sigma_y, const double _rho0,
                const DensityType _rho )
        : base_type( delta, K, G0, sigma_y )
        , rho0( _rho0 )
        , rho( _rho )
    {
        coeff = 3.0 / pi / delta / delta / delta / delta;
    }

    // Fused density update using plastic dilatation.
    KOKKOS_INLINE_FUNCTION auto density( const int i, const double s,
                                         const double xi, const double vol,
                                         const double ) const
    {
        double theta_i = coeff * s * xi * vol;

        // Update density using updated dilatation.
        // Note that this assumes zero initial plastic dilatation.
        rho( i ) = rho0 * Kokkos::exp( theta_i ); // exp(theta_i - theta_i_0)

        return theta_i;
    }
};

template <typename ParticleType>
auto createForceModel( LPS, ElasticPerfectlyPlastic, ParticleType particles,
                       const double delta, const double K, const double G,
                       const double G0, const double sigma_y,
                       const double rho0 )
{
    auto rho = particles.sliceDensity();
    using rho_type = decltype( rho );
    return ForceModel<LPS, ElasticPerfectlyPlastic, Fracture,
                      TemperatureDependent, DynamicDensity, rho_type>(
        delta, K, G, G0, sigma_y, rho0, rho );
}

template <>
struct ForceModel<LinearLPS, Elastic, NoFracture>
    : public ForceModel<LPS, Elastic, NoFracture>
{
    using base_type = ForceModel<LPS, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
};

template <>
struct ForceModel<LinearLPS, Elastic, Fracture>
    : public ForceModel<LPS, Elastic, Fracture>
{
    using base_type = ForceModel<LPS, Elastic, Fracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

} // namespace CabanaPD

#endif

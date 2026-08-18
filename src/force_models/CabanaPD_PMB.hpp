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

#ifndef FORCE_MODELS_PMB_H
#define FORCE_MODELS_PMB_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>
#include <force_models/CabanaPD_Contact.hpp>

namespace CabanaPD
{

template <typename FunctorType>
struct MechanicsModel<PMB, Elastic, FunctorType> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using mechanics_type = Elastic;

    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = PMB;
    using force_tag = PMB;

    using base_type::force_horizon;
    FunctorType K;
    double c_factor;

    MechanicsModel( PMB tag, Elastic mech, const double force_horizon,
                    const double _K )
        : MechanicsModel( tag, mech, force_horizon, ConstantProperty( _K ) )
    {
    }

    MechanicsModel( PMB tag, const double force_horizon, const double _K )
        : MechanicsModel( tag, force_horizon, ConstantProperty( _K ) )
    {
    }

    MechanicsModel( PMB tag, const double force_horizon, const FunctorType _K )
        : MechanicsModel( tag, Elastic{}, force_horizon, _K )
    {
    }

    MechanicsModel( PMB, Elastic, const double force_horizon,
                    const FunctorType _K )
        : base_type( force_horizon )
        , K( _K )
    {
        c_factor = 18.0 / ( pi * force_horizon * force_horizon * force_horizon *
                            force_horizon );
    }

    // Constructor to average from existing models.
    // FIXME: use the first model functional dependence for now.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , K( model1.K )
    {
    }

    KOKKOS_FUNCTION
    auto c( const int i, const int j ) const
    {
        return 0.5 * ( K( i ) + K( j ) ) * c_factor;
    }

    KOKKOS_FUNCTION
    int influenceType() const { return 1; }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int j, const double s,
                     const double vol, const double, const int = -1 ) const
    {
        return c( i, j ) * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int i, const int j, const double s,
                     const double xi, const double vol, const double,
                     const int = -1 ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c( i, j ) * s * s * xi * vol;
    }

    // For multi material.
    KOKKOS_INLINE_FUNCTION
    auto operator()( DilatationTag, const int, const int, const double,
                     const double, const double, const double, const int ) const
    {
        return 0.0;
    }
    KOKKOS_INLINE_FUNCTION auto operator()( DensityTag, const int,
                                            const double ) const
    {
    }
    auto getPlasticStretch( const int, const int ) const { return 0.0; }
};

template <typename ModelType>
MechanicsModel( ModelType, Elastic, const double force_horizon, const double K )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty>;

template <typename ModelType>
MechanicsModel( ModelType, const double force_horizon, const double K )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty>;

template <typename ModelType, typename FunctorType>
MechanicsModel( ModelType, Elastic, const double force_horizon,
                const FunctorType K )
    -> MechanicsModel<ModelType, Elastic, FunctorType>;

template <typename MemorySpace, typename FunctorModulus, typename FunctorYield>
struct MechanicsModel<PMB, ElasticPerfectlyPlastic, FunctorModulus,
                      FunctorYield, MemorySpace>
    : public MechanicsModel<PMB, Elastic, FunctorModulus>,
      public BasePlasticity<MemorySpace>
{
    using base_type = MechanicsModel<PMB, Elastic, FunctorModulus>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    FunctorYield s_Y;

    using base_plasticity_type::updateBonds;

    MechanicsModel( PMB tag, ElasticPerfectlyPlastic, MemorySpace,
                    const double force_horizon, const double K,
                    const double sigma_y )
        : base_type( tag, force_horizon, K )
        , base_plasticity_type()
        , s_Y( ConstantProperty( sigma_y / 3.0 / K ) )
    {
    }

    MechanicsModel( PMB tag, ElasticPerfectlyPlastic, MemorySpace,
                    const double force_horizon, const FunctorModulus K,
                    const FunctorYield _s_Y )
        : base_type( tag, force_horizon, K )
        , base_plasticity_type()
        , s_Y( _s_Y )
    {
    }

    // Constructor to average from existing models.
    // FIXME: use the first model functional dependence for now.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_plasticity_type()
        , s_Y( model1.s_Y )
    {
    }

    KOKKOS_INLINE_FUNCTION
    auto updatePlasticStretch( const int i, const double s, const double time,
                               const int n ) const
    {
        // Update bond plastic stretch.
        auto s_p = _s_p( i, n );
        // Yield in tension.
        if ( s >= s_p + s_Y( i, time ) )
            _s_p( i, n ) = s - s_Y( i, time );
        // Yield in compression.
        else if ( s <= s_p - s_Y( i, time ) )
            _s_p( i, n ) = s + s_Y( i, time );
        // else: Elastic (in between), do not modify.

        return _s_p( i, n );
    }

    auto getPlasticStretch( const int i, const int n ) const
    {
        return _s_p( i, n );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int j, const double s,
                     const double vol, const double time, const int n ) const
    {
        const double s_p = updatePlasticStretch( i, s, time, n );
        return base_type::c( i, j ) * ( s - s_p ) * vol;
    }

    // This energy calculation is only valid for pure tension or pure
    // compression.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int i, const int j, const double s,
                     const double xi, const double vol, const double time,
                     const int n ) const
    {
        auto s_p = _s_p( i, n );
        double stretch_term;
        // Yield in tension.
        if ( s >= s_p + s_Y( i, time ) )
            stretch_term = s_Y( i, time ) * ( 2.0 * s - s_Y( i, time ) );
        // Yield in compression.
        else if ( s <= s_p - s_Y( i, time ) )
            stretch_term = s_Y( i, time ) * ( -2.0 * s - s_Y( i, time ) );
        else
            // Elastic (in between).
            stretch_term = s * s;

        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * base_type::c( i, j ) * stretch_term * xi * vol;
    }
};

template <typename FunctorType>
struct MechanicsModel<LinearPMB, Elastic, FunctorType>
    : public MechanicsModel<PMB, Elastic, FunctorType>
{
    using base_type = MechanicsModel<PMB, Elastic, FunctorType>;
    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::base_type;
    using base_type::operator();

    template <typename... Args>
    MechanicsModel( LinearPMB, Args&&... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename ModelType, typename MemorySpace>
MechanicsModel( ModelType, ElasticPerfectlyPlastic, MemorySpace,
                const double force_horizon, const double K,
                const double sigma_y )
    -> MechanicsModel<ModelType, ElasticPerfectlyPlastic, ConstantProperty,
                      ConstantProperty, MemorySpace>;

template <typename ModelType, typename FunctorModulus, typename FunctorYield,
          typename MemorySpace>
MechanicsModel( ModelType, ElasticPerfectlyPlastic, MemorySpace,
                const double force_horizon, const FunctorModulus K,
                const FunctorYield sigma_y )
    -> MechanicsModel<ModelType, ElasticPerfectlyPlastic, FunctorModulus,
                      FunctorYield, MemorySpace>;

struct PMBDensity
{
};

// Force models with evolving density.
template <typename DensityType, typename CurrentDensityType,
          typename FunctorModulus, typename FunctorYield>
struct MechanicsDensityModel<PMB, ElasticPerfectlyPlastic, DensityType,
                             CurrentDensityType, FunctorModulus, FunctorYield>
    : public MechanicsModel<PMB, ElasticPerfectlyPlastic, FunctorModulus,
                            FunctorYield, typename DensityType::memory_space>
{
    using memory_space = typename DensityType::memory_space;
    using base_type =
        MechanicsModel<PMB, ElasticPerfectlyPlastic, FunctorModulus,
                       FunctorYield, memory_space>;
    using mechanics_type = typename base_type::mechanics_type;
    using force_tag = PMBDensity;

    using base_type::operator();

    // Does not use base_type::c because it is density dependent.
    using base_type::_s_p;
    using base_type::K;
    double coeff;

    double rho0;
    DensityType rho;
    CurrentDensityType rho_current;

    NormalRepulsionModel contact;

    MechanicsDensityModel( PMB model, ElasticPerfectlyPlastic mechanics,
                           const DensityType& _rho,
                           const CurrentDensityType& _rho_c, const double delta,
                           const FunctorModulus K, const FunctorYield sigma_y,
                           const double _rho0, const double contact_radius )
        : base_type( model, mechanics, memory_space{}, delta, K, sigma_y )
        , rho0( _rho0 )
        , rho( _rho )
        , rho_current( _rho_c )
        , contact( delta, contact_radius, 0.0, K.value( 0.0, 1.0 ) )
    {
        coeff = 18.0 / pi / delta / delta / delta / delta;
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    MechanicsDensityModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , rho0( model1.rho0 )
        , rho( model1.rho )
        , rho_current( model1.rho_current )
        , contact( model1.contact )
    {
        coeff = model1.coeff;
    }

    KOKKOS_INLINE_FUNCTION
    auto currentC( const int i ) const
    {
        // Initial relative density for relative density factor (HARDCODED)
        double D0 = 0.69;
        // Relative density:
        double D = rho_current( i ) / rho0;
        // Relative density factor from C. Van Nguyen et al., Journal of
        // Materials Processing Technology 226 (2015): 134-145.
        double RD = Kokkos::pow( ( D - D0 ) / ( 1.0 - D0 ), 1.46 * ( 1 - D0 ) );

        // Young's modulus from C. Van Nguyen et al., Journal of Materials
        // Processing Technology 226 (2015): 134-145.
        // double T = temperature( i );
        // double E = ( 199510 - 65.63 * T - 0.0276 * T * T - 1.754E-6 * T * T *
        // T ) * RD;
        //
        // Bulk modulus
        // double nu = 0.25; // Use bond-based model
        // double K = E / ( 3 * ( 1 - 2 * nu ) );
        //
        // return coeff * K;

        return coeff * K( i ) * RD;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const double, const int n ) const
    {
        // Update c given updated density.
        auto c_current = currentC( i );
        // Plastic stretch is already updated.
        auto s_p = base_type::getPlasticStretch( i, n );
        return c_current * ( s - s_p ) * vol;
    }

    // Update plastic dilatation.
    KOKKOS_INLINE_FUNCTION
    auto operator()( DilatationTag, const int i, const int, const double s,
                     const double xi, const double vol, const double time,
                     const int n ) const
    {
        // Update plastic stretch and then plastic dilatation.
        const auto s_p = base_type::updatePlasticStretch( i, s, time, n );
        return coeff / 6.0 * s_p * xi * vol;
    }

    // Density update using plastic dilatation.
    KOKKOS_INLINE_FUNCTION auto operator()( DensityTag, const int i,
                                            const double theta_i ) const
    {
        // Update density using plastic dilatation.
        // Note that this assumes zero initial plastic dilatation.
        rho_current( i ) = Kokkos::min( rho( i ) * Kokkos::exp( -theta_i ),
                                        rho0 ); // exp(theta_i - theta_i_0)
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        base_type::update( particles );
        rho = particles.sliceDensity();
        rho_current = particles.sliceCurrentDensity();
    }
};

template <typename DensityType, typename CurrentDensityType,
          typename FunctorModulus, typename FunctorYield>
MechanicsDensityModel( PMB, ElasticPerfectlyPlastic, DensityType rho,
                       const CurrentDensityType& rho_c, const double delta,
                       const FunctorModulus K, const FunctorYield sigma_y,
                       const double rho0, const double contact_radius )
    -> MechanicsDensityModel<PMB, ElasticPerfectlyPlastic, DensityType,
                             CurrentDensityType, FunctorModulus, FunctorYield>;

} // namespace CabanaPD

#endif

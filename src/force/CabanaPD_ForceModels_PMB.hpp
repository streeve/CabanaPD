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

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <>
struct ForceModel<PMB, Elastic, TemperatureIndependent>
    : public BasePairForceModel<NonLinearTag>
{
    using base_type = BasePairForceModel;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureIndependent;
    using linearized_type = typename base_type::linearized_type;

    using base_type::delta;

    double c;
    double K;

    ForceModel( const double delta, const double K )
        : base_type( delta )
    {
        set_param( delta, K );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        c = model.c;
        K = model.K;
    }

    void set_param( const double _delta, const double _K )
    {
        delta = _delta;
        K = _K;
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const double s, const double vol ) const
    {
        return c * s * vol;
    }

    template <class PosType, class VolType>
    KOKKOS_INLINE_FUNCTION auto force( const PosType& x, const PosType& u,
                                       const VolType& vol, const int i,
                                       const int j ) const
    {
        double xi, r, s;
        double rx, ry, rz;
        getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

        thermalStretch( s, i, j );

        const double coeff = forceCoeff( s, vol( j ) );
        Kokkos::Array<double, 3> f;

        f[0] = coeff * rx / r;
        f[1] = coeff * ry / r;
        f[2] = coeff * rz / r;
        return f;
    }

    KOKKOS_INLINE_FUNCTION
    auto energyCoeff( const double s, const double xi, const double vol ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c * s * s * xi * vol;
    }

    template <class PosType, class VolType>
    KOKKOS_INLINE_FUNCTION auto energy( const PosType& x, const PosType& u,
                                        const VolType& vol, const int i,
                                        const int j ) const
    {
        double xi, r, s;
        getDistance( x, u, i, j, xi, r, s );

        model.thermalStretch( s, i, j );

        return model.energyCoeff( s, xi, vol( j ) );
    }
};

template <>
struct ForceModel<PMB, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = base_type::thermal_type;
    using linearized_type = typename base_type::linearized_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel( const double delta, const double K, const double G0 )
        : base_type( delta, K )
    {
        set_param( delta, K, G0 );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        G0 = model.G0;
        s0 = model.s0;
        bond_break_coeff = model.bond_break_coeff;
    }

    void set_param( const double _delta, const double _K, const double _G0 )
    {
        base_type::set_param( _delta, _K );
        G0 = _G0;
        s0 = sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int, const int, const double r,
                          const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <>
struct ForceModel<LinearPMB, Elastic, TemperatureIndependent>
    : public BasePairForceModel<LinearTag>
{
    using base_type = BasePairForceModel<LinearTag>;
    using linearized_type = LinearTag;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    template <class PosType, class VolType>
    KOKKOS_INLINE_FUNCTION auto force( const PosType& x, const PosType& u,
                                       const VolType& vol, const int i,
                                       const int j ) const
    {
        double xi, linear_s;
        double xi_x, xi_y, xi_z;
        getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x, xi_y,
                                         xi_z );

        thermalStretch( linear_s, i, j );

        const double coeff = forceCoeff( linear_s, vol( j ) );
        f[0] = coeff * xi_x / xi;
        f[1] = coeff * xi_y / xi;
        f[2] = coeff * xi_z / xi;
        return f;
    }

    template <class PosType, class VolType>
    KOKKOS_INLINE_FUNCTION auto energy( const PosType& x, const PosType& u,
                                        const VolType& vol, const int i,
                                        const int j ) const
    {
        double xi, linear_s;
        getLinearizedDistance( x, u, i, j, xi, linear_s );

        model.thermalStretch( linear_s, i, j );

        return model.energyCoeff( linear_s, xi, vol( j ) );
    }
};

template <>
struct ForceModel<LinearPMB, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Fracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Fracture>;
    using linearized_type = LinearTag;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, TemperatureDependent, TemperatureType>
    : public ForceModel<PMB, Elastic, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using base_type = ForceModel<PMB, Elastic, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureDependent;
    using linearized_type = typename base_type::linearized_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;

    // Explicitly use the temperature-dependent stretch.
    using base_temperature_type::thermalStretch;

    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
        set_param( _delta, _K, _alpha, _temp0 );
    }

    void set_param( const double _delta, const double _K, const double _alpha,
                    const double _temp0 )
    {
        base_type::set_param( _delta, _K );
        base_temperature_type::set_param( _alpha, _temp0 );
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Elastic, ParticleType particles, const double delta,
                       const double K, const double alpha, const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, TemperatureDependent, temp_type>(
        delta, K, temp, alpha, temp0 );
}

template <typename TemperatureType>
struct ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>
    : public ForceModel<PMB, Fracture, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using base_type = ForceModel<PMB, Fracture, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = TemperatureDependent;
    using linearized_type = typename base_type::linearized_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Does not use the base bond_break_coeff.
    using base_type::G0;
    using base_type::s0;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_temperature_type::thermalStretch;

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K, _G0 )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
        set_param( _delta, _K, _G0, _alpha, _temp0 );
    }

    void set_param( const double _delta, const double _K, const double _G0,
                    const double _alpha, const double _temp0 )
    {
        base_type::set_param( _delta, _K, _G0 );
        base_temperature_type::set_param( _alpha, _temp0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int i, const int j, const double r,
                          const double xi ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        double bond_break_coeff =
            ( 1.0 + s0 + alpha * temp_avg ) * ( 1.0 + s0 + alpha * temp_avg );
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Fracture, ParticleType particles,
                       const double delta, const double K, const double G0,
                       const double alpha, const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Fracture, TemperatureDependent, temp_type>(
        delta, K, G0, temp, alpha, temp0 );
}

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, DynamicTemperature, TemperatureType>
    : public ForceModel<PMB, Elastic, TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type =
        ForceModel<PMB, Elastic, TemperatureDependent, TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = DynamicTemperature;
    using linearized_type = typename base_type::linearized_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::cp;
    using base_temperature_type::kappa;
    using base_temperature_type::thermal_coeff;
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_type::thermalStretch;

    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K, _temp, _alpha, _temp0 )
        , base_temperature_type( _delta, _kappa, _cp,
                                 _constant_microconductivity )
    {
        set_param( _delta, _K, _kappa, _cp, _alpha, _temp0,
                   _constant_microconductivity );
    }

    void set_param( const double _delta, const double _K, const double _kappa,
                    const double _cp, const double _alpha, const double _temp0,
                    const bool _constant_microconductivity )
    {
        base_type::set_param( _delta, _K, _alpha, _temp0 );
        base_temperature_type::set_param( _delta, _kappa, _cp,
                                          _constant_microconductivity );
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Elastic, ParticleType particles, const double delta,
                       const double K, const double kappa, const double cp,
                       const double alpha, const double temp0,
                       const bool constant_microconductivity = true )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, DynamicTemperature, temp_type>(
        delta, K, temp, kappa, cp, alpha, temp0, constant_microconductivity );
}

template <typename TemperatureType>
struct ForceModel<PMB, Fracture, DynamicTemperature, TemperatureType>
    : public ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type =
        ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = DynamicTemperature;
    using linearized_type = typename base_type::linearized_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Does not use the base bond_break_coeff.
    using base_type::G0;
    using base_type::s0;

    // Thermal parameters
    using base_temperature_type::cp;
    using base_temperature_type::kappa;
    using base_temperature_type::thermal_coeff;
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_type::thermalStretch;

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
    {
        set_param( _delta, _K, _G0, _kappa, _cp, _alpha, _temp0 );
        base_temperature_type::set_param( _delta, _kappa, _cp,
                                          _constant_microconductivity );
    }

    void set_param( const double _delta, const double _K, const double _G0,
                    const double _kappa, const double _cp, const double _alpha,
                    const double _temp0,
                    const bool _constant_microconductivity )
    {
        base_type::set_param( _delta, _K, _G0, _alpha, _temp0 );
        base_temperature_type::set_param( _delta, _kappa, _cp,
                                          _constant_microconductivity );
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Fracture, ParticleType particles,
                       const double delta, const double K, const double G0,
                       const double kappa, const double cp, const double alpha,
                       const double temp0,
                       const bool constant_microconductivity = true )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Fracture, DynamicTemperature, temp_type>(
        delta, K, G0, temp, kappa, cp, alpha, temp0,
        constant_microconductivity );
}

} // namespace CabanaPD

#endif

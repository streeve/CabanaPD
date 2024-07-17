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

#ifndef FORCE_MODELS_H
#define FORCE_MODELS_H

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
struct BaseForceModel
{
    double delta;

    BaseForceModel( const double _delta )
        : delta( _delta ){};

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double&, const int, const int ) const {}
};

// Wrap multiple models in a single object.
// TODO: this currently only supports bi-material systems.
template <typename... ModelType>
struct ForceModels
{
    // TODO: should there be a static_assert that each model form is the same
    // (PMB vs LPS)?
    ForceModels( const ModelType... models )
        : pack( Cabana::makeParameterPack( models... ) )
    {
    }

    template <std::size_t I, std::size_t J,
              std::enable_if_t<std::is_same<I, J>::type>>
    auto get()
    {
        return Cabana::get<I>( pack );
    }

    template <std::size_t I, std::size_t J,
              std::enable_if_t<!std::is_same<I, J>::type>>
    auto get()
    {
        // FIXME: only true for binary.
        return Cabana::get<2>( pack );
    }

    auto horizon( const int ) { return delta; }
    auto maxHorizon() { return delta; }

    Cabana::ParameterPack<ModelType...> pack;
};

template <typename... ModelType>
auto createForceModels( const ModelType... models )
{
    static constexpr std::size_t size = sizeof...( ModelType );

    // FIXME: only true for binary.
    if constexpr ( pack.size == 2 )
    {
        return ForceModels( models... );
    }
    else
    {
        // Create a temporary pack to extract an average.
        auto pack2 = Cabana::makeParameterPack( models... );
        using model_type = typename pack2<0>::value_type;
        model_type Model12( pack2.get<0>(), pack2.get<1>() );
        return ForceModels( models..., Model12 );
    }
}

template <typename ParticleType, typename ArrayType>
auto createForceModel( PMB, Fracture, TemperatureIndependent,
                       ParticleType particles, const ArrayType& delta,
                       const ArrayType& K, const ArrayType& G0 )
{
    auto type = particles.sliceType();
    using type_type = decltype( type );
    return ForceModel<PMB, Fracture, TemperatureIndependent, type_type>(
        delta, K, G0, type );
}

template <typename TemperatureType>
struct BaseTemperatureModel
{
    double alpha;
    double temp0;

    // Temperature field
    TemperatureType temperature;

    BaseTemperatureModel( const TemperatureType _temp, const double _alpha,
                          const double _temp0 )
        : alpha( _alpha )
        , temp0( _temp0 )
        , temperature( _temp ){};

    // Average from existing models.
    BaseTemperatureModel( const ForceModel& model1, const ForceModel& model2 )
    {
        alpha = ( model1.alpha + model2.alpha ) / 2.0;
        temp0 = ( model1.temp0 + model2.temp0 ) / 2.0;
    }

    void update( const TemperatureType _temp ) { temperature = _temp; }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double& s, const int i, const int j ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        s -= alpha * temp_avg;
    }
};

// This class stores temperature parameters needed for heat transfer, but not
// the temperature itself (stored instead in the static temperature class
// above).
struct BaseDynamicTemperatureModel
{
    double delta;

    double thermal_coeff;
    double kappa;
    double cp;
    bool constant_microconductivity;

    BaseDynamicTemperatureModel( const double _delta, const double _kappa,
                                 const double _cp,
                                 const bool _constant_microconductivity = true )
    {
        delta = _delta;
        kappa = _kappa;
        cp = _cp;
        const double d3 = _delta * _delta * _delta;
        thermal_coeff = 9.0 / 2.0 * _kappa / pi / d3;
        constant_microconductivity = _constant_microconductivity;
    }

    // Average from existing models.
    BaseDynamicTemperatureModel( const ForceModel& model1,
                                 const ForceModel& model2 )
    {
        delta = ( model1.delta + model2.delta ) / 2.0;
        kappa = ( model1.kappa + model2.kappa ) / 2.0;
        cp = ( model1.cp + model2.cp ) / 2.0;
        constant_microconductivity = model1.constant_microconductivity;
    }

    KOKKOS_INLINE_FUNCTION double microconductivity_function( double r ) const
    {
        if ( constant_microconductivity )
            return thermal_coeff;
        else
            return 4.0 * thermal_coeff * ( 1.0 - r / delta );
    }
};

template <typename PeridynamicsModelType, typename MechanicsModelType = Elastic,
          typename DamageType = Fracture,
          typename ThermalType = TemperatureIndependent, typename... DataTypes>
struct ForceModel;

} // namespace CabanaPD

#endif

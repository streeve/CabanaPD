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
    using material_type = SingleMaterial;
    double delta;

    BaseForceModel( const double _delta )
        : delta( _delta ){};

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double&, const int, const int ) const {}
};

// Wrap multiple models in a single object.
// TODO: this currently only supports bi-material systems.
template <typename MaterialType, typename ModelType1, typename ModelType2,
          typename ModelType12 = ModelType1>
struct ForceModels
{
    using material_type = MultiMaterial;
    // FIXME: improve this.
    using first_model = ModelType1;
    using base_model = typename first_model::base_model;
    using thermal_type = typename first_model::thermal_type;

    ForceModels( MaterialType t, const ModelType1 m1, ModelType2 m2 )
        : type( t )
        , model1( m1 )
        , model2( m2 )
    {
        setHorizon();

        // Construct cross terms through averaging.
        m12( m1, m2 );
    }

    ForceModels( MaterialType t, const ModelType1 m1, ModelType2 m2,
                 ModelType12 m12 )
        : type( t )
        , model1( m1 )
        , model2( m2 )
        , model12( m12 )
    {
        setHorizon();
    }

    void setHorizon()
    {
        delta = 0.0;
        if ( model1.delta > delta )
            delta = model1.delta;
        if ( model2.delta > delta )
            delta = model2.delta;
    }

    KOKKOS_INLINE_FUNCTION auto getModel( const int i, const int j ) const
    {
        const int type_i = type( i );
        const int type_j = type( j );
        return get<type_i, type_j>();
    }

    template <typename... Args>
    KOKKOS_INLINE_FUNCTION auto forceCoeff( const int i, const int j,
                                            Args... args ) const
    {
        auto model = getModel( i, j );
        return model.forceCoeff( args... );
    }

    auto horizon( const int ) { return delta; }
    auto maxHorizon() { return delta; }

    template <std::size_t I, std::size_t J>
    auto get( typename std::enable_if_t<( I == J ), int>* = 0 )
    {
        return Cabana::get<I>( pack );
    }

    template <std::size_t I, std::size_t J>
    auto get( typename std::enable_if_t<( I != J ), int>* = 0 )
    {
        // FIXME: only true for binary.
        return Cabana::get<2>( pack );
    }

    void update( const MaterialType _type ) { type = _type; }

    double delta;
    MaterialType type;
    // Kokkos::View<ModelType[2][2], typename MaterialType::memory_space>
    // models;
    ModelPack pack;
};

template <typename ParticleType, typename... ModelType>
auto createMultiForceModel( ParticleType particles, ModelType... models )
{
    auto type = particles.sliceType();
    using material_type = decltype( type );

    auto pack = Cabana::makeParameterPack( models... );
    return ForceModels<material_type, decltype( pack )>( type, pack );
}

template <typename ParticleType, typename... ModelType>
auto createMultiForceModel( ParticleType particles, AverageTag,
                            ModelType... models )
{
    auto type = particles.sliceType();
    using material_type = decltype( type );

    auto pack = Cabana::makeParameterPack( models... );
    using model_type1 = std::tuple_element_t<0, std::tuple<ModelType...>>;
    model_type1 m3( models... );
    pack = Cabana::makeParameterPack( models..., m3 );
    return ForceModels<material_type, decltype( pack )>( type, pack );
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
    BaseTemperatureModel( const BaseTemperatureModel& model1,
                          const BaseTemperatureModel& model2 )
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
    BaseDynamicTemperatureModel( const BaseDynamicTemperatureModel& model1,
                                 const BaseDynamicTemperatureModel& model2 )
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

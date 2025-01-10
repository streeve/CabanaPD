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
template <typename MaterialType, typename... ModelType>
struct ForceModels
{
    using material_type = MultiMaterial;

    ForceModels( MaterialType t, const ModelType... models )
        : type( t )
        , models( std::tuple( models... ) )
    {
        setHorizon( m1, m2 );

        models( 0, 0 ) = m1;
        models( 1, 1 ) = m2;
        ModelType m3( m1, m2 );
        models( 0, 1 ) = m3;
        models( 1, 0 ) = m3;
    }

    ForceModels( MaterialType t, const ModelType m1, const ModelType m2,
                 const ModelType m12 )
        : type( t )
    {
        setHorizon( m1, m2 );

        models( 0, 0 ) = m1;
        models( 1, 1 ) = m2;
        models( 0, 1 ) = m12;
        models( 1, 0 ) = m12;
    }

    void setHorizon( const ModelType m1, const ModelType m2 )
    {
        delta = 0.0;
        if ( m1.delta > delta )
            delta = m1.delta;
        if ( m2.delta > delta )
            delta = m2.delta;
    }

    KOKKOS_INLINE_FUNCTION auto getModel( const int i, const int j ) const
    {
        const int type_i = type( i );
        const int type_j = type( j );
        auto model = models( type_i, type_j );
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

    double delta;
    MaterialType type;
    Kokkos::View<ModelType[2][2], typename MaterialType::memory_space> models;
};

template <typename ParticleType, typename... ModelType>
auto createMultiForceModel( ParticleType particles, ModelType... models )
{
    auto type = particles.sliceType();
    using material_type = decltype( type );
    return ForceModels<material_type, ModelType...>( type, models... );
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

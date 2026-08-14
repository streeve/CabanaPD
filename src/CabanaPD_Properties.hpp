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

#ifndef PROPERTIES_H
#define PROPERTIES_H

namespace CabanaPD
{
struct ConstantProperty
{
    ConstantProperty( const double v )
        : value( v )
    {
    }
    template <typename FunctorType>
    ConstantProperty( const FunctorType p1, const FunctorType p2 )
        : value( ( p1( 0 ) + p2( 0 ) / 2.0 ) )
    {
    }

    KOKKOS_FUNCTION
    auto operator()( const int, const double = 0.0 ) const { return value; }

    template <typename ParticleType>
    void update( const ParticleType& )
    {
    }

    double value;
};

template <typename TemperatureType>
struct TemperatureDependentProperty
{
    // Constructor with temperature field.
    TemperatureDependentProperty( const TemperatureType& t )
        : temp( t )
    {
    }

    // Update with new temperature field.
    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        temp = particles.sliceTemperature();
    }

    TemperatureType temp;
};

template <typename TemperatureType>
TemperatureDependentProperty( const TemperatureType )
    -> TemperatureDependentProperty<TemperatureType>;

template <typename TemperatureType>
struct TemperatureDependentPolynomial
    : public TemperatureDependentProperty<TemperatureType>
{
    using base_type = TemperatureDependentProperty<TemperatureType>;
    using memory_space = typename TemperatureType::memory_space;

    // Constructor with polynomial coefficients and temperature field.
    template <typename ArrayType>
    TemperatureDependentPolynomial( ArrayType array, const TemperatureType& t )
        : base_type( t )
        , coeff( Kokkos::View<double*, memory_space>( "coefficients",
                                                      array.size() ) )
    {
        Kokkos::View<double*, Kokkos::HostSpace> view( array.data(),
                                                       array.size() );
        Kokkos::deep_copy( coeff, view );
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p, const double = 0.0 ) const
    {
        const auto t = temp( p );
        return value( t );
    }

    KOKKOS_FUNCTION
    auto value( const double temp ) const
    {
        double val = 0.0;
        const auto last = coeff.size() - 1;
        for ( std::size_t i = 0; i < last; i++ )
            val += coeff( i ) * temp;

        return val + coeff( last );
    }

    Kokkos::View<double*, memory_space> coeff;
    using base_type::temp;
};

template <typename ArrayType, typename TemperatureType>
TemperatureDependentPolynomial( ArrayType, const TemperatureType )
    -> TemperatureDependentPolynomial<TemperatureType>;

template <typename TemperatureType>
struct TemperatureDependentPiecewise
    : public TemperatureDependentProperty<TemperatureType>
{
    using base_type = TemperatureDependentProperty<TemperatureType>;
    using memory_space = typename TemperatureType::memory_space;

    // Constructor with XY values to linearly interpolate and temperature field.
    template <typename ArrayType>
    TemperatureDependentPiecewise( ArrayType array_x, ArrayType array_y,
                                   const TemperatureType& t )
        : base_type( t )
        , x( Kokkos::View<double*, memory_space>( "x", array_x.size() ) )
        , y( Kokkos::View<double*, memory_space>( "y", array_y.size() ) )
    {
        assert( x.size() == y.size() );

        Kokkos::View<double*, Kokkos::HostSpace> view_x( array_x.data(),
                                                         array_x.size() );
        Kokkos::deep_copy( x, view_x );
        Kokkos::View<double*, Kokkos::HostSpace> view_y( array_y.data(),
                                                         array_y.size() );
        Kokkos::deep_copy( y, view_y );
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p, const double = 0.0 ) const
    {
        const auto t = temp( p );
        return value( t );
    }

    KOKKOS_FUNCTION
    auto value( const double temp ) const
    {
        if ( temp < x( 0 ) )
            return y( 0 );

        for ( std::size_t i = 1; i < x.size() - 1; i++ )
        {
            if ( temp < x( i ) )
                return y( i - 1 ) + ( temp - x( i - 1 ) ) *
                                        ( y( i ) - y( i - 1 ) ) /
                                        ( x( i ) - x( i - 1 ) );
        }

        return y( y.size() - 1 );
    }

    Kokkos::View<double*, typename TemperatureType::memory_space> x;
    Kokkos::View<double*, typename TemperatureType::memory_space> y;
    using base_type::temp;
};

template <typename ArrayType, typename TemperatureType>
TemperatureDependentPiecewise( ArrayType, ArrayType, const TemperatureType )
    -> TemperatureDependentPiecewise<TemperatureType>;

template <typename TemperatureType, typename PropertyType>
struct BilinearPiecewise

{
    using memory_space = typename TemperatureType::memory_space;

    // Constructor with XY values to linearly interpolate and temperature field.
    template <typename ArrayType, typename View2dType>
    BilinearPiecewise( ArrayType array_x, ArrayType array_y,
                       View2dType view_val, const TemperatureType& t,
                       const PropertyType& p )
        : temp( t )
        , prop( p )
        , x( Kokkos::View<double*, memory_space>( "x", array_x.size() ) )
        , y( Kokkos::View<double*, memory_space>( "y", array_y.size() ) )
        , val( Kokkos::View<double**, memory_space>(
              "values", view_val.extent( 0 ), view_val.extent( 1 ) ) )
    {
        assert( x.size() == y.size() );
        assert( x.size() == val.extent( 0 ) );

        Kokkos::View<double*, Kokkos::HostSpace> view_x( array_x.data(),
                                                         array_x.size() );
        Kokkos::deep_copy( x, view_x );
        Kokkos::View<double*, Kokkos::HostSpace> view_y( array_y.data(),
                                                         array_y.size() );
        Kokkos::deep_copy( y, view_y );

        Kokkos::deep_copy( val, view_val );
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p, const double = 0.0 ) const
    {
        const auto t = temp( p );
        const auto p2 = prop( p );
        return value( t, p2 );
    }

    KOKKOS_FUNCTION
    auto linear( const double x0, const double x1, const double y0,
                 const double y1, const double xv ) const
    {
        if ( x0 == x1 )
            return y0;
        return y0 + ( y1 - y0 ) * ( xv - x0 ) / ( x1 - x0 );
    }

    KOKKOS_FUNCTION auto value( const double temp, const double p2 ) const
    {
        int i1 = 0;
        int i2 = 0;
        int j1 = 0;
        int j2 = 0;
        const auto max_i = x.size() - 1;
        const auto max_j = y.size() - 1;

        for ( std::size_t i = 1; i <= max_i; i++ )
        {
            if ( temp < x( i ) )
            {
                i1 = i - 1;
                i2 = i;
                break;
            }
        }
        if ( temp > x( max_i ) )
        {
            i1 = max_i;
            i2 = max_i;
        }

        for ( std::size_t j = 1; j <= max_j; j++ )
        {
            if ( p2 < y( j ) )
            {
                j1 = j - 1;
                j2 = j;
                break;
            }
        }
        if ( p2 > y( max_j ) )
        {
            j1 = max_j;
            j2 = max_j;
        }

        auto v1 =
            linear( x( i1 ), x( i2 ), val( i1, j1 ), val( i2, j1 ), temp );
        auto v2 =
            linear( x( i1 ), x( i2 ), val( i1, j2 ), val( i2, j2 ), temp );

        return linear( y( j1 ), y( j2 ), v1, v2, p2 );
    }

    Kokkos::View<double*, memory_space> x;
    Kokkos::View<double*, memory_space> y;
    Kokkos::View<double**, memory_space> val;

    TemperatureType temp;
    PropertyType prop;
};

template <typename ArrayType, typename View2dType, typename TemperatureType,
          typename PropType>
BilinearPiecewise( ArrayType, ArrayType, View2dType, const TemperatureType,
                   const PropType )
    -> BilinearPiecewise<TemperatureType, PropType>;

template <typename MemorySpace>
struct TimeDependentPolynomial
{
    // Constructor with polynomial coefficients and temperature field.
    template <typename ArrayType>
    TimeDependentPolynomial( MemorySpace, ArrayType array )
        : coeff( Kokkos::View<double*, MemorySpace>( "coefficients",
                                                     array.size() ) )
    {
        Kokkos::View<double*, Kokkos::HostSpace> view( array.data(),
                                                       array.size() );
        Kokkos::deep_copy( coeff, view );
    }

    // Return time-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int, const double time ) const
    {
        return value( time );
    }

    KOKKOS_FUNCTION
    auto value( const double time ) const
    {
        double val = 0.0;
        const auto last = coeff.size() - 1;
        for ( std::size_t i = 0; i < last; i++ )
            val += coeff( i ) * time;

        return val + coeff( last );
    }

    Kokkos::View<double*, MemorySpace> coeff;
};

template <typename ArrayType, typename MemorySpace>
TimeDependentPolynomial( MemorySpace, ArrayType array )
    -> TimeDependentPolynomial<MemorySpace>;

} // namespace CabanaPD

#endif

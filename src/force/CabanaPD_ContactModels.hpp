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

#ifndef CONTACTMODELS_H
#define CONTACTMODELS_H

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

    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta )
        , Rc( _Rc ){};
};

/* Normal repulsion */
struct NormalRepulsionModel : public ContactModel
{
    // FIXME: This is for use as the primary force model.
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using ContactModel::delta;
    using ContactModel::Rc;

    double c;
    double K;

    NormalRepulsionModel( const double delta, const double Rc, const double _K )
        : ContactModel( delta, Rc )
        , K( _K )
    {
        K = _K;
        // This could inherit from PMB (same c)
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION auto force( const PositionType& x,
                                       const PositionType& u, const int i,
                                       const int j, const double vol ) const
    {
        double xi, r, s;
        double rx, ry, rz;
        getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

        // Contact "stretch"
        const double sc = ( r - Rc ) / delta;
        // Normal repulsion uses a 15 factor compared to the PMB force
        auto coeff = 15.0 * c * sc * vol;
        auto coeff_r = coeff / r;

        Kokkos::Array<double, 3> f_i;
        f_i[0] = coeff_r * rx;
        f_i[1] = coeff_r * ry;
        f_i[2] = coeff_r * rz;
        return f_i;
    }
};

template <>
struct is_contact<NormalRepulsionModel> : public std::true_type
{
};

} // namespace CabanaPD

#endif

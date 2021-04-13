#ifndef _PHYSIC_H_
#define _PHYSIC_H_

#include <array>
#include <fstream>
#include <tuple>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>

#include <boost/iterator/zip_iterator.hpp>

#include "field.h"

using namespace boost::numeric;

#define SQ(X) ((X)*(X))

template < typename _T , std::size_t NumDimsV >
_T
energy ( field<_T,NumDimsV> const& f , ublas::vector<_T> const& E )
{
  _T H = double{0.};

  for ( typename field<_T,NumDimsV>::size_type k=0 ; k<f.size(0) ; ++k ) {
    for ( typename field<_T,NumDimsV>::size_type i=0 ; i<f.size_x() ; ++i ) {
      H += SQ( (_T(k)*f.step.dv+f.range.v_min) ) * f[k][i] * f.step.dv*f.step.dx;
    }
  }

  for ( auto it=E.begin() ; it!=E.end() ; ++it ) {
    H += SQ(*it)*f.step.dx;
  }

  return H;
}

template < typename _T , std::size_t NumDimsV >
_T
kinetic_energy ( field<_T,NumDimsV> const& f )
{
  _T Ec = double{0.};

  for ( typename field<_T,NumDimsV>::size_type k=0 ; k<f.size(0) ; ++k ) {
    for ( typename field<_T,NumDimsV>::size_type i=0 ; i<f.size_x() ; ++i ) {
      Ec += SQ( (_T(k)*f.step.dv+f.range.v_min) ) * f[k][i] * f.step.dv*f.step.dx;
    }
  }

  return Ec;
}

template <typename _T>
auto
maxwellian ( _T rho , std::vector<T_> u , std::vector<_T> T ) {
  return [=](_T z,_T vx,_T vy,_T vz) {
    return rho/( std::pow(2.*math::pi<_T>(),1.5)*T[0]*T[1]*T[2] ) * std::exp( -0.5*( SQ((vx-u[0])/T[0]) + SQ((vy-u[1])/T[1]) + SQ((vz-u[2])/T[2]) ) );
  };
}

#undef SQ

namespace factory
{

template <typename _T>
auto
computer_space_energy ( _T dz )
{
  auto __compute_energy = [dz]( const ublas::vector<_T> ux , const ublas::vector<_T> uy ) {
    // __compute_energy is a lambda which compute energy from 2 vector with a predifine dz value
    return 0.5*dz*std::inner_product(
                    std::begin(ux) , std::end(ux) , std::begin(uy) , 0.0 ,
                    std::plus<_T>() ,
                    []( const auto & x , const auto & y ){ return x*x + y*y ; }
                 );
  };
  return __compute_energy;
}

template <typename _T>
auto
computer_vperp_integral ( field3d<_T> const& f )
{
  const std::size_t Nvx = f.size(0);
  const std::size_t Nvy = f.size(1);
  const std::size_t Nvz = f.size(2);
  const std::size_t Nz  = f.size(3);
  const _T v_min = f.range.vz_min , v_max = f.range.vz_max;
  const _T x_min = f.range.z_min  , x_max = f.range.z_max;
  const _T dv_perp = f.step.dvx*f.step.dvy;

  auto compute_vperp_integral = [Nvx,Nvy,Nvz,Nz,v_min,v_max,x_min,x_max]( const complex_field<_T,3> & hf , _T current_t ) {
    field<_T,1> fdvxdvy(boost::extents[Nvz][Nz]);
    fdvxdvy.range.v_min = v_min; fdvxdvy.range.v_max = v_max; 
    fdvxdvy.range.x_min = x_min; fdvxdvy.range.x_max = x_max;
    fdvxdvy.compute_steps();

    ublas::vector<double> fvxvyvz(Nz,0.);

    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            fdvxdvy[k_z][i] = fvxvyvz[i] * dv_perp;
          }
        }
      }
    }
 
    return fdvxdvy;
  };

  return compute_vperp_integral;
}

}

#endif

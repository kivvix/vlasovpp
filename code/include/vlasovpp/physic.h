#ifndef _PHYSIC_H_
#define _PHYSIC_H_

#include <array>
#include <fstream>
#include <tuple>
#include <utility>

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
maxwellian ( _T rho , std::vector<_T> u , std::vector<_T> T ) {
  return [=](_T z,_T vx,_T vy,_T vz) {
    return rho/( std::pow(2.*math::pi<_T>(),1.5)*T[0]*T[1]*T[2] ) * std::exp( -0.5*( SQ((vx-u[0])/T[0]) + SQ((vy-u[1])/T[1]) + SQ((vz-u[2])/T[2]) ) );
  };
}

#undef SQ

namespace computer
{

template <typename _T>
auto
space_energy ( _T dz )
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
struct vperp_integral
{
  std::size_t Nvx;
  std::size_t Nvy;
  std::size_t Nvz;
  std::size_t Nz;
  _T dv_perp;

  // computational value
  ublas::vector<_T> fvxvyvz;

  // computed value
  field<_T,1> fdvxdvy;

  vperp_integral ( field3d<_T> const& f )
    : Nvx(f.size(0)) , Nvy(f.size(1)) , Nvz(f.size(2)) , Nz(f.size(3)) ,
      dv_perp(f.step.dvx*f.step.dvy) ,
      fvxvyvz(f.size(3),0.) ,
      fdvxdvy(boost::extents[f.size(2)][f.size(3)])
  {
    fdvxdvy.range.v_min = f.range.vz_min; fdvxdvy.range.v_max = f.range.vz_max; 
    fdvxdvy.range.x_min = f.range.z_min;  fdvxdvy.range.x_max = f.range.z_max;
    fdvxdvy.compute_steps();
  }

  field<_T,1> &
  operator () ( const complex_field<_T,3> & hf )
  {
    std::iota( fdvxdvy.origin() , fdvxdvy.origin() + fdvxdvy.num_elements() , 0. );

    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            fdvxdvy[k_z][i] += fvxvyvz[i] * dv_perp;
          }
        }
      }
    }
 
    return fdvxdvy;
  }

};

template <typename _T>
struct z_vz_integral
{
  std::size_t Nvx;
  std::size_t Nvy;
  std::size_t Nvz;
  std::size_t Nz;
  _T dzdvz;

  // computational value
  ublas::vector<_T> fvxvyvz;

  // computed value
  field<_T,1> fdzdvz;

  z_vz_integral ( field3d<_T> const& f )
    : Nvx(f.size(0)) , Nvy(f.size(1)) , Nvz(f.size(2)) , Nz(f.size(3)) ,
      dzdvz(f.step.dz*f.step.dvz) ,
      fvxvyvz(f.size(3),0.) ,
      fdzdvz(boost::extents[f.size(0)][f.size(1)])
  {
    fdzdvz.range.v_min = f.range.vx_min; fdzdvz.range.v_max = f.range.vx_max; 
    fdzdvz.range.x_min = f.range.vy_min; fdzdvz.range.x_max = f.range.vy_max;
    fdzdvz.compute_steps();
  }

  field<_T,1> &
  operator () ( const complex_field<_T,3> & hf )
  {
    std::iota( fdzdvz.origin() , fdzdvz.origin() + fdzdvz.num_elements() , 0. );

    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            fdzdvz[k_x][k_y] += fvxvyvz[i] * dzdvz;
          }
        }
      }
    }
 
    return fdzdvz;
  }

};

template <typename _T>
struct z_vperp_integral
{
  std::size_t Nvx;
  std::size_t Nvy;
  std::size_t Nvz;
  std::size_t Nz;
  _T dv_perp;
  _T dz;

  // computational value
  ublas::vector<_T> fvxvyvz;

  // computed value
  ublas::vector<_T> fdvxdvydz;

  z_vperp_integral ( field3d<_T> const& f )
    : Nvx(f.size(0)) , Nvy(f.size(1)) , Nvz(f.size(2)) , Nz(f.size(3)) ,
      dv_perp(f.step.dvx*f.step.dvy) ,
      dz(f.step.dz) ,
      fvxvyvz(f.size(3),0.) ,
      fdvxdvydz(f.size(2),0.)
  { ; }

  ublas::vector<_T> &
  operator () ( const complex_field<_T,3> & hf )
  {
    std::iota( std::begin(fdvxdvydz) , std::end(fdvxdvydz) , 0. );

    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            fdvxdvydz[i] += fvxvyvz[i] * dv_perp * dz;
          }
        }
      }
    }
 
    return fdvxdvydz;
  }

};

template <typename _T>
struct local_kinetic_energy
{
  std::size_t Nvx;
  std::size_t Nvy;
  std::size_t Nvz;
  std::size_t Nz;
  _T dvx,dvy,dvz;
  _T vx_min,vy_min,vz_min;
  _T dv;

  // computational value
  ublas::vector<_T> fvxvyvz;

  // computed value
  ublas::vector<_T> ec_perp;
  ublas::vector<_T> ec_par;

  local_kinetic_energy ( field3d<_T> const& f )
    : Nvx(f.size(0)) , Nvy(f.size(1)) , Nvz(f.size(2)) , Nz(f.size(3)) ,
      dvx(f.step.dvx) , dvy(f.step.dvy) , dvz(f.step.dvz) ,
      vx_min(f.range.vx_min) , vy_min(f.range.vy_min) , vz_min(f.range.vz_min) ,
      dv(f.step.dvx*f.step.dvy*f.step.dvz) ,
      fvxvyvz(f.size(3),0.) ,
      ec_perp(f.size(3),0.) ,
      ec_par(f.size(3),0.)
  { ; }

  auto
  operator () ( const complex_field<_T,3> & hf )
  {
    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      const _T vx = k_x*dvx + vx_min;
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        const _T vy = k_y*dvy + vy_min;
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          const _T vz = k_z*dvz + vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            ec_perp[i] = (vx*vx + vy*vy) * fvxvyvz[i] * dv;
            ec_par[i]  = (vz*vz) * fvxvyvz[i] * dv;
          }
        }
      }
    }
 
    return std::make_tuple( ec_perp , ec_par );
  }

};

template <typename _T>
struct hot_mass_energy
{
  std::size_t Nvx;
  std::size_t Nvy;
  std::size_t Nvz;
  std::size_t Nz;
  _T dvx,dvy,dvz;
  _T vx_min,vy_min,vz_min;
  _T dz , dv;

  // computational value
  ublas::vector<_T> fvxvyvz;

  // computed value
  _T mass;
  _T he;

  hot_mass_energy ( field3d<_T> const& f )
    : Nvx(f.size(0)) , Nvy(f.size(1)) , Nvz(f.size(2)) , Nz(f.size(3)) ,
      dvx(f.step.dvx) , dvy(f.step.dvy) , dvz(f.step.dvz) ,
      vx_min(f.range.vx_min) , vy_min(f.range.vy_min) , vz_min(f.range.vz_min) ,
      dz(f.step.dz) , dv(f.step.dvx*f.step.dvy*f.step.dvz) ,
      fvxvyvz(f.size(3),0.) ,
      mass(0.) , he(0.)
  {}

  auto
  operator () ( const complex_field<_T,3> & hf )
  {
    mass = 0.; he = 0.;

    for ( auto k_x=0u ; k_x<Nvx ; ++k_x ) {
      const _T vx = k_x*dvx + vx_min;
      for ( auto k_y=0u ; k_y<Nvy ; ++k_y ) {
        const _T vy = k_y*dvy + vy_min;
        for ( auto k_z=0u ; k_z<Nvz ; ++k_z ) {
          const _T vz = k_z*dvz + vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<Nz ; ++i ) {
            mass += fvxvyvz[i]*dz*dv;
            he += 0.5*( vx*vx + vy*vy + vz*vz )*fvxvyvz[i]*dz*dv;
          }
        }
      }
    }

    return std::make_pair(mass,he);
  }

};

} // namespace computer

#endif

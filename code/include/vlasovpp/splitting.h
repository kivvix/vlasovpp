#ifndef _SPLITTING_H_
#define _SPLITTING_H_

#include <algorithm>
#include <iterator>
#include <cmath>
#include <valarray>
#include <sstream>
#include <complex>
#include <tuple>
#include <numeric>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/constants/constants.hpp>

#include "vlasovpp/field.h"
#include "vlasovpp/complex_field.h"
#include "vlasovpp/lagrange5.h"
#include "vlasovpp/fft.h"
#include "vlasovpp/array_view.h"
#include "vlasovpp/poisson.h"

using namespace boost::numeric;

/*
template < typename _T , std::size_t NumDimsV >
struct U_type {
  std::tuple< ublas::vector<_T> , ublas::vector<_T> , complex_field<_T,NumDimsV> > data;

  //U_type ( std::tuple< ublas::vector<_T> , ublas::vector<_T> , complex_field<_T,NumDimsV> > & t )
  //  : data(t)
  //{ ; }

  const auto &
  uc () const
  { return std::get<0>(data); }
  auto &
  uc ()
  { return std::get<0>(data); }

  const auto &
  E () const
  { return std::get<1>(data); }
  auto &
  E ()
  { return std::get<1>(data); }

  const auto &
  hfh () const
  { return std::get<2>(data); }
  auto &
  hfh ()
  { return std::get<2>(data); }
};
*/

template < typename _T , std::size_t NumDimsV >
struct splitting
{
  //typedef std::tuple< ublas::vector<_T>,ublas::vector<_T>,field<_T,NumDimsV> > U_type;
  typedef void U_type;

  _T _dx,_dv;
  _T _v_min , _x_max;
  _T _rho_c;
  field<_T,NumDimsV> _tmpf0;
  field<_T,NumDimsV> _tmpf1;
  std::size_t _Nx,_Nv;
  ublas::vector<_T> kx;

  splitting ( field<_T,NumDimsV> const & fh , _T l , _T rho_c) :
    _dx(fh.step.dx) , _dv(fh.step.dv) ,
    _v_min(fh.range.v_min) , _x_max(fh.range.x_max) ,
    _rho_c(rho_c) ,
    _tmpf0(tools::array_view<const std::size_t>(fh.shape(),2)) ,
    _tmpf1(tools::array_view<const std::size_t>(fh.shape(),2)) ,
    _Nx(fh.shape()[1]) ,
    _Nv(fh.shape()[0]) ,
    kx(fh.shape()[1],0.)
  {
    _tmpf0.step = fh.step; _tmpf0.range = fh.range;
    _tmpf1.step = fh.step; _tmpf1.range = fh.range;
    //kx[0] = 1.;
    /*
    for ( auto i=1 ; i<_Nx/2 ; ++i ) { kx[i] = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nx/2 ; i<0 ; ++i ) { kx[_Nx+i] = 2.*math::pi<double>()*i/l; }
    */
    for ( auto i=0 ; i<_Nx/2 ; ++i ) { kx[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nx/2 ; i<0 ; ++i ) { kx[_Nx+i] = 2.*math::pi<double>()*i/l; }
  }

  void
  phi_a ( _T dt , ublas::vector<_T> & uc , ublas::vector<_T> & E , complex_field<_T,NumDimsV> & hfh )
  {
    // equivalent of hf of Nicolas
    /*
      $$
        \partial_t U = \{U,\mathcal{H}_f\} = \begin{cases}
          \partial_t u_c = 0 \\
          \partial_t E   = -\int v f_h\,\mathrm{d}v \\
          \partial_t f_h = -v\partial_x f_h \\
        \end{cases}
      $$
    */
    const std::complex<double> & I = std::complex<double>(0.,1.);

    // compute hdiffrho and update hfh
    ublas::vector<std::complex<_T>> hdiffrho(_Nx,0.);
    for ( auto k=0 ; k<_Nv ; ++k ) {
      double vk = k*_dv + _v_min;
      for ( auto i=0 ; i<_Nx ; ++i ) {
        // hrho_n
        hdiffrho[i] += hfh[k][i]*_dv;
        // update hfh
        hfh[k][i] *= std::exp(-I*vk*kx[i]*dt);
        // hrho_{n+1}
        hdiffrho[i] -= hfh[k][i]*_dv;
      }
    }

    // update E
    fft::spectrum_ hE(_Nx); hE.fft(E.begin());
    hE[0] = 0.;
    for ( auto i=1 ; i<_Nx ; ++i ) {
      hE[i] += I/kx[i]*hdiffrho[i];
    }
    hE.ifft(E.begin());
  }

  void
  phi_b ( _T dt , ublas::vector<_T> & uc , ublas::vector<_T> const & E , complex_field<_T,NumDimsV> & hfh )
  {
    // equivalent of HE of Nicolas
    /*
      $$
        \partial_t U = \{U,\mathcal{H}_f\} = \begin{cases}
          \partial_t u_c = 0 \\
          \partial_t E   = -\int v f_h\,\mathrm{d}v \\
          \partial_t f_h = -v\partial_x f_h \\
        \end{cases}
      $$
    */

    for ( auto k=0 ; k < _Nv ; ++k )
      { fft::ifft( &(hfh[k][0]) , &(hfh[k][0])+_Nx , &(_tmpf0[k][0]) ); }

    // faire des trucs sur `_tmpf0` : $f(x,v) = f(x,v-dt*E)$
    for ( auto k=0 ; k<_Nv ; ++k )
    {
      for ( auto i=0 ; i<_Nx ; ++i ) {
        _T  vstar = (k*_dv + _v_min) - dt*E[i]; // vstar = v - dt*E
        int kstar = std::ceil((vstar - _v_min)/_dv);

        auto N = lagrange5::generator(
                    _tmpf0[(kstar-3+_Nv)%_Nv][i],_tmpf0[(kstar-2+_Nv)%_Nv][i],_tmpf0[(kstar-1+_Nv)%_Nv][i],_tmpf0[(kstar+_Nv)%_Nv][i],_tmpf0[(kstar+1+_Nv)%_Nv][i],_tmpf0[(kstar+2+_Nv)%_Nv][i] ,
                    _dv , kstar*_dv + _v_min
                  );
        _tmpf1[k][i] = N(vstar);
      }
    }
    // update of hfh
    for ( auto k=0 ; k < _Nv ; ++k )
      { fft::fft( &(_tmpf1[k][0]) , &(_tmpf1[k][0])+_Nx , &(hfh[k][0]) ); }

    // update uc
    uc += dt*E;
  }

  void
  phi_c ( _T dt , ublas::vector<_T> & uc , ublas::vector<_T> & E , complex_field<_T,NumDimsV> & hfh )
  {
    // equivalent of Hu of Nicolas
    _T rho_c=_rho_c;
    _T curtot = std::accumulate(
        uc.begin() , uc.end() ,
        0. ,
        [rho_c]( _T s , _T ui ) { return s + rho_c*ui; }
      ) * _dx / _x_max;

    // update E
    std::transform(
        E.begin()  , E.end() ,
        uc.begin() ,
        E.begin()  ,
        [&]( _T ei , _T ui ) { return ei - dt*(_rho_c*ui - curtot); }
      );
  }
};


template < typename _T >
struct splitVP1dx3dv
{
  std::size_t _Nz;
  std::size_t _Nvx,_Nvy,_Nvz;
  _T _dz,_dvx,_dvy,_dvz;
  _T _vz_min;
  _T _volumeV;
  field3d<_T> _tmpf0;
  field3d<_T> _tmpf1;
  ublas::vector<_T> kz;

  splitVP1dx3dv ( field3d<_T> const & f , _T l ) :
    _Nz(f.shape()[3]) ,
    _Nvx(f.shape()[0]) , _Nvy(f.shape()[1]) , _Nvz(f.shape()[2]) ,
    _dz(f.step.dz) , _dvx(f.step.dvx) , _dvy(f.step.dvy) , _dvz(f.step.dvz) ,
    _vz_min(f.range.vz_min) ,
    _volumeV(f.volumeV()) ,
    _tmpf0(tools::array_view<const std::size_t>(f.shape(),4)) ,
    _tmpf1(tools::array_view<const std::size_t>(f.shape(),4)) ,
    kz(f.shape()[3],0.)
  {
    for ( auto i=0 ; i<_Nz/2 ; ++i ) { kz[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nz/2 ; i<0 ; ++i ) { kz[_Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  void
  phi_a ( _T dt , complex_field<_T,3> & hf , ublas::vector<_T> & E )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + v_z\partial_z f = 0 \\
          \partial_z E_z = \int_{\mathbb{R}^3} f(t^{n+1})\,\mathrm{d}v - 1
        \end{cases}
      $$
      with Fourier in $z$ direction for each equation
    */
    const std::complex<double> & I = std::complex<double>(0.,1.);

    // compute hdiffrho and update hfh
    ublas::vector<std::complex<_T>> hdiffrho(_Nz,0.);
    for ( auto k_x=0 ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0 ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0 ; k_z<_Nvz ; ++k_z ) {
          _T vkz = k_z*_dvz + _vz_min;
          for ( auto i=0 ; i<_Nz ; ++i ) {
            // hrho_n
            hdiffrho[i] += hf[k_x][k_y][k_z][i]*_volumeV;
            // update hf
            hf[k_x][k_y][k_z][i] *= std::exp(-I*vkz*kz[i]*dt);
            // hrho_{n+1}
            hdiffrho[i] -= hf[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    // update E
    fft::spectrum_ hE(_Nz); hE.fft(E.begin());
    hE[0] = 0.;
    for ( auto i=1 ; i<_Nz ; ++i ) {
      hE[i] += I/kz[i]*hdiffrho[i];
    }
    hE.ifft(E.begin());
  }

  void
  phi_b ( _T dt , complex_field<_T,3> & hf , ublas::vector<_T> & E )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + E_z\partial_{v_z} f = 0 \\
          \partial_z E_z = 0
        \end{cases}
      $$
      with semi-Lagrangian method in $v_z$ direction.
    */

    for ( auto k_x=0 ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0 ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0 ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    for ( auto k_x=0 ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0 ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0 ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0 ; i<_Nz ; ++i ) {
            _T  vzstar = (k_z*_dvz + _vz_min) - dt*E[i]; // vstar = v_z - dt*E
            int kstar = std::ceil((vzstar - _vz_min)/_dvz);

            auto N = lagrange5::generator(
                        _tmpf0[k_x][k_y][(kstar-3+_Nvz)%_Nvz][i],_tmpf0[k_x][k_y][(kstar-2+_Nvz)%_Nvz][i],_tmpf0[k_x][k_y][(kstar-1+_Nvz)%_Nvz][i],_tmpf0[k_x][k_y][(kstar+_Nvz)%_Nvz][i],_tmpf0[k_x][k_y][(kstar+1+_Nvz)%_Nvz][i],_tmpf0[k_x][k_y][(kstar+2+_Nvz)%_Nvz][i] ,
                        _dvz , kstar*_dvz + _vz_min
                      );
            _tmpf1[k_x][k_y][k_z][i] = N(vzstar);
          }
        }
      }
    }

    for ( auto k_x=0 ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0 ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0 ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf1[k_x][k_y][k_z].begin() , _tmpf1[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }

  }
};


template < typename _T >
struct splitVA1dx3dv
{
  std::size_t _Nz;
  std::size_t _Nvx,_Nvy,_Nvz;
  _T _dz,_dvx,_dvy,_dvz;
  _T _vx_min,_vy_min,_vz_min;
  _T _volumeV;
  field3d<_T> _tmpf0;
  field3d<_T> _tmpf1;
  ublas::vector<_T> kz;

  splitVA1dx3dv ( field3d<_T> const & f , _T l ) :
    _Nz(f.shape()[3]) ,
    _Nvx(f.shape()[0]) , _Nvy(f.shape()[1]) , _Nvz(f.shape()[2]) ,
    _dz(f.step.dz) , _dvx(f.step.dvx) , _dvy(f.step.dvy) , _dvz(f.step.dvz) ,
    _vx_min(f.range.vx_min) , _vy_min(f.range.vy_min) , _vz_min(f.range.vz_min) ,
    _volumeV(f.volumeV()) ,
    _tmpf0(tools::array_view<const std::size_t>(f.shape(),4)) ,
    _tmpf1(tools::array_view<const std::size_t>(f.shape(),4)) ,
    kz(f.shape()[3],0.)
  {
    for ( auto i=0u    ; i<_Nz/2 ; ++i ) { kz[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nz/2 ; i<0     ; ++i ) { kz[_Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  void
  phi_a ( _T dt , complex_field<_T,3> & hf , ublas::vector<_T> & Ex , ublas::vector<_T> & Ey )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + v_z\partial_z f = 0 \\
          \partial_z E_{x,y} = \int_{\mathbb{R}^3} v_{x,y} f \,\mathrm{d}v + \bar{J}_{x.y}
        \end{cases}
      $$
      with Fourier in $z$ direction for each equation
    */
    const std::complex<double> & I = std::complex<double>(0.,1.);

    // compute hdiffJ and update hfh
    ublas::vector<std::complex<_T>> hdiffJx(_Nz,0.),hdiffJy(_Nz,0.);
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      double vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        double vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            // hJ_n
            hdiffJx[i] += (vx/vz)*hf[k_x][k_y][k_z][i]*_volumeV;
            hdiffJy[i] += (vy/vz)*hf[k_x][k_y][k_z][i]*_volumeV;
            // update hf
            hf[k_x][k_y][k_z][i] *= std::exp(-I*vz*kz[i]*dt);
            // hJ_{n+1}
            hdiffJx[i] -= (vx/vz)*hf[k_x][k_y][k_z][i]*_volumeV;
            hdiffJy[i] -= (vy/vz)*hf[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    // update Ex, Ey
    fft::spectrum_ hEx(_Nz), hEy(_Nz); hEx.fft(Ex.begin()); hEy.fft(Ey.begin());
    hEx[0] = 0.; hEy[0] = 0.;
    for ( auto i=1u ; i<_Nz ; ++i ) {
      hEx[i] += I/kz[i]*hdiffJx[i];
      hEy[i] += I/kz[i]*hdiffJy[i];
    }
    hEx.ifft(Ex.begin());
    hEy.ifft(Ey.begin());
  }
  
  void phi_b_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ex )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + E_x\partial_{v_x} f = 0
        \end{cases}
      $$
      with semi-Lagrangian method in $v_x$ direction.
    */
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vxstar = (k_x*_dvx + _vx_min) - dt*Ex[i]; // vstar = v_x - dt*Ex
            int kstar = std::ceil((vxstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vxstar);
          }
        }
      }
    }
  }
  
  void phi_b_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ey )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + E_y\partial_{v_y} f = 0
        \end{cases}
      $$
      with semi-Lagrangian method in $v_x$ direction.
    */
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vystar = (k_y*_dvy + _vy_min) - dt*Ey[i]; // vstar = v_y - dt*Ey
            int kstar = std::ceil((vystar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vystar);
          }
        }
      }
    }
  }

  void
  phi_b ( _T dt , complex_field<_T,3> & hf , ublas::vector<_T> & Ex , ublas::vector<_T> & Ey )
  {
    /*
      solve :
      $$
        \begin{cases}
          \partial_t f + E_x\partial_{v_x} f + E_y\partial_{v_y} f = 0 \\
          \partial_t E_{x,y} = 0
        \end{cases}
      $$

      with splitting in time in direction $v_x$ and $v_y$
    */

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    phi_b_vx(dt,_tmpf0,_tmpf1,Ex);
    phi_b_vy(dt,_tmpf1,_tmpf0,Ey);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf0[k_x][k_y][k_z].begin() , _tmpf0[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }

    
  }
};



template < typename _T >
struct hybird1dx3dv_b0
{
  std::size_t _Nz;
  std::size_t _Nvx,_Nvy,_Nvz;
  _T _dz,_dvx,_dvy,_dvz;
  _T _vx_min,_vy_min,_vz_min;
  _T _volumeV;
  field3d<_T> _tmpf0;
  field3d<_T> _tmpf1;
  ublas::vector<_T> kz;
  ublas::vector<_T> vx,vy;
  _T _B0;
  _T _l;

  hybird1dx3dv_b0 ( field3d<_T> const & f , _T l , const _T B0 ) :
    _Nz(f.shape()[3]) ,
    _Nvx(f.shape()[0]) , _Nvy(f.shape()[1]) , _Nvz(f.shape()[2]) ,
    _dz(f.step.dz) , _dvx(f.step.dvx) , _dvy(f.step.dvy) , _dvz(f.step.dvz) ,
    _vx_min(f.range.vx_min) , _vy_min(f.range.vy_min) , _vz_min(f.range.vz_min) ,
    _volumeV(f.volumeV()) ,
    _tmpf0(tools::array_view<const std::size_t>(f.shape(),4)) ,
    _tmpf1(tools::array_view<const std::size_t>(f.shape(),4)) ,
    kz(f.shape()[3],0.) ,
    vx(f.shape()[3],0.) , vy(f.shape()[3],0.) ,
    _B0(B0) , _l(l)
  {
    for ( auto i=0u    ; i<_Nz/2 ; ++i ) { kz[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nz/2 ; i<0     ; ++i ) { kz[_Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  void
  H_E_tilde_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ex )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f + (E_x + v_yB_0) \partial_{v_x} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vxstar = vx - dt*(Ex[i] + vy*_B0); // vxstar = v_x - dt*(Ex+vy*B0)
            int kstar = std::ceil((vxstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vxstar);
          }
        }
      }
    }
  }
  void
  H_E_tilde_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ey )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f + (E_y - v_xB_0) \partial_{v_y} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vystar = vy - dt*(Ey[i] - vx*_B0); // vystar = v_y - dt*(Ey-vx*B0)
            int kstar = std::ceil((vystar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vystar);
          }
        }
      }
    }
  }

  void
  H_E_tilde ( _T dt ,
    ublas::vector<_T> & jcx , ublas::vector<_T> & jcy ,
    ublas::vector<_T> & Ex  , ublas::vector<_T> & Ey  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = E_x //
          \partial_t j_{c,y} = E_y //
          \partial_t E_{x} = 0 //
          \partial_t E_{y} = 0 //
          \partial_t f + (E_x + v_yB_0) \partial_{v_x} f + (E_y - v_xB_0) \partial_{v_y} f = 0 //
        \end{cases}
      $$
    **/

    for ( auto i=0u ; i<_Nz ; ++i ) {
      jcx[i] += dt*Ex[i];
      jcy[i] += dt*Ey[i];
    }

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    H_E_tilde_vx(dt,_tmpf0,_tmpf1,Ex);
    H_E_tilde_vy(dt,_tmpf1,_tmpf0,Ey);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf0[k_x][k_y][k_z].begin() , _tmpf0[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }
  }


  void
  H_E_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ex )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f + E_x \partial_{v_x} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vx - dt*Ex[i]; // vstar = v_x - dt*Ex
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_E_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ey )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f + E_y \partial_{v_y} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vy - dt*Ey[i]; // vstar = v_y - dt*Ey
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }

  void
  H_E ( _T dt ,
    ublas::vector<_T> & jcx , ublas::vector<_T> & jcy ,
    ublas::vector<_T> & Ex  , ublas::vector<_T> & Ey  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = E_x //
          \partial_t j_{c,y} = E_y //
          \partial_t E_{x} = 0 //
          \partial_t E_{y} = 0 //
          \partial_t f + E_x \partial_{v_x} f + E_y \partial_{v_y} f = 0 //
        \end{cases}
      $$
    **/

    for ( auto i=0u ; i<_Nz ; ++i ) {
      jcx[i] += dt*Ex[i];
      jcy[i] += dt*Ey[i];
    }

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    H_E_vx(dt,_tmpf0,_tmpf1,Ex);
    H_E_vy(dt,_tmpf1,_tmpf0,Ey);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf0[k_x][k_y][k_z].begin() , _tmpf0[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }

  }

  void
  H_jc ( _T dt ,
    ublas::vector<_T> & jcx , ublas::vector<_T> & jcy ,
    ublas::vector<_T> & Ex  , ublas::vector<_T> & Ey  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = j_{c,y}B_0 //
          \partial_t j_{c,y} = -j_{c,x}B_0 //
          \partial_t E_{x} = -j_{c,x} //
          \partial_t E_{y} = -j_{c,y} //
          \partial_t f = 0 //
        \end{cases}
      $$
    **/

    for ( auto i=0u ; i<_Nz ; ++i ) {
      // (jcx,jcy)^{n+1} = exp(B0 J dt)*(jcx,jcy)^n
      double jcxn1 =  jcx[i]*std::cos(_B0*dt) + jcy[i]*std::sin(_B0*dt);
      double jcyn1 = -jcx[i]*std::sin(_B0*dt) + jcy[i]*std::cos(_B0*dt);

      // (Ex,Ey)^{n+1} = 1/B0 J (exp(B0 J dt) - I) (jcx,jcy)^n
      Ex[i] = 1./_B0 * ( -jcx[i]*std::sin(_B0*dt)      + jcy[i]*(std::cos(_B0*dt)-1.) );
      Ey[i] = 1./_B0 * (  jcx[i]*(1.-std::cos(_B0*dt)) - jcy[i]*std::cos(_B0*dt)      );

      jcx[i] = jcxn1;
      jcy[i] = jcyn1;
    }
  }

  void
  H_f_tilde ( _T dt ,
    ublas::vector<_T> & jcx , ublas::vector<_T> & jcy ,
    ublas::vector<_T> & Ex  , ublas::vector<_T> & Ey  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = 0 //
          \partial_t j_{c,y} = 0 //
          \partial_t E_{x} = -\int v_x f\,\mathrm{d}v //
          \partial_t E_{y} = -\int v_y f\,\mathrm{d}v //
          \partial_t f + v_z\partial_z f = 0 //
        \end{cases}
      $$
    **/

    const std::complex<_T> & I = std::complex<_T>(0.,1.);

    // compute hdiffJ and update hfh
    ublas::vector<std::complex<_T>> hdiffJx(_Nz,0.),hdiffJy(_Nz,0.);
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            // hJ_n
            hdiffJx[i] += vx/vz*hf[k_x][k_y][k_z][i]*_volumeV;
            hdiffJy[i] += vy/vz*hf[k_x][k_y][k_z][i]*_volumeV;
            // update hf
            hf[k_x][k_y][k_z][i] *= std::exp(-I*vz*kz[i]*dt);
            // hJ_{n+1}
            hdiffJx[i] -= vx/vz*hf[k_x][k_y][k_z][i]*_volumeV;
            hdiffJy[i] -= vy/vz*hf[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    // update Ex, Ey
    fft::spectrum_ hEx(_Nz), hEy(_Nz); hEx.fft(Ex.begin()); hEy.fft(Ey.begin());
    hEx[0] = 0.; hEy[0] = 0.;
    for ( auto i=1u ; i<_Nz ; ++i ) {
      hEx[i] += I/kz[i]*hdiffJx[i];
      hEy[i] += I/kz[i]*hdiffJy[i];
    }
    hEx.ifft(Ex.begin());
    hEy.ifft(Ey.begin());
  }

  void
  H_f_1 ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , ublas::vector<_T> & Ex )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - v_xB_0 \partial_{v_y} f = 0 \\
          \partial_t E_x = - \int v_x f\,\mathrm{d}v
        \end{cases}
      $$
      In fact $\int v_x f\,\mathrm{d}v$ doesn't depend on $t$ (it's constant in this step)
    **/

    ublas::vector<_T> jx(_Nz,0.);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vy + dt*vx*_B0; // vstar = v_y + dt*v_x*B_0
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
            jx[i] += vx*fout[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    _T mean = 0.;
    for ( auto i=0u ; i<_Nz ; ++i ) {
      mean += jx[i]/_Nz;
    }
    //std::cout << mean << " ";

    for ( auto i=0u ; i<_Nz ; ++i ) {
      Ex[i] = Ex[i] - dt*jx[i] + dt*mean;
    }
  }
  void
  H_f_2 ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , ublas::vector<_T> & Ey )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - v_yB_0 \partial_{v_x} f = 0 \\
          \partial_t E_y = - \int v_y f\,\mathrm{d}v
        \end{cases}
      $$
      In fact $\int v_y f\,\mathrm{d}v$ doesn't depend on $t$ (it's constant in this step)
    **/

    ublas::vector<_T> jy(_Nz,0.);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vx - dt*vy*_B0; // vstar = v_x + dt*v_y*B_0
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
            jy[i] += vy*fout[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    _T mean = 0.;
    for ( auto i=0u ; i<_Nz ; ++i ) {
      mean += jy[i]/_Nz;
    }
    //std::cout << mean << "\n";

    for ( auto i=0u ; i<_Nz ; ++i ) {
      Ey[i] = Ey[i] - dt*jy[i] + dt*mean;
    }
  }
  void
  H_f_3 ( _T dt , complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - v_z\partial_{v_z} f = 0
        \end{cases}
      $$
    **/

    const std::complex<double> & I = std::complex<double>(0.,1.);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            hf[k_x][k_y][k_z][i] *= std::exp(-I*vz*kz[i]*dt);
          }
        }
      }
    }
  }

  void
  H_f ( _T dt ,
    ublas::vector<_T> & jcx , ublas::vector<_T> & jcy ,
    ublas::vector<_T> & Ex  , ublas::vector<_T> & Ey  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = 0 //
          \partial_t j_{c,y} = 0 //
          \partial_t E_{x} = -\int v_x f\,\mathrm{d}v //
          \partial_t E_{y} = -\int v_y f\,\mathrm{d}v //
          \partial_t f + v_z\partial_z f + v_yB_0\partial_{v_x} f - v_xB_0\partial_{v_y} f = 0 //
        \end{cases}
      $$
    **/

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    H_f_1(dt,_tmpf0,_tmpf1,Ex);
    H_f_2(dt,_tmpf1,_tmpf0,Ey);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf0[k_x][k_y][k_z].begin() , _tmpf0[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }

    H_f_3(dt,hf);

  }

};


template < typename _T >
struct hybird1dx3dv
{
  std::size_t _Nz;
  std::size_t _Nvx,_Nvy,_Nvz;
  _T _dz,_dvx,_dvy,_dvz;
  _T _z_min,_vx_min,_vy_min,_vz_min;
  _T _volumeV;
  field3d<_T> _tmpf0;
  field3d<_T> _tmpf1;
  ublas::vector<_T> kz;
  ublas::vector<_T> vx,vy;
  _T _B0;
  _T _l;

  hybird1dx3dv ( field3d<_T> const & f , _T l , const _T B0 ) :
    _Nz(f.shape()[3]) ,
    _Nvx(f.shape()[0]) , _Nvy(f.shape()[1]) , _Nvz(f.shape()[2]) ,
    _dz(f.step.dz) , _dvx(f.step.dvx) , _dvy(f.step.dvy) , _dvz(f.step.dvz) ,
    _z_min(f.range.z_min),_vx_min(f.range.vx_min) , _vy_min(f.range.vy_min) , _vz_min(f.range.vz_min) ,
    _volumeV(f.volumeV()) ,
    _tmpf0(tools::array_view<const std::size_t>(f.shape(),4)) ,
    _tmpf1(tools::array_view<const std::size_t>(f.shape(),4)) ,
    kz(f.shape()[3],0.) ,
    vx(f.shape()[3],0.) , vy(f.shape()[3],0.) ,
    _B0(B0) , _l(l)
  {
    for ( auto i=0u    ; i<_Nz/2 ; ++i ) { kz[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-_Nz/2 ; i<0     ; ++i ) { kz[_Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  void
  H_E_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ex )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - E_x \partial_{v_x} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vx + dt*Ex[i]; // vstar = v_x + dt*Ex
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_E_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Ey )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - E_y \partial_{v_y} f = 0
        \end{cases}
      $$
    **/
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vy + dt*Ey[i]; // vstar = v_y + dt*Ey
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }

  void
  H_E ( _T dt ,
          ublas::vector<_T> & jcx ,       ublas::vector<_T> & jcy ,
    const ublas::vector<_T> & Ex  , const ublas::vector<_T> & Ey  ,
          ublas::vector<_T> & Bx  ,       ublas::vector<_T> & By  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = 4 E_x //
          \partial_t j_{c,y} = 4 E_y //
          \partial_t E_{x} = 0 //
          \partial_t E_{y} = 0 //
          \partial_t B_{x} =  \partial_z E_y //
          \partial_t B_{y} = -\partial_z E_x //
          \partial_t f - E_x \partial_{v_x} f - E_y \partial_{v_y} f = 0 //
        \end{cases}
      $$
    **/
    static const std::complex<double> & I = std::complex<double>(0.,1.);

    // update j_{c,x|y} 
    for ( auto i=0u ; i<_Nz ; ++i ) {
      jcx[i] += 4*dt*Ex[i];
      jcy[i] += 4*dt*Ey[i];

      #if JC_condition == 0
        jcx[i] = 0.;
        jcy[i] = 0.;
      #endif
    }

    // update f
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    H_E_vx(dt,_tmpf0,_tmpf1,Ex);
    H_E_vy(dt,_tmpf1,_tmpf0,Ey);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::fft( _tmpf0[k_x][k_y][k_z].begin() , _tmpf0[k_x][k_y][k_z].end() , hf[k_x][k_y][k_z].begin() );
        }
      }
    }

    // update B
    fft::spectrum_ hBx(_Nz); hBx.fft(Bx.begin());
    fft::spectrum_ hBy(_Nz); hBy.fft(By.begin());
    fft::spectrum_ hEx(_Nz); hEx.fft(Ex.begin());
    fft::spectrum_ hEy(_Nz); hEy.fft(Ey.begin());

    for ( auto i=0 ; i<_Nz ; ++i ) {
      hBx[i] = hBx[i] + dt*I*kz[i]*hEy[i];
      hBy[i] = hBy[i] - dt*I*kz[i]*hEx[i];

      #if Bxy_condition == 0
        hBx[i] = 0.;
        hBy[i] = 0.;
      #endif
    }
    hBx.ifft(Bx.begin());
    hBy.ifft(By.begin());


    #if Bxy_condition == 0
      for ( auto i=0 ; i<_Nz ; ++i ) {
        Bx[i] = 0.;
        By[i] = 0.;
      }
    #endif
  }

  void
  H_B ( _T dt ,
    const ublas::vector<_T> & jcx , const ublas::vector<_T> & jcy ,
          ublas::vector<_T> & Ex  ,       ublas::vector<_T> & Ey  ,
    const ublas::vector<_T> & Bx  , const ublas::vector<_T> & By  ,
    const complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = 0 //
          \partial_t j_{c,y} = 0 //
          \partial_t E_{x} = -\partial_z B_y //
          \partial_t E_{y} =  \partial_z B_x //
          \partial_t B_{x} = 0 //
          \partial_t B_{y} = 0 //
          \partial_t f = 0 //
        \end{cases}
      $$
    **/
    static const std::complex<double> & I = std::complex<double>(0.,1.);

    // update E
    fft::spectrum_ hBx(_Nz); hBx.fft(Bx.begin());
    fft::spectrum_ hBy(_Nz); hBy.fft(By.begin());
    fft::spectrum_ hEx(_Nz); hEx.fft(Ex.begin());
    fft::spectrum_ hEy(_Nz); hEy.fft(Ey.begin());

    for ( auto i=0 ; i<_Nz ; ++i ) {
      hEx[i] = hEx[i] - dt*I*kz[i]*hBy[i];
      hEy[i] = hEy[i] + dt*I*kz[i]*hBx[i];
    }
    hEx.ifft(Ex.begin());
    hEy.ifft(Ey.begin());
  }

  void
  H_jc ( _T dt ,
          ublas::vector<_T> & jcx ,       ublas::vector<_T> & jcy ,
          ublas::vector<_T> & Ex  ,       ublas::vector<_T> & Ey  ,
    const ublas::vector<_T> & Bx  , const ublas::vector<_T> & By  ,
    const complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = -j_{c,y} //
          \partial_t j_{c,y} =  j_{c,x} //
          \partial_t E_{x} = -j_{c,x} //
          \partial_t E_{y} = -j_{c,y} //
          \partial_t B_{x} = 0 //
          \partial_t B_{y} = 0 //
          \partial_t f = 0 //
        \end{cases}
      $$
    **/

    for ( auto i=0u ; i<_Nz ; ++i ) {
      // (jcx,jcy)^{n+1} = exp(-J dt)*(jcx,jcy)^n
      double jcxn1 =  jcx[i]*std::cos(dt) - jcy[i]*std::sin(dt);
      double jcyn1 =  jcx[i]*std::sin(dt) + jcy[i]*std::cos(dt);

      #if JC_condition == 0
        jcxn1 = 0.;
        jcyn1 = 0.;
        jcx[i] = 0.;
        jcy[i] = 0.;
      #endif

      // (Ex,Ey)^{n+1} = (Ex,Ey)^n - J (exp(-J dt) - I) (jcx,jcy)^n
      Ex[i] += -jcx[i]*std::sin(dt)      + jcy[i]*(1.-std::cos(dt));
      Ey[i] +=  jcx[i]*(std::cos(dt)-1.) - jcy[i]*std::sin(dt);

      jcx[i] = jcxn1;
      jcy[i] = jcyn1;
    }

  }


  void
  H_f_1_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout )
  {
    // \partial_t f + vxB0\partial_{v_y} f = 0
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vy - dt*vx*_B0;
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_1_vz ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , ublas::vector<_T> & Ex , const ublas::vector<_T> & By )
  {
    // \partial_t f - vxBy\partial_{v_z} f = 0
    // \partial_t Ex = \int v_x f\,\mathrm{d}v

    ublas::vector<_T> jx(_Nz,0.);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vz + dt*vx*By[i];
            int kstar = std::ceil((vstar - _vz_min)/_dvz);

            auto N = lagrange5::generator(
                        fin[k_x][k_y][(kstar-3+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar-2+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar-1+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar  +_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar+1+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar+2+_Nvz)%_Nvz][i],
                        _dvz , kstar*_dvz + _vz_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
            jx[i] += vx*fout[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    // pour s'assurer que Ex est de moyenne nulle
    _T mean = 0.;
    for ( auto i=0u ; i<_Nz ; ++i ) {
      mean += jx[i]/_Nz;
    }

    for ( auto i=0u ; i<_Nz ; ++i ) {
      Ex[i] += dt*(jx[i] - mean);
    }
  }

  void
  H_f_2_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout )
  {
    // \partial_t f - v_yB_0\partial_{v_x} f = 0
    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vx + dt*vy*_B0;
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_2_vz ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , ublas::vector<_T> & Ey , const ublas::vector<_T> & Bx )
  {
    // \partial_t f + v_yB_x\partial_{v_z} f = 0
    // \partial_t E_y = \int v_y f\,\mathrm{d}v

    ublas::vector<_T> jy(_Nz,0.);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vz - dt*vy*Bx[i];
            int kstar = std::ceil((vstar - _vz_min)/_dvz);

            auto N = lagrange5::generator(
                        fin[k_x][k_y][(kstar-3+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar-2+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar-1+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar  +_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar+1+_Nvz)%_Nvz][i],
                        fin[k_x][k_y][(kstar+2+_Nvz)%_Nvz][i],
                        _dvz , kstar*_dvz + _vz_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
            jy[i] += vy*fout[k_x][k_y][k_z][i]*_volumeV;
          }
        }
      }
    }

    // pour s'assurer que Ex est de moyenne nulle
    _T mean = 0.;
    for ( auto i=0u ; i<_Nz ; ++i ) {
      mean += jy[i]/_Nz;
    }

    for ( auto i=0u ; i<_Nz ; ++i ) {
      Ey[i] += dt*(jy[i] - mean);
    }
  }

  void
  H_f_3_vx ( _T dt , const field3d<_T> & gin , field3d<_T> & gout , const ublas::vector<_T> & By ) {
    /**
      solve:
      $$
        \begin{cases}
          \partial_t g + v_zB_y(z+tv_z)\partial_{v_x}g = 0
        \end{cases}
      $$
    **/

    const std::complex<double> & I = std::complex<double>(0.,1.);
    fft::spectrum_ hB(_Nz); hB.fft(By.begin());
    //ublas::vector<double> tmp(_Nz,0.);
    //fft::spectrum_ htmp(_Nz);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          //for ( auto i=1u ; i<_Nz ; ++i )
          //  { htmp[i] = -I*hB[i]/kz[i]*(std::exp(I*kz[i]*vz*dt)-1.); }
          //htmp.ifft(tmp.begin());

          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T z = i*_dz + _z_min;

            std::complex<double> shBy = 0.;
            for ( auto k=1u    ; k<_Nz/2 ; ++k ) {
              shBy += -I*(hB[k]/static_cast<double>(_Nz)) / (kz[k]) * std::exp(I*kz[k]*z) * ( std::exp(I*kz[k]*vz*dt) - 1. );
            }
            for ( int k=-_Nz/2 ; k<0 ; ++k ) {
              shBy += -I*std::conj(hB[-k]/static_cast<double>(_Nz)) / (kz[_Nz+k]) * std::exp(I*kz[_Nz+k]*z) * ( std::exp(I*kz[_Nz+k]*vz*dt) - 1. );
            }
            /*
            for ( auto k=1u ; k<_Nz/2 ; ++k ) {
              shBy += -I*hB[k] / (kz[k]) * std::exp(I*kz[k]*z) * ( std::exp(I*kz[k]*vz*dt) - 1. );
            }
            */

            // shBy is in theory a real
            if ( std::imag(shBy) >= 1e-5 ) { std::cerr << "H_f_3_vx : \033[31;1m" << shBy << "\033[0m" << std::endl; }
            //std::cout << "\r" << hB[0] << " " << shBy << "   ";
            

            _T  vstar = vx - std::real(shBy);
            //_T  vstar = vx - tmp[i];
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        gin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        gin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        gin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        gin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        gin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        gin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            gout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_3_vy ( _T dt , const field3d<_T> & gin , field3d<_T> & gout , const ublas::vector<_T> & Bx ) {
    /**
      solve:
      $$
        \begin{cases}
          \partial_t g - v_zB_x(z+t_v_z)\partial_{v_y}g = 0
        \end{cases}
      $$
    **/

    const std::complex<double> & I = std::complex<double>(0.,1.);
    fft::spectrum_ hB(_Nz); hB.fft(Bx.begin());
    //ublas::vector<double> tmp(_Nz,0.);
    //fft::spectrum_ htmp(_Nz);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          //for ( auto i=1u ; i<_Nz ; ++i )
          //  { htmp[i] = -I*hB[i]/kz[i]*(std::exp(I*kz[i]*vz*dt)-1.); }
          //htmp.ifft(tmp.begin());

          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T z = i*_dz + _z_min;

            std::complex<double> shBx = 0.;
            for ( auto k=1u    ; k<_Nz/2 ; ++k ) {
              shBx += -I*(hB[k]/static_cast<double>(_Nz)) / (kz[k]) * std::exp(I*kz[k]*z) * ( std::exp(I*kz[k]*vz*dt) - 1. );
            }
            for ( int k=-_Nz/2 ; k<0 ; ++k ) {
              shBx += -I*std::conj(hB[-k]/static_cast<double>(_Nz)) / (kz[_Nz+k]) * std::exp(I*kz[_Nz+k]*z) * ( std::exp(I*kz[_Nz+k]*vz*dt) - 1. );
            }

            /*
            std::complex<double> shBx = 0.;
            for ( auto k=1u ; k<_Nz ; ++k ) {
              shBx += -I*hB[k] / (kz[k]) * std::exp(I*kz[k]*z) * ( std::exp(I*kz[k]*vz*dt) - 1. );
            }
            */

            // shBx is in theory a real
            if ( std::imag(shBx) >= 1e-5 ) { std::cerr << "H_f_3_vy : \033[31;1m" << shBx << "\033[0m" << std::endl; }
            //std::cout << "\r" << hB[0] << " " << shBx << "   ";

            _T  vstar = vy + std::real(shBx);
            //_T vstar = vy + tmp[i];
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        gin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        gin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        gin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        gin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        gin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        gin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            gout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_3_g_to_f ( _T dt , const field3d<_T> & gin , complex_field<_T,3> & hfout ) {
    /**
      solve:
      $$
        f(t^{n+1},z,v_x,v_y,v_z) = g(t^{n+1},z-\Delta t v_z,v_x,v_y,v_z)
      $$
      with an FFT
    **/

    const std::complex<double> & I = std::complex<double>(0.,1.);

    fft::spectrum_ hgin(_Nz);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          fft::fft( gin[k_x][k_y][k_z].begin() , gin[k_x][k_y][k_z].end() , hgin.begin() );
          for ( auto i=0u ; i<_Nz ; ++i ) {
            hfout[k_x][k_y][k_z][i] = hgin[i]*std::exp(-I*vz*kz[i]*dt);
          }
        }
      }
    }
  }

  void
  H_f_3_tilde_vx ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & By ) {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - (- v_zB_y)\partial_{v_x} f = 0
        \end{cases}
      $$
    **/

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vx - dt*(vz*By[i]); // vstar = v_x - dt*(vz*B_y)
            int kstar = std::ceil((vstar - _vx_min)/_dvx);

            auto N = lagrange5::generator(
                        fin[(kstar-3+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-2+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar-1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar  +_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+1+_Nvx)%_Nvx][k_y][k_z][i],
                        fin[(kstar+2+_Nvx)%_Nvx][k_y][k_z][i],
                        _dvx , kstar*_dvx + _vx_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_3_tilde_vy ( _T dt , const field3d<_T> & fin , field3d<_T> & fout , const ublas::vector<_T> & Bx ) {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f - (v_zB_x)\partial_{v_y} f = 0
        \end{cases}
      $$
    **/

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      _T vx = k_x*_dvx + _vx_min;
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        _T vy = k_y*_dvy + _vy_min;
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          for ( auto i=0u ; i<_Nz ; ++i ) {
            _T  vstar = vy + dt*(vz*Bx[i]); // vstar = v_x + dt*(v_zB_x)
            int kstar = std::ceil((vstar - _vy_min)/_dvy);

            auto N = lagrange5::generator(
                        fin[k_x][(kstar-3+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-2+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar-1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar  +_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+1+_Nvy)%_Nvy][k_z][i],
                        fin[k_x][(kstar+2+_Nvy)%_Nvy][k_z][i],
                        _dvy , kstar*_dvy + _vy_min
                      );
            fout[k_x][k_y][k_z][i] = N(vstar);
          }
        }
      }
    }
  }
  void
  H_f_3_tilde_z ( _T dt , const field3d<_T> & fin , complex_field<_T,3> & hfout ) {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t f + v_z\partial_z f = 0
        \end{cases}
      $$
      with a FFT
    **/

    const std::complex<double> & I = std::complex<double>(0.,1.);

    fft::spectrum_ hfin(_Nz);

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          _T vz = k_z*_dvz + _vz_min;
          fft::fft( fin[k_x][k_y][k_z].begin() , fin[k_x][k_y][k_z].end() , hfin.begin() );
          for ( auto i=0u ; i<_Nz ; ++i ) {
            hfout[k_x][k_y][k_z][i] = hfin[i]*std::exp(-I*vz*kz[i]*dt);
          }
        }
      }
    }
  }


  void
  H_f ( _T dt ,
    const ublas::vector<_T> & jcx , const ublas::vector<_T> & jcy ,
          ublas::vector<_T> & Ex  ,       ublas::vector<_T> & Ey  ,
    const ublas::vector<_T> & Bx  , const ublas::vector<_T> & By  ,
    complex_field<_T,3> & hf )
  {
    /**
      solve :
      $$
        \begin{cases}
          \partial_t j_{c,x} = 0 //
          \partial_t j_{c,y} = 0 //
          \partial_t E_{x} = \int v_x f\,\mathrm{d}v //
          \partial_t E_{y} = \int v_y f\,\mathrm{d}v //
          \partial_t B_{x} = 0 //
          \partial_t B_{y} = 0 //
          \partial_t f + v_z\partial_z f - (v_yB_0 - v_zB_y)\partial_{v_x} f - (-v_xB_0 + v_zB_x)\partial_{v_y} f - (v_xB_y - v_yB_x)\partial_{v_z} f = 0 //
        \end{cases}
      $$
      with `B0=1`.
    **/

    for ( auto k_x=0u ; k_x<_Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<_Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<_Nvz ; ++k_z ) {
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , _tmpf0[k_x][k_y][k_z].begin() );
        }
      }
    }

    // Hf1
    H_f_1_vy(dt,_tmpf0,_tmpf1);
    H_f_1_vz(dt,_tmpf1,_tmpf0,Ex,By);

    // Hf2
    H_f_2_vx(dt,_tmpf0,_tmpf1);
    H_f_2_vz(dt,_tmpf1,_tmpf0,Ey,Bx);

    // Hf3
    H_f_3_vx(dt,_tmpf0,_tmpf1,By);
    H_f_3_vy(dt,_tmpf1,_tmpf0,Bx);
    H_f_3_g_to_f(dt,_tmpf0,hf);
    
    /*
    // Hf3tilde
    H_f_3_tilde_vx(dt,_tmpf0,_tmpf1,By);
    H_f_3_tilde_vy(dt,_tmpf1,_tmpf0,Bx);
    H_f_3_tilde_z(dt,_tmpf0,hf);
    */
  }

};

#endif

#include <iostream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <fstream>
#include <valarray>
#include <sstream>
#include <string>
#include <list>
#include <vector>

using namespace std::string_literals;

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/constants/constants.hpp>

#include "vlasovpp/field.h"
#include "vlasovpp/complex_field.h"
#include "vlasovpp/weno.h"
#include "vlasovpp/fft.h"
#include "vlasovpp/array_view.h"
#include "vlasovpp/poisson.h"
#include "vlasovpp/rk.h"
#include "vlasovpp/config.h"
#include "vlasovpp/signal_handler.h"
#include "vlasovpp/iteration.h"
#include "vlasovpp/splitting.h"

namespace math = boost::math::constants;
const std::complex<double> & I = std::complex<double>(0.,1.);

#define SQ(X) ((X)*(X))
#define Zi(i) (i*f.step.dz+f.range.z_min)
#define Vkx(k) (k*f.step.dvx+f.range.vx_min)
#define Vky(k) (k*f.step.dvy+f.range.vy_min)
#define Vkz(k) (k*f.step.dvz+f.range.vz_min)

auto
maxwellian ( double rho , std::vector<double> u , double T ) {
  return [=](double z,double vx,double vy,double vz) {
    return rho/( std::pow(2.*math::pi<double>()*T,1.5) )*std::exp( -0.5*(SQ(vx-u[0])+SQ(vy-u[1])+SQ(vz-u[2]))/T );
  };
}

int
main ( int argc , char const * argv[] )
{
  std::string p("config.init");
  if ( argc > 1 )
    { p = argv[1]; }
  auto c = config(p);
  c.name = "vmls";

  c.create_output_directory();
  {
    std::ofstream ofconfig( c.output_dir / "config.init" );
    ofconfig << c << "\n";
    ofconfig.close();
  }

/* ------------------------------------------------------------------------- */
  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  complex_field<double,3> hf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  const double Kx = 0.5;
  f.range.vx_min = -5.; f.range.vx_max = 5.;
  f.range.vy_min = -5.; f.range.vy_max = 5.;
  f.range.vz_min = -5.; f.range.vz_max = 5.;
  f.range.z_min =  0.;  f.range.z_max = 2./Kx*math::pi<double>();
  f.compute_steps();

  double dt = 0.1;

  ublas::vector<double> vx(c.Nv,0.),vy(c.Nv,0.),vz(c.Nv,0.);
  std::generate( vx.begin() , vx.end() , [&,k=0]() mutable {return (k++)*f.step.dvx+f.range.vx_min;} );
  std::generate( vy.begin() , vy.end() , [&,k=0]() mutable {return (k++)*f.step.dvy+f.range.vy_min;} );
  std::generate( vz.begin() , vz.end() , [&,k=0]() mutable {return (k++)*f.step.dvz+f.range.vz_min;} );

  ublas::vector<double> kx(c.Nx); // beware, Nx need to be odd
  {
    double l = f.range.len_z();
    for ( auto i=0u ; i<c.Nx/2 ; ++i ) { kx[i]      = 2.*math::pi<double>()*i/l; }
    for ( int i=-c.Nx/2 ; i<0 ; ++i ) { kx[c.Nx+i] = 2.*math::pi<double>()*i/l; }
  }

  auto M1 = maxwellian(1.,{0.,0.,0.},1.);
  for (std::size_t k_x=0u ; k_x<f.size(0) ; ++k_x ) {
    for (std::size_t k_y=0u ; k_y<f.size(1) ; ++k_y ) {
      for (std::size_t k_z=0u ; k_z<f.size(2) ; ++k_z ) {
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          f[k_x][k_y][k_z][i] = M1( Zi(i),Vkx(k_x),Vky(k_y),Vkz(k_z) ) * (1. + 0.5*std::cos(Kx*Zi(i)));
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }


  const double B0 = 1.;
  ublas::vector<double> Ex(c.Nz,0.),Ey(c.Nz,0.);
  ublas::vector<double> jcx(c.Nz,0.),jcy(c.Nz,0.);


  std::vector<double> electric_energy;  electric_energy.reserve(100);
  std::vector<double> electric_energy_x; electric_energy_x.reserve(100);
  std::vector<double> electric_energy_y; electric_energy_y.reserve(100);
  std::vector<double> times; times.reserve(100);

  hybird1dx3dv_b0<double> Lie( f , f.range.len_z() , B0 );
  double current_t = 0.;
  times.push_back(0.);

  auto compute_electric_energy = [&]( const ublas::vector<double> & E ) {
    double electric_energy = 0.;
    for ( const auto & ei : E ) { electric_energy += ei*ei*f.step.dz; }
      return electric_energy;
  };
  double eex=0.,eey=0.;
  eex = compute_electric_energy(Ex); eey = compute_electric_energy(Ey);
  electric_energy_x.push_back(std::sqrt(eex));
  electric_energy_y.push_back(std::sqrt(eey));
  electric_energy.push_back(std::sqrt(eex+eey));

  while ( current_t<c.Tf ) {
    std::cout << "\r" << current_t << " / " << c.Tf << std::flush; 

/*
    Lie.H_E_tilde(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }
    Lie.H_jc(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }
    Lie.H_f_tilde(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }
*/

    Lie.H_E(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }

    Lie.H_jc(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }

    Lie.H_f(dt,jcx,jcy,Ex,Ey,hf);
    //for ( int i = 0 ; i<c.Nz ; ++i ) { jcx[i] = 0.; jcy[i] = 0.; }

    eex = compute_electric_energy(Ex); eey = compute_electric_energy(Ey);
    electric_energy.push_back( std::sqrt(eex+eey) );
    electric_energy_x.push_back(std::sqrt(eex));
    electric_energy_y.push_back(std::sqrt(eey));
    current_t += dt;
    times.push_back(current_t);
  }

  auto dt_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };
  c << monitoring::data( "ee_tilde.dat"  , electric_energy   , dt_y );
  c << monitoring::data( "eex_tilde.dat" , electric_energy_x , dt_y );
  c << monitoring::data( "eey_tilde.dat" , electric_energy_y , dt_y );

  return 0;
}

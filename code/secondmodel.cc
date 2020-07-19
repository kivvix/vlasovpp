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
  c.name = "vhll";

  c.create_output_directory();
  std::ofstream ofconfig( c.output_dir / "config.init" );
  ofconfig << c << "\n";
  ofconfig.close();

/* ------------------------------------------------------------------------- */
  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  complex_field<double,3> hf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  //c.Tf = c.Tf;
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
          //f[k_x][k_y][k_z][i] = ( SQ(Vkz(k_z))*M1(Zi(i),Vkx(k_x),Vky(k_y),Vkz(k_z)) )*(1. + 0.05*std::cos(Kx*Zi(i)));
          f[k_x][k_y][k_z][i] = M1( Zi(i),Vkx(k_x),Vky(k_y),Vkz(k_z) ) * (1. + 0.05*std::cos(Kx*Zi(i)));
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }

  poisson<double> poisson_solver(c.Nx,f.range.len_z());
  ublas::vector<double> rho(c.Nx,0.); rho = f.density();

  ublas::vector<double> Ex(c.Nz,0.),Ey(c.Nz,0.);
  //Ex = poisson_solver(rho);

  std::vector<double> ee;  ee.reserve(100);
  std::vector<double> eex; eex.reserve(100);
  std::vector<double> eey; eey.reserve(100);
  std::vector<double> times; times.reserve(100);

  splitVA1dx3dv<double> Lie( f , f.range.len_z() );
  double current_t = 0.;
  times.push_back(0.);

  auto compute_electric_energy = [&]( const ublas::vector<double> & E ) {
    double electric_energy = 0.;
    for ( const auto & ei : E ) { electric_energy += ei*ei*f.step.dz; }
      return electric_energy;
  };
  eex.push_back(compute_electric_energy(Ex));
  eey.push_back(compute_electric_energy(Ey));
  ee.push_back( std::sqrt(compute_electric_energy(Ex)+compute_electric_energy(Ey)) );

  while ( current_t<c.Tf ) {
    std::cout << "\r" << current_t << " / " << c.Tf << std::flush; 

    Lie.phi_a(dt,hf,Ex,Ey);
    Lie.phi_b(dt,hf,Ex,Ey);

    ee.push_back( std::sqrt(compute_electric_energy(Ex)+compute_electric_energy(Ey)) );
    eex.push_back(std::sqrt(compute_electric_energy(Ex)));
    eey.push_back(std::sqrt(compute_electric_energy(Ey)));
    current_t += dt;
    times.push_back(current_t);
  }

  auto dt_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };
  c << monitoring::data( "ee_secondmodel.dat"  , ee  , dt_y );
  c << monitoring::data( "eex_secondmodel.dat" , eex , dt_y );
  c << monitoring::data( "eey_secondmodel.dat" , eey , dt_y );

  return 0;
}

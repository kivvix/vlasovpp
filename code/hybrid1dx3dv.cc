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
maxwellian ( double rho , std::vector<double> u , std::vector<double> T ) {
  return [=](double z,double vx,double vy,double vz) {
    return rho/( std::pow(2.*math::pi<double>(),1.5)*T[0]*T[1]*T[2] ) * std::exp( -0.5*( SQ((vx-u[0])/T[0]) + SQ((vy-u[1])/T[1]) + SQ((vz-u[2])/T[2]) ) );
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

  const double K = 2.;
  f.range.vx_min = -3.; f.range.vx_max = 3.;
  f.range.vy_min = -3.; f.range.vy_max = 3.;
  f.range.vz_min = -3.; f.range.vz_max = 3.;
  f.range.z_min =  0.;  f.range.z_max = 2.*math::pi<double>()/K;
  f.compute_steps();
  const double v_par  = 0.2;
  const double v_perp = 0.6;
  const double nh = 0.2;

  double dt = 0.05;

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

  auto M1 = maxwellian(nh,{0.,0.,0.},{v_par,v_perp,v_perp});
  for (std::size_t k_x=0u ; k_x<f.size(0) ; ++k_x ) {
    for (std::size_t k_y=0u ; k_y<f.size(1) ; ++k_y ) {
      for (std::size_t k_z=0u ; k_z<f.size(2) ; ++k_z ) {
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          f[k_x][k_y][k_z][i] = M1( Zi(i),Vkx(k_x),Vky(k_y),Vkz(k_z) );
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }


  const double B0 = 1.;
  ublas::vector<double> Ex(c.Nz,0.),Ey(c.Nz,0.);
  ublas::vector<double> Bx(c.Nz,0.),By(c.Nz,0.);
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*i;
    Bx[i] = 1e-4 * std::sin(K*z);
  }


  ublas::vector<double> jcx(c.Nz,0.),jcy(c.Nz,0.);


  std::vector<double> times;           times.reserve(100);
  std::vector<double> electric_energy; electric_energy.reserve(100);
  std::vector<double> kinetic_energy;  kinetic_energy.reserve(100);
  std::vector<double> magnetic_energy; magnetic_energy.reserve(100);
  std::vector<double> cold_energy;     cold_energy.reserve(100);

  hybird1dx3dv<double> Lie( f , f.range.len_z() , B0 );
  double current_t = 0.;
  times.push_back(0.);

  auto compute_electric_energy = [&]( const ublas::vector<double> & Ex , const ublas::vector<double> & Ey ) {
    double electric_energy = 0.;
    for ( auto i=0u ; i<f.size_x() ; ++i ) {
      electric_energy += 0.5*( Ex[i]*Ex[i] + Ey[i]+Ey[i] )*f.step.dz;
    }
    return electric_energy;
  };
  auto compute_magnetic_energy = [&]( const ublas::vector<double> & Bx , const ublas::vector<double> & By ) {
    double magnetic_energy = 0.;
    for ( auto i=0u ; i<f.size_x() ; ++i ) {
      magnetic_energy += 0.5*( Bx[i]*Bx[i] + By[i]+By[i] )*f.step.dz;
    }
    return magnetic_energy;
  };
  auto compute_cold_energy = [&]( const ublas::vector<double> & jx , const ublas::vector<double> & jy ) {
    double cold_energy = 0.;
    for ( auto i=0u ; i<f.size_x() ; ++i ) {
      cold_energy += 0.5*( jx[i]*jx[i] + jy[i]+jy[i] )*f.step.dz;
    }
    return cold_energy;
  };
  /*
  auto compute_total_energy = [&](
            const ublas::vector<double> & jx , const ublas::vector<double> & jy ,
            const ublas::vector<double> & Ex , const ublas::vector<double> & Ey ,
            const ublas::vector<double> & Bx , const ublas::vector<double> & By ,
            const complex_field<double,3> & hf
          ) {
    double H = 0.;
    for ( auto i=0u ; i<f.size_x() ; ++i ) {
      H += 0.5*( Ex[i]*Ex[i] + Ey[i]+Ey[i] )*f.step.dz;
      H += 0.5*( Bx[i]*Bx[i] + By[i]+By[i] )*f.step.dz;
      H += 0.5*( jx[i]*jx[i] + jy[i]+jy[i] )*f.step.dz;
    }

    ublas::vector<double> fvxvyvz(c.Nz,0.);
    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double vx = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double vy = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            H += 0.5*( vx*vx + vy*vy + vz*vz )*fvxvyvz[i]*f.step.dz*f.volumeV();
          }
        }
      }
    }
    return H;
  };
  */
  auto compute_kinetic_energy = [&](const complex_field<double,3> & hf) {
    double kinetic_energy = 0.;
    ublas::vector<double> fvxvyvz(c.Nz,0.);
    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double vx = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double vy = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            kinetic_energy += 0.5*( vx*vx + vy*vy + vz*vz )*fvxvyvz[i]*f.step.dz*f.volumeV();
          }
        }
      }
    }
    return kinetic_energy;
  };
  double eex=0.,eey=0.;
  //eex = compute_electric_energy(Ex); eey = compute_electric_energy(Ey);
  //electric_energy_x.push_back(std::sqrt(eex));
  //electric_energy_y.push_back(std::sqrt(eey));
  electric_energy.push_back(compute_electric_energy(Ex,Ey));
  magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
  cold_energy.push_back(compute_cold_energy(jcx,jcy));
  kinetic_energy.push_back(compute_kinetic_energy(hf));
  //total_energy.push_back( compute_total_energy(jcx,jcy,Ex,Ey,Bx,By,hf) );

  std::cout << "kinetic energy : " << compute_kinetic_energy(hf) << std::endl; 

  while ( current_t<c.Tf ) {
    std::cout << "\r" << current_t << " / " << c.Tf << std::flush; 


    Lie.H_E(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_B(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_jc(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_f(dt,jcx,jcy,Ex,Ey,Bx,By,hf);

    electric_energy.push_back(compute_electric_energy(Ex,Ey));
    magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
    cold_energy.push_back(compute_cold_energy(jcx,jcy));
    kinetic_energy.push_back(compute_kinetic_energy(hf));
    current_t += dt;
    times.push_back(current_t);
  }

  auto writer_t_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };

  std::string name = "";//_tilde";
  c << monitoring::make_data( "ee"s+name+".dat"s  , electric_energy   , writer_t_y );
  c << monitoring::make_data( "eb"s+name+".dat"s  , magnetic_energy   , writer_t_y );
  c << monitoring::make_data( "ec"s+name+".dat"s  , cold_energy       , writer_t_y );
  c << monitoring::make_data( "ek"s+name+".dat"s  , kinetic_energy    , writer_t_y );
  //c << monitoring::data( "eex"s+name+".dat"s , electric_energy_x , writer_t_y );
  //c << monitoring::data( "eey"s+name+".dat"s , electric_energy_y , writer_t_y );
  //c << monitoring::data( "H"s+name+".dat"s   , total_energy      , writer_t_y );

  return 0;
}

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
#include <tuple>
#include <functional>
#include <utility>

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
  c.name = "vmhls";

  c.create_output_directory();

/* ------------------------------------------------------------------------- */
  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  complex_field<double,3> hf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  const double K = c.K;
  f.range.vx_min = -3.6; f.range.vx_max = 3.6;
  f.range.vy_min = -3.6; f.range.vy_max = 3.6;
  f.range.vz_min = -1.2; f.range.vz_max = 1.2;
  f.range.z_min =  0.;  f.range.z_max = 2.*math::pi<double>()/K;
  f.compute_steps();
  const double v_par  = c.v_par;
  const double v_perp = c.v_perp;
  const double nh = c.nh;

  double dt = std::min(c.dt0,f.step.dz);
  {
    std::ofstream ofconfig( c.output_dir / ("config_"s + c.name + ".init"s) );
    ofconfig << c << "\n";
    ofconfig.close();
  }

  // ublas::vector<double> vx(c.Nv,0.),vy(c.Nv,0.),vz(c.Nv,0.);
  // std::generate( vx.begin() , vx.end() , [&,k=0]() mutable {return (k++)*f.step.dvx+f.range.vx_min;} );
  // std::generate( vy.begin() , vy.end() , [&,k=0]() mutable {return (k++)*f.step.dvy+f.range.vy_min;} );
  // std::generate( vz.begin() , vz.end() , [&,k=0]() mutable {return (k++)*f.step.dvz+f.range.vz_min;} );

  ublas::vector<double> kx(c.Nx); // beware, Nx need to be odd
  {
    double l = f.range.len_z();
    for ( auto i=0u ; i<c.Nx/2 ; ++i ) { kx[i]      = 2.*math::pi<double>()*i/l; }
    for ( int i=-c.Nx/2 ; i<0 ; ++i ) { kx[c.Nx+i] = 2.*math::pi<double>()*i/l; }
  }

  // projection in some plan to see anisotropy in v
  field<double,1> fvxz(boost::extents[c.Nvy][c.Nvz]);
  field<double,1> fvyz(boost::extents[c.Nvx][c.Nvz]);
  ublas::vector<double> int_f_init(c.Nvz,0.), int_f_end(c.Nvz,0.);
  fvxz.range.v_min = f.range.vy_min; fvxz.range.v_max = f.range.vy_max;
  fvxz.range.x_min = f.range.vz_min; fvxz.range.x_max = f.range.vz_max;
  fvxz.compute_steps();

  fvyz.range.v_min = f.range.vx_min; fvyz.range.v_max = f.range.vx_max;
  fvyz.range.x_min = f.range.vz_min; fvyz.range.x_max = f.range.vz_max;
  fvyz.compute_steps();

  auto M1 = maxwellian(nh,{0.,0.,0.},{v_perp,v_perp,v_par});
  for (std::size_t k_x=0u ; k_x<f.size(0) ; ++k_x ) {
    for (std::size_t k_y=0u ; k_y<f.size(1) ; ++k_y ) {
      for (std::size_t k_z=0u ; k_z<f.size(2) ; ++k_z ) {
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          f[k_x][k_y][k_z][i] = M1( Zi(i),Vkx(k_x),Vky(k_y),Vkz(k_z) );

          fvxz[k_y][k_z] += f[k_x][k_y][k_z][i]*f.step.dvx*f.step.dz;
          fvyz[k_x][k_z] += f[k_x][k_y][k_z][i]*f.step.dvy*f.step.dz;
          int_f_init[k_z] += f[k_x][k_y][k_z][i]*f.step.dvx*f.step.dvy*f.step.dz;
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }
  fvxz.write(c.output_dir/"fvxz_init.dat");
  fvyz.write(c.output_dir/"fvyz_init.dat");

  auto writer_z_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<f.step.dz*(count++)+f.range.z_min<<" "<<y;
    return ss.str();
  };
  c << monitoring::make_data( "int_f_init.dat" , int_f_init , writer_z_y );

  const double B0 = 1.;
  ublas::vector<double> Ex(c.Nz,0.),Ey(c.Nz,0.);
  ublas::vector<double> Bx(c.Nz,0.),By(c.Nz,0.);
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*i;
    Bx[i] = c.alpha * std::sin(K*z);
  }


  ublas::vector<double> jcx(c.Nz,0.),jcy(c.Nz,0.);


  std::vector<double> times;           times.reserve(100);
  std::vector<double> electric_energy; electric_energy.reserve(100);
  std::vector<double> kinetic_energy;  kinetic_energy.reserve(100);
  std::vector<double> magnetic_energy; magnetic_energy.reserve(100);
  std::vector<double> cold_energy;     cold_energy.reserve(100);
  std::vector<double> mass;            mass.reserve(100);
  std::vector<double> Exmax;           Exmax.reserve(100);
  std::vector<double> Eymax;           Eymax.reserve(100);
  std::vector<double> Bxmax;           Bxmax.reserve(100);
  std::vector<double> Bymax;           Bymax.reserve(100);

  hybird1dx3dv<double> Lie( f , f.range.len_z() , B0 );
  double current_t = 0.;
  times.push_back(0.);

  auto __compute_energy = [&]( const ublas::vector<double> & ux , const ublas::vector<double> & uy , double dz ) {
    return std::inner_product( ux.begin() , ux.end() , uy.begin() , 0.0 ,
                 std::plus<double>() ,
                 [&]( const auto & x , const auto & y ){ return 0.5*(x*x + y*y)*dz ; } );
  };
  auto compute_electric_energy = [&]( const ublas::vector<double> & Ex , const ublas::vector<double> & Ey ) {
    return __compute_energy(Ex,Ey,f.step.dz);
  };
  auto compute_magnetic_energy = [&]( const ublas::vector<double> & Bx , const ublas::vector<double> & By ) {
    return __compute_energy(Bx,By,f.step.dz);
  };
  auto compute_cold_energy = [&]( const ublas::vector<double> & jx , const ublas::vector<double> & jy ) {
    return __compute_energy(jx,jy,f.step.dz);
  };

  auto compute_mass_kinetic_energy = [&](const complex_field<double,3> & hf) {
    double kinetic_energy = 0.;
    double mass = 0.;
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
            mass += fvxvyvz[i]*f.step.dz*f.volumeV();
          }
        }
      }
    }
    return std::make_pair(mass,kinetic_energy);
  };

  double m, ek;

  electric_energy.push_back(compute_electric_energy(Ex,Ey));
  magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
  cold_energy.push_back(compute_cold_energy(jcx,jcy));
  std::tie(m,ek) = compute_mass_kinetic_energy(hf);
  kinetic_energy.push_back(ek);
  mass.push_back(m);
  Exmax.push_back(*std::max_element(Ex.begin(),Ex.end()));
  Eymax.push_back(*std::max_element(Ey.begin(),Ey.end()));
  Bxmax.push_back(*std::max_element(Bx.begin(),Bx.end()));
  Bymax.push_back(*std::max_element(By.begin(),By.end()));
  
  monitoring::reactive_monitoring<std::vector<double>> moni( c.output_dir/("energy_"s + c.name + ".dat"s) , times , {&electric_energy,&magnetic_energy,&cold_energy,&kinetic_energy,&mass,&Exmax,&Eymax,&Bxmax,&Bymax} );

  //total_energy.push_back( compute_total_energy(jcx,jcy,Ex,Ey,Bx,By,hf) );

  while ( current_t<c.Tf ) {
    std::cout << "\r" << current_t << " / " << c.Tf << std::flush;


    Lie.H_E(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_B(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_jc(dt,jcx,jcy,Ex,Ey,Bx,By,hf);
    Lie.H_f(dt,jcx,jcy,Ex,Ey,Bx,By,hf);

    electric_energy.push_back(compute_electric_energy(Ex,Ey));
    magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
    cold_energy.push_back(compute_cold_energy(jcx,jcy));
    std::tie(m,ek) = compute_mass_kinetic_energy(hf);
    kinetic_energy.push_back(ek);
    mass.push_back(m);
    Exmax.push_back(*std::max_element(Ex.begin(),Ex.end()));
    Eymax.push_back(*std::max_element(Ey.begin(),Ey.end()));
    Bxmax.push_back(*std::max_element(Bx.begin(),Bx.end()));
    Bymax.push_back(*std::max_element(By.begin(),By.end()));

    current_t += dt;
    times.push_back(current_t);
    moni.push();
  }

  auto writer_t_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };


  auto pfvxz = fvxz.origin() , pfvyz = fvyz.origin();
  for ( auto i=0u ; i<fvxz.num_elements() ; ++i ) {
    pfvxz[i] = 0.;
    pfvyz[i] = 0.;
  }

  for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
    for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
      for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
        fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          fvxz[k_y][k_z] += f[k_x][k_y][k_z][i]*f.step.dvx*f.step.dz;
          fvyz[k_x][k_z] += f[k_x][k_y][k_z][i]*f.step.dvy*f.step.dz;
          int_f_end[k_z] += f[k_x][k_y][k_z][i]*f.step.dvx*f.step.dvy*f.step.dz;
        }
      }
    }
  }


  std::string name = "_tilde";
  fvxz.write(c.output_dir/("fvxz_end_"s + c.name + ".dat"s));
  fvyz.write(c.output_dir/("fvyz_end_"s + c.name + ".dat"s));
  c << monitoring::make_data( "int_f_end"s + c.name + ".dat" , int_f_end , writer_z_y );

  c << monitoring::make_data( "ee"s + c.name + ".dat"s , electric_energy , writer_t_y );
  c << monitoring::make_data( "eb"s + c.name + ".dat"s , magnetic_energy , writer_t_y );
  c << monitoring::make_data( "ec"s + c.name + ".dat"s , cold_energy     , writer_t_y );
  c << monitoring::make_data( "ek"s + c.name + ".dat"s , kinetic_energy  , writer_t_y );
  c << monitoring::make_data( "m"s + c.name + ".dat"s  , mass            , writer_t_y );

  return 0;
}

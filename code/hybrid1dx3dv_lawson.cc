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
  c.name = "vmhll";

  c.create_output_directory();
  {
    std::ofstream ofconfig( c.output_dir / ("config_"s + c.name + ".init"s) );
    ofconfig << c << "\n";
    ofconfig.close();
  }

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

  double dt = c.dt0;

  ublas::vector<double> vx(c.Nv,0.),vy(c.Nv,0.),vz(c.Nv,0.);
  std::generate( vx.begin() , vx.end() , [&,k=0]() mutable {return (k++)*f.step.dvx+f.range.vx_min;} );
  std::generate( vy.begin() , vy.end() , [&,k=0]() mutable {return (k++)*f.step.dvy+f.range.vy_min;} );
  std::generate( vz.begin() , vz.end() , [&,k=0]() mutable {return (k++)*f.step.dvz+f.range.vz_min;} );

  ublas::vector<double> Kz(c.Nz); // beware, Nz need to be odd
  {
    double l = f.range.len_z();
    for ( auto i=0u ; i<c.Nz/2 ; ++i ) { Kz[i]      = 2.*math::pi<double>()*i/l; }
    for ( int i=-c.Nz/2 ; i<0 ; ++i ) { Kz[c.Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  // projection in some plan to see anisotropy in v
  field<double,1> fvxz(boost::extents[c.Nvy][c.Nvz]);
  field<double,1> fvyz(boost::extents[c.Nvx][c.Nvz]);
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
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }
  fvxz.write(c.output_dir/"fvxz_init.dat");
  fvyz.write(c.output_dir/"fvyz_init.dat");

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
  std::vector<double> mass;            mass.reserve(100);

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
  monitoring::reactive_monitoring<std::vector<double>> moni( c.output_dir/("energy_"s + c.name + ".dat"s) , times , {&electric_energy,&magnetic_energy,&cold_energy,&kinetic_energy} );

  ublas::vector<std::complex<double>> hjcx(c.Nz,0.), hjcx1(c.Nz,0.), hjcx2(c.Nz,0.),
                                      hjcy(c.Nz,0.), hjcy1(c.Nz,0.), hjcy2(c.Nz,0.),
                                      hEx(c.Nz,0.), hEx1(c.Nz,0.), hEx2(c.Nz,0.),
                                      hEy(c.Nz,0.), hEy1(c.Nz,0.), hEy2(c.Nz,0.),
                                      hBx(c.Nz,0.), hBx1(c.Nz,0.), hBx2(c.Nz,0.),
                                      hBy(c.Nz,0.), hBy1(c.Nz,0.), hBy2(c.Nz,0.);
  complex_field<double,3> hf1(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]), hf2(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  field3d<double> dvf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  fft::fft(jcx.begin(),jcx.end(),hjcx.begin());
  fft::fft(jcy.begin(),jcy.end(),hjcy.begin());
  fft::fft(Ex.begin(),Ex.end(),hEx.begin());
  fft::fft(Ey.begin(),Ey.end(),hEy.begin());
  fft::fft(Bx.begin(),Bx.end(),hBx.begin());
  fft::fft(By.begin(),By.end(),hBy.begin());

  while ( current_t<c.Tf ) {
    std::cout << "\r" << current_t << " / " << c.Tf << std::flush;

    /* Lawson(RK(3,3)) */
    // FIRST STAGE //////////////////////////////////////////////////
    {
      // compute $\int v_x \hat{f}\,\mathrm{d}v$ et $\int v_y \hat{f}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += v_x*hf[k_x][k_y][k_z][i]*f.volumeV();
              hjhy[i] += v_y*hf[k_x][k_y][k_z][i]*f.volumeV();
            }
          }
        }
      }

      // compute hjcx1,hjcy1,hBx1,hBy1,hEx1,hEy1 (all spatial values)
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx1[i] = -0.970142500145332*dt*(-1.*hjhx[i] + I*Kz[i]*hBy[i])*(std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.970142500145332*dt*(-1.*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)))*(I*Kz[i]*hBx[i] + hjhy[i]) + 0.970142500145332*hEx[i]*(std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.970142500145332*hEy[i]*(-1.*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx[i]*(0.378732187481834*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.621267812518167*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.378732187481834*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.621267812518167*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hjcy1[i] = -1.0*dt*(-1.*hjhx[i] + I*Kz[i]*hBy[i])*(0.970142500145332*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.970142500145332*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*dt*(I*Kz[i]*hBx[i] + hjhy[i])*(0.970142500145332*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.970142500145332*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.970142500145332*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.970142500145332*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.378732187481834*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.621267812518167*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.378732187481834*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.621267812518167*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hBx1[i] = 1.0*I*Kz[i]*dt*hEy[i] + 1.0*hBx[i];
        // ---
        hBy1[i] = -1.0*I*Kz[i]*dt*hEx[i] + 1.0*hBy[i];
        // ---
        hEx1[i] = -1.0*dt*(-1.*hjhx[i] + I*Kz[i]*hBy[i])*(0.621267812518167*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*dt*(I*Kz[i]*hBx[i] + hjhy[i])*(0.621267812518166*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481833*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.621267812518167*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.621267812518166*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481833*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.242535625036333*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.242535625036333*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.242535625036333*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hEy1[i] = 1.0*dt*(-1.*hjhx[i] + I*Kz[i]*hBy[i])*(0.621267812518167*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*dt*(I*Kz[i]*hBx[i] + hjhy[i])*(0.621267812518166*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481833*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEx[i]*(0.621267812518167*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.621267812518166*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481833*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.242535625036333*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.242535625036333*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy[i]*(0.242535625036333*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
      }

      // compute hf1

      // compute iFFT(hf)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
          }
        }
      }
      fft::ifft(hEx.begin(),hEx.end(),Ex.begin());
      fft::ifft(hEy.begin(),hEy.end(),Ey.begin());
      fft::ifft(hBx.begin(),hBx.end(),Bx.begin());
      fft::ifft(hBy.begin(),hBy.end(),By.begin());
      // compute approximation of (E×vB)∂ᵥf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              double velocity_vx = Ex[i] + v_y*B0 - v_z*By[i];
              double velocity_vy = Ey[i] - v_x*B0 + v_z*Bx[i];
              double velocity_vz = v_x*By[i] - v_y*Bx[i];
              dvf[k_x][k_y][k_z][i] = weno3d::weno_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf1
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( f[k_x][k_y][k_z].begin() , f[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf1[k_x][k_y][k_z][i] = 1.0*(dt*hfvxvyvz[i] + hf[k_x][k_y][k_z][i])*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }

    } // end first stage

    // SECOND STAGE /////////////////////////////////////////////////
    {
      // compute $\int v_x \hat{f}^{(1)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(1)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += v_x*hf1[k_x][k_y][k_z][i]*f.volumeV();
              hjhy[i] += v_y*hf1[k_x][k_y][k_z][i]*f.volumeV();
            }
          }
        }
      }

      // compute hjcx2,hjcy2,hBx2,hBy2,hEx2,hEy2 (all spatial values)
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx2[i] = 0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.970142500145332*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.242535625036333*dt*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)))*(I*Kz[i]*hBx1[i] + hjhy[i]) - 1.0*hEx1[i]*(0.242535625036333*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.727606875108999*hEx[i]*(std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.242535625036333*hEy1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 0.727606875108999*hEy[i]*(-1.*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx1[i]*(0.0946830468704584*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.155316953129542*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx[i]*(0.284049140611375*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.465950859388625*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy1[i]*(0.0946830468704584*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.155316953129542*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.284049140611375*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.465950859388625*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hjcy2[i] = -0.242535625036333*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.970142500145332*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.242535625036333*hEx1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.727606875108999*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.727606875108999*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEy1[i]*(0.242535625036333*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.727606875108999*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.727606875108999*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx1[i]*(0.0946830468704584*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.155316953129542*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.284049140611375*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.465950859388625*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy1[i]*(0.0946830468704584*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.155316953129542*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.284049140611375*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.465950859388625*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hBx2[i] = 0.25*I*Kz[i]*dt*hEy1[i] + 0.25*hBx1[i] + 0.75*hBx[i];
        // ---

        hBy2[i] = -0.25*I*Kz[i]*dt*hEx1[i] + 0.25*hBy1[i] + 0.75*hBy[i];
        // ---
        hEx2[i] = -0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.621267812518166*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.621267812518167*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx1[i]*(0.155316953129542*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0946830468704584*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.465950859388625*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.284049140611375*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEy1[i]*(0.155316953129542*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0946830468704584*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.465950859388625*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.284049140611375*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcx1[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.18190171877725*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.18190171877725*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcy1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.18190171877725*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.18190171877725*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hEy2[i] = -0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.621267812518166*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.621267812518167*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx1[i]*(0.155316953129542*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0946830468704584*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEx[i]*(0.465950859388625*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.284049140611375*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy1[i]*(0.155316953129542*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0946830468704584*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.465950859388625*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.284049140611375*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.0606339062590832*hjcx1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.18190171877725*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.18190171877725*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcy1[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy[i]*(0.18190171877725*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.18190171877725*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
      }

      // compute hf2

      // compute iFFT(hf1)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( hf1[k_x][k_y][k_z].begin() , hf1[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
          }
        }
      }
      fft::ifft(hEx1.begin(),hEx1.end(),Ex.begin());
      fft::ifft(hEy1.begin(),hEy1.end(),Ey.begin());
      fft::ifft(hBx1.begin(),hBx1.end(),Bx.begin());
      fft::ifft(hBy1.begin(),hBy1.end(),By.begin());
      // compute approximation of (E×vB)∂ᵥf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              double velocity_vx = Ex[i] + v_y*B0 - v_z*By[i];
              double velocity_vy = Ey[i] - v_x*B0 + v_z*Bx[i];
              double velocity_vz = v_x*By[i] - v_y*Bx[i];
              dvf[k_x][k_y][k_z][i] = weno3d::weno_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf2
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( f[k_x][k_y][k_z].begin() , f[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf2[k_x][k_y][k_z][i] = (0.75*hf[k_x][k_y][k_z][i] + 0.25*(dt*hfvxvyvz[i] + hf1[k_x][k_y][k_z][i])*std::exp(1.5*I*Kz[i]*dt*v_z))*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }

    } // end second stage

    // THRID STAGE //////////////////////////////////////////////////
    {
      // compute $\int v_x \hat{f}^{(2)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(2)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += v_x*hf2[k_x][k_y][k_z][i]*f.volumeV();
              hjhy[i] += v_y*hf2[k_x][k_y][k_z][i]*f.volumeV();
            }
          }
        }
      }

      // update hjcx,hjcy,hBx,hBy,hEx,hEy (all spatial values)
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        auto hjcx_tmp = -0.646761666763555*dt*(-1.*hjhx[i] + I*Kz[i]*hBy2[i])*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.646761666763555*dt*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)))*(I*Kz[i]*hBx2[i] + hjhy[i]) + 0.646761666763555*hEx2[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.323380833381777*hEx[i]*(std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.646761666763555*hEy2[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 0.323380833381777*hEy[i]*(-1.*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx2[i]*(0.252488124987889*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.414178541678778*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx[i]*(0.126244062493945*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.207089270839389*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy2[i]*(0.252488124987889*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.414178541678778*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.126244062493945*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.207089270839389*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        auto hjcy_tmp = -0.666666666666667*dt*(-1.*hjhx[i] + I*Kz[i]*hBy2[i])*(0.970142500145332*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.970142500145332*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.666666666666667*dt*(I*Kz[i]*hBx2[i] + hjhy[i])*(0.970142500145332*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx2[i]*(0.646761666763555*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.646761666763555*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.323380833381777*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.323380833381777*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy2[i]*(0.646761666763555*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.646761666763555*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.323380833381777*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.323380833381777*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx2[i]*(0.252488124987889*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.414178541678778*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.126244062493945*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.207089270839389*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy2[i]*(0.252488124987889*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.414178541678778*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.126244062493945*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.207089270839389*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        auto hBx_tmp = 0.666666666666667*I*Kz[i]*dt*hEy2[i] + 0.666666666666667*hBx2[i] + 0.333333333333333*hBx[i];
        // ---
        auto hBy_tmp = -0.666666666666667*I*Kz[i]*dt*hEx2[i] + 0.666666666666667*hBy2[i] + 0.333333333333333*hBy[i];
        // ---
        auto hEx_tmp = -0.666666666666667*dt*(-1.*hjhx[i] + I*Kz[i]*hBy2[i])*(0.621267812518167*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.666666666666667*dt*(I*Kz[i]*hBx2[i] + hjhy[i])*(0.621267812518166*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481833*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx2[i]*(0.414178541678778*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.252488124987889*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.207089270839389*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.126244062493944*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy2[i]*(0.414178541678778*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.252488124987889*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.207089270839389*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.126244062493944*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx2[i]*(0.161690416690889*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.161690416690889*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.0808452083454443*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0808452083454443*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy2[i]*(0.161690416690889*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.161690416690889*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.0808452083454443*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0808452083454443*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---

        auto hEy_tmp = 0.666666666666667*dt*(-1.*hjhx[i] + I*Kz[i]*hBy2[i])*(0.621267812518167*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.666666666666667*dt*(I*Kz[i]*hBx2[i] + hjhy[i])*(0.621267812518166*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481833*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEx2[i]*(0.414178541678778*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.252488124987889*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEx[i]*(0.207089270839389*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.126244062493944*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy2[i]*(0.414178541678778*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.252488124987889*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.207089270839389*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.126244062493944*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx2[i]*(0.161690416690889*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.161690416690889*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.0808452083454443*std::cos(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0808452083454443*std::cos(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy2[i]*(0.161690416690889*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.161690416690889*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy[i]*(0.0808452083454443*std::sin(1.5*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0808452083454443*std::sin(1.5*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));

          hjcx[i] = hjcx_tmp;
          hjcy[i] = hjcy_tmp;
          hBx[i] = hBx_tmp;
          hBy[i] = hBy_tmp;
          hEx[i] = hEx_tmp;
          hEy[i] = hEy_tmp;
      }

      // update hf

      // compute iFFT(hf2)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( hf2[k_x][k_y][k_z].begin() , hf2[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
          }
        }
      }
      fft::ifft(hEx2.begin(),hEx2.end(),Ex.begin());
      fft::ifft(hEy2.begin(),hEy2.end(),Ey.begin());
      fft::ifft(hBx2.begin(),hBx2.end(),Bx.begin());
      fft::ifft(hBy2.begin(),hBy2.end(),By.begin());
      // compute approximation of (E×vB)∂ᵥf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              double velocity_vx = Ex[i] + v_y*B0 - v_z*By[i];
              double velocity_vy = Ey[i] - v_x*B0 + v_z*Bx[i];
              double velocity_vz = v_x*By[i] - v_y*Bx[i];
              dvf[k_x][k_y][k_z][i] = weno3d::weno_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                    + weno3d::weno_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        double v_x = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          double v_y = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( f[k_x][k_y][k_z].begin() , f[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf[k_x][k_y][k_z][i] = (0.333333333333333*hf[k_x][k_y][k_z][i]*std::exp(0.5*I*Kz[i]*dt*v_z) + 0.666666666666667*(dt*hfvxvyvz[i] + hf2[k_x][k_y][k_z][i])*std::exp(1.0*I*Kz[i]*dt*v_z))*std::exp(-1.5*I*Kz[i]*dt*v_z);
            }
          }
        }
      }
    } // end thrid stage


    fft::ifft(hEx.begin(),hEx.end(),Ex.begin());
    fft::ifft(hEy.begin(),hEy.end(),Ey.begin());
    fft::ifft(hBx.begin(),hBx.end(),Bx.begin());
    fft::ifft(hBy.begin(),hBy.end(),By.begin());
    fft::ifft(hjcx.begin(),hjcx.end(),jcx.begin());
    fft::ifft(hjcy.begin(),hjcy.end(),jcy.begin());

    electric_energy.push_back(compute_electric_energy(Ex,Ey));
    magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
    cold_energy.push_back(compute_cold_energy(jcx,jcy));
    std::tie(m,ek) = compute_mass_kinetic_energy(hf);
    kinetic_energy.push_back(ek);
    mass.push_back(m);

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
        }
      }
    }
  }

  fvxz.write(c.output_dir/("fvxz_end_"s + c.name + ".dat"s));
  fvyz.write(c.output_dir/("fvyz_end_"s + c.name + ".dat"s));

  c << monitoring::make_data( "ee"s + c.name + ".dat"s , electric_energy , writer_t_y );
  c << monitoring::make_data( "eb"s + c.name + ".dat"s , magnetic_energy , writer_t_y );
  c << monitoring::make_data( "ec"s + c.name + ".dat"s , cold_energy     , writer_t_y );
  c << monitoring::make_data( "ek"s + c.name + ".dat"s , kinetic_energy  , writer_t_y );
  c << monitoring::make_data( "m"s + c.name + ".dat"s  , mass            , writer_t_y );

  return 0;
}


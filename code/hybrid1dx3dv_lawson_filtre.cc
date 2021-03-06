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
#include <iomanip>

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
  c.create_output_directory();

  c.name = "vmhllf";

  std::string escape;
  if ( argc > 2 ) { std::size_t line = std::stoul(argv[2]); std::stringstream sescape; sescape << "\033[" << line << ";0H"; escape = sescape.str(); }
  else { escape = "\r"; }

/* ------------------------------------------------------------------------- */
  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  complex_field<double,3> hf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  const double K = c.K;
  f.range.vx_min = -3.6; f.range.vx_max = 3.6;
  f.range.vy_min = -3.6; f.range.vy_max = 3.6;
  f.range.vz_min = -2.0; f.range.vz_max = 2.0;
  f.range.z_min =  0.;  f.range.z_max = 2.*math::pi<double>()/K;
  f.compute_steps();
  const double v_par  = c.v_par;
  const double v_perp = c.v_perp;
  const double nh = c.nh;

  double dt = c.dt0; /*std::min({
        c.dt0,
        f.step.dz,
        f.step.dvx/f.range.vy_max,
        f.step.dvy/f.range.vx_max,
        f.step.dvz
      });*/
  c.dt0 = dt;
  {
    std::ofstream ofconfig( c.output_dir / ("config_"s + c.name + ".init"s) );
    ofconfig << c << "\n";
    ofconfig.close();
  }

  // ublas::vector<double> vx(c.Nv,0.),vy(c.Nv,0.),vz(c.Nv,0.);
  // std::generate( vx.begin() , vx.end() , [&,k=0]() mutable {return (k++)*f.step.dvx+f.range.vx_min;} );
  // std::generate( vy.begin() , vy.end() , [&,k=0]() mutable {return (k++)*f.step.dvy+f.range.vy_min;} );
  // std::generate( vz.begin() , vz.end() , [&,k=0]() mutable {return (k++)*f.step.dvz+f.range.vz_min;} );

  ublas::vector<double> Kz(c.Nz); // beware, Nz need to be odd
  {
    double l = f.range.len_z();
    for ( auto i=0u ; i<c.Nz/2 ; ++i ) { Kz[i]     = 2.*math::pi<double>()*i/l; }
    for ( int i=-c.Nz/2 ; i<0 ; ++i ) { Kz[c.Nz+i] = 2.*math::pi<double>()*i/l; }
  }

  auto M1 = maxwellian(nh,{0.,0.,0.},{v_perp,v_perp,v_par});
  for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
    const double vx = k_x*f.step.dvx + f.range.vx_min;
    for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
      const double vy = k_y*f.step.dvy + f.range.vy_min;
      for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
        const double vz = k_z*f.step.dvz + f.range.vz_min;
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          const double z = i*f.step.dz + f.range.z_min;
          f[k_x][k_y][k_z][i] = M1( z,vx,vy,vz );
          //f[k_x][k_y][k_z][i] = M1( z,vx,vy,vz )*( 1.0 + c.alpha*std::cos(K*z) );
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }

  const double B0 = c.B0;
  ublas::vector<double> jcx(c.Nz,0.) , jcy(c.Nz,0.);
  ublas::vector<double> Ex(c.Nz,0.)  , Ey(c.Nz,0.);
  ublas::vector<double> Bx(c.Nz,0.)  , By(c.Nz,0.);
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*i;
    Bx[i] = c.alpha * std::sin(K*z);
    //Bx[i] = 0.;
  }

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

  ublas::vector<double> fdvxdvydz(c.Nvz,0.);
  ublas::vector<double> vxfdv(c.Nz,0.), vyfdv(c.Nz,0.), vzfdv(c.Nz,0.);
  ublas::vector<double> ec_perp(c.Nz,0.), ec_vz(c.Nz,0.);
  ublas::vector<double> rho_h(c.Nz,0.);
  field<double,1> fdvxdvy(boost::extents[c.Nvz][c.Nz]); // 2d field, 1dz-1dvz

  auto compute_vperp_integral = [&]( const complex_field<double,3> & hf , double current_t ) {
    field<double,1> fdvxdvy(boost::extents[c.Nvz][c.Nz]); // 2d field, 1dz-1dvz
    fdvxdvy.range.v_min = f.range.vz_min; fdvxdvy.range.v_max = f.range.vz_max;
    fdvxdvy.range.x_min = f.range.z_min;  fdvxdvy.range.x_max = f.range.z_max;
    fdvxdvy.compute_steps();

    ublas::vector<double> fvxvyvz(c.Nz,0.);

    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double v_x = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double v_y = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            fdvxdvy[k_z][i] = fvxvyvz[i]*f.step.dvx*f.step.dvy;
          }
        }
      }
    }

    return fdvxdvy;
  };
  auto compute_integrals = [&]( const complex_field<double,3> & hf , double current_t ) {
    ublas::vector<double> fdvxdvydz(c.Nvz,0.);
    ublas::vector<double> vxfdv(c.Nz,0.);
    ublas::vector<double> vyfdv(c.Nz,0.);
    ublas::vector<double> vzfdv(c.Nz,0.);

    ublas::vector<double> fvxvyvz(c.Nz,0.);

    double c_ = std::cos(B0*current_t), s_ = std::sin(B0*current_t);

    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double w_1 = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double w_2 = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            fdvxdvydz[k_z] += fvxvyvz[i]*f.step.dz*f.step.dvx*f.step.dvy;
            vxfdv[i] += ( w_1*c_ - w_2*s_ )*fvxvyvz[i]*f.volumeV();
            vyfdv[i] += ( w_1*s_ + w_2*c_ )*fvxvyvz[i]*f.volumeV();
            vzfdv[i] += vz*fvxvyvz[i]*f.volumeV();
          }
        }
      }
    }

    return std::make_tuple(fdvxdvydz,vxfdv,vyfdv,vzfdv);
  };
  auto compute_local_kinetic_energy = [&]( const complex_field<double,3> & hf ) {
    ublas::vector<double> ec_perp(c.Nz,0.);
    ublas::vector<double> ec_vz(c.Nz,0.);

    ublas::vector<double> fvxvyvz(c.Nz,0.);

    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double w_1 = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double w_2 = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            ec_perp[i] += (w_1*w_1 + w_2*w_2)*fvxvyvz[i]*f.volumeV();
            ec_vz[i]   += (vz*vz)*fvxvyvz[i]*f.volumeV();
          }
        }
      }
    }

    return std::make_tuple( ec_perp , ec_vz );
  };
  auto compute_rho_h = [&]( const complex_field<double,3> & hf ) {
    ublas::vector<double> rho(c.Nz,0.);

    ublas::vector<double> fvxvyvz(c.Nz,0.);

    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double vx = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double vy = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            rho[i] += fvxvyvz[i]*f.volumeV();
          }
        }
      }
    }

    return rho;
  };

  auto printer__vz_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss.precision(15);
    ss<<(count++)*f.step.dvz + f.range.vz_min<<" "<<y;
    return ss.str();
  };
  auto printer__z_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss.precision(15);
    ss<<(count++)*f.step.dz + f.range.z_min<<" "<<y;
    return ss.str();
  };

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

  auto max_abs = []( const ublas::vector<double> u ) {
    return std::abs(*std::max_element(
        u.begin() , u.end() ,
        [](double a , double b){ return std::abs(a)<std::abs(b); }
      ));
  };
  
  double m, ek;

  electric_energy.push_back(compute_electric_energy(Ex,Ey));
  magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
  cold_energy.push_back(compute_cold_energy(jcx,jcy));
  std::tie(m,ek) = compute_mass_kinetic_energy(hf);
  kinetic_energy.push_back(ek);
  mass.push_back(m);
  Exmax.push_back( max_abs(Ex) );
  Eymax.push_back( max_abs(Ey) );
  Bxmax.push_back( max_abs(Bx) );
  Bymax.push_back( max_abs(By) );

  monitoring::reactive_monitoring<std::vector<double>> moni(
    c.output_dir/("energy_"s + c.name + ".dat"s) ,
    times ,
    {&electric_energy,&magnetic_energy,&cold_energy,&kinetic_energy,&mass,&Exmax,&Eymax,&Bxmax,&Bymax}
  );

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

  #define _hjfx(hf) ( w_1*c_ - w_2*s_ )*hf[k_x][k_y][k_z][i]*f.volumeV()
  #define _hjfy(hf) ( w_1*s_ + w_2*c_ )*hf[k_x][k_y][k_z][i]*f.volumeV()
  #define _velocity_vx(Ex,Ey,Bx,By) -( Ex[i]*c_ + Ey[i]*s_ + v_z*Bx[i]*s_ - v_z*By[i]*c_)
  #define _velocity_vy(Ex,Ey,Bx,By) -(-Ex[i]*s_ + Ey[i]*c_ + v_z*Bx[i]*c_ + v_z*By[i]*s_)
  #define _velocity_vz(Ex,Ey,Bx,By) -(-Bx[i]*( w_1*s_ + w_2*c_ ) + By[i]*( w_1*c_ - w_2*s_ ))

  std::size_t iteration_t = 0;
  while ( current_t<c.Tf ) {
    std::cout << escape << std::setw(8) << current_t << " / " << c.Tf << " [" << iteration_t << "]" << std::flush;

    /* Lawson(RK(3,3)) */
    // FIRST STAGE //////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*current_t), s_ = std::sin(B0*current_t);

      // compute $\int v_x \hat{f}\,\mathrm{d}v$ et $\int v_y \hat{f}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += _hjfx(hf);
              hjhy[i] += _hjfy(hf);
            }
          }
        }
      }
      // keep zero mean
      hjhx[0] = 0.0;
      hjhy[0] = 0.0;

      // compute hjcx1,hjcy1,hBx1,hBy1,hEx1,hEy1 (all spatial values)
      //#pragma omp parallel for
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
      // keep zero mean
      hjcx1[0] = 0.0;
      hjcy1[0] = 0.0;
      hBx1[0] = 0.0;
      hBy1[0] = 0.0;
      hEx1[0] = 0.0;
      hEy1[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx1[i] = 0.;
          hjcy1[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx1[i] = 0.;
          hBy1[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx1[i] = 0.;
          hEy1[i] = 0.;
        }
      #endif


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
      //#pragma omp parallel for collapse(4)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              const double w_1 = k_x*f.step.dvx + f.range.vx_min;
              const double w_2 = k_y*f.step.dvy + f.range.vy_min;
              const double v_z = k_z*f.step.dvz + f.range.vz_min;

              const double velocity_vx = _velocity_vx(Ex,Ey,Bx,By);
              const double velocity_vy = _velocity_vy(Ex,Ey,Bx,By);
              const double velocity_vz = _velocity_vz(Ex,Ey,Bx,By);

              dvf[k_x][k_y][k_z][i] = + weno3d::d_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf1
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      //#pragma omp parallel for collapse(3) firstprivate(hfvxvyvz)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf1[k_x][k_y][k_z][i] = 1.0*(dt*hfvxvyvz[i] + hf[k_x][k_y][k_z][i])*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }

    } // end first stage

    // SECOND STAGE /////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*(current_t+dt)), s_ = std::sin(B0*(current_t+dt));

      // compute $\int v_x \hat{f}^{(1)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(1)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += _hjfx(hf1);
              hjhy[i] += _hjfy(hf1);
            }
          }
        }
      }
      // keep zero mean
      hjhx[0] = 0.0;
      hjhy[0] = 0.0;

      // compute hjcx2,hjcy2,hBx2,hBy2,hEx2,hEy2 (all spatial values)
      //#pragma omp parallel for
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx2[i] = 0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.970142500145332*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.242535625036333*dt*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)))*(I*Kz[i]*hBx1[i] + hjhy[i]) - 1.0*hEx1[i]*(0.242535625036333*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.727606875108999*hEx[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.242535625036333*hEy1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 0.727606875108999*hEy[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx1[i]*(0.0946830468704584*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.155316953129542*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx[i]*(0.284049140611375*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.465950859388625*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy1[i]*(0.0946830468704584*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.155316953129542*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.284049140611375*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.465950859388625*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hjcy2[i] = -0.242535625036333*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.970142500145332*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.970142500145332*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.242535625036333*hEx1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.727606875108999*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.727606875108999*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEy1[i]*(0.242535625036333*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.242535625036333*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.727606875108999*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.727606875108999*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcx1[i]*(0.0946830468704584*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.155316953129542*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.284049140611375*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.465950859388625*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy1[i]*(0.0946830468704584*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.155316953129542*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.284049140611375*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.465950859388625*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hBx2[i] = 0.25*I*Kz[i]*dt*hEy1[i] + 0.25*hBx1[i] + 0.75*hBx[i];
        // ---
        hBy2[i] = -0.25*I*Kz[i]*dt*hEx1[i] + 0.25*hBy1[i] + 0.75*hBy[i];
        // ---

        hEx2[i] = -0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.621267812518166*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.621267812518167*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx1[i]*(0.155316953129542*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0946830468704584*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx[i]*(0.465950859388625*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.284049140611375*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEy1[i]*(0.155316953129542*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0946830468704584*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.465950859388625*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.284049140611375*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcx1[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.18190171877725*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.18190171877725*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcy1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hjcy[i]*(0.18190171877725*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.18190171877725*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
        // ---
        hEy2[i] = -0.25*dt*(-1.*hjhx[i] + I*Kz[i]*hBy1[i])*(0.621267812518166*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.378732187481834*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.25*dt*(I*Kz[i]*hBx1[i] + hjhy[i])*(0.621267812518167*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.378732187481834*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEx1[i]*(0.155316953129542*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.0946830468704584*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hEx[i]*(0.465950859388625*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.284049140611375*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy1[i]*(0.155316953129542*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.0946830468704584*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 1.0*hEy[i]*(0.465950859388625*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.284049140611375*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 0.0606339062590832*hjcx1[i]*(-1.*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)) + std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcx[i]*(0.18190171877725*std::cos(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) - 0.18190171877725*std::cos(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) + 0.0606339062590832*hjcy1[i]*(std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.))) - 1.0*hjcy[i]*(0.18190171877725*std::sin(0.75*dt*std::sqrt(-0.222222222222222*std::sqrt(17.) + 2.)) + 0.18190171877725*std::sin(0.75*dt*std::sqrt(0.222222222222222*std::sqrt(17.) + 2.)));
      }
      // keep zero mean
      hjcx2[0] = 0.0;
      hjcy2[0] = 0.0;
      hBx2[0] = 0.0;
      hBy2[0] = 0.0;
      hEx2[0] = 0.0;
      hEy2[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx2[i] = 0.;
          hjcy2[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx2[i] = 0.;
          hBy2[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx2[i] = 0.;
          hEy2[i] = 0.;
        }
      #endif

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
      //#pragma omp parallel for collapse(4)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              const double w_1 = k_x*f.step.dvx + f.range.vx_min;
              const double w_2 = k_y*f.step.dvy + f.range.vy_min;
              const double v_z = k_z*f.step.dvz + f.range.vz_min;
              
              const double velocity_vx = _velocity_vx(Ex,Ey,Bx,By);
              const double velocity_vy = _velocity_vy(Ex,Ey,Bx,By);
              const double velocity_vz = _velocity_vz(Ex,Ey,Bx,By);

              dvf[k_x][k_y][k_z][i] = + weno3d::d_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf2
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      //#pragma omp parallel for collapse(3) firstprivate(hfvxvyvz)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf2[k_x][k_y][k_z][i] = (0.75*hf[k_x][k_y][k_z][i] + 0.25*(dt*hfvxvyvz[i] + hf1[k_x][k_y][k_z][i])*std::exp(1.0*I*Kz[i]*dt*v_z))*std::exp(-0.5*I*Kz[i]*dt*v_z);
            }
          }
        }
      }

    } // end second stage

    // THRID STAGE //////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*(current_t+0.5*dt)), s_ = std::sin(B0*(current_t+0.5*dt));

      // compute $\int v_x \hat{f}^{(2)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(2)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hjhx[i] += _hjfx(hf2);
              hjhy[i] += _hjfy(hf2);
            }
          }
        }
      }
      // keep zero mean
      hjhx[0] = 0.0;
      hjhy[0] = 0.0;

      // update hjcx,hjcy,hBx,hBy,hEx,hEy (all spatial values)
      //#pragma omp parallel for
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
      // keep zero mean
      hjcx[0] = 0.0;
      hjcy[0] = 0.0;
      hBx[0] = 0.0;
      hBy[0] = 0.0;
      hEx[0] = 0.0;
      hEy[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx[i] = 0.;
          hjcy[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx[i] = 0.;
          hBy[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx[i] = 0.;
          hEy[i] = 0.;
        }
      #endif

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
      //#pragma omp parallel for collapse(4)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              const double w_1 = k_x*f.step.dvx + f.range.vx_min;
              const double w_2 = k_y*f.step.dvy + f.range.vy_min;
              const double v_z = k_z*f.step.dvz + f.range.vz_min;

              const double velocity_vx = _velocity_vx(Ex,Ey,Bx,By);
              const double velocity_vy = _velocity_vy(Ex,Ey,Bx,By);
              const double velocity_vz = _velocity_vz(Ex,Ey,Bx,By);

              dvf[k_x][k_y][k_z][i] = + weno3d::d_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf
      ublas::vector<std::complex<double>> hfvxvyvz(c.Nz,0.0);
      //#pragma omp parallel for collapse(3) firstprivate(hfvxvyvz)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
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
    Exmax.push_back( max_abs(Ex) );
    Eymax.push_back( max_abs(Ey) );
    Bxmax.push_back( max_abs(Bx) );
    Bymax.push_back( max_abs(By) );

    ++iteration_t;
    current_t += dt;
    times.push_back(current_t);
    moni.push();

    if ( iteration_t % 1000 == 0 )
    {
      std::tie(fdvxdvydz,vxfdv,vyfdv,vzfdv) = compute_integrals( hf , current_t );
      std::stringstream filename; filename << "fdvxdvydz_" << c.name << "_" << iteration_t << ".dat";
      c << monitoring::make_data( filename.str() , fdvxdvydz , printer__vz_y );

      fdvxdvy = compute_vperp_integral( hf , current_t );
      filename.str("");
      filename << "fdvxdvy_" << c.name << "_" << iteration_t << ".dat";
      fdvxdvy.write( c.output_dir / filename.str() );

      filename.str("");
      filename << "jhxyz_" << c.name << "_" << iteration_t << ".dat";
      auto printer__z_jh = [&,count=0] (auto const& y) mutable {
        std::stringstream ss; ss<<(count)*f.step.dz + f.range.z_min<<" "<<vxfdv[count]<<" "<<vyfdv[count]<<" "<<vzfdv[count];
        ++count;
        return ss.str();
      };
      c << monitoring::make_data( filename.str() , vxfdv , printer__z_jh );

      std::tie(ec_perp,ec_vz) = compute_local_kinetic_energy( hf );
      filename.str("");
      filename << "keh_"<< c.name << "_" << iteration_t << ".dat";
      auto printer__z_ec = [&,count=0] (auto const& y) mutable {
        std::stringstream ss; ss<<(count)*f.step.dz + f.range.z_min<<" "<<ec_perp[count]<<" "<<ec_vz[count];
        ++count;
        return ss.str();
      };
      c << monitoring::make_data( filename.str() , ec_perp , printer__z_ec );
      c << monitoring::make_data( filename.str() , ec_perp , printer__z_y );

      c << monitoring::make_data( filename.str() , ec_perp , printer__z_y );

      filename.str("");
      filename << "EBjxy_"<< c.name << "_" << iteration_t << ".dat";
      auto printer__z_EBxy = [&,count=0] (auto const& y) mutable {
        std::stringstream ss; ss<<(count)*f.step.dz + f.range.z_min<<" "<<Ex[count]<<" "<<Ey[count]<<" "<<Bx[count]<<" "<<By[count]<<" "<<jcx[count]<<" "<<jcy[count];
        ++count;
        return ss.str();
      };
      c << monitoring::make_data( filename.str() , Ex , printer__z_EBxy );
    }

  }

  auto writer_t_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };

  c << monitoring::make_data( "ee"s + c.name + ".dat"s , electric_energy , writer_t_y );
  c << monitoring::make_data( "eb"s + c.name + ".dat"s , magnetic_energy , writer_t_y );
  c << monitoring::make_data( "ec"s + c.name + ".dat"s , cold_energy     , writer_t_y );
  c << monitoring::make_data( "ek"s + c.name + ".dat"s , kinetic_energy  , writer_t_y );
  c << monitoring::make_data( "m"s + c.name + ".dat"s  , mass            , writer_t_y );

  return 0;
}


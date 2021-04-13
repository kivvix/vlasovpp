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

  c.name = "vmhllf_dp43";

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

  //double dt = c.dt0;
  /*std::min({
        c.dt0,
        f.step.dz,
        f.step.dvx/f.range.vy_max,
        f.step.dvy/f.range.vx_max,
        f.step.dvz
      });*/
  //c.dt0 = dt;
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
  field<double,1> fdvxdvy(boost::extents[c.Nvz][c.Nz]);

  auto compute_vperp_integral = [&]( const complex_field<double,3> & hf , double current_t ) {
    field<double,1> fdvxdvy(boost::extents[c.Nvz][c.Nz]); // 2d field, 1dz-1dvz
    fdvxdvy.range.v_min = f.range.vz_min; fdvxdvy.range.v_max = f.range.vz_max;
    fdvxdvy.range.x_min = f.range.z_min;  fdvxdvy.range.x_max = f.range.z_max;
    fdvxdvy.compute_steps();
 
    ublas::vector<double> fvxvyvz(c.Nz,0.);
 
    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
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
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
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
    std::stringstream ss; ss<<(count++)*f.step.dvz + f.range.vz_min<<" "<<y;
    return ss.str();
  };
  auto printer__z_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<(count++)*f.step.dz + f.range.z_min<<" "<<y;
    return ss.str();
  };

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

  std::vector<iteration_4d::iteration<double>> success_iter; success_iter.reserve(100);
  std::vector<iteration_4d::iteration<double>> iterations;   iterations.reserve(100);
  monitoring::reactive_monitoring< std::vector<iteration_4d::iteration<double>>> moni_success_iter(
    c.output_dir/("success_iter_"s + c.name + ".dat"s) ,
    success_iter ,
    { } // yes it's strange but I hack this class to get monitoring with iteration_4d::iteration<double>
  );
  monitoring::reactive_monitoring< std::vector<iteration_4d::iteration<double>>> moni_iter(
    c.output_dir/("iterations_"s + c.name + ".dat"s) ,
    iterations ,
    { } // yes it's strange but I hack this class to get monitoring with iteration_4d::iteration<double>
  );

  ublas::vector<std::complex<double>> hjcx(c.Nz,0.),  hjcy(c.Nz,0.),  hEx(c.Nz,0.),  hEy(c.Nz,0.),  hBx(c.Nz,0.),  hBy(c.Nz,0.);
  ublas::vector<std::complex<double>> hjcx1(c.Nz,0.), hjcy1(c.Nz,0.), hEx1(c.Nz,0.), hEy1(c.Nz,0.), hBx1(c.Nz,0.), hBy1(c.Nz,0.);
  ublas::vector<std::complex<double>> hjcx2(c.Nz,0.), hjcy2(c.Nz,0.), hEx2(c.Nz,0.), hEy2(c.Nz,0.), hBx2(c.Nz,0.), hBy2(c.Nz,0.);
  ublas::vector<std::complex<double>> hjcx3(c.Nz,0.), hjcy3(c.Nz,0.), hEx3(c.Nz,0.), hEy3(c.Nz,0.), hBx3(c.Nz,0.), hBy3(c.Nz,0.);
  ublas::vector<std::complex<double>> hjcx4(c.Nz,0.), hjcy4(c.Nz,0.), hEx4(c.Nz,0.), hEy4(c.Nz,0.), hBx4(c.Nz,0.), hBy4(c.Nz,0.);
  ublas::vector<std::complex<double>> hjcx5(c.Nz,0.), hjcy5(c.Nz,0.), hEx5(c.Nz,0.), hEy5(c.Nz,0.), hBx5(c.Nz,0.), hBy5(c.Nz,0.);

  complex_field<double,3> hf1(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]),
                          hf2(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]),
                          hf3(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]),
                          hf4(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]),
                          hf5(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  field3d<double> dvf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  // init h. variables
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

  auto next_snapshot = c.snaptimes.begin();

  const double dt_cfl_maxwell = 2.0*std::sqrt(2.0)/c.Nz;
  iteration_4d::iteration<double> iter;
  iter.dt = c.dt0;
  times.push_back(iter.current_time);
  success_iter.push_back(iter);
  
  while ( iter.current_time<c.Tf ) {
    const double current_t = iter.current_time;
    const double dt = iter.dt;
    //std::cout << escape << std::setw(8) << current_t << " / " << c.Tf << " [" << iteration_t << "]" << std::flush;
    std::cout << "\r" << iteration_4d::time(iter) << std::flush;

    /* Lawson(RK4(3)) */
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
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx1[i] = -0.485071250072666*dt*(1.0*I*Kz[i]*hBx[i] + 1.0*hjhy[i])*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.485071250072666*dt*(1.0*I*Kz[i]*hBy[i] - 1.0*hjhx[i])*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.970142500145332*hEx[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.970142500145332*hEy[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hjcy1[i] = 0.5*dt*(1.0*I*Kz[i]*hBx[i] + 1.0*hjhy[i])*(0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.5*dt*(1.0*I*Kz[i]*hBy[i] - 1.0*hjhx[i])*(0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hBx1[i] = 0.5*I*Kz[i]*dt*hEy[i] + 1.0*hBx[i];
        // ---
        hBy1[i] = -0.5*I*Kz[i]*dt*hEx[i] + 1.0*hBy[i];
        // ---
        hEx1[i] = 0.5*dt*(1.0*I*Kz[i]*hBx[i] + 1.0*hjhy[i])*(0.621267812518166*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481833*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.5*dt*(1.0*I*Kz[i]*hBy[i] - 1.0*hjhx[i])*(0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481833*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hEy1[i] = 0.5*dt*(1.0*I*Kz[i]*hBx[i] + 1.0*hjhy[i])*(0.621267812518166*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481833*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.5*dt*(1.0*I*Kz[i]*hBy[i] - 1.0*hjhx[i])*(0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx[i]*(0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481833*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
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
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::spectrum_ hfvxvyvz(c.Nz);
            hfvxvyvz.fft(dvf[k_x][k_y][k_z].begin());
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf1[k_x][k_y][k_z][i] = 1.0*(0.5*dt*hfvxvyvz[i] + 1.0*hf[k_x][k_y][k_z][i])*std::exp(-0.5*I*Kz[i]*dt*v_z);
            }
          }
        }
      }

    } // end first stage

    // SECOND STAGE /////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*(current_t+0.5*dt)), s_ = std::sin(B0*(current_t+0.5*dt));

      // compute $\int v_x \hat{f}^{(1)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(1)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
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
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx2[i] = 0.970142500145332*hEx[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.970142500145332*hEy[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hjcy2[i] = 1.0*hEx[i]*(0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hBx2[i] = 0.5*I*Kz[i]*dt*hEy1[i] + 1.0*hBx[i];
        // ---
        hBy2[i] = -0.5*I*Kz[i]*dt*hEx1[i] + 1.0*hBy[i];
        // ---
        hEx2[i] = -0.5*dt*(1.0*I*Kz[i]*hBy1[i] - 1.0*hjhx[i]) + 1.0*hEx[i]*(0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481833*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hEy2[i] = 0.5*dt*(1.0*I*Kz[i]*hBx1[i] + 1.0*hjhy[i]) - 1.0*hEx[i]*(0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481833*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
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
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf2[k_x][k_y][k_z][i] = 0.5*dt*hfvxvyvz[i] + 1.0*hf[k_x][k_y][k_z][i]*std::exp(-0.5*I*Kz[i]*dt*v_z);
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
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx3[i] = -0.970142500145332*dt*(1.0*I*Kz[i]*hBx2[i] + 1.0*hjhy[i])*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.970142500145332*dt*(1.0*I*Kz[i]*hBy2[i] - 1.0*hjhx[i])*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.970142500145332*hEx[i]*(1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.970142500145332*hEy[i]*(1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.378732187481834*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hjcy3[i] = 1.0*dt*(1.0*I*Kz[i]*hBx2[i] + 1.0*hjhy[i])*(0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*dt*(1.0*I*Kz[i]*hBy2[i] - 1.0*hjhx[i])*(0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.970142500145332*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.970142500145332*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.970142500145332*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.970142500145332*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.378732187481834*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.621267812518167*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.378732187481834*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.621267812518167*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hBx3[i] = 1.0*I*Kz[i]*dt*hEy2[i] + 1.0*hBx[i];
        // ---
        hBy3[i] = -1.0*I*Kz[i]*dt*hEx2[i] + 1.0*hBy[i];
        // ---
        hEx3[i] = 1.0*dt*(1.0*I*Kz[i]*hBx2[i] + 1.0*hjhy[i])*(0.621267812518166*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481833*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*dt*(1.0*I*Kz[i]*hBy2[i] - 1.0*hjhx[i])*(0.621267812518167*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481834*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.621267812518167*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481834*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481833*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.242535625036333*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hEy3[i] = 1.0*dt*(1.0*I*Kz[i]*hBx2[i] + 1.0*hjhy[i])*(0.621267812518166*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481833*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*dt*(1.0*I*Kz[i]*hBy2[i] - 1.0*hjhx[i])*(0.621267812518167*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481834*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx[i]*(0.621267812518167*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.378732187481834*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy[i]*(0.621267812518166*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.378732187481833*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx[i]*(0.242535625036333*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.242535625036333*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.242535625036333*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.242535625036333*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
      }
      // keep zero mean
      hjcx3[0] = 0.0;
      hjcy3[0] = 0.0;
      hBx3[0] = 0.0;
      hBy3[0] = 0.0;
      hEx3[0] = 0.0;
      hEy3[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx3[i] = 0.;
          hjcy3[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx3[i] = 0.;
          hBy3[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx3[i] = 0.;
          hEy3[i] = 0.;
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
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf3[k_x][k_y][k_z][i] = 1.0*dt*hfvxvyvz[i]*std::exp(-0.5*I*Kz[i]*dt*v_z) + 1.0*hf[k_x][k_y][k_z][i]*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }
    } // end thrid stage

    // FOURTH STAGE //////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*(current_t+dt)), s_ = std::sin(B0*(current_t+dt));

      // compute $\int v_x \hat{f}^{(2)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(2)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
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
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx4[i] = 0.323380833381777*hEx1[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.646761666763555*hEx2[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.323380833381777*hEx[i]*(1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.323380833381777*hEy1[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.646761666763555*hEy2[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.323380833381777*hEy[i]*(1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx1[i]*(0.126244062493945*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.207089270839389*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx2[i]*(0.252488124987889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.414178541678778*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.333333333333333*hjcx3[i] - 1.0*hjcx[i]*(0.126244062493945*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.207089270839389*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.126244062493945*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.207089270839389*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.252488124987889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.414178541678778*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.126244062493945*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.207089270839389*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---

        hjcy4[i] = 1.0*hEx1[i]*(0.323380833381777*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.323380833381777*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx2[i]*(0.646761666763555*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.646761666763555*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx[i]*(0.323380833381777*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.323380833381777*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.323380833381777*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.323380833381777*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.646761666763555*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.646761666763555*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEy[i]*(0.323380833381777*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.323380833381777*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.126244062493945*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.207089270839389*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.252488124987889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.414178541678778*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.126244062493945*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.207089270839389*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.126244062493945*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.207089270839389*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.252488124987889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.414178541678778*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.333333333333333*hjcy3[i] - 1.0*hjcy[i]*(0.126244062493945*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.207089270839389*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hBx4[i] = 0.166666666666667*I*Kz[i]*dt*hEy3[i] + 0.333333333333333*hBx1[i] + 0.666666666666667*hBx2[i] + 0.333333333333333*hBx3[i] - 0.333333333333333*hBx[i];
        // ---
        hBy4[i] = -0.166666666666667*I*Kz[i]*dt*hEx3[i] + 0.333333333333333*hBy1[i] + 0.666666666666667*hBy2[i] + 0.333333333333333*hBy3[i] - 0.333333333333333*hBy[i];
        // ---
        hEx4[i] = -0.166666666666667*dt*(1.0*I*Kz[i]*hBy3[i] - 1.0*hjhx[i]) + 1.0*hEx1[i]*(0.207089270839389*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.126244062493944*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx2[i]*(0.414178541678778*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.252488124987889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.333333333333333*hEx3[i] - 1.0*hEx[i]*(0.207089270839389*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.126244062493944*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.207089270839389*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.126244062493944*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.414178541678778*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.252488124987889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEy[i]*(0.207089270839389*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.126244062493944*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.0808452083454443*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0808452083454443*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.161690416690889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.161690416690889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.0808452083454443*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0808452083454443*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.0808452083454443*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0808452083454443*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.161690416690889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.161690416690889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.0808452083454443*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0808452083454443*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hEy4[i] = 0.166666666666667*dt*(1.0*I*Kz[i]*hBx3[i] + 1.0*hjhy[i]) - 1.0*hEx1[i]*(0.207089270839389*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.126244062493944*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx2[i]*(0.414178541678778*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.252488124987889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.207089270839389*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.126244062493944*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.207089270839389*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.126244062493944*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.414178541678778*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.252488124987889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.333333333333333*hEy3[i] - 1.0*hEy[i]*(0.207089270839389*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.126244062493944*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.0808452083454443*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0808452083454443*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.161690416690889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.161690416690889*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.0808452083454443*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0808452083454443*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy1[i]*(0.0808452083454443*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0808452083454443*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy2[i]*(0.161690416690889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.161690416690889*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.0808452083454443*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0808452083454443*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
      }
      // keep zero mean
      hjcx4[0] = 0.0;
      hjcy4[0] = 0.0;
      hBx4[0] = 0.0;
      hBy4[0] = 0.0;
      hEx4[0] = 0.0;
      hEy4[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx4[i] = 0.;
          hjcy4[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx4[i] = 0.;
          hBy4[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx4[i] = 0.;
          hEy4[i] = 0.;
        }
      #endif

      // update hf

      // compute iFFT(hf3)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( hf3[k_x][k_y][k_z].begin() , hf3[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
          }
        }
      }
      fft::ifft(hEx3.begin(),hEx3.end(),Ex.begin());
      fft::ifft(hEy3.begin(),hEy3.end(),Ey.begin());
      fft::ifft(hBx3.begin(),hBx3.end(),Bx.begin());
      fft::ifft(hBy3.begin(),hBy3.end(),By.begin());
      // compute approximation of (E×vB)∂ᵥf
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
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf4[k_x][k_y][k_z][i] = 0.166666666666667*dt*hfvxvyvz[i] + 0.333333333333333*hf1[k_x][k_y][k_z][i]*std::exp(-0.5*I*Kz[i]*dt*v_z) + 0.666666666666667*hf2[k_x][k_y][k_z][i]*std::exp(-0.5*I*Kz[i]*dt*v_z) + 0.333333333333333*hf3[k_x][k_y][k_z][i] - 0.333333333333333*hf[k_x][k_y][k_z][i]*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }
    } // end fourth stage

    // FIFTH STAGE //////////////////////////////////////////////////
    {
      double c_ = std::cos(B0*(current_t+dt)), s_ = std::sin(B0*(current_t+dt));

      // compute $\int v_x \hat{f}^{(2)}\,\mathrm{d}v$ et $\int v_y \hat{f}^{(2)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.0), hjhy(c.Nz,0.0);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
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
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx5[i] = 0.194028500029066*hEx1[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.388057000058133*hEx2[i]*(1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.194028500029066*hEx[i]*(1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 1.0*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.194028500029066*hEy1[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 0.388057000058133*hEy2[i]*(1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.194028500029066*hEy[i]*(1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 1.0*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx1[i]*(0.0757464374963667*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.124253562503633*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx2[i]*(0.151492874992733*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.248507125007267*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.2*hjcx3[i] + 0.4*hjcx4[i] - 1.0*hjcx[i]*(0.0757464374963667*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.124253562503633*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.0757464374963667*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.124253562503633*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.151492874992733*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.248507125007267*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.0757464374963667*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.124253562503633*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hjcy5[i] = 1.0*hEx1[i]*(0.194028500029066*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.194028500029066*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx2[i]*(0.388057000058133*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.388057000058133*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx[i]*(0.194028500029066*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.194028500029066*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.194028500029066*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.194028500029066*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.388057000058133*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.388057000058133*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEy[i]*(0.194028500029066*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.194028500029066*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.0757464374963667*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.124253562503633*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.151492874992733*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.248507125007267*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.0757464374963667*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.124253562503633*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.0757464374963667*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.124253562503633*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.151492874992733*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.248507125007267*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.2*hjcy3[i] + 0.4*hjcy4[i] - 1.0*hjcy[i]*(0.0757464374963667*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.124253562503633*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hBx5[i] = 0.1*I*Kz[i]*dt*hEy4[i] + 0.2*hBx1[i] + 0.4*hBx2[i] + 0.2*hBx3[i] + 0.4*hBx4[i] - 0.2*hBx[i];
        // ---
        hBy5[i] = -0.1*I*Kz[i]*dt*hEx4[i] + 0.2*hBy1[i] + 0.4*hBy2[i] + 0.2*hBy3[i] + 0.4*hBy4[i] - 0.2*hBy[i];
        // ---
        hEx5[i] = -0.1*dt*(1.0*I*Kz[i]*hBy4[i] - 1.0*hjhx[i]) + 1.0*hEx1[i]*(0.124253562503633*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0757464374963667*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx2[i]*(0.248507125007267*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.151492874992733*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.2*hEx3[i] + 0.4*hEx4[i] - 1.0*hEx[i]*(0.124253562503633*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0757464374963667*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.124253562503633*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0757464374963667*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.248507125007267*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.151492874992733*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEy[i]*(0.124253562503633*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0757464374963667*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.0485071250072666*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0485071250072666*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.0970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.0485071250072666*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0485071250072666*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy1[i]*(0.0485071250072666*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0485071250072666*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy2[i]*(0.0970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy[i]*(0.0485071250072666*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0485071250072666*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
        // ---
        hEy5[i] = 0.1*dt*(1.0*I*Kz[i]*hBx4[i] + 1.0*hjhy[i]) - 1.0*hEx1[i]*(0.124253562503633*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0757464374963667*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hEx2[i]*(0.248507125007267*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.151492874992733*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEx[i]*(0.124253562503633*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0757464374963667*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy1[i]*(0.124253562503633*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0757464374963667*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hEy2[i]*(0.248507125007267*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.151492874992733*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 0.2*hEy3[i] + 0.4*hEy4[i] - 1.0*hEy[i]*(0.124253562503633*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0757464374963667*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx1[i]*(0.0485071250072666*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0485071250072666*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcx2[i]*(0.0970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0970142500145332*std::cos(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcx[i]*(0.0485071250072666*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) - 0.0485071250072666*std::cos(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy1[i]*(0.0485071250072666*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0485071250072666*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) - 1.0*hjcy2[i]*(0.0970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0970142500145332*std::sin(0.75*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1))) + 1.0*hjcy[i]*(0.0485071250072666*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(1 - 0.111111111111111*std::sqrt(17))) + 0.0485071250072666*std::sin(1.5*std::sqrt(2)*dt*std::sqrt(0.111111111111111*std::sqrt(17) + 1)));
      }
      // keep zero mean
      hjcx5[0] = 0.0;
      hjcy5[0] = 0.0;
      hBx5[0] = 0.0;
      hBy5[0] = 0.0;
      hEx5[0] = 0.0;
      hEy5[0] = 0.0;

      #if JC_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hjcx5[i] = 0.;
          hjcy5[i] = 0.;
        }
      #endif
      #if Bxy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hBx5[i] = 0.;
          hBy5[i] = 0.;
        }
      #endif
      #if Exy_condition == 0
        for ( auto i=0u ; i<c.Nz ; ++i ) {
          hEx5[i] = 0.;
          hEy5[i] = 0.;
        }
      #endif

      // update hf

      // compute iFFT(hf4)
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( hf4[k_x][k_y][k_z].begin() , hf4[k_x][k_y][k_z].end() , f[k_x][k_y][k_z].begin() );
          }
        }
      }
      fft::ifft(hEx4.begin(),hEx4.end(),Ex.begin());
      fft::ifft(hEy4.begin(),hEy4.end(),Ey.begin());
      fft::ifft(hBx4.begin(),hBx4.end(),Bx.begin());
      fft::ifft(hBy4.begin(),hBy4.end(),By.begin());
      // compute approximation of (E×vB)∂ᵥf
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
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::fft( dvf[k_x][k_y][k_z].begin() , dvf[k_x][k_y][k_z].end() , hfvxvyvz.begin() );
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              hf5[k_x][k_y][k_z][i] = 0.1*dt*hfvxvyvz[i] + 0.2*hf1[k_x][k_y][k_z][i]*std::exp(-0.5*I*Kz[i]*dt*v_z) + 0.4*hf2[k_x][k_y][k_z][i]*std::exp(-0.5*I*Kz[i]*dt*v_z) + 0.2*hf3[k_x][k_y][k_z][i] + 0.4*hf4[k_x][k_y][k_z][i] - 0.2*hf[k_x][k_y][k_z][i]*std::exp(-1.0*I*Kz[i]*dt*v_z);
            }
          }
        }
      }
    } // end fifth stage


    // ---- compute local error of the iteration ----------------------
    iter.jcx_error(hjcx5,hjcx4,f.step.dz);
    iter.jcy_error(hjcy5,hjcy4,f.step.dz);
    iter.Bx_error(hBx5,hBx4,f.step.dz);
    iter.By_error(hBy5,hBy4,f.step.dz);
    iter.Ex_error(hEx5,hEx4,f.step.dz);
    iter.Ey_error(hEy5,hEy4,f.step.dz);
    
    iter.fh_error(hf5,hf4,f.step.dz*f.step.dvx*f.step.dvy*f.step.dvz);

    iter.success = ( iter.error() <= c.tol ); // || (  iter.dt <= 0.501*dt_cfl_maxwell  );

    std::cout << " -- " << iteration_4d::error(iter) << std::flush;
    iterations.push_back(iter);
    moni_iter.push();

    //if ( iter.dt < 1e-2 ) { iter.success = true; } // if dt is too small iteration is accepted

    if ( iter.success ) {
      // iteration is accepted, copy all variables for the next iteration
      success_iter.push_back(iter);

      // copy space variables
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        hjcx[i] = hjcx4[i];
        hjcy[i] = hjcy4[i];
        hBx[i]  = hBx4[i];
        hBy[i]  = hBy4[i];
        hEx[i]  = hEx4[i];
        hEy[i]  = hEy4[i];
      }
      // copy phase space variable
      std::copy( hf4.data() , hf4.data()+hf4.num_elements() , hf.data() );

      fft::ifft(hEx.begin(),hEx.end(),Ex.begin());
      fft::ifft(hEy.begin(),hEy.end(),Ey.begin());
      fft::ifft(hBx.begin(),hBx.end(),Bx.begin());
      fft::ifft(hBy.begin(),hBy.end(),By.begin());
      fft::ifft(hjcx.begin(),hjcx.end(),jcx.begin());
      fft::ifft(hjcy.begin(),hjcy.end(),jcy.begin());

      // monitoring
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

      ++(iter.iter); // increment current time and iteration number
      iter.current_time += iter.dt;
      times.push_back(iter.current_time);
      moni.push();
      moni_success_iter.push();
    } else {
      ++iter.iter;
    }

    // compute new dt
    double dt_opt = std::pow( c.tol/(iter.error()) , 0.25 )*iter.dt;
    iter.dt = std::min( std::max( dt_opt , 0.5*dt_cfl_maxwell ) , 3.0*dt_cfl_maxwell );
        
    //if ( next_snapshot != c.snaptimes.end() && iter.current_time+iter.dt > *next_snapshot ) { iter.dt = *next_snapshot - iter.current_time; }
    if ( iter.current_time+iter.dt > c.Tf ) { iter.dt = c.Tf - iter.current_time; }
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



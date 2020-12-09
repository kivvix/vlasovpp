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
  c.create_output_directory();

  c.name = "vmhls";

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
    for ( auto i=0u ; i<c.Nx/2 ; ++i ) { kx[i]     = 2.*math::pi<double>()*i/l; }
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
  for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
    const double vx = k_x*f.step.dvx + f.range.vx_min;
    for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
      const double vy = k_y*f.step.dvy + f.range.vy_min;
      for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
        const double vz = k_z*f.step.dvz + f.range.vz_min;
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          const double z = i*f.step.dz + f.range.z_min;
          f[k_x][k_y][k_z][i] = M1( z,vx,vy,vz )*( 1.0 + c.alpha*std::cos(K*z) );

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

  const double B0 = c.B0;
  ublas::vector<double> Ex(c.Nz,0.),Ey(c.Nz,0.);
  ublas::vector<double> Bx(c.Nz,0.),By(c.Nz,0.);
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*i;
    //Bx[i] = c.alpha * std::sin(K*z);
    Bx[i] = 0.;
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

  std::vector<double> velocitiy_vx_max;  velocitiy_vx_max.reserve(100);
  std::vector<double> velocitiy_vy_max;  velocitiy_vy_max.reserve(100);
  std::vector<double> velocitiy_vz_max;  velocitiy_vz_max.reserve(100);

  ublas::vector<double> fdvxdvydz(c.Nvz,0.);
  ublas::vector<double> vxfdv(c.Nz,0.), vyfdv(c.Nz,0.), vzfdv(c.Nz,0.);
  ublas::vector<double> ec_perp(c.Nz,0.), ec_vz(c.Nz,0.);
  ublas::vector<double> rho_h(c.Nz,0.);

  auto compute_integrals = [&]( const complex_field<double,3> & hf , double current_t ) {
    ublas::vector<double> fdvxdvydz(c.Nvz,0.);
    ublas::vector<double> vxfdv(c.Nz,0.);
    ublas::vector<double> vyfdv(c.Nz,0.);
    ublas::vector<double> vzfdv(c.Nz,0.);

    ublas::vector<double> fvxvyvz(c.Nz,0.);

    double c_ = std::cos(B0*current_t), s_ = std::sin(B0*current_t);

    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double vx = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double vy = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            fdvxdvydz[k_z] += fvxvyvz[i]*f.step.dz*f.step.dvx*f.step.dvy;
            vxfdv[i] += vx*fvxvyvz[i]*f.volumeV();
            vyfdv[i] += vy*fvxvyvz[i]*f.volumeV();
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
      double vx = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double vy = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double vz = k_z*f.step.dvz + f.range.vz_min;
          fft::ifft( hf[k_x][k_y][k_z].begin() , hf[k_x][k_y][k_z].end() , fvxvyvz.begin() );
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            ec_perp[i] += (vx*vx + vy*vy)*fvxvyvz[i]*f.volumeV();
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
    std::stringstream ss; ss<<(count++)*f.step.dvz + f.range.vz_min<<" "<<y;
    return ss.str();
  };
  auto printer__z_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<(count++)*f.step.dz + f.range.z_min<<" "<<y;
    return ss.str();
  };

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

  velocitiy_vx_max.push_back( 0. );
  velocitiy_vy_max.push_back( 0. );
  velocitiy_vz_max.push_back( 0. );

  monitoring::reactive_monitoring<std::vector<double>> moni( c.output_dir/("energy_"s + c.name + ".dat"s) , times , {&electric_energy,&magnetic_energy,&cold_energy,&kinetic_energy,&mass,&Exmax,&Eymax,&Bxmax,&Bymax} );

  monitoring::reactive_monitoring<std::vector<double>> moni_velocity( c.output_dir/("velocity_"s + c.name + ".dat"s) , times , {&velocitiy_vx_max,&velocitiy_vy_max,&velocitiy_vz_max} );

  //total_energy.push_back( compute_total_energy(jcx,jcy,Ex,Ey,Bx,By,hf) );

  #define _velocity_vx(Ex,Ey,Bx,By)  Ex[i]*c_ + Ey[i]*s_ + v_z*Bx[i]*s_ - v_z*By[i]*c_
  #define _velocity_vy(Ex,Ey,Bx,By) -Ex[i]*s_ + Ey[i]*c_ + v_z*Bx[i]*c_ + v_z*By[i]*s_
  #define _velocity_vz(Ex,Ey,Bx,By) -Bx[i]*( w_1*s_ + w_2*c_ ) + By[i]*( w_1*c_ - w_2*s_ )

  std::size_t iteration_t = 0;
  while ( current_t<c.Tf ) {
    std::cout << escape << std::setw(8) << current_t << " / " << c.Tf << " [" << iteration_t << "]" << std::flush;


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
    Exmax.push_back( max_abs(Ex) );
    Eymax.push_back( max_abs(Ey) );
    Bxmax.push_back( max_abs(Bx) );
    Bymax.push_back( max_abs(By) );


    double max_velocity_vx = 0.;
    double max_velocity_vy = 0.;
    double max_velocity_vz = 0.;

    double c_ = std::cos(B0*current_t), s_ = std::sin(B0*current_t);
    for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
      double v_x = k_x*f.step.dvx + f.range.vx_min;
      double w_1 = k_x*f.step.dvx + f.range.vx_min;
      for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
        double v_y = k_y*f.step.dvy + f.range.vy_min;
        double w_2 = k_y*f.step.dvy + f.range.vy_min;
        for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
          double v_z = k_z*f.step.dvz + f.range.vz_min;
          for ( auto i=0u ; i<c.Nz ; ++i ) {
            double velocity_vx = _velocity_vx(Ex,Ey,Bx,By);
            double velocity_vy = _velocity_vy(Ex,Ey,Bx,By);
            double velocity_vz = _velocity_vz(Ex,Ey,Bx,By);

            max_velocity_vx = std::max(std::abs(velocity_vx),max_velocity_vx);
            max_velocity_vy = std::max(std::abs(velocity_vy),max_velocity_vy);
            max_velocity_vz = std::max(std::abs(velocity_vz),max_velocity_vz);
          }
        }
      }
    }

    velocitiy_vx_max.push_back( max_velocity_vx );
    velocitiy_vy_max.push_back( max_velocity_vy );
    velocitiy_vz_max.push_back( max_velocity_vz );

    ++iteration_t;
    current_t += dt;
    times.push_back(current_t);
    moni.push();
    moni_velocity.push();

    if ( iteration_t % 1000 == 0 )
    {
      std::tie(fdvxdvydz,vxfdv,vyfdv,vzfdv) = compute_integrals( hf , current_t );
      std::stringstream filename; filename << "fdvxdvydz_" << c.name << "_" << iteration_t << ".dat";
      c << monitoring::make_data( filename.str() , fdvxdvydz , printer__vz_y );
      filename.str("");
      filename; filename << "jhxyz_" << c.name << "_" << iteration_t << ".dat";

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

      rho_h = compute_rho_h( hf );
      filename.str("");
      filename << "rhoh_"<< c.name << "_" << iteration_t << ".dat";
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

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
#include "vlasovpp/physic.h"
#include "vlasovpp/tool.h"


namespace math = boost::math::constants;
const std::complex<double> & I = std::complex<double>(0.,1.);

#define SQ(X) ((X)*(X))
#define Zi(i) (i*f.step.dz+f.range.z_min)
#define Vkx(k) (k*f.step.dvx+f.range.vx_min)
#define Vky(k) (k*f.step.dvy+f.range.vy_min)
#define Vkz(k) (k*f.step.dvz+f.range.vz_min)

int
main ( int argc , char const * argv[] )
{
  // load configuration
  auto c = config(argc,argv,"{{ simu_name }}");
  save_config(c);

/* ---------------------------------------------------------------- */
  // physics variables initilization

  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  complex_field<double,3> hf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  ublas::vector<double> jcx(c.Nz,0.) , jcy(c.Nz,0.);
  ublas::vector<double> Ex(c.Nz,0.)  , Ey(c.Nz,0.);
  ublas::vector<double> Bx(c.Nz,0.)  , By(c.Nz,0.);

  // range in velocity are done by jinja parameter
  f.range.vx_min = {{ frange.vx_min }}; f.range.vx_max = {{ frange.vx_max }};
  f.range.vy_min = {{ frange.vy_min }}; f.range.vy_max = {{ frange.vy_max }};
  f.range.vz_min = {{ frange.vz_min }}; f.range.vz_max = {{ frange.vz_max }};
  // range in space are done from config (K parameter)
  f.range.z_min =  0.;  f.range.z_max = 2.*math::pi<double>()/c.K;

  f.compute_steps();

  const double v_par  = c.v_par;
  const double v_perp = c.v_perp;
  const double nh = c.nh;
  const double B0 = c.B0;


  ublas::vector<double> Kz(c.Nz);
  {
    const double L = f.range.len_z();
    for ( auto i=0u ; i<c.Nz/2 ; ++i ) { Kz[i]      = 2.*math::pi<double>()*i/L; }
    for ( auto i=-c.Nz/2 ; i<0 ; ++i ) { Kz[c.Nz+i] = 2.*math::pi<double>()*i/L; }
  }

  // initial condition
  auto M1 = maxwellian(nh,{0.,0.,0.},{v_perp,v_perp,v_par});
  for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
    for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
      for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          const double vx = k_x*f.step.dvx + f.range.vx_min;
          const double vy = k_y*f.step.dvy + f.range.vy_min;
          const double vz = k_z*f.step.dvz + f.range.vz_min;
          const double z  = i*f.step.dz + f.range.z_min;
          f[k_x][k_y][k_z][i] = M1( z,vx,vy,vz );
        }
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),hf[k_x][k_y][k_z].begin());
      }
    }
  }
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*i;
    Bx[i] = c.alpha * std::sin(c.K*z);
  }

  // end physics variables initialization
/* ---------------------------------------------------------------- */
  // monitorring initialization

  std::vector<double> times;           times.reserve(100);
  std::vector<double> electric_energy; electric_energy.reserve(100);
  std::vector<double> kinetic_energy;  kinetic_energy.reserve(100);
  std::vector<double> magnetic_energy; magnetic_energy.reserve(100);
  std::vector<double> cold_energy;     cold_energy.reserve(100);
  std::vector<double> mass;            mass.reserve(100);
  std::vector<double> Bxmax;           Bxmax.reserve(100);
  std::vector<double> Bymax;           Bymax.reserve(100);
  std::vector<double> Exmax;           Exmax.reserve(100);
  std::vector<double> Eymax;           Eymax.reserve(100);

  ublas::vector<double> fdvxdvydz(c.Nvz,0.);
  ublas::vector<double> vxfdv(c.Nz,0.), vyfdv(c.Nz,0.), vzfdv(c.Nz,0.);
  ublas::vector<double> ec_perp(c.Nz,0.), ec_vz(c.Nz,0.);
  ublas::vector<double> rho_h(c.Nz,0.);
  field<double,1> fdvxdvy(boost::extents[c.Nvz][c.Nz]);

  // init all computer functor for each monitoring value
  auto compute_vperp_integral       = computer::vperp_integral( f );
  auto compute_z_vperp_integral     = computer::z_vperp_integral( f );
  auto compute_local_kinetic_energy = computer::local_kinetic_energy( f );
  auto compute_electric_energy      = computer::space_energy( f.step.dz );
  auto compute_magnetic_energy      = computer::space_energy( f.step.dz );
  auto compute_cold_energy          = computer::space_energy( f.step.dz );
  auto compute_hot_mass_energy      = computer::hot_mass_energy( f );

  auto printer_z_data  = factory::printer__x_data( f.range.z_min  , f.step.dz  );
  auto printer_vz_data = factory::printer__x_data( f.range.vz_min , f.step.dvz );


  electric_energy.push_back( compute_electric_energy(Ex,Ey) );
  magnetic_energy.push_back( compute_magnetic_energy(Bx,By) );
  cold_energy.push_back( compute_cold_energy(jcx,jcy) );
  compute_hot_mass_energy(hf);
  kinetic_energy.push_back( compute_hot_mass_energy.he );
  mass.push_back( compute_hot_mass_energy.mass );
  Bxmax.push_back( max_abs(Bx) );
  Bymax.push_back( max_abs(By) );
  Exmax.push_back( max_abs(Ex) );
  Eymax.push_back( max_abs(Ey) );

  monitoring::reactive_monitoring<std::vector<double>> moni(
    c.output_dir/("energy_"s + c.name + ".dat"s) ,
    times ,
    {&electric_energy,&magnetic_energy,&cold_energy,&kinetic_energy,&mass,&Exmax,&Eymax,&Bxmax,&Bymax}
  );

  // end monitorring initialization
/* ---------------------------------------------------------------- */
  // substage initialization

  //{# declaration of each substage variables #}
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.jcx }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.jcy }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.Bx  }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.By  }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.Ex  }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};
  ublas::vector<std::complex<double>> {%- for (lhs,_) in schemes %} {{ lhs.Ey  }}(c.Nz,0.) {{ "," if not loop.last else "" }}{% endfor %};

  complex_field<double,3> {%- for (lhs,_) in schemes[:-1] %} {{ lhs.fh }}(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]) {{ ", " if not loop.last else "" }}{% endfor %};
  field3d<double> dvf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  // init Fourier variables (just for first loop and first stage)
  fft::fft( jcx.begin() , jcx.end() , {{ schemes[-1][0].jcx }}.begin() );
  fft::fft( jcy.begin() , jcy.end() , {{ schemes[-1][0].jcy }}.begin() );
  fft::fft(  Ex.begin() ,  Ex.end() , {{ schemes[-1][0].Ex  }}.begin() );
  fft::fft(  Ey.begin() ,  Ey.end() , {{ schemes[-1][0].Ey  }}.begin() );
  fft::fft(  Bx.begin() ,  Bx.end() , {{ schemes[-1][0].Bx  }}.begin() );
  fft::fft(  By.begin() ,  By.end() , {{ schemes[-1][0].By  }}.begin() );

  #define _hjfx(hf) ( w_1*c_ - w_2*s_ )*hf[k_x][k_y][k_z][i]*f.volumeV()
  #define _hjfy(hf) ( w_1*s_ + w_2*c_ )*hf[k_x][k_y][k_z][i]*f.volumeV()
  #define _velocity_vx(Ex,Ey,Bx,By) -( Ex[i]*c_ + Ey[i]*s_ + v_z*Bx[i]*s_ - v_z*By[i]*c_)
  #define _velocity_vy(Ex,Ey,Bx,By) -(-Ex[i]*s_ + Ey[i]*c_ + v_z*Bx[i]*c_ + v_z*By[i]*s_)
  #define _velocity_vz(Ex,Ey,Bx,By) -(-Bx[i]*( w_1*s_ + w_2*c_ ) + By[i]*( w_1*c_ - w_2*s_ ))
  {% macro compute_hjfx( hf ) -%}
    ( w_1*c_ - w_2*s_ )*{{ hf }}[k_x][k_y][k_z][i]*f.volumeV()
  {%- endmacro %}
  {% macro compute_hjfy( hf ) -%}
    ( w_1*s_ + w_2*c_ )*{{ hf }}[k_x][k_y][k_z][i]*f.volumeV()
  {%- endmacro %}

  {% macro compute_velocity_vx( Ex , Ey , Bx , By ) -%}
    -( {{ Ex }}[i]*c_ + {{ Ey }}[i]*s_ + v_z*{{ Bx }}[i]*s_ - v_z*{{ By }}[i]*c_)
  {%- endmacro %}
  {% macro compute_velocity_vy( Ex , Ey , Bx , By ) -%}
    -(-{{ Ex }}[i]*s_ + {{ Ey }}[i]*c_ + v_z*{{ Bx }}[i]*c_ + v_z*{{ By }}[i]*s_)
  {%- endmacro %}
  {% macro compute_velocity_vz( Ex , Ey , Bx , By ) -%}
    -(-{{ Bx }}[i]*( w_1*s_ + w_2*c_ ) + {{ By }}[i]*( w_1*c_ - w_2*s_ ))
  {%- endmacro %}

  auto next_snapshot = c.snaptimes.begin();

  iteration_4d::iteration<double> iter;
  iter.dt = c.dt0;
  times.push_back(iter.current_time);
  const double dt_cfl_maxwell = 2.0*std::sqrt(2.0)/c.Nz;
  
  while ( iter.current_time<c.Tf )
  {
    const double current_t = iter.current_time;
    const double dt = iter.dt;
    std::cout << "\r" << iteration_4d::time(iter) << std::flush;

    // {# write here only one stage and loop with jinja2 #}
    {% for (lhs,rhs) in schemes %}
    //////////////////////////////////////////////////////////////////
    { // begin stage {{ loop.index }}
      double c_ = std::cos(B0*( current_t + {{ lhs.dt }}*dt )) ,
             s_ = std::sin(B0*( current_t + {{ lhs.dt }}*dt )) ;

      // compute $\int v_x \hat{f}^{(stage)}\,\mathrm{d}v$ and $\int v_y \hat{f}^{(stage)}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.) , hjhy(c.Nz,0.);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=1u ; i<c.Nz ; ++i ) {
              //hjhx[i] += _hjfx({{ schemes[loop.index0-1][0].fh }});
              //hjhy[i] += _hjfy({{ schemes[loop.index0-1][0].fh }});
              hjhx[i] += {{ compute_hjfx(schemes[loop.index0-1][0].fh) }};
              hjhy[i] += {{ compute_hjfy(schemes[loop.index0-1][0].fh) }};
            }
          }
        }
      }
      // keep zero mean
      hjhx[0] = 0.0;
      hjhy[0] = 0.0;

      // --> compute hjcx(stage),hjcy(stage),hBx(stage),hBy(stage),hEx(stage),hEy(stage) (all spatial values)
      for ( auto i=1u ; i<c.Nz ; ++i ) {
        {% if not loop.last %}
          {{ lhs.jcx }}[i] = {{ rhs.jcx }};
          // ---
          {{ lhs.jcy }}[i] = {{ rhs.jcy }};
          // ---
          {{ lhs.Bx  }}[i] = {{ rhs.Bx  }};
          // ---
          {{ lhs.By  }}[i] = {{ rhs.By  }};
          // ---
          {{ lhs.Ex  }}[i] = {{ rhs.Ex  }};
          // ---
          {{ lhs.Ey  }}[i] = {{ rhs.Ey  }};
        {% else %}
          auto {{ lhs.jcx }}_tmp = {{ rhs.jcx }};
          // ---
          auto {{ lhs.jcy }}_tmp = {{ rhs.jcy }};
          // ---
          auto {{ lhs.Bx  }}_tmp = {{ rhs.Bx  }};
          // ---
          auto {{ lhs.By  }}_tmp = {{ rhs.By  }};
          // ---
          auto {{ lhs.Ex  }}_tmp = {{ rhs.Ex  }};
          // ---
          auto {{ lhs.Ey  }}_tmp = {{ rhs.Ey  }};

          {{ lhs.jcx }}[i] = {{ lhs.jcx }}_tmp;
          {{ lhs.jcy }}[i] = {{ lhs.jcy }}_tmp;
          {{ lhs.Bx  }}[i] = {{ lhs.Bx  }}_tmp;
          {{ lhs.By  }}[i] = {{ lhs.By  }}_tmp;
          {{ lhs.Ex  }}[i] = {{ lhs.Ex  }}_tmp;
          {{ lhs.Ey  }}[i] = {{ lhs.Ey  }}_tmp;
        {% endif %}
      }
      // keep zero mean
      {{ lhs.jcx }}[0] = 0.0;
      {{ lhs.jcy }}[0] = 0.0;
      {{ lhs.Bx  }}[0] = 0.0;
      {{ lhs.By  }}[0] = 0.0;
      {{ lhs.Ex  }}[0] = 0.0;
      {{ lhs.Ey  }}[0] = 0.0;

      // --> compute hf(stage)
      // iFFT of hf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            fft::ifft( {{ schemes[loop.index0-1][0].fh }}[k_x][k_y][k_z].begin() ,
                       {{ schemes[loop.index0-1][0].fh }}[k_x][k_y][k_z].end()   ,
                       f[k_x][k_y][k_z].begin()
                    );
          }
        }
      }
      // iFFT of hEx, hEy, hBx and hBy
      fft::ifft({{ schemes[loop.index0-1][0].Ex }}.begin(),{{ schemes[loop.index0-1][0].Ex }}.end(),Ex.begin());
      fft::ifft({{ schemes[loop.index0-1][0].Ey }}.begin(),{{ schemes[loop.index0-1][0].Ey }}.end(),Ey.begin());
      fft::ifft({{ schemes[loop.index0-1][0].Bx }}.begin(),{{ schemes[loop.index0-1][0].Bx }}.end(),Bx.begin());
      fft::ifft({{ schemes[loop.index0-1][0].By }}.begin(),{{ schemes[loop.index0-1][0].By }}.end(),By.begin());

      // compute approximation of (E×vB)∂ᵥf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              const double w_1 = k_x*f.step.dvx + f.range.vx_min;
              const double w_2 = k_y*f.step.dvy + f.range.vy_min;
              const double v_z = k_z*f.step.dvz + f.range.vz_min;

              //const double velocity_vx = _velocity_vx(Ex,Ey,Bx,By);
              //const double velocity_vy = _velocity_vy(Ex,Ey,Bx,By);
              //const double velocity_vz = _velocity_vz(Ex,Ey,Bx,By);

              const double velocity_vx = {{ compute_velocity_vx('Ex','Ey','Bx','By') }};
              const double velocity_vy = {{ compute_velocity_vy('Ex','Ey','Bx','By') }};
              const double velocity_vz = {{ compute_velocity_vz('Ex','Ey','Bx','By') }};

              dvf[k_x][k_y][k_z][i] = + weno3d::d_vx(velocity_vx,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vy(velocity_vy,f,k_x,k_y,k_z,i)
                                      + weno3d::d_vz(velocity_vz,f,k_x,k_y,k_z,i);
            }
          }
        }
      }

      // update hf
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            const double v_z = k_z*f.step.dvz + f.range.vz_min;
            fft::spectrum_ hfvxvyvz(c.Nz);
            hfvxvyvz.fft(dvf[k_x][k_y][k_z].begin());
            for ( auto i=0u ; i<c.Nz ; ++i ) {
              {{ lhs.fh }}[k_x][k_y][k_z][i] = {{ rhs.fh }};
            }
          }
        }
      }

    } // end stage {{ loop.index }}
    //////////////////////////////////////////////////////////////////
    {% endfor %}

    fft::ifft(hEx.begin(),hEx.end(),Ex.begin());
    fft::ifft(hEy.begin(),hEy.end(),Ey.begin());
    fft::ifft(hBx.begin(),hBx.end(),Bx.begin());
    fft::ifft(hBy.begin(),hBy.end(),By.begin());
    fft::ifft(hjcx.begin(),hjcx.end(),jcx.begin());
    fft::ifft(hjcy.begin(),hjcy.end(),jcy.begin());

    electric_energy.push_back(compute_electric_energy(Ex,Ey));
    magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
    cold_energy.push_back(compute_cold_energy(jcx,jcy));
    compute_hot_mass_energy(hf);
    kinetic_energy.push_back(compute_hot_mass_energy.he);
    mass.push_back(compute_hot_mass_energy.mass);
    Exmax.push_back( max_abs(Ex) );
    Eymax.push_back( max_abs(Ey) );
    Bxmax.push_back( max_abs(Bx) );
    Bymax.push_back( max_abs(By) );

    ++iter.iter;
    iter.current_time += iter.dt;
    times.push_back(current_t);
    moni.push();

    if ( iter.iter % 1000 == 0 ) {
      std::stringstream filename;

      /*
      filename.str("");
      compute_vperp_integral( hf );
      filename << "fdvxdvy_" << c.name << "_" << iter.iter << ".dat";
      compute_vperp_integral.fdvxdvy.write( c.output_dir / filename.str() );
      */

      filename.str("");
      compute_z_vperp_integral( hf );
      filename << "fdvxdvydz_" << c.name << "_" << iter.iter << ".dat";
      c << monitoring::make_data( filename.str() , compute_z_vperp_integral.fdvxdvydz , printer_vz_data );
    } // end monitoring %1000

  } // end time loop


  auto writer_t_y = [&,count=0] (auto const& y) mutable {
    std::stringstream ss; ss<<times[count++]<<" "<<y;
    return ss.str();
  };

  // this data are already in energy_XXX.dat
  c << monitoring::make_data( "ee"s + c.name + ".dat"s , electric_energy , writer_t_y );
  c << monitoring::make_data( "eb"s + c.name + ".dat"s , magnetic_energy , writer_t_y );
  c << monitoring::make_data( "ec"s + c.name + ".dat"s , cold_energy     , writer_t_y );
  c << monitoring::make_data( "ek"s + c.name + ".dat"s , kinetic_energy  , writer_t_y );
  c << monitoring::make_data( "m"s  + c.name + ".dat"s , mass            , writer_t_y );


return 0;
}

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

{% for i in range(0,expLt_mat|count) %}
  {% for j in range(0,expLt_mat[i]|count) %}
std::complex<double>
matrixExpr{{ i }}{{ j }} ( double t , double k ) {
  return {{ expLt_mat[i][j] }};
}
  {% endfor %}
{% endfor %}

int
main ( int argc , char const * argv[] )
{
  // load configuration
  auto c = config(argc,argv,"{{ simu_name }}");
  c.create_output_directory();
  save_config(c);

/* ---------------------------------------------------------------- */
  // physics variables initilization

  field3d<double> f(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  {%- if is_embeded %}
  complex_field<double,3> {{ un.fh }}(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  {% else %}
  complex_field<double,3> {{ schemes[-1][0].fh }}(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);
  {% endif %}

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

  const double B0 = c.B0;

  ublas::vector<double> Kz(c.Nz);
  {
    const double L = f.range.len_z();
    for ( auto i=0u ; i<c.Nz/2 ; ++i ) { Kz[i]      = 2.*math::pi<double>()*i/L; }
    for ( auto i=-c.Nz/2 ; i<0 ; ++i ) { Kz[c.Nz+i] = 2.*math::pi<double>()*i/L; }
  }

  // initial condition
  auto M1 = maxwellian(c.nh,{0.,0.,0.},{c.v_perp,c.v_perp,c.v_par});
  for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
    for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
      for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
        for (std::size_t i=0u ; i<f.size_x() ; ++i ) {
          const double vx = static_cast<double>(k_x)*f.step.dvx + f.range.vx_min;
          const double vy = static_cast<double>(k_y)*f.step.dvy + f.range.vy_min;
          const double vz = static_cast<double>(k_z)*f.step.dvz + f.range.vz_min;
          const double z  = static_cast<double>(i)*f.step.dz + f.range.z_min;
          f[k_x][k_y][k_z][i] = M1( z,vx,vy,vz );
        }
        {%- if is_embeded %}
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),{{ un.fh }}[k_x][k_y][k_z].begin());
        {% else %}
        fft::fft(f[k_x][k_y][k_z].begin(),f[k_x][k_y][k_z].end(),{{ schemes[-1][0].fh }}[k_x][k_y][k_z].begin());
        {% endif %}
      }
    }
  }
  for ( auto i=0u ; i<c.Nz ; ++i ) {
    double z = f.range.z_min + f.step.dz*static_cast<double>(i);
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
  auto compute_vperp_integral       = computer::vperp_integral<double>( f );
  auto compute_z_vperp_integral     = computer::z_vperp_integral<double>( f );
  auto compute_z_vz_integral        = computer::z_vz_integral<double>( f );
  auto compute_local_kinetic_energy = computer::local_kinetic_energy<double>( f );
  auto compute_electric_energy      = computer::space_energy( f.step.dz );
  auto compute_magnetic_energy      = computer::space_energy( f.step.dz );
  auto compute_cold_energy          = computer::space_energy( f.step.dz );
  auto compute_hot_mass_energy      = computer::hot_mass_energy<double>( f );

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
  {%- if is_embeded %}
  ublas::vector<std::complex<double>> {{ un.jcx }}(c.Nz,0.);
  ublas::vector<std::complex<double>> {{ un.jcy }}(c.Nz,0.);
  ublas::vector<std::complex<double>> {{ un.Bx  }}(c.Nz,0.);
  ublas::vector<std::complex<double>> {{ un.By  }}(c.Nz,0.);
  ublas::vector<std::complex<double>> {{ un.Ex  }}(c.Nz,0.);
  ublas::vector<std::complex<double>> {{ un.Ey  }}(c.Nz,0.);
  {% endif %}

  {% if is_embeded %}
  complex_field<double,3> {%- for (lhs,_) in schemes %} {{ lhs.fh }}(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]) {{ ", " if not loop.last else "" }}{% endfor %};
  {% else %}
  complex_field<double,3> {%- for (lhs,_) in schemes[:-1] %} {{ lhs.fh }}(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]) {{ ", " if not loop.last else "" }}{% endfor %};
  {% endif %}

  field3d<double> dvf(boost::extents[c.Nvx][c.Nvy][c.Nvz][c.Nz]);

  // init Fourier variables (just for first loop and first stage)
  {% if is_embeded %}
  fft::fft( jcx.begin() , jcx.end() , {{ un.jcx }}.begin() );
  fft::fft( jcy.begin() , jcy.end() , {{ un.jcy }}.begin() );
  fft::fft(  Ex.begin() ,  Ex.end() , {{ un.Ex  }}.begin() );
  fft::fft(  Ey.begin() ,  Ey.end() , {{ un.Ey  }}.begin() );
  fft::fft(  Bx.begin() ,  Bx.end() , {{ un.Bx  }}.begin() );
  fft::fft(  By.begin() ,  By.end() , {{ un.By  }}.begin() );
  {% else %}
  fft::fft( jcx.begin() , jcx.end() , {{ schemes[-1][0].jcx }}.begin() );
  fft::fft( jcy.begin() , jcy.end() , {{ schemes[-1][0].jcy }}.begin() );
  fft::fft(  Ex.begin() ,  Ex.end() , {{ schemes[-1][0].Ex  }}.begin() );
  fft::fft(  Ey.begin() ,  Ey.end() , {{ schemes[-1][0].Ey  }}.begin() );
  fft::fft(  Bx.begin() ,  Bx.end() , {{ schemes[-1][0].Bx  }}.begin() );
  fft::fft(  By.begin() ,  By.end() , {{ schemes[-1][0].By  }}.begin() );
  {% endif %}

  {% macro compute_hjfx( hf ) -%}
    ( w_1*c_ - w_2*s_ )*{{ hf }}[k_x][k_y][k_z][i]*f.volumeV()
  {%- endmacro %}
  {% macro compute_hjfy( hf ) -%}
    ( w_1*s_ + w_2*c_ )*{{ hf }}[k_x][k_y][k_z][i]*f.volumeV()
  {%- endmacro %}

  {% macro velocity_vx(Bx,By,Ex,Ey) -%}
      -{{ Ex }}[i]*c_ + {{ Ey }}[i]*s_ + v_z*{{ Bx }}[i]*s_ - v_z*{{ By }}[i]*c_;
  {%- endmacro %}
  {% macro velocity_vx(Bx,By,Ex,Ey) -%}
    -(-{{ Ex }}[i]*s_ + {{ Ey }}[i]*c_ + v_z*{{ Bx }}[i]*c_ + v_z*{{ By }}[i]*s_);
  {%- endmacro %}
  {% macro velocity_vx(Bx,By,Ex,Ey) -%}
    -(-{{ Bx }}[i]*( w_1*s_ + w_2*c_ ) + {{ By }}[i]*( w_1*c_ - w_2*s_ ));
  {%- endmacro %}

  iteration_4d::iteration<double> iter;
  iter.dt = c.dt0;
  times.push_back(iter.current_time);
  
  while ( iter.current_time<c.Tf )
  {
    std::cout << "\r" << iteration_4d::time(iter) << std::flush;
    const double current_t = iter.current_time;
    const double dt = iter.dt;

    // {# write here only one stage and loop with jinja2 #}
    {% for (lhs,rhs) in schemes %}
    //////////////////////////////////////////////////////////////////
    { // begin stage {{ loop.index }}
      double c_ = std::cos(B0*( current_t + {{ lhs.dt }}*dt )) ,
             s_ = std::sin(B0*( current_t + {{ lhs.dt }}*dt )) ;

      // compute $\int v_x {{ schemes[loop.index0-1][0].fh }}\,\mathrm{d}v$ and $\int v_y {{ schemes[loop.index0-1][0].fh }}\,\mathrm{d}v$
      ublas::vector<std::complex<double>> hjhx(c.Nz,0.) , hjhy(c.Nz,0.);
      for ( auto k_x=0u ; k_x<c.Nvx ; ++k_x ) {
        const double w_1 = k_x*f.step.dvx + f.range.vx_min;
        for ( auto k_y=0u ; k_y<c.Nvy ; ++k_y ) {
          const double w_2 = k_y*f.step.dvy + f.range.vy_min;
          for ( auto k_z=0u ; k_z<c.Nvz ; ++k_z ) {
            for ( auto i=1u ; i<c.Nz ; ++i ) {
              hjhx[i] += {{ compute_hjfx(schemes[loop.index0-1][0].fh) }};
              hjhy[i] += {{ compute_hjfy(schemes[loop.index0-1][0].fh) }};
            }
          }
        }
      }
      // keep zero mean
      hjhx[0] = 0.0;
      hjhy[0] = 0.0;

      // --> compute {{ lhs.jcx }}, {{ lhs.jcy }}, {{ lhs.Bx }}, {{ lhs.By }}, {{ lhs.Ex }}, {{ lhs.Ey }} (all spatial values)
      for ( auto i=1u ; i<c.Nz ; ++i ) {
        {% if not loop.last or is_embeded %}
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

      // --> compute {{ lhs.fh }}
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
              const double w_1 = static_cast<double>(k_x)*f.step.dvx + f.range.vx_min;
              const double w_2 = static_cast<double>(k_y)*f.step.dvy + f.range.vy_min;
              const double v_z = static_cast<double>(k_z)*f.step.dvz + f.range.vz_min;

              const double velocity_vx = {{ velocity_vx(schemes[loop.index0-1][0].Bx,schemes[loop.index0-1][0].By,schemes[loop.index0-1][0].Ex,schemes[loop.index0-1][0].Ey) }};
              const double velocity_vy = {{ velocity_vy(schemes[loop.index0-1][0].Bx,schemes[loop.index0-1][0].By,schemes[loop.index0-1][0].Ex,schemes[loop.index0-1][0].Ey) }};
              const double velocity_vz = {{ velocity_vz(schemes[loop.index0-1][0].Bx,schemes[loop.index0-1][0].By,schemes[loop.index0-1][0].Ex,schemes[loop.index0-1][0].Ey) }};

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

    {% if is_embeded %}
    // ---- begin compute local error of the iteration ---------------
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
    // ---- end compute local error of the iteration -----------------

    if (iter.success) {
      success_iter.push_back(iter);

      // copy space variables
      for ( auto i=0u ; i<c.Nz ; ++i ) {
        {{ un.jcx }}[i] = {{ schemes[-2][0].jcx }}[i];
        {{ un.jcy }}[i] = {{ schemes[-2][0].jcy }}[i];
        {{ un.Bx  }}[i] = {{ schemes[-2][0].Bx  }}[i];
        {{ un.By  }}[i] = {{ schemes[-2][0].By  }}[i];
        {{ un.Ex  }}[i] = {{ schemes[-2][0].Ex  }}[i];
        {{ un.Ey  }}[i] = {{ schemes[-2][0].Ey  }}[i];
      }
      // copy phase space variable
      std::copy(
        {{ schemes[-2][0].fh }}.data() , {{ schemes[-2][0].fh }}.data()+{{ schemes[-2][0].fh }}.num_elements() ,
        {{ un.fh }}.data()
      );

      fft::ifft( {{ un.jcx }}.begin() , {{ un.jcx }}.end() , jcx.begin() );
      fft::ifft( {{ un.jcy }}.begin() , {{ un.jcy }}.end() , jcy.begin() );
      fft::ifft( {{ un.Bx  }}.begin() , {{ un.Bx  }}.end() , Bx.begin()  );
      fft::ifft( {{ un.By  }}.begin() , {{ un.By  }}.end() , By.begin()  );
      fft::ifft( {{ un.Ex  }}.begin() , {{ un.Ex  }}.end() , Ex.begin()  );
      fft::ifft( {{ un.Ey  }}.begin() , {{ un.Ey  }}.end() , Ey.begin()  );

      cold_energy.push_back(compute_cold_energy(jcx,jcy));
      magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
      electric_energy.push_back(compute_electric_energy(Ex,Ey));
      compute_hot_mass_energy({{ un.fh }});
      kinetic_energy.push_back( compute_hot_mass_energy.he );
      mass.push_back(compute_hot_mass_energy.mass);
      Bxmax.push_back( max_abs(Bx) );
      Bymax.push_back( max_abs(By) );
      Exmax.push_back( max_abs(Ex) );
      Eymax.push_back( max_abs(Ey) );

      iter.current_time += iter.dt;
      times.push_back(iter.current_time);
      moni.push();
    }
    ++iter.iter;

    // compute new dt
    double dt_opt = std::pow( c.tol/(iter.error()) , 0.25 )*iter.dt;
    //iter.dt = std::min( std::max( dt_opt , 0.5*dt_cfl_maxwell ) , 3.0*dt_cfl_maxwell );
    iter.dt = dt_opt;
    {% else %}
    {# trouver un moyen de ne pas avoir de copier-coller ici #}
    fft::ifft( {{ schemes[-1][0].jcx }}.begin() , {{ schemes[-1][0].jcx }}.end() , jcx.begin() );
    fft::ifft( {{ schemes[-1][0].jcy }}.begin() , {{ schemes[-1][0].jcy }}.end() , jcy.begin() );
    fft::ifft( {{ schemes[-1][0].Bx  }}.begin() , {{ schemes[-1][0].Bx  }}.end() , Bx.begin()  );
    fft::ifft( {{ schemes[-1][0].By  }}.begin() , {{ schemes[-1][0].By  }}.end() , By.begin()  );
    fft::ifft( {{ schemes[-1][0].Ex  }}.begin() , {{ schemes[-1][0].Ex  }}.end() , Ex.begin()  );
    fft::ifft( {{ schemes[-1][0].Ey  }}.begin() , {{ schemes[-1][0].Ey  }}.end() , Ey.begin()  );

    cold_energy.push_back(compute_cold_energy(jcx,jcy));
    magnetic_energy.push_back(compute_magnetic_energy(Bx,By));
    electric_energy.push_back(compute_electric_energy(Ex,Ey));
    compute_hot_mass_energy({{ schemes[-1][0].fh }});
    kinetic_energy.push_back( compute_hot_mass_energy.he );
    mass.push_back(compute_hot_mass_energy.mass);
    Bxmax.push_back( max_abs(Bx) );
    Bymax.push_back( max_abs(By) );
    Exmax.push_back( max_abs(Ex) );
    Eymax.push_back( max_abs(Ey) );

    ++iter.iter;
    iter.current_time += iter.dt;
    times.push_back(iter.current_time);
    moni.push();
    {% endif %}

    if ( iter.iter % 1000 == 0 ) {
      std::stringstream filename;
      
      filename.str("");
      compute_vperp_integral( hf );
      filename << "fdvxdvy_" << c.name << "_" << iter.iter << ".dat";
      compute_vperp_integral.fdvxdvy.write( c.output_dir / filename.str() );
      
      filename.str("");
      compute_z_vz_integral( hf );
      filename << "fdzdvz_" << c.name << "_" << iter.iter << ".dat";
      compute_z_vz_integral.fdzdvz.write( c.output_dir / filename.str() );
      
      filename.str("");
      compute_z_vperp_integral( hf );
      filename << "fdvxdvydz_" << c.name << "_" << iter.iter << ".dat";
      c << monitoring::make_data( filename.str() , compute_z_vperp_integral.fdvxdvydz , printer_vz_data );
    } // end monitoring %1000


    if ( iter.current_time+iter.dt > c.Tf ) { iter.dt = c.Tf - iter.current_time; }
  } // end time loop
  std::cout << "\r" << iteration_4d::time(iter) << std::endl;

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

  {
    std::stringstream filename;
    
    filename.str("");
    compute_vperp_integral( hf );
    filename << "fdvxdvy_" << c.name << "_Tf.dat";
    compute_vperp_integral.fdvxdvy.write( c.output_dir / filename.str() );
    
    filename.str("");
    compute_z_vz_integral( hf );
    filename << "fdzdvz_" << c.name << "_Tf.dat";
    compute_z_vz_integral.fdzdvz.write( c.output_dir / filename.str() );
    
    filename.str("");
    compute_z_vperp_integral( hf );
    filename << "fdvxdvydz_" << c.name << "_Tf.dat";
    c << monitoring::make_data( filename.str() , compute_z_vperp_integral.fdvxdvydz , printer_vz_data );
  }

return 0;
}

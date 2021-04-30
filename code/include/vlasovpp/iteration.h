#include <ostream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <sstream>
#include <iomanip>

#include <boost/numeric/ublas/matrix.hpp>

#include "vlasovpp/complex_field.h"

namespace iteration {
using namespace boost::numeric;

template < typename _T >
struct iteration {
  std::size_t iter;
  _T dt;
  _T current_time;
  _T Lhfh=0., LE=0., Luc=0.;
  bool success;

  template < typename Iterator >
  static _T
  error ( Iterator first1 , Iterator last1 , Iterator first2 , _T integrator_step )
  {
    return std::sqrt(std::inner_product(
      first1 , last1 , first2 , _T{0.} ,
      std::plus<_T>{} ,
      [&] ( const auto & a , const auto & b ) { return std::pow(std::abs( a - b ),2)*integrator_step; }
    ));
  }

  //template < std::size_t NumDimsV >
  _T
  hfh_error ( const complex_field<_T,1> & hfh1 , const complex_field<_T,1> & hfh2 , _T integrator_step )
  {
    Lhfh = error( hfh1.origin() , hfh1.origin()+hfh1.num_elements() , hfh2.origin() , integrator_step );
    success = false;
    return Lhfh;
  }

  _T
  E_error ( const ublas::vector<_T> & E1 , const ublas::vector<_T> & E2 , _T integrator_step )
  {
    LE = error( E1.begin() , E1.end() , E2.begin() , integrator_step );
    success = false;
    return LE;
  }

  _T
  uc_error ( const ublas::vector<_T> & uc1 , const ublas::vector<_T> & uc2 , _T integrator_step )
  {
    Luc = error( uc1.begin() , uc1.end() , uc2.begin() , integrator_step );
    success = false;
    return Luc;
  }

  const _T &
  increment ()
  {
    current_time += dt;
    return current_time;
  }

  _T
  error () const
  { return Lhfh + LE + Luc; }

};

template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const iteration<_T> & iter )
{
  os.precision(15);
  os << iter.iter << " " << iter.dt << " " << iter.current_time << " " << iter.error() << " " << iter.Lhfh << " " << iter.LE << " " << iter.Luc << " " << std::noboolalpha << iter.success;
  return os;
}

template< typename _T >
struct __time_iteration
{
  std::size_t i;
  _T t;
  _T dt;

  __time_iteration ( std::size_t _i , _T _t , _T _dt )
    : i(_i) , t(_t) , dt(_dt)
  { ; }
};
template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const __time_iteration<_T> & ti )
{
  os.precision(15);
  os << " [" << std::setw(6) << ti.i << "] " << std::setw(8) << ti.t << " (" << std::setw(9) << ti.dt << ")";
  return os;
}

template < typename _T >
__time_iteration<_T>
time ( const iteration<_T> & iter )
{
  return __time_iteration<_T>(iter.iter,iter.current_time,iter.dt);
}

template< typename _T >
struct __error_iteration
{
  bool success;
  _T Lhfh,LE;

  __error_iteration ( bool _success , _T _Lhfh , _T _LE )
    : success(_success) , Lhfh(_Lhfh) , LE(_LE)
  { ; }
};
template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const __error_iteration<_T> & ei )
{
  using namespace std::string_literals;
  os.precision(15);
  os << ((ei.success)?"\033[92m"s:"\033[31m"s) << std::setw(10) << ei.Lhfh << " " << std::setw(10) << ei.LE << "\033[0m";
  return os;
}

template < typename _T >
__error_iteration<_T>
error ( const iteration<_T> & iter )
{
  return __error_iteration<_T>(iter.success,iter.Lhfh,iter.LE);
}

} // namespace iteration

namespace iteration_4d {
using namespace boost::numeric;

template < typename _T >
struct iteration
{
  std::size_t iter=0;
  _T dt=0.1;
  _T current_time=0.;
  _T Ljcx=0., Ljcy=0., LBx=0., LBy=0., LEx=0., LEy=0., Lfh=0.;
  bool success;

  template < typename Iterator >
  _T
  error ( Iterator first1 , Iterator last1 , Iterator first2 , _T integrator_step )
  {
    _T L = std::sqrt(std::real(std::inner_product(
      first1 , last1 , first2 , _T{0.} ,
      std::plus<_T>{} ,
      [&] ( const auto & a , const auto & b ) { return std::pow(std::abs( a - b ),2)*integrator_step; }
    )));
    success = false;
    return L;
  }

  template <typename Container>
  _T
  jcx_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    Ljcx = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return Ljcx;
  }
  template <typename Container>
  _T
  jcy_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    Ljcy = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return Ljcy;
  }
  template <typename Container>
  _T
  Bx_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    LBx = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return LBx;
  }
  template <typename Container>
  _T
  By_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    LBy = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return LBy;
  }
  template <typename Container>
  _T
  Ex_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    LEx = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return LEx;
  }
  template <typename Container>
  _T
  Ey_error ( const Container & X1 , const Container & X2 , _T integrator_step )
  {
    LEy = error( X1.begin() , X1.end() , X2.begin() , integrator_step );
    return LEy;
  }

  _T
  fh_error ( const complex_field<_T,3> & X1 , const complex_field<_T,3> & X2 , _T integrator_step )
  {
    Lfh = error( X1.origin() , X1.origin()+X1.num_elements() , X2.origin() , integrator_step );
    return Lfh;
  }

  iteration<_T> &
  operator ++ ()
  {
    ++iter;
    current_time += dt;
    return *this;
  }
  iteration<_T>
  operator ++ (int)
  {
    auto tmp = *this;
    ++iter;
    current_time += dt;
    return tmp;
  }

  _T
  error () const
  { return Ljcx + Ljcy + LBx + LBy + LEx + LEy + Lfh; }
};


template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const iteration<_T> & iter )
{
  os.precision(15);
  os << iter.iter << " " << iter.dt << " " << iter.current_time << " "
     << iter.error() << " "
     << iter.Ljcx << " " << iter.Ljcy << " "
     << iter.LBx  << " " << iter.LBy  << " "
     << iter.LEx  << " " << iter.LEy  << " "
     << iter.Lfh  << " "
     << std::noboolalpha << iter.success;
  return os;
}

template< typename _T >
struct __time_iteration
{
  std::size_t i;
  _T t;
  _T dt;

  __time_iteration ( std::size_t _i , _T _t , _T _dt )
    : i(_i) , t(_t) , dt(_dt)
  { ; }
};
template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const __time_iteration<_T> & ti )
{
  os.precision(15);
  os << " [" << std::setw(6) << ti.i << "] " << std::setw(8) << ti.t << " (" << std::setw(9) << ti.dt << ")";
  return os;
}

template < typename _T >
__time_iteration<_T>
time ( const iteration<_T> & iter )
{
  return __time_iteration<_T>(iter.iter,iter.current_time,iter.dt);
}

template< typename _T >
struct __error_iteration
{
  bool success;
  _T Ljcx, Ljcy, LBx, LBy, LEx, LEy, Lfh;

  __error_iteration ( bool _success , _T _Ljcx, _T _Ljcy, _T _LBx, _T _LBy, _T _LEx, _T _LEy, _T _Lfh )
    : success(_success) , Ljcx(_Ljcx), Ljcy(_Ljcy), LBx(_LBx), LBy(_LBy), LEx(_LEx), LEy(_LEy), Lfh(_Lfh)
  { ; }
};
template < typename CharT , typename Traits = std::char_traits<CharT> , typename _T >
std::basic_ostream<CharT,Traits> &
operator << ( std::basic_ostream<CharT,Traits> & os , const __error_iteration<_T> & ei )
{
  using namespace std::string_literals;
  os.precision(15);
  os << ((ei.success)?"\033[92m"s:"\033[31m"s)
     << std::setw(5) << ei.Ljcx << " " << std::setw(5) << ei.Ljcy << " "
     << std::setw(5) << ei.LBx  << " " << std::setw(5) << ei.LBy  << " "
     << std::setw(5) << ei.LEx  << " " << std::setw(5) << ei.LEy  << " "
     << std::setw(5) << ei.Lfh
     << "\033[0m";
  return os;
}

template < typename _T >
__error_iteration<_T>
error ( const iteration<_T> & iter )
{
  return __error_iteration<_T>(iter.success,iter.Ljcx,iter.Ljcy,iter.LBx,iter.LBy,iter.LEx,iter.LEy,iter.Lfh);
}

} // namespace iteration_4d

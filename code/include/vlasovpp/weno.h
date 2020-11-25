#ifndef _WENO_H_
#define _WENO_H_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <type_traits>
#include <tuple>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>

#include <boost/iterator/zip_iterator.hpp>
#include "boost/tuple/tuple.hpp"

#include "field.h"
#include "array_view.h"

namespace weno {

using namespace boost::numeric;
#define SQ(X) ((X)*(X))

/**
  @fn template < class IteratorTuple > auto local_flux ( boost::zip_iterator<IteratorTuple> & f_it )
  @brief Compute local flux $\left( f_{i,k+\frac{1}{2}}^2 , f_{i,k+\frac{1}{2}}^-\right)$
  @param `boost::zip_iterator<IteratorTuple> f_it` a `boost::zip_iterator` of length 6 on $(f_{i,k-2},\dots,f_{i,k+3})$ values
  @return a pair of flux plus and minus
**/
//template < typename IteratorTuple >
template < typename _T >
auto
//local_flux ( boost::zip_iterator<IteratorTuple> const& f_it )
local_flux ( _T fim2 , _T fim1 , _T fi , _T fip1 , _T fip2 , _T fip3 )
{
  //typedef typename std::remove_cv<typename std::remove_reference<decltype(f_it->template get<0>())>::type>::type _T; // récupération du type stocké dans le premier itérateur du zip_iterator (sans const ni reference)

  const static _T epsi = 1e-6;

  // NB pour optimisation minuscule : 2 termes sont calculés 2 fois :
  // * `(13./12.)*SQ( f_it->template get<1>() - 2.*f_it->template get<2>() + f_it->template get<3>() )` dans `w0p` et `w2m`
  // * `(13./12.)*SQ( f_it->template get<2>() - 2.*f_it->template get<3>() + f_it->template get<4>() )` dans `w2p` et `w1m`

  // $f_{i,k+1/2}^+$
  //_T w0p = (13./12.)*SQ( f_it->template get<0>() - 2.*f_it->template get<1>() + f_it->template get<2>() ) + 0.25*SQ(    f_it->template get<0>() - 4.*f_it->template get<1>() + 3.*f_it->template get<2>() );
  _T w0p = (13./12.)*SQ( fim2 - 2.*fim1 + fi   ) + 0.25*SQ(    fim2 - 4.*fim1 + 3.*fi   );
  _T w1p = (13./12.)*SQ( fim1 - 2.*fi   + fip1 ) + 0.25*SQ(    fim1           -    fip1 );
  _T w2p = (13./12.)*SQ( fi   - 2.*fip1 + fip2 ) + 0.25*SQ( 3.*fi   - 4.*fip1 +    fip2 );

  w0p = 0.1/(SQ( epsi + w0p )); w1p = 0.6/(SQ( epsi + w1p )); w2p = 0.3/(SQ( epsi + w2p ));

  _T sum_wp = w0p+w1p+w2p;
  w0p /= sum_wp; w1p /= sum_wp; w2p /= sum_wp;

  _T fikp12p = w0p*( (2./6.)*fim2 - (7./6.)*fim1 + (11./6.)*fi   )
             + w1p*(-(1./6.)*fim1 + (5./6.)*fi   +  (2./6.)*fip1 )
             + w2p*( (2./6.)*fi   + (5./6.)*fip1 -  (1./6.)*fip2 );

  // $f_{i,k+1/2}^-$
  _T w0m = (13./12.)*SQ( fip1 - 2.*fip2 + fip3 ) + 0.25*SQ( 3.*fip1 - 4.*fip2 +    fip3 );
  _T w1m = (13./12.)*SQ( fi   - 2.*fip1 + fip2 ) + 0.25*SQ(    fi             -    fip2 );
  _T w2m = (13./12.)*SQ( fim1 - 2.*fi   + fip1 ) + 0.25*SQ(    fim1 - 4.*fi   + 3.*fip1 );

  w0m = 0.1/SQ( epsi + w0m ); w1m = 0.6/SQ( epsi + w1m ); w2m = 0.3/SQ( epsi + w2m );

  _T sum_wm = w0m+w1m+w2m;
  w0m /= sum_wm; w1m /= sum_wm; w2m /= sum_wm;

  _T fikp12m = w2m*(-(1./6.)*fim1 + (5./6.)*fi   + (2./6.)*fip1 )
             + w1m*( (2./6.)*fi   + (5./6.)*fip1 - (1./6.)*fip2 )
             + w0m*((11./6.)*fip1 - (7./6.)*fip2 + (2./6.)*fip3 );
  
  // return a pair of fluxes, one loop on the data structure for two fluxes
  return std::make_pair(fikp12p,fikp12m);
}

#undef SQ


/**
  @fn template < typename _T , std::size_t NumDimsV > boost::multi_array<std::pair<_T,_T>,NumDimsV+1> flux ( flied<_T,NumDimsV> const& u )
  @brief Compute flux array $\left( f_{i,k+\frac{1}{2}}^2 , f_{i,k+\frac{1}{2}}^-\right)_{i,k}$
  @param `flied<_T,NumDimsV> const& u` the field to transport
  @return a `multi_array<std::pair<_T,_T>,NumDimsV+1>` of all flux (plus and minus)
**/
template < typename _T , std::size_t NumDimsV >
boost::multi_array<std::pair<_T,_T>,NumDimsV+1>
flux ( field<_T,NumDimsV> const& u )
{
  boost::multi_array<std::pair<_T,_T>,NumDimsV+1> fikp12(tools::array_view<const std::size_t>(u.shape(),NumDimsV));

  for ( std::size_t k=0 ; k<2 ; ++k ) {
    std::size_t i=0;
    auto begin = u.begin_border_stencil(k);
    auto end   = std::get<0>(u.end_border_stencil(k));
    boost::detail::multi_array::array_iterator<_T, const _T*, mpl_::size_t<NumDimsV>, const _T&, boost::iterators::random_access_traversal_tag> it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3;
    std::tie(it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3) = begin;
    //for ( auto it = u.begin_border_stencil(k) ; it != u.end_border_stencil(k) ; ++it , ++i ) {
    for (  ; it_im2 != end ; ++it_im2,++it_im1,++it_i,++it_ip1,++it_ip2,++it_ip3 , ++i ) {
      //fikp12[k][i] = local_flux(it);
      fikp12[k][i] = local_flux(*it_im2,*it_im1,*it_i,*it_ip1,*it_ip2,*it_ip3);
    }
  }
  for ( std::size_t k=2 ; k<u.size(0)-3 ; ++k ) {
    std::size_t i=0;
    auto begin = u.begin_stencil(k);
    auto end   = std::get<0>(u.end_stencil(k));
    boost::detail::multi_array::array_iterator<_T, const _T*, mpl_::size_t<NumDimsV>, const _T&, boost::iterators::random_access_traversal_tag> it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3;
    std::tie(it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3) = begin;
    //for ( auto it = u.begin_stencil(k) ; it != u.end_stencil(k) ; ++it , ++i ) {
    for (  ; it_im2 != end ; ++it_im2,++it_im1,++it_i,++it_ip1,++it_ip2,++it_ip3 , ++i ) {
      //fikp12[k][i] = local_flux(it);
      fikp12[k][i] = local_flux(*it_im2,*it_im1,*it_i,*it_ip1,*it_ip2,*it_ip3);
    }
  }
  for ( std::size_t k=u.size(0)-3 ; k<u.size(0) ; ++k ) {
    std::size_t i=0;
    auto begin = u.begin_border_stencil(k);
    auto end   = std::get<0>(u.end_border_stencil(k));
    boost::detail::multi_array::array_iterator<_T, const _T*, mpl_::size_t<NumDimsV>, const _T&, boost::iterators::random_access_traversal_tag> it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3;
    std::tie(it_im2,it_im1,it_i,it_ip1,it_ip2,it_ip3) = begin;
    //for ( auto it = u.begin_border_stencil(k) ; it != u.end_border_stencil(k) ; ++it , ++i ) {
    for ( ; it_im2 != end ; ++it_im2,++it_im1,++it_i,++it_ip1,++it_ip2,++it_ip3 , ++i ) {
      //fikp12[k][i] = local_flux(it);
      fikp12[k][i] = local_flux(*it_im2,*it_im1,*it_i,*it_ip1,*it_ip2,*it_ip3);
    }
  }
  return fikp12;
}

/**
  @fn template < typename _T , std::size_t NumDimsV > auto trp_v ( field<_T,NumDimsV> const& u , ublas::vector<_T> const& E )
  @brief Transport a field `u` at velocity `E`
  @param `flied<_T,NumDimsV> const& u` the field to transport
  @param `ublas::vector<_T> const& E` the velocity in $v$-direction
  @return a `field<_T,NumDimsV>` a field with transported data at velocity `E` in $v$-direction
**/
template < typename _T , std::size_t NumDimsV >
auto
trp_v ( field<_T,NumDimsV> const& u , ublas::vector<_T> const& E )
{
  field<_T,NumDimsV> trp(tools::array_view<const std::size_t>(u.shape(),NumDimsV+1)); // !?? NumDimsV+1

  ublas::vector<_T> Em(E.size()) , Ep(E.size());

  for ( auto i=0 ; i<E.size() ; ++i ) {
    Ep(i) = std::max(E(i),0.);
    Em(i) = std::min(E(i),0.);
  }

  auto fikp12 = flux(u);

  { auto k=0,km1=trp.size(0)-1;
    for ( auto i=0 ; i<trp.size(1) ; ++i ) {
      trp[k][i] = ( Ep(i)*(fikp12[k][i].first  - fikp12[km1][i].first)
                  + Em(i)*(fikp12[k][i].second - fikp12[km1][i].second) )/u.step.dv;
    }
  }
  for ( auto k=1 ; k<trp.size(0) ; ++k ) {
    for ( auto i=0 ; i<trp.size(1) ; ++i ) {
      trp[k][i] = ( Ep(i)*(fikp12[k][i].first  - fikp12[k-1][i].first)
                  + Em(i)*(fikp12[k][i].second - fikp12[k-1][i].second) )/u.step.dv;
    }
  }

  return trp;
}

} // namespace weno

namespace wenolin {

using namespace boost::numeric;

/**
  @fn template < class IteratorTuple > auto local_flux ( boost::zip_iterator<IteratorTuple> & f_it )
  @brief Compute local flux $\left( f_{i,k+\frac{1}{2}}^2 , f_{i,k+\frac{1}{2}}^-\right)$
  @param `boost::zip_iterator<IteratorTuple> f_it` a `boost::zip_iterator` of length 6 on $(f_{i,k-2},\dots,f_{i,k+3})$ values
  @return a pair of flux plus and minus
**/
template < typename IteratorTuple >
auto
local_flux ( boost::zip_iterator<IteratorTuple> const& f_it )
{
  typedef typename std::remove_cv<typename std::remove_reference<decltype(f_it->template get<0>())>::type>::type _T; // récupération du type stocké dans le premier itérateur du zip_iterator (sans const ni reference)

  // $f_{i,k+1/2}^+$
  _T w0p = 0.1, w1p = 0.6, w2p = 0.3;

  _T fikp12p = w0p*( (2./6.)*f_it->template get<0>() - (7./6.)*f_it->template get<1>() + (11./6.)*f_it->template get<2>() )
             + w1p*(-(1./6.)*f_it->template get<1>() + (5./6.)*f_it->template get<2>() +  (2./6.)*f_it->template get<3>() )
             + w2p*( (2./6.)*f_it->template get<2>() + (5./6.)*f_it->template get<3>() -  (1./6.)*f_it->template get<4>() );

  // $f_{i,k+1/2}^-$
  _T w0m = 0.1, w1m = 0.6, w2m = 0.3;

  _T fikp12m = w2m*(-(1./6.)*f_it->template get<1>() + (5./6.)*f_it->template get<2>() + (2./6.)*f_it->template get<3>() )
             + w1m*( (2./6.)*f_it->template get<2>() + (5./6.)*f_it->template get<3>() - (1./6.)*f_it->template get<4>() )
             + w0m*((11./6.)*f_it->template get<3>() - (7./6.)*f_it->template get<4>() + (2./6.)*f_it->template get<5>() );
  
  // return a pair of fluxes, one loop on the data structure for two fluxes
  return std::make_pair(fikp12p,fikp12m);
}


/**
  @fn template < typename _T , std::size_t NumDimsV > boost::multi_array<std::pair<_T,_T>,NumDimsV+1> flux ( flied<_T,NumDimsV> const& u )
  @brief Compute flux array $\left( f_{i,k+\frac{1}{2}}^2 , f_{i,k+\frac{1}{2}}^-\right)_{i,k}$
  @param `flied<_T,NumDimsV> const& u` the field to transport
  @return a `multi_array<std::pair<_T,_T>,NumDimsV+1>` of all flux (plus and minus)
**/
template < typename _T , std::size_t NumDimsV >
boost::multi_array<std::pair<_T,_T>,NumDimsV+1>
flux ( field<_T,NumDimsV> const& u )
{
  boost::multi_array<std::pair<_T,_T>,NumDimsV+1> fikp12(tools::array_view<const std::size_t>(u.shape(),NumDimsV));

  for ( std::size_t k=0 ; k<2 ; ++k ) {
    std::size_t i=0;
    for ( auto it = u.begin_border_stencil(k) ; it != u.end_border_stencil(k) ; ++it , ++i ) {
      fikp12[k][i] = local_flux(it);
    }
  }
  for ( std::size_t k=2 ; k<u.size(0)-3 ; ++k ) {
    std::size_t i=0;
    for ( auto it = u.begin_stencil(k) ; it != u.end_stencil(k) ; ++it , ++i ) {
      fikp12[k][i] = local_flux(it);
    }
  }
  for ( std::size_t k=u.size(0)-3 ; k<u.size(0) ; ++k ) {
    std::size_t i=0;
    for ( auto it = u.begin_border_stencil(k) ; it != u.end_border_stencil(k) ; ++it , ++i ) {
      fikp12[k][i] = local_flux(it);
    }
  }
  return fikp12;
}

/**
  @fn template < typename _T , std::size_t NumDimsV > auto trp_v ( field<_T,NumDimsV> const& u , ublas::vector<_T> const& E )
  @brief Transport a field `u` at velocity `E`
  @param `flied<_T,NumDimsV> const& u` the field to transport
  @param `ublas::vector<_T> const& E` the velocity in $v$-direction
  @return a `field<_T,NumDimsV>` a field with transported data at velocity `E` in $v$-direction
**/
template < typename _T , std::size_t NumDimsV >
auto
trp_v ( field<_T,NumDimsV> const& u , ublas::vector<_T> const& E )
{
  field<_T,NumDimsV> trp(tools::array_view<const std::size_t>(u.shape(),NumDimsV+1)); // !?? NumDimsV+1

  ublas::vector<_T> Em(E.size()) , Ep(E.size());

  for ( auto i=0 ; i<E.size() ; ++i ) {
    Ep(i) = std::max(E(i),0.);
    Em(i) = std::min(E(i),0.);
  }

  auto fikp12 = flux(u);

  { auto k=0,km1=trp.size(0)-1;
    for ( auto i=0 ; i<trp.size(1) ; ++i ) {
      trp[k][i] = ( Ep(i)*(fikp12[k][i].first  - fikp12[km1][i].first)
                  + Em(i)*(fikp12[k][i].second - fikp12[km1][i].second) )/u.step.dv;
    }
  }
  for ( auto k=1 ; k<trp.size(0) ; ++k ) {
    for ( auto i=0 ; i<trp.size(1) ; ++i ) {
      trp[k][i] = ( Ep(i)*(fikp12[k][i].first  - fikp12[k-1][i].first)
                  + Em(i)*(fikp12[k][i].second - fikp12[k-1][i].second) )/u.step.dv;
    }
  }

  return trp;
}

} // namespace wenolin

namespace weno3d {

  using namespace boost::numeric;
  #define SQ(X) ((X)*(X))

  template < typename _T >
  _T
  dv_weno ( _T velocity , _T uim3 , _T uim2 , _T uim1 , _T ui , _T uip1 , _T uip2 , _T uip3 , _T dx ) {
    _T vp = std::max(velocity,0.);
    _T vm = std::min(velocity,0.);

    #ifdef linearized_WENO
      auto fip12 = wenolin::local_flux(uim2,uim1,ui,uip1,uip2,uip3);
      auto fim12 = wenolin::local_flux(uim3,uim2,uim1,ui,uip1,uip2);
    #else
      auto fip12 = weno::local_flux(uim2,uim1,ui,uip1,uip2,uip3);
      auto fim12 = weno::local_flux(uim3,uim2,uim1,ui,uip1,uip2);
    #endif

    return ( vp*(fip12.first - fim12.first) + vm*(fip12.second - fim12.second) )/dx;
    //return ( vp*(ui-uim1) + vm*(uip1-ui) )/dx; //upwind
  }

  std::tuple<std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t>
  periodic_index ( const std::size_t N , const std::size_t k ) {
    std::size_t km3 = static_cast<std::size_t>(k-3), km2 = static_cast<std::size_t>(k-2), km1 = static_cast<std::size_t>(k-1),
                kp1 = static_cast<std::size_t>(k+1), kp2 = static_cast<std::size_t>(k+2), kp3 = static_cast<std::size_t>(k+3);
    if ( k < 3u ) {
      km3 = static_cast<std::size_t>(( k-3 +N)%N); km2 = static_cast<std::size_t>(( k-2 +N)%N); km1 = static_cast<std::size_t>(( k-1 +N)%N);
    }
    if ( k >= N-3 ) {
      kp1 = static_cast<std::size_t>(( k+1 )%N); kp2 = static_cast<std::size_t>(( k+2 )%N); kp3 = static_cast<std::size_t>(( k+3 )%N);
    }
    return std::make_tuple(km3,km2,km1,k,kp1,kp2,kp3);
  }

  template < typename _T >
  _T
  weno_vx ( _T velocity , field3d<_T> const& u , std::size_t k_x , std::size_t k_y , std::size_t k_z , std::size_t i ) {
    const std::size_t Nvx = u.size(0);
    std::size_t kxm3, kxm2, kxm1, kx, kxp1, kxp2, kxp3;
    std::tie(kxm3,kxm2,kxm1,kx,kxp1,kxp2,kxp3) = periodic_index(Nvx,k_x);

    return dv_weno( velocity , u[kxm3][k_y][k_z][i] , u[kxm2][k_y][k_z][i] , u[kxm1][k_y][k_z][i] , u[kx][k_y][k_z][i] , u[kxp1][k_y][k_z][i] , u[kxp2][k_y][k_z][i] , u[kxp3][k_y][k_z][i] , u.step.dvx );
  }

  template < typename _T >
  _T
  weno_vy ( _T velocity , field3d<_T> const& u , std::size_t k_x , std::size_t k_y , std::size_t k_z , std::size_t i ) {
    const std::size_t Nvy = u.size(1);
    std::size_t kym3, kym2, kym1, ky, kyp1, kyp2, kyp3;
    std::tie(kym3,kym2,kym1,ky,kyp1,kyp2,kyp3) = periodic_index(Nvy,k_y);

    return dv_weno( velocity , u[k_x][kym3][k_z][i] , u[k_x][kym2][k_z][i] , u[k_x][kym1][k_z][i] , u[k_x][ky][k_z][i] , u[k_x][kyp1][k_z][i] , u[k_x][kyp2][k_z][i] , u[k_x][kyp3][k_z][i] , u.step.dvy );
  }

  template < typename _T >
  _T
  weno_vz ( _T velocity , field3d<_T> const& u , std::size_t k_x , std::size_t k_y , std::size_t k_z , std::size_t i ) {
    const std::size_t Nvz = u.size(2);
    std::size_t kzm3, kzm2, kzm1, kz, kzp1, kzp2, kzp3;
    std::tie(kzm3,kzm2,kzm1,kz,kzp1,kzp2,kzp3) = periodic_index(Nvz,k_z);

    return dv_weno( velocity , u[k_x][k_y][kzm3][i] , u[k_x][k_y][kzm2][i] , u[k_x][k_y][kzm1][i] , u[k_x][k_y][kz][i] , u[k_x][k_y][kzp1][i] , u[k_x][k_y][kzp2][i] , u[k_x][k_y][kzp3][i] , u.step.dvz );
  }

}

#endif

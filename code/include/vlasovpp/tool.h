#ifndef _TOOL_H_
#define _TOOL_H_

template <typename Container>
auto
max_abs ( const Container & u )
{
  return std::abs(*std::max_element(
      std::begin(u) , std::end(u) ,
      [](const auto & a,const auto & b){ return std::abs(a)<std::abs(b); }
    ));
}

namespace factory
{

template <typename _T>
auto
printer__x_data ( _T x_min , _T x_step , std::size_t precision=15 )
{
  // define a printer which writes data depending of a direction define by its minimal value and step
  auto __printer_x_y = [=,count=0] (auto const& y) mutable {
    std::stringstream ss; ss.precision(precision);
    ss<< (count++)*x_step + x_min << " " << y;
    return ss.str();
  };
  return __printer_x_y;
}

}

#endif

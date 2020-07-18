#include "miMaS/config.h"

#define converter( type , str_to_type ) template <> \
type \
convertor::operator () ( std::string && key , type && default_value ) \
{ \
  auto it = map_config.find(std::move(key)); \
  if ( it != map_config.end() ) { return str_to_type(it->second); }\
  return std::move(default_value); \
}\
template <> \
type \
convertor::operator () ( std::string && key , const type & default_value ) \
{ \
  auto it = map_config.find(std::move(key)); \
  if ( it != map_config.end() ) { return str_to_type(it->second); }\
  return default_value; \
}

  converter(int,std::stoi)
  converter(long,std::stol)
  converter(unsigned long,std::stoul)
  converter(unsigned long long,std::stoull)
  converter(float,std::stof)
  converter(double,std::stod)
  converter(long double, std::stold)
#undef converter

template <>
std::string
convertor::operator () ( std::string && key , std::string && default_value )
{
  auto it = map_config.find(std::move(key));
  if ( it != map_config.end() ) { return it->second; }
  return std::move(default_value);
}
template <>
std::string
convertor::operator () ( std::string && key , const std::string & default_value )
{
  auto it = map_config.find(std::move(key));
  if ( it != map_config.end() ) { return it->second; }
  return default_value;
}

bool
config::create_output_directory () const
{ return fs::create_directories(output_dir); }

std::ostream &
operator << ( std::ostream & os , const config & c )
{
  os << "Nx " << c.Nx << "\n"
     << "Nv " << c.Nv << "\n"

     << "Nz "  << c.Nz << "\n"
     << "Nvx " << c.Nvx << "\n"
     << "Nvy " << c.Nvy << "\n"
     << "Nvz " << c.Nvz << "\n"

     << "Tc " << c.Tc << "\n"
     << "ui " << c.ui << "\n" 
     << "alpha " << c.alpha << "\n"

     << "Tf " << c.Tf << "\n"
     << "tol " << c.tol << "\n"
     << "output_dir " << c.output_dir.string();
  return os;
}

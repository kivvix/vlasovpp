#include "vlasovpp/config.h"

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

config::config( fs::path && path_config )
  : name("")
{
  using namespace std::string_literals;

  convertor convert(std::move(path_config));

  dt0 = convert("dt0",0.1);

  Nx = convert("Nx",135);
  Nv = convert("Nv",256);

  Nz  = convert("Nz" ,27);
  Nvx = convert("Nvx",16);
  Nvy = convert("Nvy",16);
  Nvz = convert("Nvz",27);

  Tc = convert("Tc",0.01);
  ui = convert("ui",3.4);

  B0 = convert("B0",1.0);

  alpha = convert("alpha",0.2);

  nh     = convert("nh"    ,0.2);
  v_perp = convert("v_perp",0.6);
  v_par  = convert("v_par" ,0.2);
  K      = convert("K"     ,2.0);

  Tf = convert("Tf",10.0);
  tol = convert("tol",1e-5);

  std::stringstream data; data << convert("snaptimes","50 100 150"s); // get string of snaptimes, default times are 50 100 and 150
  std::transform(
      std::istream_iterator<std::string>(data) , // iterator on stringstream which contains snapshot times
      std::istream_iterator<std::string>()     , // end iterator on input stream (this is a default iterator)
      std::back_inserter(snaptimes)            , // insert at the end of snaptimes vector
      [] ( std::string const& s ) {              // lambda function which converts string to double
        return std::stod(s);
      }
    );
  std::sort( std::begin(snaptimes) , std::end(snaptimes) );

  output_dir = convert("output_dir","."s);
}

config::config( int argc , const char** argv , std::string name_ )
  : name(name_)
{
  using namespace std::string_literals;

  fs::path path_config = "config.init";
  if ( argc>1 ) {
    path_config = argv[1];
  }

  convertor convert(std::move(path_config));

  dt0 = convert("dt0",0.1);

  Nx = convert("Nx",135);
  Nv = convert("Nv",256);

  Nz  = convert("Nz" ,27);
  Nvx = convert("Nvx",16);
  Nvy = convert("Nvy",16);
  Nvz = convert("Nvz",27);

  Tc = convert("Tc",0.01);
  ui = convert("ui",3.4);

  B0 = convert("B0",1.0);

  alpha = convert("alpha",0.2);

  nh     = convert("nh"    ,0.2);
  v_perp = convert("v_perp",0.6);
  v_par  = convert("v_par" ,0.2);
  K      = convert("K"     ,2.0);

  Tf = convert("Tf",10.0);
  tol = convert("tol",1e-5);

  std::stringstream data; data << convert("snaptimes","50 100 150"s); // get string of snaptimes, default times are 50 100 and 150
  std::transform(
      std::istream_iterator<std::string>(data) , // iterator on stringstream which contains snapshot times
      std::istream_iterator<std::string>()     , // end iterator on input stream (this is a default iterator)
      std::back_inserter(snaptimes)            , // insert at the end of snaptimes vector
      [] ( std::string const& s ) {              // lambda function which converts string to double
        return std::stod(s);
      }
    );
  std::sort( std::begin(snaptimes) , std::end(snaptimes) );

  output_dir = convert("output_dir","."s);
}

bool
config::create_output_directory () const
{ return fs::create_directories(output_dir); }

std::ostream &
operator << ( std::ostream & os , const config & c )
{
  if ( !c.name.empty() ) { os << "name " << c.name <<"\n"; }
  
  os << "dt0 " << c.dt0 << "\n"

     << "Nx " << c.Nx << "\n"
     << "Nv " << c.Nv << "\n"

     << "Nz "  << c.Nz << "\n"
     << "Nvx " << c.Nvx << "\n"
     << "Nvy " << c.Nvy << "\n"
     << "Nvz " << c.Nvz << "\n"

     << "Tc " << c.Tc << "\n"
     << "ui " << c.ui << "\n"

     << "B0 " << c.B0 << "\n"

     << "alpha " << c.alpha << "\n"

     << "nh " << c.nh << "\n"
     << "v_perp " << c.v_perp << "\n"
     << "v_par "  << c.v_par  << "\n"
     << "K " << c.K << "\n"

     << "Tf " << c.Tf << "\n"
     << "tol " << c.tol << "\n";
  os << "snaptimes ";
  std::copy( std::begin(c.snaptimes) , std::end(c.snaptimes) , std::ostream_iterator<double>(os," ") );
  os << "\n"
     << "output_dir " << c.output_dir.string();
  return os;
}

using namespace std::string_literals;

void
save_config ( const config & c ) {
  std::ofstream of( c.output_dir / ("config_"s + c.name + ".init"s) );
  of << c << "\n";
  of.close();
}

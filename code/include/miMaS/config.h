#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include <fstream>
#include <map>
#include <tuple>
#include <utility>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <vector>

#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  else
#    error "no filesystem support ='("
#  endif
#endif

/**
  struct convertor
  read config file of "key value" and store it in map<string,string>
  and can convert any string into any base type with operator()
  example :
  ```cpp
    convertor convert("config");
    int i = convert("i",42); // if `i` isn't in config file, use the default value `42`
  ```
**/
struct convertor {
  std::map<std::string,std::string> map_config;

  convertor ()
  { ; }
  template < typename OPENABLE >
  convertor ( OPENABLE input )
  {
    std::ifstream ifs(input);
    std::string key,value;
    while ( ifs >> key >> value ) {
      map_config[key] = value;
    }
    ifs.close();
  }

  // default operator () for unknow type
  // for other type see config.cc

  template < typename _T >
  _T operator () ( std::string && key , _T && default_value )
  { return std::move(default_value); }

  template < typename _T >
  _T operator () ( std::string && key , const _T & default_value )
  { return default_value; }
};

#define converter( type ) template <> \
type \
convertor::operator () ( std::string && , type && ); \
template <> \
type \
convertor::operator () ( std::string && , const type & )

converter(int);
converter(long);
converter(unsigned long);
converter(unsigned long long);
converter(float);
converter(double);
converter(long double);
converter(std::string);
#undef converter

/**
  struct config
  store configuration of simulation and some function to export data
**/
struct config {
  std::string name;
  std::size_t Nx, Nv; // for 1dx1dv
  std::size_t Nz, Nvx,Nvy,Nvz; // for 1dx3dv
  double Tc;
  double Tf;
  double ui;
  double alpha;
  double tol;
  fs::path output_dir;

  template <typename OPENABLE>
  config ( OPENABLE path_config )
    : name("")
  {
    using namespace std::string_literals;

    convertor convert(path_config);

    Nx = convert("Nx",135);
    Nv = convert("Nv",256);

    Nz  = convert("Nz" ,27);
    Nvx = convert("Nvx",16);
    Nvy = convert("Nvy",16);
    Nvz = convert("Nvz",27);
    
    Tc = convert("Tc",0.01);
    ui = convert("ui",3.4);
    alpha = convert("alpha",0.2);

    Tf = convert("Tf",10.0);
    tol = convert("tol",1e-5);
    output_dir = convert("output_dir","."s);
  }

  bool
  create_output_directory () const;
};

// export configuration
std::ostream &
operator << ( std::ostream & , const config & );

// temporary object to prepare export of data
// just get a filename, a container and a function to write into file
namespace monitoring
{

template < typename Container , typename Writer >
struct data
{
  fs::path file;
  const Container * dat;
  Writer writer;

  data ( fs::path && _file , const Container & _dat ,  Writer _writer )
    : file(std::move(_file)) , dat(&_dat) , writer(_writer)
  { ; }
};

template < typename Container , typename Writer >
data<Container,Writer>
make_data ( fs::path && _file , const Container & _dat , Writer _writer ) {
  return data<Container,Writer>(std::move(_file),_dat,_writer);
}

template < typename Container , typename Writer >
struct reactive_monitoring
{
  std::ofstream file;
  const Container * ptr_time;
  std::vector<Container *> arr_data;
  std::size_t index;

  template < typename FileConstructible >
  reactive_monitoring ( FileConstructible && _file , const Container & _time , std::initializer_list<Container*> _arr_data )
    : file(std::move(_file)) , ptr_time(&time) , arr_data(_arr_data.size()) , index(0)
  {
    auto it = arr_data.begin();
    for ( auto ptr : _arr_data ) {
      *it = ptr;
      ++it;
    }
  }

  void
  push () {
    std::transform( std::begin(*ptr_time)+index , std::end(*ptr_time) ,
        std::ostream_iterator<std::string>(file,"\n") ,
        [&]( const auto & t ) mutable {
          std::stringstream ss; ss << t;
          for ( const auto & data : arr_data ) {
            ss << " " << data->at(index);
          }
          ++index;
          return ss.str();
        }
      );
    file << std::flush;
  }
};

} // namespace monitoring

/** export data into output file, example :
  ```cpp
    config c("config");

    // i_y is a lambda function to write "index data", and it will be call on each element of container `ee`
    i_y = [count=0]( const auto & y ) mutable { std::stringstream ss; ss<<count++<<" "<<y; return ss.str(); };
    c << monitoring::data( "ee.dat" , ee , i_y );
  ```
**/
template < typename Container , typename Writer >
void
operator << ( const config & c , const monitoring::data<Container,Writer> & dat )
{
  std::ofstream of( c.output_dir / dat.file );
  std::transform( std::begin(*dat.dat) , std::end(*dat.dat) , std::ostream_iterator<std::string>(of,"\n") , dat.writer );
  of.close();
}

#endif

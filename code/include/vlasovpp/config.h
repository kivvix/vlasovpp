#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include <fstream>
#include <sstream>
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
     //namespace fs = std::__fs::filesystem;
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
  //template < typename OPENABLE >
  //convertor ( OPENABLE input )
  convertor ( fs::path && input )
  {
    std::ifstream ifs(std::move(input));
    // it is not efficiency code but it has no dependance to boost or other lib
    // and we don't care because this is just to read configuration file
    for ( std::string line ; std::getline(ifs,line) ; ) {
      std::stringstream sline; sline<<line; // convert line to stringstream
      std::stringstream key,value;
      sline.get(*key.rdbuf(),' '); sline.get(*value.rdbuf()); // get first field in key, and the rest into value (even if there is spaces)
      map_config[key.str()] = value.str();
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
  double dt0;
  std::size_t Nx, Nv; // for 1dx1dv
  std::size_t Nz, Nvx,Nvy,Nvz; // for 1dx3dv
  double Tc;
  double Tf;
  double ui;
  double B0;
  double alpha;
  double nh;
  double v_par,v_perp;
  double K;
  double tol;
  std::vector<double> snaptimes;
  fs::path output_dir;

  config ( fs::path && );

  bool
  create_output_directory () const;
};

// export configuration
std::ostream &
operator << ( std::ostream & , const config & );

namespace monitoring
{

// temporary object to prepare export of data
// just get a filename, a container and a function to write into file
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

// for older version of `g++` without indication of type like `data<std::vector<double>>`
// this function works like `std::make_pair`
template < typename Container , typename Writer >
data<Container,Writer>
make_data ( fs::path && _file , const Container & _dat , Writer _writer ) {
  return data<Container,Writer>(std::move(_file),_dat,_writer);
}

// class to frequently push data in output files 
// /!\ WARNING all containers to save needs to be of the same type
template < typename Container >
struct reactive_monitoring
{
  std::ofstream file;
  const Container * ptr_time; // pointer to column pointer
  std::vector<Container *> arr_data; // vector of pointer to all contairers to save
  std::size_t index; // index of the last data pushed (we can't save an iterator because iterator are illed when values are pushed)

  template < typename FileConstructible >
  reactive_monitoring ( FileConstructible && _file , const Container & _time , std::initializer_list<Container*> _arr_data )
    : file(std::move(_file)) , ptr_time(&_time) , arr_data(_arr_data.size()) , index(0)
  {
    // copy pointer to each container in `arr_data`
    auto it = arr_data.begin();
    for ( auto ptr : _arr_data ) {
      *it = ptr;
      ++it;
    }
  }
  ~reactive_monitoring ()
  {
    file.close(); // don't forget to close the file
  }

  // 'cause every time we push, I get this feeling...
  // Every time a `reactive_monitoring` is `push` it print data since the last `index` value to the end of containers
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

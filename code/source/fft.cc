#include "vlasovpp/fft.h"

bool spectrum_::initied = []() {
#ifdef _GLIBCXX_PARALLEL
  return static_cast<bool>(fftw_init_threads(void));
#else
  return false;
#endif
}();


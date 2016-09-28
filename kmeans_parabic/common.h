#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <limits>
//
extern char kSegmentFaultCauser[];

#define CHECK(a) if (!(a)) {                                            \
    std::cerr << "CHECK failed "                                        \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_EQ(a, b) if (!((a) == (b))) {                             \
    std::cerr << "CHECK_EQ failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GT(a, b) if (!((a) > (b))) {                              \
    std::cerr << "CHECK_GT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LT(a, b) if (!((a) < (b))) {                              \
    std::cerr << "CHECK_LT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GE(a, b) if (!((a) >= (b))) {                             \
    std::cerr << "CHECK_GE failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LE(a, b) if (!((a) <= (b))) {             \
    std::cerr << "CHECK_LE failed "                     \
              << __FILE__ << ":" << __LINE__ << "\n"    \
              << #a << " = " << (a) << "\n"             \
              << #b << " = " << (b) << "\n";            \
    *kSegmentFaultCauser = '\0';                        \
  }                                                     \
                                                        \


enum LogSeverity { INFO, WARNING, ERROR, FATAL };

class Logger {
 public:
  Logger(LogSeverity ls, const std::string& file, int line)
      : ls_(ls), file_(file), line_(line)
  {}
  std::ostream& stream() const {
    return std::cerr << file_ << " (" << line_ << ") : ";
  }
  ~Logger() {
    if (ls_ == FATAL) {
      *::kSegmentFaultCauser = '\0';
    }
  }
 private:
  LogSeverity ls_;
  std::string file_;
  int line_;
};

#define LOG(ls) Logger(ls, __FILE__, __LINE__).stream()

// Basis POD types.
typedef int                 int32;
#ifdef COMPILER_MSVC
typedef __int64             int64;
#else
typedef long long           int64;
#endif

namespace kmeanst {
using std::vector;
using std::string;
using std::sqrt;
using std::istringstream;
using std::ifstream;
struct IndexValue {
  int index;
  double value;
  IndexValue() {}
  IndexValue(int i, double v) : index(i), value(v) {
  }
};
// Generate a random float value in the range of [0,1) from the
// uniform distribution.
inline double RandDouble() {
  return rand() / static_cast<double>(RAND_MAX);
}

// Generate a random integer value in the range of [0,bound) from the
// uniform distribution.
inline int RandInt(int bound) {
  // NOTE: Do NOT use rand() % bound, which does not approximate a
  // discrete uniform distribution will.
  return static_cast<int>(RandDouble() * bound);
}

// Steaming output facilities for GSL matrix, GSL vector and STL
// vector.
std::ostream& operator << (std::ostream& out, vector<double>& v);


//void IntToString(int i32, string* key);
//int StringToInt(const char* key, int size);
//void Int64ToString(int64 i64, string* key);
//int64 StringToInt64(const char* key, int size);

//void DoubleToString(double d, string* key);
//double StringToDouble(const char* key, int size);

// A maximum heap to store only Top N maximum elements of inserted elements.
// struct Cmp {bool operator()(double a, double b) {return a > b;}}
// TopN n(2); n.Insert(5); n.Insert(3); n.Insert(2);
// vector<double> v; n.Extract(&v);
// CHECK_EQ(3, v[0]);
// CHECK_EQ(5, v[1]);

}  // namespace learning_psc

#endif  // _OPENSOURCE_PSC_COMMON_H__

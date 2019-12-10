#pragma once
#ifndef normal_h
#define normal_h
#include "init.hpp"
class Normal : public Init {
   public:
    Normal(dtype, dtype);
    Matrix weights(int, int) const override;

   private:
    dtype _mean;
    dtype _std;
};
#endif

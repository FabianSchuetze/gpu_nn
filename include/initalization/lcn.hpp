#pragma once
#ifndef lcn_h
#define lcn_h
#include "init.hpp"
class LCN : public Init {
   public:
    LCN(dtype);
    Matrix weights(int, int) const override;

   private:
    dtype _value;
};
#endif

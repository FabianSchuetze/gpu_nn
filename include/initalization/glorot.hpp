#pragma once
#ifndef glorot_h
#define glorot_h
#include "init.hpp"
class Glorot : public Init {
   public:
    Glorot() : Init("Glorot") { ; };
    Matrix weights(int, int) const override;
};
#endif

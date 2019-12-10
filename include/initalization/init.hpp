#pragma once
#ifndef init_h
#define init_h
#include <string>
#include "../common.h"
class Init {
   public:
    explicit Init(const std::string& name) : _name(name) { ; };
    virtual Matrix weights(int, int) const = 0;

   protected:
    std::string _name;
};
#endif

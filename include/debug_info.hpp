#ifndef debug_info_hpp
#define debug_info_hpp
#include <algorithm>
#include <fstream>
#include <memory>
#include <deque>
//#include "network.h"
#include "layer/layer.h"
#include "storage.h"

class DebugInfo {
   public:
    DebugInfo(std::string, std::string);
    void backward_debug_info(const std::vector<std::shared_ptr<Storage>>&,
                             const std::deque<std::shared_ptr<Layer>>&, int);
    void forward_debug_info(const std::vector<std::shared_ptr<Storage>>&,
                            const std::deque<std::shared_ptr<Layer>>&, int);
    void print_layers(const std::deque<std::shared_ptr<Layer>>& layers);
    bool is_set();

   private:
    typedef std::shared_ptr<Storage> SharedStorage;
    std::ofstream _ffw_stream;
    std::ofstream _bwd_stream;
    bool _is_set;

    void _print_layers(const std::deque<std::shared_ptr<Layer>>& layers,
                       std::ostream& stream);
};
#endif

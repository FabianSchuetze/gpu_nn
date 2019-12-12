#include "../../include/layer/input.h"

Input::Input(Features features)
    : Layer("Input"),
      _features(features),
      _channels(0),
      _img(0, 0) {
    _out_dim.push_back(_features.get());
}

Input::Input(Channels channels, ImageShape img)
    : Layer("Input"), _features(0), _channels(channels), _img(img) {
    _out_dim.push_back(_channels.get());
    _out_dim.push_back(_img.first());
    _out_dim.push_back(_img.second());
};

void Input::forward_gpu(const std::shared_ptr<Storage>&,
                        std::shared_ptr<Storage>&, const std::string&) {}
void Input::forward_cpu(const std::shared_ptr<Storage>&,
                        std::shared_ptr<Storage>&, const std::string&) {}
void Input::backward_gpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {};
void Input::backward_cpu(const SharedStorage&, const SharedStorage&,
                         SharedStorage&) {};


//std::vector<int> Input::output_dimension() {
    //return _out_dim;
//}

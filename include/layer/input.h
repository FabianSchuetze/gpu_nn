#pragma once
#ifndef input_h
#define input_h
#include "layer.h"
class Input : public Layer {
   public:
    explicit Input(Features);
    Input(Channels, ImageShape);
    virtual ~Input() = default;
    //std::vector<int> output_dimension() const override;
    void forward_gpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void forward_cpu(const std::shared_ptr<Storage>&,
                     std::shared_ptr<Storage>&, const std::string&) override;
    void backward_gpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;
    void backward_cpu(const SharedStorage&, const SharedStorage&,
                      SharedStorage&) override;

   private:
    Features _features;
    Channels _channels;
    ImageShape _img;
};
#endif

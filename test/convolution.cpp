#include <eigen-git-mirror/Eigen/Core>
//#define CATCH_CONFIG_MAIN
#include "../third_party/catch/catch.hpp"
#include "../include/common.h"
#include "../include/neural_network.h"
#include <sys/time.h>
int main() {
    srand((unsigned int) time(0));
    Layer* inp1 = new Convolution(FilterShape(2,2), Pad(1), Stride(1),
            Filters(3), ImageShape(5,5), Channels(3));
    //Softmax s1;
    //inp1 = &s1;
    Matrix in = Matrix::Random(6, 5);
    Matrix out = Matrix::Zero(6, 5);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    inp1->forward_gpu(storage_in, storage_out, "train");
    Vector sum = storage_out->return_data_const().colwise().sum();
}

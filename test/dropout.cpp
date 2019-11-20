#define CATCH_CONFIG_MAIN
#include "../include/neural_network.h"
//#include "../include/layer/dense.h"
#include <eigen-git-mirror/Eigen/Core>
#include <memory>
//#include "../include/common.h"
//#include "../include/layer/layer.h"
#include "../third_party/catch/catch.hpp"
//#include "../include/math.h"
#include <sys/time.h>
//#include "../include/storage.h"

using std::make_shared;
using std::shared_ptr;
using std::vector;
typedef std::shared_ptr<Storage> SharedStorage;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
TEST_CASE("Dense forward_gpu", "[gpu]") {
    srand((unsigned int)time(0));
    // NOT A GOOD IDEAS AS IT REQUIRES TO KNOW THE NUMBER OF ELEMENTS
    // BEFOREHAND
    Layer* inp1 = new Dropout(5, 5, 0.5);
    Matrix in = Matrix::Random(5, 5);
    Matrix out = Matrix::Zero(5, 5);
    //dtype begin = out(0, 0);
    std::shared_ptr<Storage> storage_in = std::make_shared<Storage>(in);
    std::shared_ptr<Storage> storage_out = std::make_shared<Storage>(out);
    std::cout << storage_in->return_data_const() << std::endl;
    inp1->forward_gpu(storage_in, storage_out);
    std::cout << storage_out->return_data_const() << std::endl;
    //dtype end = storage_out->return_data_const()(0, 0);
    //REQUIRE(begin != end);
}

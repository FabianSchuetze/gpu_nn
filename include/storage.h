#ifndef storage_h
#define storage_h
#include <eigen-git-mirror/Eigen/Dense>
#include <memory>
#include "common.h"
class Storage {
   public:
    Storage();
    Storage(const Matrix&);
    // Storage(const Eigen::MatrixXd&&) // I need to provide that!
    ~Storage();
    const dtype* cpu_pointer_const();
    const dtype* gpu_pointer_const();
    void update_cpu_data(Matrix);
    void update_gpu_data(const dtype);
    dtype* cpu_pointer();
    dtype* gpu_pointer();
    int get_rows() { return _data.rows(); }
    int get_cols() { return _data.cols(); }
    Matrix& return_data();
    const Matrix& return_data_const();

   private:
    Matrix _data;
    dtype* _cpu_pointer;
    dtype* _gpu_pointer;
    std::string recent_head;
    void initialize_gpu_memory();
    void sync_to_cpu();
    void sync_to_gpu();
};
#endif

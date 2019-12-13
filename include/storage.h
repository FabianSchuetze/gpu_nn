#ifndef storage_h
#define storage_h
#include <eigen-git-mirror/Eigen/Dense>
#include <memory>
#include "common.h"
class Storage {
   public:
    explicit Storage();
    explicit Storage(const Matrix&);
    // Storage(const Storage&) = delete;
    Storage& operator=(Storage other) = delete;
    // Storage(const Eigen::MatrixXd&&) // I need to provide that!
    ~Storage();
    const dtype* cpu_pointer_const();
    const dtype* gpu_pointer_const();
    void update_cpu_data(const dtype*);
    void update_cpu_data(Matrix);
    void update_gpu_data(const dtype);
    void update_gpu_data(const dtype*);
    void update_gpu_data(const dtype*, const unsigned int, const unsigned int);
    dtype* cpu_pointer();
    dtype* gpu_pointer();
    int get_rows() { return _data.rows(); }
    int get_cols() { return _data.cols(); }
    Matrix& return_data();
    const Matrix& return_data_const();
    Matrix copy_data();
    bool is_set();

   private:
    Matrix _data;
    dtype* _cpu_pointer;
    dtype* _gpu_pointer;
    std::string recent_head;
    void initialize_gpu_memory();
    void sync_to_cpu();
    void sync_to_gpu();
};

bool same_size(const std::shared_ptr<Storage>&,
               const std::shared_ptr<Storage>&);
#endif

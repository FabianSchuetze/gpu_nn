#ifndef storage_h
#define storage_h
#include <eigen-git-mirror/Eigen/Dense>
class Storage {
   public:
    Storage(const Eigen::MatrixXd&);
    ~Storage();
    const double* cpu_pointer_const();
    const double* gpu_pointer_const();
    double* cpu_pointer();
    double* gpu_pointer();
    int get_rows() { return _data.rows(); }
    int get_cols() { return _data.cols(); }
    Eigen::MatrixXd return_data();
    const Eigen::MatrixXd return_data_const();

   private:
    Eigen::MatrixXd _data;
    void initialize_gpu_memory();
    void sync_to_cpu();
    void sync_to_gpu();
    double* _cpu_pointer;
    double* _gpu_pointer;
    std::string recent_head;
};
#endif

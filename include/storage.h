#ifndef storage_h
#define storage_h
#include <eigen-git-mirror/Eigen/Dense>
class Storage{
    public:
        Storage(const Eigen::MatrixXd&);
        double* cpu_pointer() {return _cpu_pointer;};
        double* gpu_pointer() {return _gpu_pointer;};
        int get_rows() {return _data.rows();}
        int get_cols() {return _data.cols();}
        void copy_to_cpu();
        Eigen::MatrixXd return_data() {return _data;};
    private:
        Eigen::MatrixXd _data;
        void initialize_gpu_memory();
        double* _cpu_pointer;
        double* _gpu_pointer;

};

#endif

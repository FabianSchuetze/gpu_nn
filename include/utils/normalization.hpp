#include <eigen-git-mirror/Eigen/Core>
#include "../common.h"
// typedef float dtype;
// typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;

class GCN {
   public:
    GCN(int rows, int cols, int channels);
    Matrix transform(const Matrix&);

   private:
    void reshape_images(const Matrix&);
    Matrix reshape(Matrix&);
    Matrix inv_reshape(Matrix&, int);

    int _rows;
    int _cols;
    int _channels;
};

class StandardNormalization {
   public:
    StandardNormalization();
    Matrix transform(const Matrix&);

   private:
    typedef Eigen::RowVector<float, Eigen::Dynamic> RowVector;
    Vector colwise_std(const Matrix& diff);
};

class ZCAWhitening {
   public:
    ZCAWhitening();
    void fit(const Matrix&);
    Matrix transform(const Matrix&);

   private:
    void compute_covariance(const Eigen::MatrixXd&);
    void solve_eigensystem(const Eigen::MatrixXd& varcov);
    Eigen::MatrixXd eigenvectors;
    Eigen::MatrixXd eigenvalues;
    Eigen::MatrixXd cov;
    Eigen::MatrixXd weights;
};

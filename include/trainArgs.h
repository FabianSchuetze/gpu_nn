#ifndef GUARD_trainArgs_h
#define GUARD_trainArgs_h
#include <fstream>
#include <list>
#include <memory>
#include <vector>
#include "common.h"
#include "gradient_descent/gradient_descent.h"
#include "storage.h"
#include "threadsafe_queue.hpp"

class trainArgs {
   private:
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;
    typedef std::shared_ptr<Storage> SharedStorage;

   public:
    trainArgs() = default;
    trainArgs(const Matrix&, const Matrix&, Epochs, Patience, BatchSize,
              std::shared_ptr<GradientDescent>&,
              std::deque<std::shared_ptr<Layer>>&,
              Shuffle shuffle);
    int iter_since_update() { return _iter_since_update; }
    void reset_iter_since_update() { _iter_since_update = 0; }
    void advance_iter_since_update() { _iter_since_update++; };
    int patience() { return _patience; }
    void reset_total_iter() { _total_iter = 0; }
    int cummulative_iter() { return _cum_iter; }
    void advance_cumulative_iter(int step) { _cum_iter += step; }
    int total_iter() { return _total_iter; }
    void advance_total_iter() { _total_iter += _batch_size; };
    int max_total_iter() { return _x_train.rows(); }
    int epochs() { return _epochs; }
    int current_epoch() { return _current_epoch; }
    void advance_epoch() { _current_epoch++; }
    int batch_size() { return _batch_size; }
    dtype& best_error() { return _best_error; }
    const Matrix& x_train() { return _x_train; }
    const Matrix& y_train() { return _y_train; }
    const Matrix& x_val() { return _x_val; }
    const Matrix& y_val() { return _y_val; }
    const SharedStorage& y_val_shared() { return _y_val_shared; }
    std::vector<std::vector<SharedStorage>>& optimizer() { return _optimizer; }
    threadsafe_queue<std::pair<SharedStorage, SharedStorage>> data_queue;
    const bool shuffle() {return _shuffle;}

   private:
    Matrix _x_train;
    Matrix _x_val;
    Matrix _y_train;
    Matrix _y_val;
    SharedStorage _y_val_shared;
    std::vector<std::vector<SharedStorage>> _optimizer;

    int _iter_since_update;
    int _total_iter;
    int _current_epoch;
    int _cum_iter;
    dtype _best_error;
    int _batch_size;
    int _epochs;
    int _patience;
    bool _shuffle;
    void train_test_split(const Matrix&, const Matrix&, dtype);
    void create_optimizers(const std::shared_ptr<GradientDescent>&,
                           const std::deque<std::shared_ptr<Layer>>&);
};
#endif

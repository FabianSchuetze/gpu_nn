#ifndef GUARD_trainArgs_h
#define GUARD_trainArgs_h
#include <memory>
#include <vector>
#include "common.h"
#include "storage.h"
#include "threadsafe_queue.hpp"

class trainArgs {
   private:
    typedef std::vector<std::shared_ptr<Storage>> VecSharedStorage;
    typedef std::shared_ptr<Storage> SharedStorage;

   public:
    trainArgs() = default;
    trainArgs(const Matrix&, const Matrix&, Epochs, Patience, BatchSize);
    //int iter_since_update() { return _iter_since_update; }
    //void reset_iter_since_update() { _iter_since_update = 0; }
    void reset_total_iter() { _total_iter = 0; }
    int total_iter() { return _total_iter; }
    void advance_total_iter() { _total_iter += _batch_size;};
    int max_total_iter() {return _x_train.rows();}
    int epochs() { return _epochs; }
    int current_epoch() { return _current_epoch; }
    void advance_epoch() { _current_epoch++; }
    int patience() { return _patience; }
    int batch_size() { return _batch_size; }
    //dtype best_erorr() { return _best_error; }
    const Matrix& x_train() { return _x_train; }
    const Matrix& y_train() { return _y_train; }
    const Matrix& x_val() { return _x_val; }
    const Matrix& y_val() { return _y_val; }
    const SharedStorage& y_val_shared() {return _y_val_shared;}
    threadsafe_queue<std::pair<SharedStorage, SharedStorage>> data_queue;

   private:
    Matrix _x_train;
    Matrix _x_val;
    Matrix _y_train;
    Matrix _y_val;

    SharedStorage _y_val_shared;
    
    //int _iter_since_update;
    int _total_iter;
    int _current_epoch;
    //int _not_improved;
    //dtype _best_error;
    //dtype _current_error;
    int _batch_size;
    int _epochs;
    int _patience;
    void train_test_split(const Matrix&, const Matrix&, dtype);
};
#endif

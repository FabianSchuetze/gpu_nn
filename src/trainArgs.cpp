#include "../include/trainArgs.h"
#include <memory>
#include "../include/network.h"
using Eigen::all;
using std::make_shared;
using std::vector;

trainArgs::trainArgs(const Matrix& features, const Matrix& target,
                     Epochs __epochs, Patience __patience,
                     BatchSize __batch_size,
                     std::shared_ptr<GradientDescent>& sgd,
                     std::deque<std::shared_ptr<Layer>>& layers)
    : _x_train(),
      _x_val(),
      _y_train(),
      _y_val(),
      _iter_since_update(0),
      _total_iter(0),
      _current_epoch(0),
      //_not_improved(0),
      _best_error(std::numeric_limits<double>::infinity()),
      //_current_error(0),
      _batch_size(__batch_size.get()),
      _epochs(__epochs.get()),
      _patience(__patience.get()) {
    train_test_split(features, target, 0.1);
    _y_val_shared = std::make_shared<Storage>(_y_val.transpose());
    create_optimizers(sgd, layers);
    // create_backup(layers);
};

void trainArgs::create_backup(const std::deque<std::shared_ptr<Layer>>& layers) {
    for (std::shared_ptr<Layer> layer : layers) {
        if (layer->n_paras() > 0) {
            for (SharedStorage store : layer->return_parameters()) {
                Matrix tmp = store->copy_data();
                _param_bkp.push_back(std::make_shared<Storage>(tmp));
            }
        }
    }
}

void trainArgs::create_optimizers(
    const std::shared_ptr<GradientDescent>& sgd,
    const std::deque<std::shared_ptr<Layer>>& layers) {
    if (sgd->name() == "Momentum")
        for (std::shared_ptr<Layer> layer : layers) {
            if (layer->n_paras() > 0) {
                std::vector<SharedStorage> helper;
                for (SharedStorage store : layer->return_gradients()) {
                    int rows = store->get_rows();
                    int cols = store->get_cols();
                    Matrix tmp = Matrix::Zero(rows, cols);
                    helper.push_back(std::make_shared<Storage>(tmp));
                }
                _optimizer.push_back(helper);
            }
        }
}

void trainArgs::train_test_split(const Matrix& features, const Matrix& target,
                                 dtype validation_fraction) {
    vector<int> indices(features.rows());
    for (int i = 0; i < features.rows(); i++) indices[i] = i;
    int cutoff = indices.size() * (1 - validation_fraction);
    vector<int> train_set(indices.begin(), indices.begin() + cutoff);
    vector<int> test_set(indices.begin() + cutoff, indices.end());
    Matrix test = features(train_set, all);
    _x_train = features(train_set, all);
    _y_train = target(train_set, all);
    if (validation_fraction > 0.) {
        _x_val = features(test_set, all);
        _y_val = target(test_set, all);
    }
}

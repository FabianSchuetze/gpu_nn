#ifndef threadsafe_queue_hpp
#define threadsafe_queue_hpp
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template <typename T>
class threadsafe_queue {
   private:
    mutable std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;

   public:
    threadsafe_queue() {}
    // threadsafe_queue(threadsafe_queue const& other) = delete;
    void push(T new_value) {
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push(new_value);
        data_cond.notify_one();
    }

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        value = data_queue.front();
        //data_queue.pop();
    }

    void pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_queue.pop();
    }

    T wait_and_pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        return data_queue.front();
        //return r<T> res(std::make_shared<T>(data_queue.front()));
        //data_queue.pop();
        //return res;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) return false;
        value = data_queue.front();
        data_queue.pop();
        return true;
    }
    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) return std::shared_ptr<T>();
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    }
    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.size();
    }
};
#endif

#pragma once

#include <functional>
// #include <memory>

struct hardware_context_mt {
    void (*lock_block)(int bid);
    int (*mark_block_visit)(int bid, int tid);
    void (*unlock_block)(int bid);
    void (*wait_block_sync)(int bid, int visited,
                            std::function<void()> callable);
    void (*lock_all)();
    int (*mark_all_visit)(int bid, int tid);
    void (*unlock_all)();
    void (*wait_all_sync)(int visited, std::function<void()> callable);
    void (*init)();
};

class hardware_context {
  public:
    // hardware_context(hardware_context_mt *impl) : impl_(impl){};
    void lock_block(int bid) { impl_->lock_block(bid); }
    int mark_block_visit(int bid, int tid) {
        return impl_->mark_block_visit(bid, tid);
    }
    void unlock_block(int bid) { impl_->unlock_block(bid); }
    void wait_block_sync(
        int bid, int visited,
        std::function<void()> callable = []() -> void {}) {
        impl_->wait_block_sync(bid, visited, callable);
    }
    void lock_all() { impl_->lock_all(); }
    int mark_all_visit(int bid, int tid) {
        return impl_->mark_all_visit(bid, tid);
    }
    void unlock_all() { impl_->unlock_all(); }
    void wait_all_sync(
        int visited, std::function<void()> callable = []() -> void {}) {
        impl_->wait_all_sync(visited, callable);
    }
    void *global_var = nullptr;

    hardware_context_mt* impl_;
};

static hardware_context global_hardware_ctx;

void global_hardware_init(hardware_context_mt *impl) {
    global_hardware_ctx.impl_ = impl;
    impl->init();
}
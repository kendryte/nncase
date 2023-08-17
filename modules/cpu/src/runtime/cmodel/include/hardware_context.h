#pragma once

#include <iostream>
#include <memory>
#include <functional>

struct hardware_context_impl;

class hardware_context {
  public:
    hardware_context();
    void lock_block(int bid);
    int mark_block_visit(int bid, int tid);
    void unlock_block(int bid);
    void wait_block_sync(
        int bid, int visited, std::function<void()> callable = []() -> void {});
    void lock_all();
    int mark_all_visit(int bid, int tid);
    void unlock_all();
    void wait_all_sync(
        int visited, std::function<void()> callable = []() -> void {});
    void *global_var = nullptr;

  private:
    std::unique_ptr<hardware_context_impl> impl_;
};

extern std::unique_ptr<hardware_context> global_hardware_ctx;

void global_hardware_init();
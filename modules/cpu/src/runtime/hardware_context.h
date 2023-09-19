#pragma once

#include <functional>
#include <memory>
#include <nncase/runtime/cpu/compiler_defs.h>

struct hardware_context_impl;

BEGIN_NS_NNCASE_RT_MODULE(cpu)
namespace hardware_context {
void init();

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

extern std::unique_ptr<hardware_context_impl> impl_;
} // namespace hardware_context
END_NS_NNCASE_RT_MODULE
#pragma once

#include <cstddef>

class thread_context {
  public:
    thread_context(size_t bid, size_t tid) : bid_(bid), tid_(tid) {}

    size_t bid() noexcept { return bid_; }

    size_t tid() noexcept { return tid_; }

  private:
    size_t bid_;
    size_t tid_;
};
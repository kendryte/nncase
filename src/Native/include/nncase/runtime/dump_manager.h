#pragma once
#include <fstream>
#include <iostream>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>
#include <sstream>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API dump_manager {

  private:
    bool append_;
    int count_ = 1;
    std::string current_op_;
    std::string dump_root_;

  public:
    void set_current_op(const std::string &op) { current_op_ = op; }

    std::string get_current_op() { return current_op_; }

    std::string dump_path();

    std::string get_dump_root() { return dump_root_; }

    std::ofstream get_stream() { return get_stream(dump_path()); }

    std::ofstream get_stream(const std::string &path);

    int get_count() { return count_; }

    void incr_count() {
        count_++;
        append_ = false;
    }

    void set_append(bool app) { append_ = app; }

    void set_dump_root(std::string root);

    void dump_op(nncase::runtime::stackvm::tensor_function_t tensor_funct);

    void dump_op(const std::string &op);

    void dump_output(nncase::value_t value);

    void dump_input(nncase::value_t value, std::string name);
};

END_NS_NNCASE_RUNTIME
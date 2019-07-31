#pragma once
#include "../../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    namespace k210
    {
        class kpu_upload : public node
        {
        public:
            DEFINE_NODE_OPCODE(op_k210_kpu_upload);

            input_connector &input() { return input_at(0); }
            output_connector &output() { return output_at(0); }

            kpu_upload(shape_t input_shape)
            {
                add_input("input", dt_uint8, input_shape);
                add_output("output", dt_uint8, input_shape, mem_k210_kpu);
            }
        };

        class kpu_download : public node
        {
        public:
            DEFINE_NODE_OPCODE(op_k210_kpu_download);

            input_connector &input() { return input_at(0); }
            output_connector &output() { return output_at(0); }

            kpu_download(shape_t input_shape)
            {
                add_input("input", dt_uint8, input_shape);
                add_output("output", dt_uint8, input_shape);
            }
        };
    }
}
}

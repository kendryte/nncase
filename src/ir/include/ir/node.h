#pragma once
#include "connectors.h"
#include "opcode.h"
#include <vector>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace ir
{
    class node
    {
    public:
        node() = default;
        node(node &) = delete;
        virtual ~node();

        xtl::span<input_connector> inputs() noexcept { return input_connectors_; }
        xtl::span<output_connector> outputs() noexcept { return output_connectors_; }

        input_connector &input_at(size_t index) { return input_connectors_.at(index); }
        output_connector &output_at(size_t index) { return output_connectors_.at(index); }

        virtual node_opcode opcode() const noexcept = 0;

    protected:
        template <class TName, class TShape>
        input_connector &add_input(TName &&name, datatype_t type, TShape &&shape)
        {
            return input_connectors_.emplace_back(*this, std::forward<TName>(name), type, std::forward<TShape>(shape));
        }

        template <class TName, class TShape>
        output_connector &add_output(TName &&name, datatype_t type, TShape &&shape, memory_type_t memory_type = mem_main)
        {
            return output_connectors_.emplace_back(*this, std::forward<TName>(name), type, std::forward<TShape>(shape), memory_type);
        }

    private:
        std::vector<input_connector> input_connectors_;
        std::vector<output_connector> output_connectors_;
    };
}
}

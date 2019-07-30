#pragma once
#include "input_node.h"
#include "node.h"
#include "output_node.h"
#include <memory>
#include <vector>

namespace nncase
{
namespace ir
{
    class graph
    {
    public:
        xtl::span<std::unique_ptr<node>> nodes() noexcept { return nodes_; }
        xtl::span<input_node *> inputs() noexcept { return inputs_; }
        xtl::span<output_node *> outputs() noexcept { return outputs_; }

        template <class T, class... TArgs>
        T *emplace(TArgs &&... args)
        {
            auto node = static_cast<T *>(nodes_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
            if constexpr (std::is_same_v<T, input_node>)
                inputs_.emplace_back(node);
            else if constexpr (std::is_same_v<T, output_node>)
                outputs_.emplace_back(node);
            return node;
        }

        void assign_names();
        void collect();

    private:
        std::vector<std::unique_ptr<node>> nodes_;
        std::vector<input_node *> inputs_;
        std::vector<output_node *> outputs_;
    };
}
}

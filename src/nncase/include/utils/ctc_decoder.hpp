#pragma once

#include <vector>

namespace nncase {
    namespace utils {
        template<size_t AlphabetsNum>
        class greedy_decoder
        {
        public:
            template<typename TContainer>
            std::vector<size_t> decode(const TContainer& logits, size_t max_time_step)
            {
                std::vector<size_t> output;
                size_t last_idx = AlphabetsNum;
                for (size_t time_step = 0; time_step < max_time_step; time_step++)
                {
                    auto start = logits.begin() + time_step * (AlphabetsNum + 1);

                    auto max = *start;
                    size_t max_idx = 0;
                    for (size_t idx = 0; idx < AlphabetsNum + 1; ++idx)
                    {
                        auto value = *(start + idx);
                        if (value > max)
                        {
                            max = value;
                            max_idx = idx;
                        }
                    }

                    if (max_idx != last_idx)
                    {
                        last_idx = max_idx;
                        if (max_idx != AlphabetsNum)
                            output.emplace_back(max_idx);
                    }
                }

                return output;
            }
        };
    }
}
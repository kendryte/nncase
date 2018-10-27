#pragma once

#include "../layer.hpp"

namespace nncase
{
	namespace layers
	{
		template<size_t rows, size_t cols>
		class matmul : public layer
		{
		public:
			using layer::layer;

			virtual void forward(forward_ctx& ctx) override
			{
				// load weights
				auto W = ctx.get_weights(get_name() + "/W", rows * cols);

                vec_t output(ctx.width * cols, static_cast<float>(0));

                auto out_p = output.begin();
                auto i_p = ctx.inout.cbegin()
                auto w_p = W.cbegin();
                for (size_t oc = 0; oc < cols; oc++)
                {
                    for (size_t or = 0; or < ctx.width; or++)
                    {
                        float sum = 0;
                        for (size_t i = 0; i < rows; i++)
                            sum += i_p[or * ctx.width + i] * w_p[i * cols + oc];
                        *out_p++ = sum;
                    }
                }

                ctx.inout.swap(output);
			}
		};
	}
}
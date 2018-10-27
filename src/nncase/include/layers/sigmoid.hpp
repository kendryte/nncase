#pragma once

#include "../layer.hpp"

namespace nncase
{
	namespace layers
	{
		template<size_t dim>
		class sigmoid : public layer
		{
		public:
            using layer::layer;

			virtual void forward(forward_ctx& ctx) override
			{
				auto inout_p = ctx.inout.begin();
				for (size_t od = 0; od < dim; od++)
				{
					for (size_t x = 0; x < ctx.width; x++)
					{
						const auto i = *inout_p;
						*inout_p++ = 1 / (1 + std::exp(-i));
					}
				}
			}
		};
	}
}
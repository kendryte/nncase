#pragma once

#include "../layer.hpp"

namespace nncase
{
	namespace layers
	{
		template<size_t dim>
		class bias_add : public layer
		{
		public:
			using layer::layer;

			virtual void forward(forward_ctx& ctx) override
			{
				// load weights
				auto bias = ctx.get_weights(get_name() + "/b", dim);

				auto inout_p = ctx.inout.begin();
				auto bias_p = bias.cbegin();
				for (size_t od = 0; od < dim; od++)
				{
					const auto b = *bias_p++;

					for (size_t x = 0; x < ctx.width; x++)
						*inout_p++ = *inout_p + b;
				}
			}
		};
	}
}
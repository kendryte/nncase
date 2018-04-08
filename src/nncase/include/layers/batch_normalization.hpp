#pragma once

#include "../layer.hpp"

namespace nncase
{
	namespace layers
	{
		template<size_t dim>
		class batch_normalization : public layer
		{
		public:
			batch_normalization(const std::string& name, float epsilon)
				:layer(name), epsilon_(epsilon)
			{
			}

			virtual void forward(forward_ctx& ctx) override
			{
				// load weights
				auto mean = ctx.get_weights(get_name() + "/moving_mean", dim);
				auto variance = ctx.get_weights(get_name() + "/moving_variance", dim);
				auto gamma = ctx.get_weights(get_name() + "/gamma", dim);
				auto beta = ctx.get_weights(get_name() + "/beta", dim);

				auto inout_p = ctx.inout.begin();
				auto mean_p = mean.cbegin();
				auto variance_p = variance.cbegin();
				auto gamma_p = gamma.cbegin();
				auto beta_p = beta.cbegin();
				for (size_t od = 0; od < dim; od++)
				{
					const auto m = *mean_p++;
					const auto v = std::sqrt(*variance_p++ + epsilon_);
					const auto g = *gamma_p++;
					const auto b = *beta_p++;

					for (size_t x = 0; x < ctx.width; x++)
						*inout_p++ = (*inout_p - m) / v * g + b;
				}
			}
		private:
			const float epsilon_;
		};
	}
}
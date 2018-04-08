#pragma once

#include "../layer.hpp"

namespace nncase
{
	namespace layers
	{
		template<size_t dim>
		class leaky_relu : public layer
		{
		public:
			leaky_relu(const std::string& name, float alpha = 0.2f)
				:layer(name), alpha_(alpha)
			{
			}

			virtual void forward(forward_ctx& ctx) override
			{
				auto inout_p = ctx.inout.begin();
				for (size_t od = 0; od < dim; od++)
				{
					for (size_t x = 0; x < ctx.width; x++)
					{
						const auto i = *inout_p;
						*inout_p++ = i > 0 ? i : i * alpha_;
					}
				}
			}
		private:
			const float alpha_;
		};
	}
}
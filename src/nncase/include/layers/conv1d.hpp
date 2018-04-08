#pragma once

#include "../layer.hpp"
#include <array>

namespace nncase
{
	namespace layers
	{
		namespace details
		{
			template<uint8_t kernel_size, size_t input_dim>
			size_t padding(size_t width, vec_t& inout)
			{
				const auto padding_size = kernel_size - 1;
				const auto padded_width = width + kernel_size - 1;
				if (padding_size)
				{
					inout.resize(padded_width * input_dim);
					for (int id = input_dim - 1; id > 0; id--)
					{
						const auto begin = inout.begin() + id * width;
						std::copy(begin, begin + width, begin + id * padding_size);
					}
				}

				return padded_width;
			}

			template<typename T, size_t N, size_t offset = 0, typename TIt>
			std::array<T, N> load(TIt& begin, T bias = {})
			{
				std::array<T, N> arr;
				for (size_t i = offset; i < N; i++)
					arr[i] = *begin++ + bias;
				return arr;
			}

			template<typename T, size_t N, typename TIt>
			void shift_load(std::array<T, N>& arr, TIt& begin, T bias = {})
			{
				for (size_t i = 0; i < N - 1; i++)
					arr[i] = arr[i + 1];

				arr.back() = *begin++ + bias;
			}
		}

		template<uint8_t kernel_size, size_t input_dim, size_t output_dim>
		class conv1d : public layer
		{
		public:
			using layer::layer;

			virtual void forward(forward_ctx& ctx) override
			{
				// load weights
				auto W = ctx.get_weights(get_name() + "/W", kernel_size * input_dim * output_dim);

				// padding
				const auto padded_width = details::padding<kernel_size, input_dim>(ctx.width, ctx.inout);

				vec_t output(ctx.width * output_dim, static_cast<float>(0));

				auto out_p = output.begin();
				auto w_p = W.cbegin();
				for (size_t od = 0; od < output_dim; od++)
				{
					for (size_t id = 0; id < input_dim; id++)
					{
						const auto w_arr = details::load<float, kernel_size>(w_p);
						auto i_p = ctx.inout.cbegin() + id * padded_width;
						auto i_arr = details::load<float, kernel_size, 1>(i_p);

						for (size_t x = 0; x < ctx.width; x++)
						{
							details::shift_load<float>(i_arr, i_p);

							float sum = 0;
							for (size_t k = 0; k < kernel_size; k++)
								sum += i_arr[k] * w_arr[k];

							out_p[x] += sum;
						}
					}

					out_p += ctx.width;
				}

				ctx.inout.swap(output);
			}
		};
	}
}
#pragma once

#include <vector>
#include <string>

namespace nncase
{
	typedef std::vector<float> vec_t;

	struct qvec_t
	{
		std::vector<uint8_t> vec;
		float min, max;
	};

	struct forward_ctx
	{
		vec_t inout;
		size_t width;

		virtual vec_t get_weights(const std::string& name, size_t size) = 0;
	};
}
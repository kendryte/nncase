#pragma once
#include "datatypes.hpp"

namespace nncase
{
	class layer
	{
	public:
		layer(std::string name)
			:name_(name)
		{
		}

		const std::string& get_name() const noexcept { return name_; }

		virtual void forward(forward_ctx& ctx) = 0;
	private:
		std::string name_;
	};
}
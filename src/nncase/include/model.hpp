#pragma once
#include "datatypes.hpp"
#include "layer.hpp"
#include "io/weights_container.h"
#include <list>
#include <memory>
#include <unordered_map>

namespace nncase
{
	class sequential
	{
		class forward_ctx : public nncase::forward_ctx
		{
		public:
			forward_ctx(weights_container& weights_container)
				:weights_container_(weights_container)
			{
			}

			virtual vec_t get_weights(const std::string& name, size_t size) override
			{
				return weights_container_.get_weights(name, size);
			}
		private:
			weights_container & weights_container_;
		};

	public:
		template<typename TLayer, typename ...TArgs>
		void emplace_back(TArgs&&... args)
		{
			auto layer = std::make_shared<TLayer>(std::forward<TArgs>(args)...);
			layers_.emplace_back(layer);
			named_layers_.emplace(layer->get_name(), layer);
		}

		void inferrence(vec_t& inout, size_t width) const
		{
			weights_container_->begin_inferrence();

			forward_ctx ctx(*weights_container_);
			ctx.inout = std::move(inout);
			ctx.width = width;

			for (auto& layer : layers_)
				layer->forward(ctx);

			inout = std::move(ctx.inout);
		}

		template<typename TContainer, typename ...TArgs>
		void set_weights_container(TArgs&&... args)
		{
			weights_container_.reset(new TContainer(std::forward<TArgs>(args)...));
		}
	private:
		std::list<std::shared_ptr<layer>> layers_;
		std::unordered_map<std::string, std::shared_ptr<layer>> named_layers_;
		std::unique_ptr<weights_container> weights_container_;
	};
}
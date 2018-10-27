#pragma once
#include "../datatypes.hpp"

namespace nncase
{
	class weights_container
	{
	public:
		virtual void begin_inferrence() = 0;
		virtual vec_t get_weights(const std::string& name, size_t size) = 0;
	};

	template<typename TFileStream>
	class file_weights_container : public weights_container
	{
	public:
		template<typename ...TArgs>
		file_weights_container(TArgs&&... args)
			:filestream_(std::forward<TArgs>(args)...)
		{
		}

		virtual void begin_inferrence() override
		{
			filestream_.seekg(0);
		}

		virtual vec_t get_weights(const std::string& name, size_t size) override
		{
			vec_t weights(size);
			filestream_.read(reinterpret_cast<char*>(weights.data()), size * sizeof(decltype(weights.front())));
			return weights;
		}
	private:
		TFileStream filestream_;
	};

    class zero_weights_container : public weights_container
    {
    public:
        virtual void begin_inferrence() override
        {
        }

        virtual vec_t get_weights(const std::string& name, size_t size) override
        {
            vec_t weights(size);
            return weights;
        }
    };
}
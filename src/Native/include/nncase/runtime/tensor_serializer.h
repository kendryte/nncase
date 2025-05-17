#pragma once
#include <nlohmann/json.hpp>
#include <nncase/llm/paged_attention_config.h>
#include <nncase/runtime/result.h>
#include <nncase/tensor.h>

BEGIN_NS_NNCASE_RUNTIME

NNCASE_API result<datatype_t>
deserialize_datatype(const nlohmann::json &json) noexcept;
NNCASE_API result<tensor>
deserialize_tensor(const nlohmann::json &root) noexcept;
NNCASE_API result<llm::paged_attention_config>
deserialize_paged_attention_config(const nlohmann::json &json) noexcept;
NNCASE_API result<intptr_t>
deserialize_paged_attention_kv_cache(const nlohmann::json &json) noexcept;
NNCASE_API result<intptr_t>
deserialize_reference(const nlohmann::json &json) noexcept;

END_NS_NNCASE_RUNTIME
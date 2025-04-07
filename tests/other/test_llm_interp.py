# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import nncase
import numpy as np


def test_llm_interp(request):
  num_blocks = 12
  layers = 24
  num_kv_head = 2
  kv = 2
  head_dim = 256
  block_size = 32

  # alloc kv caches
  kvcaches = [[[nncase.RuntimeTensor.from_numpy(np.random.rand(num_blocks, layers, num_kv_head, kv, head_dim //
                                                64, block_size, 64).astype(np.float32)) for core in range(32)] for die in range(2)] for chip in range(1)]
  # set up kv caches
  paged_kv = nncase.PagedAttentionKVCacheForEvaluator()
  paged_kv = nncase.PagedAttentionKVCache()
  paged_kv.kv_caches = kvcaches
  paged_kv.block_tables = [nncase.RuntimeTensor.from_numpy(np.random.randint(0, 100, [8, 4, 3]))]
  paged_kv.slot_mapping = [nncase.RuntimeTensor.from_numpy(np.random.randint(0, 100, [8, 4, 3]))]

  # create model
  model = nncase.AutoModelForCausalLM('xxxx.kmodel', None, None, None)

  input_ids = nncase.RuntimeTensor.from_numpy(np.random.randint(0, 100, [100]).astype(np.int64))
  position_ids = nncase.RuntimeTensor.from_numpy(np.random.randint(0, 100, [100]).astype(np.int64))
  hidden_states = model(input_ids, None, position_ids, paged_kv)


if __name__ == "__main__":
  pytest.main(['-vv', __file__])

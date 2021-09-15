/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

[[vk::binding(0)]] Buffer<float> input;
[[vk::binding(1)]] RWBuffer<float> output;

void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= {{ length }})
		return;

	float v = input[id.x];
## if unary_op == "unary_abs"
	v = abs(v);
## if unary_op == "unary_acos"
	v = acos(v);
## if unary_op == "unary_asin"
	v = asin(v);
## else if unary_op == "unary_ceil"
	v = ceil(v);
## else if unary_op == "unary_cos"
	v = cos(v);
## else if unary_op == "unary_exp"
	v = exp(v);
## else if unary_op == "unary_floor"
	v = floor(v);
## else if unary_op == "unary_log"
	v = log(v);
## else if unary_op == "unary_neg"
	v = -v;
## else if unary_op == "unary_round"
	v = round(v);
## else if unary_op == "unary_rsqrt"
	v = rsqrt(v);
## else if unary_op == "unary_sin"
	v = sin(v);
## else if unary_op == "unary_sqrt"
	v = sqrt(v);
## else if unary_op == "unary_square"
	v = v * v;
## else if unary_op == "unary_tanh"
	v = tanh(v);
## else
	unsupported unary: {{ unary_op }}
## endif
	output[id.x] = v;
}

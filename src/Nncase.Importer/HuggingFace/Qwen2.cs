// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml;
using CommunityToolkit.HighPerformance;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Buffer = System.Buffer;
using F = Nncase.IR.F;
using Tuple = System.Tuple;

namespace Nncase.Importer
{
    public partial class HuggingFaceImporter
    {
        protected (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) Qwen2CreateInputs()
        {
            _ = (long)_config!["hidden_size"];
            _inputs = new List<Var>();
            _dynVarMap = new Dictionary<string, Var>();
            var varMap = new Dictionary<Var, Expr[]>();

            var bucketOptions = CompileSession.CompileOptions.ShapeBucketOptions;
            _fixVarMap = bucketOptions.FixVarMap;

            // local test set
            // _fixVarMap["sequence_length"] = 10;
            // _fixVarMap["history_len"] = 0;
            if (!_fixVarMap.ContainsKey("sequence_length"))
            {
                _dynVarMap["sequence_length"] = new Var(
                    "sequence_length",
                    new TensorType(DataTypes.Int32, Shape.Scalar));
            }

            if (!_fixVarMap.ContainsKey("history_len"))
            {
                _dynVarMap["history_len"] = new Var(
                    "history_len",
                    new TensorType(DataTypes.Int32, Shape.Scalar));
            }

            if (!_fixVarMap.ContainsKey("batch_size"))
            {
                _dynVarMap["batch_size"] = new Var(
                    "batch_size",
                    new TensorType(DataTypes.Int32, Shape.Scalar));
            }

            var inputIdsShapeExpr = new Expr[] { _dynVarMap["batch_size"], _dynVarMap["sequence_length"] };
            var attentionMaskShapeExpr = new Expr[]
            {
                1,
                1,
                _dynVarMap["sequence_length"],
                _dynVarMap["sequence_length"],
            };
            var positionIdsShapeExpr = new Expr[] { _dynVarMap["batch_size"], _dynVarMap["sequence_length"] };
            var pastKeyValueShapeExpr = new Expr[] { 24, 2, 1, _dynVarMap["history_len"], 2, 64 };

            var inputIds = new Var(
                "input_ids",
                new TensorType(DataTypes.Int32, new Shape(Dimension.Unknown, Dimension.Unknown)));

            var attentionMask = new Var(
                "attention_mask",
                new TensorType(
                    DataTypes.Float32,
                    new Shape(1, 1, Dimension.Unknown, Dimension.Unknown)));
            var positionIds = new Var(
                "position_ids",
                new TensorType(DataTypes.Float32, new Shape(Dimension.Unknown, Dimension.Unknown)));
            var pastKeyValue = new Var(
                "past_key_values",
                new TensorType(DataTypes.Float32, new Shape(24, 2, 1, Dimension.Unknown, 2, 64)));

            _inputs.Add(inputIds);
            _inputs.Add(attentionMask);
            _inputs.Add(positionIds);
            _inputs.Add(pastKeyValue);
            varMap[inputIds] = inputIdsShapeExpr;
            varMap[attentionMask] = attentionMaskShapeExpr;
            varMap[positionIds] = positionIdsShapeExpr;
            varMap[pastKeyValue] = pastKeyValueShapeExpr;
            return (_inputs, varMap);
        }

        private Expr Qwen2CreateOutputs() {
            // TODO: use self.config.output_attention to judge wether output kvache
            var lm_head = _outputs!["lm_head"];
            Expr? kv_cache = null;
            if (_outputs.ContainsKey("kv_cache"))
            {
                kv_cache = _outputs["kv_cache"];
            }

            if (kv_cache is null)
            {
                return lm_head;
            }

            return new IR.Tuple([lm_head, kv_cache]);
        }

        // private Tuple<Call, HuggingFaceUtils.DynamicCache> VisitQwen2ForCausalLM()
        private void VisitQwen2ForCausalLM()
        {
            if (_constTensors == null)
            {
                throw new ArgumentNullException(nameof(_constTensors));
            }

            // architecture: "QWenForCausalLM"
            /*
             Qwen2ForCausalLM
             (
                (model): Qwen2Model
                (
                    (embed_tokens): Embedding(151936, 896)
                    (layers): ModuleList
                    (
                        (0-23): 24 x Qwen2DecoderLayer
                        (
                            (self_attn): Qwen2SdpaAttention
                            (
                                (q_proj): Linear(in_features=896, out_features=896, bias=True)
                                (k_proj): Linear(in_features=896, out_features=128, bias=True)
                                (v_proj): Linear(in_features=896, out_features=128, bias=True)
                                (o_proj): Linear(in_features=896, out_features=896, bias=False)
                                (rotary_emb): Qwen2RotaryEmbedding()
                            )
                            (mlp): Qwen2MLP
                            (
                                (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
                                (up_proj): Linear(in_features=896, out_features=4864, bias=False)
                                (down_proj): Linear(in_features=4864, out_features=896, bias=False)
                                (act_fn): SiLU()
                            )
                            (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
                            (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
                        )
                    )
                    (norm): Qwen2RMSNorm((896,), eps=1e-06)
                    (rotary_emb): Qwen2RotaryEmbedding()
                )
                (lm_head): Linear(in_features=896, out_features=151936, bias=False)
            )
            */

            var input_ids = _inputs[0];
            var attention_mask = _inputs[1];
            var position_ids = _inputs[2];
            var pastKeyValues = _inputs[3];

            // var (lastHiddenStates, pastKeyValues, allSelfAttns, allHiddenStates) = Qwen2Model(input_ids,
            //     inputEmbeds: null, new HuggingFaceUtils.DynamicCache(), cachePosition: null, positionIds: null,
            //     useCache: false, outputAttentions: false, outputHiddenStates: false);
            var (lastHiddenStates, allSelfAttnsKV) = Qwen2Model(
                input_ids,
                attention_mask,
                position_ids,
                pastKeyValues);
            var lmHead = Linear(
                lastHiddenStates,
                _constTensors["model.embed_tokens.weight"]);

            _outputs!["lm_head"] = lmHead;
            Call attentionKVCache = null;

            //TODO: using config.output_attentions to judge whether need kv cache
            if (CheckNeedKVcache(allSelfAttnsKV))
            {
                attentionKVCache = Concat(allSelfAttnsKV.ToArray(), 0);
            }

            if (attentionKVCache is not null) {
                _outputs!["kv_cache"] = attentionKVCache;
            }
        }

        // private Tuple<Call, HuggingFaceUtils.DynamicCache, List<Call>, List<Call>> Qwen2Model(
        //     Expr input_ids,
        //     Call? inputEmbeds,
        //     HuggingFaceUtils.DynamicCache? pastKeyValues,
        //     Expr? cachePosition,
        //     Call? positionIds,
        //     bool? useCache = false,
        //     bool? outputAttentions = false,
        //     bool? outputHiddenStates = false
        // )
        private Tuple<Call, List<Call>> Qwen2Model(
            Var input_ids,
            Var attentionMask,
            Var positionIds,
            Var pastKeyValues)
        {
            /*
             * 1.1 embedding
             */
            // if (inputEmbeds == null)
            // {
            var embedTokensWeight = _constTensors["model.embed_tokens.weight"];
            if (_config!.Keys.Contains("pad_token_id"))
            {
                // embedTokensWeight[(int)_config["pad_token_ids"]] = new float[embedTokensWeight.Shape[-1].FixedValue];
                for (int i = 0; i < embedTokensWeight.Shape[-1].FixedValue; i++)
                {
                    embedTokensWeight[(int)_config["pad_token_id"], (int)i] = 0;
                }
            }

            var inputEmbeds = Gather(embedTokensWeight, 0, input_ids);

            // }

            // if (useCache == true && pastKeyValues == null)
            // {
            //     pastKeyValues = new HuggingFaceUtils.DynamicCache();
            // }

            // if (cachePosition == null)
            // {
            //     if (pastKeyValues != null)
            //     {
            //         var pastSeenTokens = pastKeyValues.GetSeqLength();
            //         int sequenceLength =
            //             inputEmbeds.CheckedShape[1].FixedValue; // 假设 inputEmbeds 的第二个维度是 sequenceLength
            //         var cachePositionList = Enumerable.Range(pastSeenTokens, pastSeenTokens + sequenceLength).ToArray();
            //         cachePosition = Tensor.FromArray(cachePositionList);
            //     }
            // }
            //
            // if (positionIds == null)
            // {
            //     positionIds = Unsqueeze(cachePosition, 0);
            // }

            // TODO : _update_causal_mask
            // causal_mask = self._update_causal_mask(
            //     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            // )
            // Call? causalMask = null;
            var hiddenStates = inputEmbeds;
            var positionEmbeddings = RotaryEmbedding(hiddenStates, positionIds);

            var allHiddenStates = new List<Call>();
            var allSelfAttns = new List<Call>();
            /*
            * 1.2 DecodeLayer * _config["num_hidden_layers"]
            * DecodeLayer:
            *
            */
            _ = new List<Tuple<Call, Call>>();
            for (int i = 0; i < (int)(long)_config["num_hidden_layers"]; i++)
            {
                // if (outputAttentions == true)
                // {
                allHiddenStates.Add(hiddenStates);

                // }

                // var (hiddenStatesTmp, selfAttenWeights) = DecodeLayer(i, hiddenStates, causalMask, positionIds,
                //     pastKeyValues, outputAttentions,
                //     useCache, cachePosition, positionEmbeddings);
                var (hiddenStatesTmp, selfAttenKV) = DecodeLayer(
                    i,
                    hiddenStates,
                    attentionMask,
                    pastKeyValues,
                    positionEmbeddings);

                hiddenStates = hiddenStatesTmp;

                // if (outputAttentions == true)
                // {
                allSelfAttns.Add(selfAttenKV);

                // }
            }

            var lastHiddenStates = Qwen2LayerNorm(hiddenStates, "model.norm.weight");

            // if (outputAttentions == true)
            // {
            // allHiddenStates.Add(lastHiddenStates);
            // }

            // return Tuple.Create(lastHiddenStates, pastKeyValues, allHiddenStates, allSelfAttns);
            return Tuple.Create(lastHiddenStates, allSelfAttns);
        }

        // calc torch.nn.linear (i.e. x*transpose(W)+b)
        private Call Linear(Expr expr, Tensor weight, Tensor bias = null)
        {
            var transposed_weight = F.Tensors.Transpose(weight, new int[] { 1, 0 });
            var result = F.Math.MatMul(expr, transposed_weight);
            if (bias != null)
            {
                result = F.Math.Add(result, bias);
            }

            return result;
        }

        private bool CheckNeedKVcache(List<Call> allSelfAttnsKV)
        {
            foreach (var call in allSelfAttnsKV)
            {
                if (call is null)
                {
                    return false;
                }
            }

            return true;
        }

        // private Tuple<Call, Call> DecodeLayer(int count, Call hiddenStates, Call? attentionMask, Call positionIds,
        //     HuggingFaceUtils.DynamicCache pastKeyValues, bool? outputAttentions, bool? useCache, Expr cachePosition,
        //     Tuple<Call, Call> positionEmbeddings)
        private Tuple<Call, Call> DecodeLayer(
            int count,
            Call hiddenStates,
            Var? attentionMask,
            Var pastKeyValues,
            Tuple<Call, Call> positionEmbeddings)
        {
            var residual = hiddenStates;
            hiddenStates = Qwen2LayerNorm(
                hiddenStates,
                $"model.layers.{count}.input_layernorm.weight");

            // self attention
            var (hiddenStatesTmp, selfAttenKV) = Qwen2SelfAtten(
                count,
                hiddenStates,
                attentionMask,
                positionEmbeddings);
            hiddenStates = hiddenStatesTmp;
            hiddenStates = residual + hiddenStates;

            // fully Connected
            residual = hiddenStates;
            hiddenStates = Qwen2LayerNorm(
                hiddenStates,
                $"model.layers.{count}.post_attention_layernorm.weight");
            hiddenStates = Qwen2Mlp(count, hiddenStates);
            hiddenStates = residual + hiddenStates;

            var output = hiddenStates;

            // if (outputAttentions == true && selfAttenKV is not null)
            // {
            return Tuple.Create<Call, Call>(output, selfAttenKV);

            // }

            // return Tuple.Create<Call, Call>(output, null);
        }

        private Call Qwen2Mlp(int count, Call hiddenStates)
        {
            var gateProjW = _constTensors![$"model.layers.{count}.mlp.gate_proj.weight"];
            var upProjW = _constTensors![$"model.layers.{count}.mlp.up_proj.weight"];
            var downProjW = _constTensors![$"model.layers.{count}.mlp.down_proj.weight"];

            var tmp = Linear(hiddenStates, gateProjW);
            return Linear(
                Sigmoid(tmp) * tmp * Linear(hiddenStates, upProjW),
                downProjW);
        }

        // Qwen2RMSNorm : Qwen2LayerNorm : input_layernorm
        private Call Qwen2LayerNorm(Call hiddenStates, string layerName)
        {
            // fit layernorm partten 5
            var weight = _constTensors![$"{layerName}"];
            var bias = Tensor.FromScalar(0f, weight.Shape);
            int axis = -1;
            return F.NN.LayerNorm(axis, 1e-6F, hiddenStates, weight, bias, false);
        }

        // Qwen2Attention : SelfAtten
        // llama config find in : https://www.restack.io/p/transformer-models-answer-llama-config-json-cat-ai
        private Tuple<Call, Call> Qwen2SelfAtten(
            int count,
            Call hiddenStates,
            Var attentionMask,
            Tuple<Call, Call> positionEmbeddings)
        {
            var head_dim = (int)(long)_config!["hidden_size"] / (int)(long)_config["num_attention_heads"];
            if (_config!.Keys.Contains("head_dim"))
            {
                head_dim = (int)(long)_config["head_dim"];
            }

            var batch_size = ShapeOf(hiddenStates)[0];
            var seq_len = ShapeOf(hiddenStates)[1];

            // batch_size, seq_len, num_heads, head_dim
            var hidden_shape = Stack(new IR.Tuple(batch_size, seq_len, -1L, (long)head_dim), 0);

            // hidden_shape.Add(new Dimension(hidden_dim));
            // hidden_shape = Concat(new IR.Tuple(hidden_shape, Tensor.FromScalar(hidden_dim)), 0);
            var qProjW = _constTensors![$"model.layers.{count}.self_attn.q_proj.weight"];
            var qProjB = _constTensors![$"model.layers.{count}.self_attn.q_proj.bias"];
            var queryStates = Linear(hiddenStates, qProjW, qProjB);
            queryStates = Reshape(queryStates, hidden_shape);
            queryStates = Transpose(queryStates, new int[] { 0, 2, 1, 3 });

            var kProjW = _constTensors![$"model.layers.{count}.self_attn.k_proj.weight"];
            var kProjB = _constTensors![$"model.layers.{count}.self_attn.k_proj.bias"];
            var keyStates = Linear(hiddenStates, kProjW, kProjB);
            keyStates = Reshape(keyStates, hidden_shape);
            keyStates = Transpose(keyStates, new int[] { 0, 2, 1, 3 });

            var vProjW = _constTensors![$"model.layers.{count}.self_attn.k_proj.weight"];
            var vProjB = _constTensors![$"model.layers.{count}.self_attn.k_proj.bias"];
            var valueStates = Linear(hiddenStates, vProjW, vProjB);
            valueStates = Reshape(valueStates, hidden_shape);
            valueStates = Transpose(valueStates, new int[] { 0, 2, 1, 3 });

            var (cos, sin) = positionEmbeddings;

            // apply_rotary_pos_emb
            (queryStates, keyStates) = ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);

            // TODO: sliding window
            // var slidingWindow = 0;
            // if (_config!.Keys.Contains("use_sliding_window") && _config!["use_sliding_window"] != null &&
            //     count >= (int)_config!["max_window_layers"])
            // {
            //     slidingWindow = (int)_config!["sliding_window"];
            // }

            // qwen use sdpa attention
            var (hiddenStatesTmp, selfAttenWeight) = SdpaAttention(
                queryStates,
                keyStates,
                valueStates,
                attentionMask, /*TODO: 这里可能不需要,如果使用输入*/
                0.0f,
                false);
            hiddenStates = hiddenStatesTmp;

            // inputShape.Add(-1);
            var inputShape = Stack(new IR.Tuple(batch_size, seq_len, -1L), 0);
            hiddenStates = IR.F.Tensors.Reshape(hiddenStates, inputShape);
            var oProjW = _constTensors![$"model.layers.{count}.self_attn.o_proj.weight"];
            hiddenStates = F.Math.MatMul(hiddenStates, oProjW);

            return Tuple.Create(hiddenStates, selfAttenWeight);
        }

        private Tuple<Call, Call> SdpaAttention(
            Call queryStates,
            Call keyStates,
            Call valueStates,
            Expr? attentionMask,
            float? scaling,
            bool? isCausal)
        {
            /*
             * def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
                   L, S = query.size(-2), key.size(-2)
                   scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                   attn_bias = torch.zeros(L, S, dtype=query.dtype)
                   if is_causal:
                       assert attn_mask is None
                       temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                       attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                       attn_bias.to(query.dtype)

                   if attn_mask is not None:
                       if attn_mask.dtype == torch.bool:
                           attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                       else:
                           attn_bias += attn_mask

                   if enable_gqa:
                       key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                       value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                   attn_weight = query @ key.transpose(-2, -1) * scale_factor
                   attn_weight += attn_bias
                   attn_weight = torch.softmax(attn_weight, dim=-1)
                   attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                   return attn_weight @ value
             */
            var casualMask = attentionMask;
            if (attentionMask != null)
            {
                casualMask = Slice(
                    casualMask,
                    new[] { 0, 0, 0, 0 },
                    Stack(new IR.Tuple(-1, -1, -1, ShapeOf(keyStates)[-2]), 0),
                    new[] { -1 },
                    new[] { 1 });
            }

            if (isCausal == null)
            {
                isCausal = casualMask == null && queryStates.CheckedShape[2].FixedValue > 1;
            }

            var (l, s) = (ShapeOf(queryStates)[-2], ShapeOf(keyStates)[-2]);
            var scaleFactor = 1.0f / F.Math.Sqrt(Cast(ShapeOf(queryStates)[-1], DataTypes.Float32));
            var attnBias = (Call)F.Tensors.Broadcast(Tensor.FromScalar(0f), F.Tensors.Stack(new IR.Tuple(l, s), 0));

            // TODO: 也许需要处理attentionMask为空的情况,在这里生成下三角为0 ,其他为-inf的功能.
            // if (isCausal == true)
            // {
            //     var tempMask = (Call)Tensor.FromScalar(0f, new Shape(l, s));
            // }
            if (attentionMask != null)
            {
                attnBias = Binary(BinaryOp.Add, attnBias, attentionMask);
            }

            var attnWeight =
                IR.F.Math.MatMul(
                    queryStates,
                    Transpose(keyStates, ShapeExprUtility.GetPermutation(keyStates, [-2, -1]))) * scaleFactor;
            attnWeight += attnBias;
            attnWeight = Softmax(attnWeight, -1);
            var attnOutput = F.Math.MatMul(attnWeight, valueStates);
            attnOutput = Transpose(attnOutput, ShapeExprUtility.GetPermutation(attnOutput, [1, 2]));
            return Tuple.Create(attnOutput, (Call)null);
        }

        private Tuple<Call, Call> ApplyRotaryPosEmb(Call q, Call k, Call cos, Call sin)
        {
            cos = Unsqueeze(cos, Tensor.From<long>(new long[] { 1 }));
            sin = Unsqueeze(sin, Tensor.From<long>(new long[] { 1 }));

            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            var qEmbed = Binary(
                BinaryOp.Add,
                Binary(BinaryOp.Mul, q, cos),
                Binary(BinaryOp.Mul, RotateHalf(q), sin));
            var kEmbed = Binary(
                BinaryOp.Add,
                Binary(BinaryOp.Mul, k, cos),
                Binary(BinaryOp.Mul, RotateHalf(k), sin));
            return Tuple.Create(qEmbed, kEmbed);
        }

        private Tuple<Call, Call> RotaryEmbedding(Expr x, Expr positionIds)
        {
            // rope type not in config, so it is default. :_compute_default_rope_parameters
            // if "dynamic" in self.rope_type:
            //      self._dynamic_frequency_update(position_ids, device=x.device)
            var (inv_freq, _) = RoPEInit("default");

            // var a = x.CheckedShape[0];
            var invFreq = Tensor.FromArray(inv_freq.ToArray()); // Unsqueeze(Unsqueeze(Tensor.FromArray(inv_freq.ToArray()), new[] { 0 }),new[] { -1 });
            var invFreq_float = Cast(invFreq, DataTypes.Float32);
            var invFreqExpanded = Unsqueeze(invFreq_float, Tensor.From<long>(new long[] { 0, 2 }));
            var batch_size = ShapeOf(positionIds)[0];
            var dim_div_2 = ShapeOf(invFreq)[0];
            var shape_tensor = F.Tensors.Stack(new IR.Tuple(batch_size, dim_div_2, 1L), 0);

            invFreqExpanded = Expand(invFreqExpanded, shape_tensor);

            // var invFreqExpanded = Broadcast(
            //     inv_freq_tensor,
            //     new Dimension[] { x.CheckedShape[0], inv_freq.Count, 1 });
            var positionIdsExpanded = Unsqueeze(positionIds, Tensor.From<long>(new long[] { 1 }));

            var freqs = F.Math.MatMul(invFreqExpanded, positionIdsExpanded);
            freqs = Transpose(freqs, new int[] { 0, 2, 1 });

            // F.Tensors.Transpose(F.Math.MatMul(invFreqExpanded, positionIdsExpanded),new Dimension[] { 0, 2, 1 });
            var emb = F.Tensors.Concat(new IR.Tuple(freqs, freqs), -1);
            var cos = F.Math.Unary(UnaryOp.Cos, emb);
            var sin = F.Math.Unary(UnaryOp.Sin, emb);

            // TODO: add attention scaling
            return Tuple.Create(cos, sin);
        }

        private Tuple<List<double>, float> RoPEInit(string type = "default")
        {
            switch (type)
            {
                case "default":
                    return HuggingFaceUtils.ComputeDefaultRopeParameters(_config);
                default:
                    throw new NotImplementedException($"RoPE function {type} need to impl");
            }
        }

        // def rotate_half(x):
        // """Rotates half the hidden dims of the input."""
        // x1 = x[..., : x.shape[-1] // 2]
        // x2 = x[..., x.shape[-1] // 2 :]
        // return torch.cat((-x2, x1), dim=-1)
        private Call RotateHalf(Expr x)
        {
            var xS3 = Cast(ShapeOf(x)[3], DataTypes.Int32);
            var x1 = Slice(
                x,
                new[] { 0, 0, 0, 0 },
                F.Tensors.Stack(new IR.Tuple(-1, -1, -1, xS3 / 2), 0),
                new[] { 1, 1, 1, 1 },
                new[] { 1, 1, 1, 1 });
            var x2 = Slice(
                x,
                F.Tensors.Stack(new IR.Tuple(0, 0, 0, xS3 / 2), 0),
                F.Tensors.Stack(new IR.Tuple(-1, -1, -1, -1), 0),
                new[] { 1, 1, 1, 1 },
                new[] { 1, 1, 1, 1 });
            return Concat(new IR.Tuple(Binary(BinaryOp.Mul, x2, -1.0f), x1), -1);
        }
    }
}

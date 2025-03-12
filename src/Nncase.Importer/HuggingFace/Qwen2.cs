// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
            var hiddenSize = (long)_config!["hidden_size"];
            var numsHiddenLayers = (long)_config!["num_hidden_layers"];
            var num_attention_heads = (long)_config!["num_attention_heads"];
            var headDim = hiddenSize / num_attention_heads;
            if (_config.ContainsKey("head_dim"))
            {
                headDim = (long)_config["head_dim"];
            }

            var numKVHeads = (long)_config!["num_key_value_heads"];

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
                _dynVarMap["batch_size"],
                _dynVarMap["sequence_length"],
            };
            var positionIdsShapeExpr = new Expr[] { _dynVarMap["batch_size"], _dynVarMap["sequence_length"] };

            // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
            var pastKeyValueShapeExpr = new Expr[] { numsHiddenLayers, 2, _dynVarMap["batch_size"], numKVHeads, _dynVarMap["history_len"], headDim };

            var inputIds = new Var(
                "input_ids",
                new TensorType(DataTypes.Int32, new Shape(Dimension.Unknown, Dimension.Unknown)));

            var attentionMask = new Var(
                "attention_mask",
                new TensorType(
                    DataTypes.Float32,
                    new Shape(Dimension.Unknown, Dimension.Unknown)));
            var positionIds = new Var(
                "position_ids",
                new TensorType(DataTypes.Float32, new Shape(Dimension.Unknown, Dimension.Unknown)));

            // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
            var pastKeyValue = new Var(
                "past_key_values",
                new TensorType(DataTypes.Float32, new Shape((int)numsHiddenLayers, 2, Dimension.Unknown, (int)numKVHeads, Dimension.Unknown, (int)headDim)));

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

        private Expr Qwen2CreateOutputs()
        {
            // TODO: use self.config.output_attention to judge wether output kvache
            var lm_head = _outputs!["lm_head"];
            Expr? out_attention = null;
            Expr? kvcache = null;
            if (_outputs.ContainsKey("out_attention"))
            {
                out_attention = _outputs["out_attention"];
            }

            if (_outputs.ContainsKey("kvcache"))
            {
                kvcache = _outputs["kvcache"];
            }

            if ((out_attention is null) && (kvcache is null))
            {
                return lm_head;
            }

            if ((out_attention is not null) && (kvcache is null))
            {
                return new IR.Tuple([lm_head, out_attention]);
            }

            if ((out_attention is null) && (kvcache is not null))
            {
                return new IR.Tuple([lm_head, kvcache]);
            }

            return new IR.Tuple([lm_head, out_attention!, kvcache!]);
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
            var (lastHiddenStates, allSelfAttns, allSelfKV) = Qwen2Model(
                input_ids,
                attention_mask,
                position_ids,
                pastKeyValues);
            var lmHead = Linear(
                lastHiddenStates,
                _constTensors["model.embed_tokens.weight"]);

            _outputs!["lm_head"] = lmHead;
            var outAttention = (Call)null;
            var kvCache = (Call)null;

            // TODO: using config.output_attentions to judge whether need kv cache
            if (CheckNeedOutput(allSelfAttns))
            {
                outAttention = Concat(new IR.Tuple(allSelfAttns.ToArray()), 0);
            }

            if (CheckNeedOutput(allSelfKV))
            {
                kvCache = Concat(new IR.Tuple(allSelfKV.ToArray()), 0);
            }

            if (outAttention is not null)
            {
                _outputs!["out_attention"] = outAttention;
            }

            if (kvCache is not null)
            {
                _outputs!["kvcache"] = kvCache;
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
        private Tuple<Expr, List<Expr>, List<Expr>> Qwen2Model(
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
            var embedTokensWeight = _constTensors!["model.embed_tokens.weight"];
            if (_config!.Keys.Contains("pad_token_id"))
            {
                // embedTokensWeight[(int)_config["pad_token_ids"]] = new float[embedTokensWeight.Shape[-1].FixedValue];
                for (int i = 0; i < embedTokensWeight.Shape[-1].FixedValue; i++)
                {
                    embedTokensWeight[(int)_config["pad_token_id"], (int)i] = 0;
                }
            }
            Expr? inputEmbeds;
            if (input_ids.CheckedShape.Rank>2){
                System.Console.WriteLine("input_ids rank >2 ,regard input_id as embedding...");
                inputEmbeds = input_ids;
            }
            else{
            inputEmbeds = Gather(embedTokensWeight, 0, input_ids);
            }

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

            // TODO : _update_casualMask
            // casualMask = self._update_casualMask(
            //     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            // )
            // Call? casualMask = null;
            // TODO: outputAttentions need by config.output_attentions
            var historyLen = ShapeOf(pastKeyValues)[-2];
            var seqLen = ShapeOf(inputEmbeds)[1];
            var cachePosition = Range(Cast(historyLen, DataTypes.Int32), Cast(historyLen + seqLen, DataTypes.Int32), 1);
            var casualMask = UpdatecasualMask(attentionMask, inputEmbeds, cachePosition, pastKeyValues, outputAttentions: false);
            var hiddenStates = inputEmbeds;
            var positionEmbeddings = RotaryEmbedding(hiddenStates, positionIds);

            var allHiddenStates = new List<Expr>();
            var allSelfAttns = new List<Expr>();
            var allKVcaches = new List<Expr>();
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

                // var (hiddenStatesTmp, selfAttenWeights) = DecodeLayer(i, hiddenStates, casualMask, positionIds,
                //     pastKeyValues, outputAttentions,
                //     useCache, cachePosition, positionEmbeddings);
                var (hiddenStatesTmp, outAttention, currentKV) = DecodeLayer(
                    i,
                    hiddenStates,
                    casualMask,
                    pastKeyValues,
                    cachePosition,
                    positionEmbeddings);

                hiddenStates = hiddenStatesTmp;

                // if (outputAttentions == true)
                // {
                allSelfAttns.Add(outAttention);
                allKVcaches.Add(currentKV);

                // }
            }

            Expr lastHiddenStates = Qwen2LayerNorm(hiddenStates, "model.norm.weight");

            // if (outputAttentions == true)
            // {
            // allHiddenStates.Add(lastHiddenStates);
            // }

            // return Tuple.Create(lastHiddenStates, pastKeyValues, allHiddenStates, allSelfAttns);
            return Tuple.Create(lastHiddenStates, allSelfAttns, allKVcaches);
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

        /*
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        */
        private Expr Prepare4dCausalAttentionMaskWithCachePosition(
                                    Expr attentionMask,
                                    Expr seqLen,
                                    Expr targtLen,
                                    DataType dtype,
                                    Expr cachePosition,
                                    Expr batchSize,
                                    Expr pastKeyValues)
        {
            var casualMask = (Expr)null!;
            if (attentionMask.CheckedShape.Rank == 4)
            {
                casualMask = attentionMask;
            }
            else
            {
                // get the min value for current dtype
                var valueRangeType = typeof(ValueRange<>).MakeGenericType(dtype.CLRType);
                PropertyInfo fullProperty = valueRangeType.GetProperty("Full", BindingFlags.Public | BindingFlags.Static)!;
                object fullRangeInstance = fullProperty.GetValue(null)!;
                PropertyInfo minProperty = valueRangeType.GetProperty("Min")!;
                var min = minProperty.GetValue(fullRangeInstance);
                var minValue = (Expr)null!;
                var implicitConversionMethod = typeof(IR.Expr).GetMethods(BindingFlags.Public | BindingFlags.Static).
                FirstOrDefault(m => m.Name == "op_Implicit" && m.GetParameters()[0].ParameterType == min!.GetType());
                if (implicitConversionMethod != null)
                {
                    minValue = (Expr)implicitConversionMethod.Invoke(null, [min])!;
                }
                else
                {
                    throw new InvalidOperationException("Cannot find implicit conversion method for " + min!.GetType());
                }

                var mask_shape = Stack(new IR.Tuple([seqLen, targtLen]), 0);
                casualMask = F.Tensors.ConstantOfShape(mask_shape, minValue);

                /*
                    min_dtype = torch.finfo(dtype).min
                causal_mask = torch.full(
                    (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
                )
                diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                */
                var diagonalAttendMask = Range(0, targtLen, 1) > Reshape(cachePosition, new int[] { -1, 1 });

                // TODO: maybe consider:
                /*
                 if config.sliding_window is not None:
                    # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                    # the check is needed to verify is current checkpoint was trained with sliding window or not
                    if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                        sliding_attend_mask = torch.arange(target_length, device=device) <= (
                            cache_position.reshape(-1, 1) - config.sliding_window
                        )
                        diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
                */
                casualMask = casualMask * Cast(diagonalAttendMask, casualMask.CheckedDataType);

                // casualMask = casualMask[None, None, :, :].expand(batch_size, 1, -1, -1)
                var expandShape = Stack(new IR.Tuple(Cast(batchSize, DataTypes.Int32), 1, -1, -1), 0);
                casualMask = Unsqueeze(casualMask, new int[] { 0, 0 });
                casualMask = Expand(casualMask, expandShape);
                /*
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                        causal_mask.device
                    )
                */
                var maskLength = ShapeOf(attentionMask)[-1];
                var paddingMask = Slice(
                    casualMask,
                    new[] { 0, 0, 0, 0 },
                    Stack(new IR.Tuple(-1, -1, -1, Cast(maskLength, DataTypes.Int32)), 0),
                    new[] { -1 },
                    new[] { 1, 1, 1, 1 });
                paddingMask += Unsqueeze(attentionMask, new int[] { 1, 2 });

                /*
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                */
                paddingMask = Equal(paddingMask, 0.0f);
                var maskPart = Slice(
                    casualMask,
                    new[] { 0, 0, 0, 0 },
                    Stack(new IR.Tuple(-1, -1, -1, Cast(maskLength, DataTypes.Int32)), 0),
                    new[] { -1 },
                    new[] { 1, 1, 1, 1 });

                var minDtypeMatrix = ConstantOfShape(ShapeOf(maskPart), minValue);

                maskPart = Where(paddingMask, minDtypeMatrix, maskPart);

                // TODO: for dynamic cache, maskLength== sequence length == target length
                //  just return maskPart
                var leftPart = Slice(
                    casualMask,
                    Stack(new IR.Tuple(0, 0, 0, Cast(maskLength, DataTypes.Int32)), 0),
                    Stack(new IR.Tuple(-1, -1, -1, -1), 0),
                    new[] { -1 },
                    new[] { 1, 1, 1, 1 });
                casualMask = Concat(new IR.Tuple(maskPart, leftPart), -1);
            }

            return casualMask;
        }

        private Expr UpdatecasualMask(Expr attentionMask, Expr inputsEmbeds, Expr cachePosition, Expr pastKeyValues, bool outputAttentions = false)
        {
            /*
            # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
            */
            // TODO:consider flash attention v2
            var targetLength = ShapeOf(attentionMask)[-1];
            var batchSize = ShapeOf(inputsEmbeds)[0];
            var dataType = inputsEmbeds.CheckedDataType;

            var seqLen = ShapeOf(inputsEmbeds)[1];
            Expr casualMask = Prepare4dCausalAttentionMaskWithCachePosition(
                                                attentionMask,
                                                seqLen,
                                                targetLength,
                                                dataType,
                                                cachePosition,
                                                batchSize,
                                                pastKeyValues);
            /*
            if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
            ):
            # Attend to all tokens in fully masked rows in the casualMask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            casualMask = AttentionMaskConverter._unmask_unattended(casualMask, min_dtype)
            */

            // TODO: maybe need upon
            return casualMask;
        }

        private bool CheckNeedOutput(List<Expr> allSelfAttns)
        {
            if (allSelfAttns.Length() == 0)
            {
                return false;
            }

            foreach (var call in allSelfAttns)
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
        private Tuple<Expr, Expr, Expr> DecodeLayer(
            int count,
            Expr hiddenStates,
            Expr attentionMask,
            Var pastKeyValues,
            Expr cachePosition,
            Tuple<Expr, Expr> positionEmbeddings)
        {
            var residual = hiddenStates;
            hiddenStates = Qwen2LayerNorm(
                hiddenStates,
                $"model.layers.{count}.input_layernorm.weight");

            // TODO: using `config.attn_implementation` to choose attention implementation
            // self attention
            var (hiddenStatesTmp, outAttention, currentKV) = Qwen2SelfAtten(
                count,
                hiddenStates,
                attentionMask,
                pastKeyValues,
                cachePosition,
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
            return Tuple.Create<Expr, Expr, Expr>(output, outAttention, currentKV);

            // }

            // return Tuple.Create<Call, Call>(output, null);
        }

        private Call Qwen2Mlp(int count, Expr hiddenStates)
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
        private Call Qwen2LayerNorm(Expr hiddenStates, string layerName)
        {
            // fit layernorm partten 5
            var weight = _constTensors![$"{layerName}"];
            var bias = Tensor.FromScalar(0f, weight.Shape);
            int axis = -1;
            return F.NN.LayerNorm(axis, 1e-6F, hiddenStates, weight, bias, false);
        }

        // Qwen2Attention : SelfAtten
        // llama config find in : https://www.restack.io/p/transformer-models-answer-llama-config-json-cat-ai
        private Tuple<Expr, Expr, Expr> Qwen2SelfAtten(
            int count,
            Expr hiddenStates,
            Expr attentionMask,
            Expr paskKeyValues,
            Expr cachePosition,
            Tuple<Expr, Expr> positionEmbeddings)
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

            // batch_size, num_heads, seq_len, head_dim
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

            // update kv with cache
            (keyStates, valueStates) = UpdateKVWithCache(count, keyStates, valueStates, paskKeyValues);

            // TODO: sliding window
            // var slidingWindow = 0;
            // if (_config!.Keys.Contains("use_sliding_window") && _config!["use_sliding_window"] != null &&
            //     count >= (int)_config!["max_window_layers"])
            // {
            //     slidingWindow = (int)_config!["sliding_window"];
            // }

            // qwen use sdpa attention
            // var (hiddenStatesTmp, selfAttenWeight) = SdpaAttention(
            //     queryStates,
            //     keyStates,
            //     valueStates,
            //     attentionMask, /*TODO: 这里可能不需要,如果使用输入*/
            //     0.0f,
            //     false);

            // qwen2 use eager_attention_forward
            var (hiddenStatesTmp, selfAttenWeight) = EagerAttentionForward(
                queryStates,
                keyStates,
                valueStates,
                attentionMask,
                0.0f);

            hiddenStates = hiddenStatesTmp;

            // inputShape.Add(-1);
            var inputShape = Stack(new IR.Tuple(batch_size, seq_len, -1L), 0);
            hiddenStates = IR.F.Tensors.Reshape(hiddenStates, inputShape);
            var oProjW = _constTensors![$"model.layers.{count}.self_attn.o_proj.weight"];
            hiddenStates = F.Math.MatMul(hiddenStates, oProjW);

            // TODO: using config to judge weher need collect kv
            var mergedKeyValue = MergeKV(keyStates, valueStates);

            return Tuple.Create(hiddenStates, selfAttenWeight, mergedKeyValue);
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
                    new[] { 1, 1, 1, 1 });
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

        private Tuple<Expr, Expr> EagerAttentionForward(Expr query, Expr key, Expr value, Expr? attentionMask, float scaling)
        {
            /*
                key_states = repeat_kv(key, module.num_key_value_groups)
                value_states = repeat_kv(value, module.num_key_value_groups)

                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()

                return attn_output, attn_weights
            */
            int numKVGroups = (int)((long)_config!["num_attention_heads"] / (long)_config!["num_key_value_heads"]);
            var keyStates = RepeatKV(key, numKVGroups);
            var valueStates = RepeatKV(value, numKVGroups);
            Expr attnWeights = F.Math.MatMul(query, Transpose(keyStates, ShapeExprUtility.GetPermutation(keyStates, [2, 3]))) * scaling;
            if (attentionMask is not null)
            {
                var causalMask = Slice(
                        attentionMask,
                        new[] { 0, 0, 0, 0 },
                        Stack(new IR.Tuple(-1, -1, -1, Cast(ShapeOf(keyStates)[-2], DataTypes.Int32)), 0),
                        new[] { -1 },
                        new[] { 1, 1, 1, 1 });
                attnWeights += causalMask;
            }

            attnWeights = Softmax(attnWeights, -1);
            Expr attnOutput = F.Math.MatMul(attnWeights, valueStates);
            attnOutput = Transpose(attnOutput, ShapeExprUtility.GetPermutation(attnOutput, [1, 2]));

            // TODO: base on config to decide output attnWeights or not
            return Tuple.Create(attnOutput, attnWeights);
        }

        private Expr RepeatKV(Expr hiddenStates, int nRep)
        {
            /*
                batch, num_key_value_heads, slen, head_dim = hidden_states.shape
                if n_rep == 1:
                    return hidden_states
                hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
                return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
            */
            if (nRep == 1)
            {
                return hiddenStates;
            }

            var batch_size = Cast(ShapeOf(hiddenStates)[0], DataTypes.Int32);
            var numKVHeads = Cast(ShapeOf(hiddenStates)[1], DataTypes.Int32);
            var seqLen = Cast(ShapeOf(hiddenStates)[2], DataTypes.Int32);
            var headDim = Cast(ShapeOf(hiddenStates)[3], DataTypes.Int32);
            hiddenStates = Unsqueeze(hiddenStates, new int[] { 2 });

            hiddenStates = Expand(hiddenStates, Stack(new IR.Tuple(batch_size, numKVHeads, nRep, seqLen, headDim), 0));
            hiddenStates = Reshape(hiddenStates, Stack(new IR.Tuple(batch_size, numKVHeads * nRep, seqLen, headDim), 0));
            return hiddenStates;
        }

        private Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin)
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

        private Tuple<Expr, Expr> RotaryEmbedding(Expr x, Expr positionIds)
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
            Expr cos = F.Math.Unary(UnaryOp.Cos, emb);
            Expr sin = F.Math.Unary(UnaryOp.Sin, emb);

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

        private Tuple<Call, Call> UpdateKVWithCache(int layerIdx, Call k, Call v, Expr pastKeyValues)
        {
            // dynamic cache update kvcache
            /*
                # Update the number of seen tokens
                if layer_idx == 0:
                    self._seen_tokens += key_states.shape[-2]

                # Update the cache
                if key_states is not None:
                    if len(self.key_cache) <= layer_idx:
                        # There may be skipped layers, fill them with empty lists
                        for _ in range(len(self.key_cache), layer_idx):
                            self.key_cache.append([])
                            self.value_cache.append([])
                        self.key_cache.append(key_states)
                        self.value_cache.append(value_states)
                    elif (
                        len(self.key_cache[layer_idx]) == 0
                    ):  # fills previously skipped layers; checking for tensor causes errors
                        self.key_cache[layer_idx] = key_states
                        self.value_cache[layer_idx] = value_states
                    else:
                        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            */
            // past_key_values shape: [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
            var pastKeyValuesCurrentLayer = Gather(pastKeyValues, 0, layerIdx);
            var pastKeyCurrentLayer = Gather(pastKeyValuesCurrentLayer, 0, 0);
            var pastValueCurrentLayer = Gather(pastKeyValuesCurrentLayer, 0, 1);

            // [batch_size, num_heads, past_seq_length, head_dim]
            var key_states = Concat(new IR.Tuple(pastKeyCurrentLayer, k), -2);
            var value_states = Concat(new IR.Tuple(pastValueCurrentLayer, v), -2);

            return Tuple.Create(key_states, value_states);
        }

        private Expr MergeKV(Expr key, Expr value)
        {
            // [batchsize, num_heads, seq_length, head_dim]  ->[1,2,batchsize, num_heads, seq_length, head_dim]
            var keyStates = Unsqueeze(key, new[] { 0 });
            var valueStates = Unsqueeze(value, new[] { 0 });
            var mergedKeyValue = Concat(new IR.Tuple(keyStates, valueStates), 0);
            return Unsqueeze(mergedKeyValue, new[] { 0 });
        }
    }
}

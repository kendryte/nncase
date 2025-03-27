// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Xml;
using CommunityToolkit.HighPerformance;
using DryIoc;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
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
    // public partial class HuggingFaceImporter
    // {
    public class Qwen2 : HuggingFaceModel
    {
        private ModelInitContext? _context;

        public override void Initialize(ModelInitContext context, string dir)
        {
            base.Initialize(context, dir);
            _context = context;
        }

        // private Tuple<Call, HuggingFaceUtils.DynamicCache> VisitQwen2ForCausalLM()
        public override void VisitForCausalLM()
        {
            if (_context!.ConstTensors == null)
            {
                throw new ArgumentNullException(nameof(_context.ConstTensors));
            }

            Var input_ids = _context.Inputs![0]!;
            var attention_mask = _context.Inputs[1];
            var position_ids = _context.Inputs[2];
            var pastKeyValues = _context.Inputs[3];

            var (lastHiddenStates, allSelfKV, allHiddenStates, allSelfAttns) = LLMModel(
                input_ids,
                attention_mask,
                position_ids,
                pastKeyValues);

            var lmHead = Linear(lastHiddenStates, _context.ConstTensors["model.embed_tokens.weight"]);

            _context.Outputs!.Add("logits", lmHead);

            if (_context.CompileSession!.CompileOptions.HuggingFaceOptions.OutputAttentions)
            {
                var outAttention = Concat(new IR.Tuple(allSelfAttns.ToArray()), 0);
                _context.Outputs!["outAttention"] = outAttention;
            }

            if (_context.CompileSession.CompileOptions.HuggingFaceOptions.UseCache)
            {
                var kvCache = Concat(new IR.Tuple(allSelfKV.ToArray()), 0);
                _context.Outputs!["kvCache"] = kvCache;
            }

            if (_context.CompileSession.CompileOptions.HuggingFaceOptions.OutputHiddenStates)
            {
                var hiddenStates = Concat(new IR.Tuple(allHiddenStates.ToArray()), 0);
                _context.Outputs!["hiddenStates"] = hiddenStates;
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
        public override Tuple<Expr, List<Expr>, List<Expr>, List<Expr>> LLMModel(
            Expr input_ids,
            Expr? attentionMask,
            Expr? positionIds,
            Expr? pastKeyValues)
        {
            /*
             * 1.1 embedding
             */
            // if (inputEmbeds == null)
            // {
            var embedTokensWeight = _context!.ConstTensors!["model.embed_tokens.weight"];
            var zeroValue = 0.0f;
            object asTypeZero = zeroValue;
            if (embedTokensWeight.ElementType != DataTypes.Float32)
            {
                var method = embedTokensWeight.ElementType.CLRType.GetMethod(
                                            "op_Explicit",
                                            BindingFlags.Public | BindingFlags.Static,
                                            null,
                                            new Type[] { typeof(float) },
                                            null);

                if (method != null)
                {
                    asTypeZero = method.Invoke(null, [zeroValue])!;
                }
                else
                {
                    throw new InvalidOperationException($"cannot get float convert for type:{embedTokensWeight.ElementType}");
                }
            }

            if (_context.Config!.Keys.Contains("pad_token_id"))
            {
                for (var i = 0; i < embedTokensWeight.Shape[-1].FixedValue; i++)
                {
                    embedTokensWeight[(long)_context.Config["pad_token_id"], i] = asTypeZero!;
                }
            }

            Expr? inputEmbeds;
            if (input_ids.CheckedShape.Rank > 2 && input_ids.CheckedDataType.IsFloat())
            {
                System.Console.WriteLine("input_ids rank >2 && dtype==float32 ,regard input_id as embedding...");
                inputEmbeds = input_ids;
            }
            else
            {
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
            // TODO : _update_casualMask
            // casualMask = self._update_casualMask(
            //     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            // )
            // Call? casualMask = null;
            // TODO: outputAttentions need by config.output_attentions
            Expr historyLen = 0L;
            if (pastKeyValues != null)
            {
                historyLen = ShapeOf(pastKeyValues)[-2];
            }

            var seqLen = ShapeOf(inputEmbeds)[1];
            var cachePosition = Range(historyLen, historyLen + seqLen, 1L);
            var casualMask = UpdatecasualMask(attentionMask, inputEmbeds, cachePosition, pastKeyValues, outputAttentions: false);
            var hiddenStates = inputEmbeds;
            if (positionIds == null)
            {
                positionIds = Cast(Unsqueeze(cachePosition, 0), hiddenStates.CheckedDataType);
            }

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
            for (int i = 0; i < (int)(long)_context.Config["num_hidden_layers"]; i++)
            {
                if (_context.CompileSession!.CompileOptions.HuggingFaceOptions.OutputHiddenStates)
                {
                    allHiddenStates.Add(Unsqueeze(hiddenStates, new long[] { 0 }));
                }

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

                if (_context.CompileSession.CompileOptions.HuggingFaceOptions.OutputAttentions)
                {
                    allSelfAttns.Add(Unsqueeze(outAttention, new[] { 0L }));
                }

                if (_context.CompileSession.CompileOptions.HuggingFaceOptions.UseCache)
                {
                    allKVcaches.Add(currentKV);
                }
            }

            // the last one
            Expr lastHiddenStates = LLMLayerNorm(hiddenStates, "model.norm.weight");

            if (_context.CompileSession!.CompileOptions.HuggingFaceOptions.OutputHiddenStates)
            {
                allHiddenStates.Add(Unsqueeze(lastHiddenStates, new long[] { 0 }));
            }

            return Tuple.Create(lastHiddenStates, allKVcaches, allHiddenStates, allSelfAttns);

            // return Tuple.Create(lastHiddenStates, allSelfAttns, allKVcaches);
        }

        public override Tuple<Expr, Expr, Expr> DecodeLayer(
            int count,
            Expr hiddenStates,
            Expr? attentionMask,
            Expr? pastKeyValues,
            Expr cachePosition,
            Tuple<Expr, Expr> positionEmbeddings)
        {
            var residual = hiddenStates;
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.input_layernorm.weight");

            // TODO: using `config.attn_implementation` to choose attention implementation
            // self attention
            var (hiddenStatesTmp, outAttention, currentKV) = LLMSelfAttention(
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
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.post_attention_layernorm.weight");

            hiddenStates = LLMMlp(count, hiddenStates);

            hiddenStates = residual + hiddenStates;

            var output = hiddenStates;

            return Tuple.Create<Expr, Expr, Expr>(output, outAttention, currentKV);
        }

        public override Tuple<Expr, Expr, Expr> LLMSelfAttention(
            int count,
            Expr hiddenStates,
            Expr? attentionMask,
            Expr? paskKeyValues,
            Expr cachePosition,
            Tuple<Expr, Expr> positionEmbeddings)
        {
            var head_dim = (long)_context!.Config!["hidden_size"] / (long)_context.Config["num_attention_heads"];
            if (_context.Config!.Keys.Contains("head_dim"))
            {
                head_dim = (long)_context.Config["head_dim"];
            }

            var batch_size = ShapeOf(hiddenStates)[0];
            var seq_len = ShapeOf(hiddenStates)[1];

            // batch_size, seq_len, num_heads, head_dim
            var (queryStates, keyStates, valueStates) = QKVCompute(count, hiddenStates, batch_size, seq_len, head_dim);

            var (cos, sin) = positionEmbeddings;

            // apply_rotary_pos_emb
            (queryStates, keyStates) = ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);

            // update kv with cache
            if (paskKeyValues != null)
            {
                (keyStates, valueStates) = UpdateKVWithCache(count, keyStates, valueStates, paskKeyValues);
            }

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
            float scaling = (float)(1.0f / System.Math.Sqrt((double)head_dim));
            var (hiddenStatesTmp, selfAttenWeight) = EagerAttentionForward(
                queryStates,
                keyStates,
                valueStates,
                attentionMask,
                scaling);

            hiddenStates = hiddenStatesTmp;

            // inputShape.Add(-1);
            var inputShape = Stack(new IR.Tuple(batch_size, seq_len, -1L), 0L);
            hiddenStates = IR.F.Tensors.Reshape(hiddenStates, inputShape);
            var oProjW = _context.ConstTensors![$"model.layers.{count}.self_attn.o_proj.weight"];
            hiddenStates = Linear(hiddenStates, oProjW);

            // TODO: using config to judge weher need collect kv
            var mergedKeyValue = MergeKV(keyStates, valueStates);

            return Tuple.Create(hiddenStates, selfAttenWeight, mergedKeyValue);
        }
    }
}

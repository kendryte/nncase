// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;

namespace Nncase.Importer
{
    public class DeepSeekV3 : HuggingFaceModel
    {
        public override Tuple<Expr, Expr> DecodeLayer(
            int count,
            Expr hiddenStates,
            Expr pastKeyValues,
            Tuple<Expr, Expr> positionEmbeddings)
        {
            var residual = hiddenStates;
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.input_layernorm.weight");

            // TODO: using `config.attn_implementation` to choose attention implementation
            // self attention
            (hiddenStates, pastKeyValues) = LLMSelfAttention(
                count,
                hiddenStates,
                pastKeyValues,
                positionEmbeddings);

            // hiddenStates = IR.F.Tensors.Transpose(hiddenStates, new[] { 0, 2, 1 });
            // ModelUtils.CheckShape(hiddenStates);
            hiddenStates = residual + hiddenStates;

            // fully Connected
            residual = hiddenStates;
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.post_attention_layernorm.weight");

            // if (Config.ContainsKey("n_routed_experts")
            //     && count >= Config.GetNestedValue<long>("first_k_dense_replace")
            //     && count % Config.GetNestedValue<long>("moe_layer_freq") == 0)
            // {
            //     throw new NotImplementedException("MOE is not supported yet");
            // }
            // else
            // {
            hiddenStates = LLMMlp(count, hiddenStates);

            // }
            hiddenStates = residual + hiddenStates;

            var output = hiddenStates;

            return System.Tuple.Create<Expr, Expr>(output, pastKeyValues);
        }

        public override Tuple<float[], float> RoPEInit(Dictionary<string, object> config)
        {
            var type = config.GetNestedValue<string>("rope_scaling", "type");

            return ModelUtils.RoPEInit(config, type);
        }

        public Call DeepSeekQKVCompute(int count, Expr hiddenStates)
        {
            // public Tuple<Expr, Expr, Expr, Expr, Expr> DeepSeekQKVCompute(int count, Expr hiddenStates, Dimension seqLen, Dimension headDim)
            // {


            // var queryStates = Linear(hiddenStates, qAProjW, null, null, wAScaleQ, $"model.layers.{count}.self_attn.q_a_proj");

            // queryStates = LLMLayerNorm(queryStates, $"model.layers.{count}.self_attn.q_a_layernorm.weight");

            // queryStates = Linear(queryStates, qBProjW, null, null, wBScaleQ, $"model.layers.{count}.self_attn.q_b_proj");

            // queryStates = IR.F.Tensors.Reshape(queryStates, new RankedShape(seqLen, -1L, qkNopeHeadDim + qkRopeHeadDim));
            // queryStates = IR.F.Tensors.Transpose(queryStates, new long[] { 1, 0, 2 });
            // var qResSplit = IR.F.Tensors.Split(queryStates, -1, new RankedShape(qkNopeHeadDim, qkRopeHeadDim)).With(metadata: new IRMetadata() { OutputNames = new[] { "split qNope and qPe" } });
            // var (qNope, qPe) = (qResSplit[0], qResSplit[1]);


            var kvAProjWithMqaW = GetWeight($"model.layers.{count}.self_attn.kv_a_proj_with_mqa.weight")!;
            var kvAProjWithMqaWScale = GetWeight($"model.layers.{count}.self_attn.kv_a_proj_with_mqa.weight_scale_inv")!;
            var compressedKVAll = Linear(hiddenStates, kvAProjWithMqaW, null, null, kvAProjWithMqaWScale, $"model.layers.{count}.self_attn.kv_a_proj_with_mqa");

            // var compressedKVSplit = IR.F.Tensors.Split(compressedKVAll, -1, new RankedShape(kvLoraRank, qkRopeHeadDim)).With(metadata: new IRMetadata() { OutputNames = new[] { "split compressedKV and kPe" } });
            // var compressedKV = compressedKVSplit[0];
            // var kPe = compressedKVSplit[1];
            // kPe = IR.F.Tensors.Reshape(kPe, new RankedShape(seqLen, 1, qkRopeHeadDim));
            // kPe = IR.F.Tensors.Transpose(kPe, new long[] { 1, 0, 2 });

            // var kv = LLMLayerNorm(compressedKV, $"model.layers.{count}.self_attn.kv_a_layernorm.weight");
            // kv = Linear(kv, kvBProjW, null, null, wBScaleKV, $"model.layers.{count}.self_attn.kv_b_proj");
            // kv = IR.F.Tensors.Reshape(kv, new RankedShape(seqLen, -1L, qkNopeHeadDim + vHeadDim));
            // kv = IR.F.Tensors.Transpose(kv, new long[] { 1, 0, 2 });

            // var kvResSplit = IR.F.Tensors.Split(kv, -1, new RankedShape(qkNopeHeadDim, vHeadDim)).With(metadata: new IRMetadata() { OutputNames = new[] { "split kNope and valueStates" } });
            // var (kNope, valueStates) = (kvResSplit[0], kvResSplit[1]);

            // return System.Tuple.Create(qNope, qPe, kNope, kPe, valueStates);
            return compressedKVAll;
        }

        public override Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, long unSqueezeDim = 1)
        {
            cos = IR.F.Tensors.Unsqueeze(cos, Tensor.From<long>(new long[] { 0 }));
            sin = IR.F.Tensors.Unsqueeze(sin, Tensor.From<long>(new long[] { 0 }));

            var qShape = q.CheckedShape;
            q = IR.F.Tensors.Reshape(q, new[] { qShape[0], qShape[1], qShape[2] / 2L, 2L });
            q = IR.F.Tensors.Transpose(q, new long[] { 0, 1, 3, 2 });
            q = IR.F.Tensors.Reshape(q, qShape);

            var kShape = k.CheckedShape;
            k = IR.F.Tensors.Reshape(k, new[] { kShape[0], kShape[1], kShape[2] / 2L, 2L });
            k = IR.F.Tensors.Transpose(k, new long[] { 0, 1, 3, 2 });
            k = IR.F.Tensors.Reshape(k, kShape);

            var qEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, q, cos),
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(q), sin));
            var kEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, k, cos),
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(k), sin));
            return System.Tuple.Create(qEmbed, kEmbed);
        }

        public override Tuple<Expr, Expr> LLMSelfAttention(
                int count,
                Expr hiddenStates,
                Expr paskKeyValues,
                Tuple<Expr, Expr> positionEmbeddings)
        {
            var head_dim = (long)Context!.Config!["hidden_size"] / (long)Context.Config["num_attention_heads"];
            if (Context.Config!.Keys.Contains("head_dim"))
            {
                head_dim = (long)Context.Config["head_dim"];
            }

            var pagedAttentionConfig = (IPagedAttentionConfig)Context.ImportOptions!.HuggingFaceOptions.Config;

            // var batch_size = hiddenStates.CheckedShape[0];
            var seq_len = hiddenStates.CheckedShape[0];

            // [seq_len, kv_lora_rank + qk_rope_head_dim]
            var compressedKV = DeepSeekQKVCompute(count, hiddenStates);
            var queryStates = hiddenStates;
            var qAProjW = GetWeight($"model.layers.{count}.self_attn.q_a_proj.weight")!;
            var qAProjScaleQ = GetWeight($"model.layers.{count}.self_attn.q_a_proj.weight_scale_inv");
            var qALayerNormW = GetWeight($"model.layers.{count}.self_attn.q_a_layernorm.weight")!;
            var qBProjW = GetWeight($"model.layers.{count}.self_attn.q_b_proj.weight")!;
            var qBProjScaleQ = GetWeight($"model.layers.{count}.self_attn.q_b_proj.weight_scale_inv");
            var kvALayerNormW = GetWeight($"model.layers.{count}.self_attn.kv_a_layernorm.weight")!;
            var kvBProjW = GetWeight($"model.layers.{count}.self_attn.kv_b_proj.weight")!;
            var kvBProjScaleQ = GetWeight($"model.layers.{count}.self_attn.kv_b_proj.weight_scale_inv");
            var qkNopeHeadDim = Config.GetNestedValue<long>("qk_nope_head_dim");
            var qkRopeHeadDim = Config.GetNestedValue<long>("qk_rope_head_dim");
            var kvLoraRank = Config.GetNestedValue<long>("kv_lora_rank");
            var vHeadDim = Config.GetNestedValue<long>("v_head_dim");
            // [seq_len, hidden_size]
            AttentionDimKind[] qSrcLayout = [AttentionDimKind.Seq, AttentionDimKind.Dim];
            AttentionDimKind[] kvSrcLayout = [AttentionDimKind.Seq, AttentionDimKind.Dim];
            {
                AttentionDimKind[] kvDestLayout = { AttentionDimKind.Dim, AttentionDimKind.Seq };
                var padedK = seq_len is DimVar ? IR.F.NN.Pad(compressedKV, new(new(0, ((long)seq_len.Metadata.Range!.Value.Max) - seq_len), new(0, 0)), PadMode.Constant, Tensor.Zero(DataTypes.Float32)) : compressedKV;
                var kvPerms = ModelUtils.GetLayoutPerm(kvSrcLayout, kvDestLayout);
                var (kvLanes, kvPackedAxis) = ModelUtils.GetQKVPackParams(pagedAttentionConfig, kvDestLayout);
                var transK = IR.F.Tensors.Transpose(padedK, kvPerms);
                var castK = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(transK, pagedAttentionConfig.KVPrimType) : transK;
                var packedK = kvLanes.Length > 0 ? IR.F.Tensors.Pack(castK, kvLanes, kvPackedAxis) : castK;
                paskKeyValues = IR.F.NN.UpdatePagedAttentionKVCache(packedK, paskKeyValues, AttentionCacheKind.Key, count, kvDestLayout);
            }

            var scaling = Tensor.FromScalar((float)(1.0f / System.Math.Sqrt((double)head_dim)));

            // var mergedKeyValue = MergeKV(keyStates, valueStates);
            AttentionDimKind[] qDestLayout = { AttentionDimKind.Dim, AttentionDimKind.Seq };
            // var qPerm = ModelUtils.GetLayoutPerm(qSrcLayout, qDestLayout);
            // var (qLanes, qPackedAxis) = ModelUtils.GetQKVPackParams(pagedAttentionConfig, qDestLayout);
            // var padedQ = seq_len is DimVar ? IR.F.NN.Pad(queryStates, new(new(0, ((long)seq_len.Metadata.Range!.Value.Max) - seq_len), new(0, 0)), PadMode.Constant, Tensor.Zero(DataTypes.Float32)) : queryStates;
            // var transQ = IR.F.Tensors.Transpose(padedQ, qPerm);
            var castQ = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(queryStates, pagedAttentionConfig.KVPrimType) : queryStates;
            // var packedQ = qLanes.Length > 0 ? IR.F.Tensors.Pack(castQ, qLanes, qPackedAxis) : castQ;

            // cpu : [q_head, max_query_len, max_seq_len + 1 ]<primtype>
            var extra_size = pagedAttentionConfig.KVPrimType.SizeInBytes * (long)Context.Config["num_attention_heads"] * Context.ImportOptions.HuggingFaceOptions.MaxModelLen * (Context.ImportOptions.HuggingFaceOptions.MaxModelLen + 1);

            // xpu : 10 mb.
            if (Context.CompileSession!.Target.Name == "xpu")
            {
                extra_size = 10 * 1024 * 1024;
            }

            var output = IR.F.NN.MLAPagedAttention(
                castQ,
                paskKeyValues,
                IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, [extra_size]),
                scaling.CastTo(pagedAttentionConfig.KVPrimType, CastMode.KDefault),
                qAProjW,
                qAProjW, //qAProjScaleQ,
                qALayerNormW,
                qBProjW,
                qBProjW, //qBProjScaleQ,
                kvBProjW,
                kvBProjW, //kvBProjScaleQ,
                kvALayerNormW,
                count,
                qDestLayout,
                (int)(long)Context.Config["hidden_size"],
                (int)(long)Context.Config["num_attention_heads"],
                (int)kvLoraRank,
                (int)qkNopeHeadDim,
                (int)qkRopeHeadDim,
                (int)vHeadDim);

            // output = qLanes.Length > 0 ? IR.F.Tensors.Unpack(output, qLanes, qPackedAxis) : output;
            // output = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(output, DataTypes.Float32) : output;
            // output = IR.F.Tensors.Transpose(output, ModelUtils.GetLayoutPerm(qDestLayout, qSrcLayout));
            // output = seq_len is DimVar ? IR.F.Tensors.Slice(output, new[] { 0 }, new Dimension[] { seq_len }, new[] { 1 }, new[] { 1 }) : output;
            output = IR.F.Tensors.Transpose(output, new[] { 1, 0, 2 });

            output = IR.F.Tensors.Reshape(output, new RankedShape(seq_len, -1L));
            var oProjW = GetWeight($"model.layers.{count}.self_attn.o_proj.weight")!;

            var ifScaleO = GetWeight($"model.layers.{count}.self_attn.o_proj.input_scale");
            var wScaleO = GetWeight($"model.layers.{count}.self_attn.o_proj.weight_scale");
            ModelUtils.CheckShape(output);
            output = Linear(output, oProjW, null, ifScaleO, wScaleO, $"model.layers.{count}.self_attn.o_proj");
            return System.Tuple.Create(output, paskKeyValues);
        }
    }
}

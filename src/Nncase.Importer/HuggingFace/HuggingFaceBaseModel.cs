// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using Tuple = System.Tuple;
using TypeCode = Nncase.Runtime.TypeCode;

namespace Nncase.Importer;

/// <summary>
/// This model architectures base on LlamaForCausalLM.
/// </summary>
public abstract class HuggingFaceModel
{
    protected ModelInitContext? Context { get; private set; }

    public virtual void Initialize(ModelInitContext context, string dir)
    {
        Context = context;
    }

    public virtual (IEnumerable<IVar> Inputs, Dictionary<IVar, Dimension[]> VarMap) CreateInputs()
    {
        var hiddenSize = (long)Context!.Config!["hidden_size"];
        _ = (long)Context.Config!["num_hidden_layers"];
        var num_attention_heads = (long)Context.Config!["num_attention_heads"];
        _ = hiddenSize / num_attention_heads;
        if (Context.Config.ContainsKey("head_dim"))
        {
            _ = (long)Context.Config["head_dim"];
        }

        _ = (long)Context.Config!["num_key_value_heads"];

        Context.Inputs = [];
        Context.DynVarMap = new Dictionary<string, DimVar>();
        var varMap = new Dictionary<IVar, Dimension[]>();

        var bucketOptions = Context.CompileSession!.CompileOptions.ShapeBucketOptions;
        Context.FixVarMap = bucketOptions.FixVarMap;

        // local test set
        // _fixVarMap["sequence_length"] = 10;
        // _fixVarMap["history_len"] = 0;
        // TODO: control by config file
        if (!Context.FixVarMap.ContainsKey("sequence_length"))
        {
            Context.DynVarMap["sequence_length"] = new DimVar("sequence_length");
            Context.DynVarMap["sequence_length"].Metadata.Range = new(bucketOptions.RangeInfo["sequence_length"].Min, bucketOptions.RangeInfo["sequence_length"].Max);
        }

        // if (!_fixVarMap.ContainsKey("history_len"))
        // {
        //     _dynVarMap["history_len"] = new DimVar("history_len");
        //     _dynVarMap["history_len"].Metadata.Range=new (4096, 8192);
        // }
        // if (!Context.FixVarMap.ContainsKey("batch_size"))
        // {
        //     Context.DynVarMap["batch_size"] = new DimVar("batch_size");
        //     Context.DynVarMap["batch_size"].Metadata.Range = new(1, 4);
        // }
        var inputIdsShapeExpr = new Dimension[]
        {
            // Context.FixVarMap.ContainsKey("batch_size") ? Context.FixVarMap["batch_size"] : Context.DynVarMap["batch_size"],
            Context.FixVarMap.ContainsKey("sequence_length") ? Context.FixVarMap["sequence_length"] : Context.DynVarMap["sequence_length"],
        };

        // var attentionMaskShapeExpr = new Expr[]
        // {
        //         1L, // _dynVarMap["batch_size"],
        //         20L, // _dynVarMap["sequence_length"]
        // };
        // var positionIdsShapeExpr = new Expr[] {
        //                                         1L, // _dynVarMap["batch_size"],
        //                                         20L, // _dynVarMap["sequence_length"]
        //                                         };

        // // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        // var pastKeyValueShapeExpr = new Expr[] { numsHiddenLayers,
        //                                              2L,
        //                                              1L, // _dynVarMap["batch_size"],
        //                                              numKVHeads,
        //                                              0, // _dynVarMap["history_len"],
        //                                              headDim, };
        var inputIds = new Var(
            "input_ids",
            new TensorType(DataTypes.Int64, new RankedShape(inputIdsShapeExpr)));

        // var attentionMask = new Var(
        //     "attention_mask",
        //     new TensorType(
        //         DataTypes.Float32,
        //         new RankedShape(
        //             1L, // _dynVarMap["batch_size"],
        //             20L)));
        // var positionIds = new Var(
        //     "position_ids",
        //     new TensorType(DataTypes.Float32, new RankedShape(
        //                                     1L, // _dynVarMap["batch_size"],
        //                                     20L)));

        // // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        // var pastKeyValue = new Var(
        //     "past_key_values",
        //     new TensorType(DataTypes.Float32, new RankedShape(
        //         numsHiddenLayers,
        //         2L,
        //         1L, // _dynVarMap["batch_size"],
        //         numKVHeads,
        //         0, // _dynVarMap["history_len"],
        //         headDim)));
        var pastKeyValue = new Var("kvCache", TensorType.Scalar(
            new ReferenceType(new PagedAttentionKVCacheType { Config = (IPagedAttentionConfig)Context.ImportOptions!.HuggingFaceOptions.Config })));

        Context.Inputs.Add(inputIds);
        Context.Inputs.Add(null); // attentionMask
        Context.Inputs.Add(null); // positionIds
        Context.Inputs.Add(pastKeyValue); // pastKeyValue

        // _inputs.Add(attentionMask);
        // _inputs.Add(positionIds);
        // _inputs.Add(pastKeyValue);
        varMap[inputIds] = inputIdsShapeExpr;
        if (!Context.FixVarMap.ContainsKey("sequence_length"))
        {
            varMap[Context.DynVarMap["sequence_length"]] = [Context.DynVarMap["sequence_length"]];
        }

        // varMap[attentionMask] = attentionMaskShapeExpr;
        // varMap[positionIds] = positionIdsShapeExpr;
        // varMap[pastKeyValue] = pastKeyValueShapeExpr;
        var inputs = new List<IVar> { };

        // for the input is optional
        foreach (var input in Context.Inputs)
        {
            if (input != null)
            {
                inputs.Add(input);
            }
        }

        Context.CompileSession.CompileOptions.ShapeBucketOptions.VarMap = varMap;
        return (inputs, varMap);
    }

    // public abstract Expr CreateOutputs();
    public virtual BaseExpr CreateOutputs()
    {
        // TODO: use self.config.output_attention to judge wether output kvache
        Expr? logits = null;
        Expr? lastHiddenStates = null;
        Expr? hiddenStates = null;

        if (Context!.ImportOptions!.HuggingFaceOptions.OutputLogits)
        {
            logits = Context.Outputs["logits"];
        }
        else
        {
            lastHiddenStates = Context.Outputs["lastHiddenStates"];
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            hiddenStates = Context.Outputs["hiddenStates"];
        }

        var output = new List<Expr?> { logits, lastHiddenStates, hiddenStates };
        output.RemoveAll(item => item == null);

        return new IR.Tuple([.. output!]);
    }

    public virtual Expr RepeatKV(Expr hiddenStates, long nRep)
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

        var batch_size = hiddenStates.CheckedShape[0];
        var numKVHeads = hiddenStates.CheckedShape[1];
        var seqLen = hiddenStates.CheckedShape[2];
        var headDim = hiddenStates.CheckedShape[3];
        hiddenStates = IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 2 });

        var tmp = new RankedShape(batch_size, numKVHeads, nRep, seqLen, headDim);
        hiddenStates = IR.F.Tensors.Expand(hiddenStates, tmp);
        hiddenStates = IR.F.Tensors.Reshape(hiddenStates, new RankedShape(batch_size, numKVHeads * nRep, seqLen, headDim));
        return hiddenStates;
    }

    public virtual System.Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, long unSqueezeDim = 1)
    {
        // q_embed = (q * cos) + (rotate_half(q) * sin)
        // k_embed = (k * cos) + (rotate_half(k) * sin)
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

    // def rotate_half(x):
    // """Rotates half the hidden dims of the input."""
    // x1 = x[..., : x.shape[-1] // 2]
    // x2 = x[..., x.shape[-1] // 2 :]
    // return torch.cat((-x2, x1), dim=-1)
    public virtual Call RotateHalf(Expr x)
    {
        var xS3 = x.CheckedShape[^1];
        var x1 = IR.F.Tensors.Slice(
            x,
            new[] { 0L },
            new RankedShape(xS3 / 2L),
            new[] { -1L },
            new[] { 1L });
        var x2 = IR.F.Tensors.Slice(
            x,
            new RankedShape(xS3 / 2L),
            new RankedShape(xS3),
            new[] { -1L },
            new[] { 1L });

        return IR.F.Tensors.Concat(new IR.Tuple(IR.F.Math.Neg(x2), x1), -1);
    }

    public virtual Call LLMLayerNorm(Expr hiddenStates, string layerName)
    {
        // originType->fp32->dolayernorm->origintype
        // fit layernorm partten 5
        var originDtype = hiddenStates.CheckedDataType;
        hiddenStates = IR.F.Tensors.Cast(hiddenStates, DataTypes.Float32);

        Expr weight = Context!.ConstTensors![$"{layerName}"];

        weight = IR.F.Tensors.Cast(weight, DataTypes.Float32);
        var bias = Tensor.FromScalar(0f, (RankedShape)weight.CheckedShape);
        int axis = -1;

        float eps = 1e-6F;
        if (Context!.Config!.ContainsKey("rms_norm_eps"))
        {
            eps = (float)Context!.Config!.GetNestedValue<double>("rms_norm_eps");
        }

        return IR.F.Tensors.Cast(IR.F.NN.LayerNorm(axis, eps, hiddenStates, weight, bias, false), originDtype);
    }

    public virtual Call Linear(Expr expr, Tensor weight, Tensor? bias = null, Tensor? scaleIf = null, Tensor? scaleW = null, string layerName = "")
    {
        if (scaleIf is not null && scaleW is not null)
        {
            // TODO: only support by tensor quant now!
            if (scaleIf.Rank > 1 || scaleW.Rank > 1)
            {
                throw new NotImplementedException("only support by tensor quant now: ");
            }

            var qScaleA = 1.0f / scaleIf.ToArray<float>()[0];
            var qScaleB = 1.0f / scaleW.ToArray<float>()[0];
            var deqScaleA = 1.0f / qScaleA;
            var deqScaleB = 1.0f / qScaleB;

            var qInput = expr.CheckedDataType switch
            {
                var t when t == DataTypes.BFloat16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, (BFloat16)qScaleA),
                var t when t == DataTypes.Float16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, (Half)qScaleA),
                _ => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA),
            };

            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var transposed_weight = IR.F.Tensors.Transpose(weight, new long[] { 1, 0 }).Evaluate().AsTensor();
            var qWeights = IR.F.Tensors.Cast(transposed_weight, DataTypes.Float8E4M3);
            var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeights, expr.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            var result = expr.CheckedDataType switch
            {
                var t when t == DataTypes.BFloat16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, (BFloat16)(deqScaleA * deqScaleB)),
                var t when t == DataTypes.Float16 => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, (Half)(deqScaleA * deqScaleB)),
                _ => Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA * deqScaleB),
            };
            if (bias != null)
            {
                bias = bias.CastTo(expr.CheckedDataType);
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else if (scaleIf is null && scaleW is not null)
        {
            long[] axes = new long[] { expr.CheckedShape.Rank - 1 };
            var max = Nncase.IR.F.Tensors.ReduceMax(expr, axes, float.MinValue, true);
            var min = Nncase.IR.F.Tensors.ReduceMin(expr, axes, float.MaxValue, true);
            var limit = Nncase.IR.F.Math.Max(Nncase.IR.F.Math.Abs(max), Nncase.IR.F.Math.Abs(min));
            if (limit.CheckedDataType != DataTypes.Float32)
            {
                limit = Nncase.IR.F.Tensors.Cast(limit, DataTypes.Float32);
            }

            var qScaleA = Nncase.IR.F.Math.Div((float)Float8E4M3.MaxNormal, limit);
            var deqScaleA = Nncase.IR.F.Math.Div(1.0f, qScaleA);
            var deqScaleB = scaleW;

            if (qScaleA.CheckedDataType != expr.CheckedDataType)
            {
                qScaleA = Nncase.IR.F.Tensors.Cast(qScaleA, expr.CheckedDataType);
            }

            var qInput = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, expr, qScaleA);
            qInput = Nncase.IR.F.Tensors.Cast(qInput, DataTypes.Float8E4M3);
            var transposed_weight = IR.F.Tensors.Transpose(weight, new long[] { 1, 0 }).Evaluate().AsTensor();
            var qWeights = IR.F.Tensors.Cast(transposed_weight, DataTypes.Float8E4M3);
            var qMatmul = Nncase.IR.F.Math.MatMul(qInput, qWeights, expr.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });

            if (deqScaleA.CheckedDataType != expr.CheckedDataType)
            {
                deqScaleA = Nncase.IR.F.Tensors.Cast(deqScaleA, expr.CheckedDataType);
            }

            var result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, qMatmul, deqScaleA);

            if (deqScaleB.Rank == 2)
            {
                long[] dims = System.Linq.Enumerable.Range(0, qMatmul.CheckedShape.Rank).Select(i => 1L).ToArray();
                dims[dims.Length - 1] = deqScaleB.Shape[0].FixedValue;
                deqScaleB = Tensor.From<float>(deqScaleB.ToArray<float>(), dims);
                if (deqScaleB.ElementType != expr.CheckedDataType)
                {
                    deqScaleB = deqScaleB.CastTo(expr.CheckedDataType);
                }
            }

            result = Nncase.IR.F.Math.Binary(Nncase.BinaryOp.Mul, result, deqScaleB);
            if (bias != null)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
        else
        {
            var transposed_weight = IR.F.Tensors.Transpose(weight, new long[] { 1, 0 });
            var result = IR.F.Math.MatMul(expr, transposed_weight, expr.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { layerName } });
            if (bias != null)
            {
                result = IR.F.Math.Add(result, bias);
            }

            return result;
        }
    }

    public virtual Tuple<Expr, Expr> DecodeLayer(
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
        hiddenStates = residual + hiddenStates;

        // fully Connected
        residual = hiddenStates;
        hiddenStates = LLMLayerNorm(
            hiddenStates,
            $"model.layers.{count}.post_attention_layernorm.weight");

        hiddenStates = LLMMlp(count, hiddenStates);

        hiddenStates = residual + hiddenStates;

        var output = hiddenStates;

        return System.Tuple.Create<Expr, Expr>(output, pastKeyValues);
    }

    public virtual Call LLMMlp(int count, Expr hiddenStates)
    {
        var gateProjW = Context!.ConstTensors![$"model.layers.{count}.mlp.gate_proj.weight"];
        var upProjW = Context.ConstTensors![$"model.layers.{count}.mlp.up_proj.weight"];
        var downProjW = Context.ConstTensors![$"model.layers.{count}.mlp.down_proj.weight"];
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.gate_proj.input_scale", out var ifScaleGate);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.gate_proj.weight_scale", out var wScaleGate);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.up_proj.input_scale", out var ifScaleUp);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.up_proj.weight_scale", out var wScaleUp);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.down_proj.input_scale", out var ifScaleDown);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.mlp.down_proj.weight_scale", out var wScaleDown);

        var tmp = Linear(hiddenStates, gateProjW, null, ifScaleGate, wScaleGate, $"model.layers.{count}.mlp.gate_proj");
        if (Context!.Config!.ContainsKey("hidden_act"))
        {
            var actType = Context!.Config!.GetNestedValue<string>("hidden_act");
            tmp = ModelUtils.ActFunc(tmp, actType);
        }

        return Linear(tmp * Linear(hiddenStates, upProjW, null, ifScaleUp, wScaleUp, $"model.layers.{count}.mlp.up_proj"), downProjW, null, ifScaleDown, wScaleDown, $"model.layers.{count}.mlp.down_proj");
    }

    public virtual Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Dimension seqLen, Dimension headDim)
    {
        var hidden_shape = new RankedShape(seqLen, -1L, headDim);

        var qProjW = Context!.ConstTensors![$"model.layers.{count}.self_attn.q_proj.weight"];
        Tensor? qProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.q_proj.bias"))
        {
            qProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.q_proj.bias"];
        }

        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.q_proj.input_scale", out var ifScaleQ);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.q_proj.weight_scale", out var wScaleQ);
        var queryStates = Linear(hiddenStates, qProjW, qProjB, ifScaleQ, wScaleQ, $"model.layers.{count}.self_attn.q_proj");
        queryStates = IR.F.Tensors.Reshape(queryStates, hidden_shape);

        // batch_size, num_heads, seq_len, head_dim
        queryStates = IR.F.Tensors.Transpose(queryStates, new long[] { 1, 0, 2 });

        var kProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.weight"];
        Tensor? kProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.k_proj.bias"))
        {
            kProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.bias"];
        }

        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.k_proj.input_scale", out var ifScaleK);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.k_proj.weight_scale", out var wScaleK);
        var keyStates = Linear(hiddenStates, kProjW, kProjB, ifScaleK, wScaleK, $"model.layers.{count}.self_attn.k_proj");
        keyStates = IR.F.Tensors.Reshape(keyStates, hidden_shape);
        keyStates = IR.F.Tensors.Transpose(keyStates, new long[] { 1, 0, 2 });

        var vProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.weight"];
        Tensor? vProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.v_proj.bias"))
        {
            vProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.bias"];
        }

        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.v_proj.input_scale", out var ifScaleV);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.v_proj.weight_scale", out var wScaleV);
        var valueStates = Linear(hiddenStates, vProjW, vProjB, ifScaleV, wScaleV, $"model.layers.{count}.self_attn.v_proj");
        valueStates = IR.F.Tensors.Reshape(valueStates, hidden_shape);
        valueStates = IR.F.Tensors.Transpose(valueStates, new long[] { 1, 0, 2 });
        return System.Tuple.Create(queryStates, keyStates, valueStates);
    }

    public virtual Tuple<Expr, Expr> EagerAttentionForward(Expr query, Expr key, Expr value, Expr? attentionMask, float scaling)
    {
        var numKVGroups = (long)Context!.Config!["num_attention_heads"] / (long)Context.Config!["num_key_value_heads"];
        var keyStates = RepeatKV(key, numKVGroups);
        var valueStates = RepeatKV(value, numKVGroups);
        var scalingExpr = IR.F.Tensors.Cast(Tensor.FromScalar(scaling), query.CheckedDataType);
        Expr attnWeights = IR.F.Math.MatMul(query, IR.F.Tensors.Transpose(keyStates, ShapeExprUtility.GetPermutation(keyStates, [2, 3])), query.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { "EagerAttentionForward0" } }) * scalingExpr;
        if (attentionMask is not null)
        {
            var causalMask = IR.F.Tensors.Slice(
                    attentionMask,
                    new[] { 0L },
                    new RankedShape(keyStates.CheckedShape[^2]),
                    new[] { 3L },
                    new[] { 1L });

            attnWeights += causalMask;
        }

        attnWeights = IR.F.Tensors.Cast(IR.F.NN.Softmax(IR.F.Tensors.Cast(attnWeights, DataTypes.Float32), 3L), valueStates.CheckedDataType);

        Expr attnOutput = IR.F.Math.MatMul(attnWeights, valueStates, query.CheckedDataType).With(metadata: new IRMetadata() { OutputNames = new[] { "EagerAttentionForward1" } });
        attnOutput = IR.F.Tensors.Transpose(attnOutput, ShapeExprUtility.GetPermutation(attnOutput, [1, 2]));

        // TODO: base on config to decide output attnWeights or not
        return System.Tuple.Create(attnOutput, attnWeights);
    }

    public virtual Tuple<Expr, Expr> RotaryEmbedding(Expr x, Expr kvObject, float[] invFreq, float attentionScaling)
    {
        var positionIds = IR.F.NN.GetPositionIds(IR.F.Shapes.AsTensor(x.CheckedShape[0]), kvObject);
        positionIds = IR.F.Tensors.Unsqueeze(IR.F.Tensors.Cast(positionIds, DataTypes.Float32), [1]);
        var invFreqExpanded = invFreq.Concat(invFreq).ToArray();
        var emb = IR.F.Math.Mul(invFreqExpanded, positionIds).With(metadata: new IRMetadata()
        {
            OutputNames = ["RotaryEmbedding"],
        });

        // add attention scaling
        Expr cos = IR.F.Math.Unary(UnaryOp.Cos, emb) * attentionScaling;
        Expr sin = IR.F.Math.Unary(UnaryOp.Sin, emb) * attentionScaling;

        cos = IR.F.Tensors.Cast(cos, x.CheckedDataType);
        sin = IR.F.Tensors.Cast(sin, x.CheckedDataType);

        return System.Tuple.Create(cos, sin);
    }

    public virtual Tuple<Call, Call> UpdateKVWithCache(int layerIdx, Call k, Call v, Expr pastKeyValues)
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
        var pastKeyValuesCurrentLayer = IR.F.Tensors.Gather(pastKeyValues, 0, (long)layerIdx);
        var pastKeyCurrentLayer = IR.F.Tensors.Gather(pastKeyValuesCurrentLayer, 0, 0L);
        var pastValueCurrentLayer = IR.F.Tensors.Gather(pastKeyValuesCurrentLayer, 0, 1L);

        // [batch_size, num_heads, past_seq_length, head_dim]
        var key_states = IR.F.Tensors.Concat(new IR.Tuple(pastKeyCurrentLayer, k), -2);
        var value_states = IR.F.Tensors.Concat(new IR.Tuple(pastValueCurrentLayer, v), -2);

        return System.Tuple.Create(key_states, value_states);
    }

    public virtual Expr MergeKV(Expr key, Expr value)
    {
        // [batchsize, num_heads, seq_length, head_dim]  ->[1,2,batchsize, num_heads, seq_length, head_dim]
        var keyStates = IR.F.Tensors.Unsqueeze(key, new long[] { 0 });
        var valueStates = IR.F.Tensors.Unsqueeze(value, new long[] { 0 });
        var mergedKeyValue = IR.F.Tensors.Concat(new IR.Tuple(keyStates, valueStates), 0);
        return IR.F.Tensors.Unsqueeze(mergedKeyValue, new long[] { 0 });
    }

    public virtual Expr Prepare4dCausalAttentionMaskWithCachePosition(
                            Expr? attentionMask,
                            Dimension seqLen,
                            Dimension targtLen,
                            DataType dtype,
                            Expr cachePosition,
                            Dimension batchSize,
                            Expr? pastKeyValues)
    {
        Expr? casualMask;
        if (attentionMask != null && attentionMask.CheckedShape.Rank == 4)
        {
            Console.WriteLine("attention_mask is already 4D, no need to prepare 4D causal mask.");
            casualMask = attentionMask;
        }
        else
        {
            var mask_shape = new RankedShape([seqLen, targtLen]);
            Tensor minValue;

            // get the min value for current dtype
            FieldInfo? minValueField = dtype.CLRType.GetField("MinValue", BindingFlags.Public | BindingFlags.Static);
            if (minValueField != null)
            {
                var min = minValueField.GetValue(null)!;
                minValue = Tensor.FromScalar(dtype, min, [1L]);
            }
            else
            {
                PropertyInfo? minValueProperty = dtype.CLRType.GetProperty("MinValue", BindingFlags.Public | BindingFlags.Static);
                if (minValueProperty != null)
                {
                    var min = minValueProperty.GetValue(null)!;
                    minValue = Tensor.FromScalar(dtype, min, [1L]);
                }
                else
                {
                    throw new InvalidOperationException($"cannot get current dtype's min value:{dtype.CLRType}");
                }
            }

            casualMask = IR.F.Tensors.ConstantOfShape(mask_shape, minValue);

            /*
                min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            ;
            */
            var diagonalAttendMask = IR.F.Tensors.Range(0L, IR.F.Shapes.AsTensor(targtLen), 1L) > IR.F.Tensors.Reshape(cachePosition, new long[] { -1, 1 });

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

            // casualMask = casualMask * IR.F.Tensors.Cast(diagonalAttendMask, casualMask.CheckedDataType);
            casualMask = IR.F.Tensors.Where(diagonalAttendMask, casualMask, IR.F.Tensors.Cast(0f, casualMask.CheckedDataType));

            // casualMask = casualMask[None, None, :, :].expand(batch_size, 1, -1, -1)
            var expandShape = new RankedShape(batchSize, 1L, seqLen, targtLen);
            casualMask = IR.F.Tensors.Unsqueeze(casualMask, new long[] { 0, 1 });
            casualMask = IR.F.Tensors.Expand(casualMask, expandShape);
            /*
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
            */
            if (attentionMask != null)
            {
                var maskLength = attentionMask.CheckedShape[^1];
                var paddingMask = IR.F.Tensors.Slice(
                    casualMask,
                    new[] { 0L, 0L, 0L, 0L },
                    new RankedShape(maskLength),
                    new[] { 0L, 1L, 2L, 3L },
                    new[] { 1L, 1L, 1L, 1L });
                paddingMask += IR.F.Tensors.Unsqueeze(attentionMask, new long[] { 1, 2 });

                /*
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                */
                paddingMask = IR.F.Math.Equal(paddingMask, 0.0f);
                var maskPart = IR.F.Tensors.Slice(
                    casualMask,
                    new[] { 0L },
                    new RankedShape(maskLength),
                    new[] { -1L },
                    new[] { 1L });

                var minDtypeMatrix = IR.F.Tensors.ConstantOfShape(maskPart.CheckedShape, minValue);

                maskPart = IR.F.Tensors.Where(paddingMask, minDtypeMatrix, maskPart);

                // TODO: for dynamic cache, maskLength== sequence length == target length
                //  just return maskPart
                var leftPart = IR.F.Tensors.Slice(
                    casualMask,
                    new RankedShape(maskLength),
                    new RankedShape(casualMask.CheckedShape[^1]),
                    new[] { -1L },
                    new[] { 1L });
                casualMask = IR.F.Tensors.Concat(new IR.Tuple(maskPart, leftPart), -1);
            }
        }

        return casualMask;
    }

    public virtual Expr UpdatecasualMask(Expr? attentionMask, Expr inputsEmbeds, Expr cachePosition, Expr? pastKeyValues, bool outputAttentions = false)
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
        Dimension historyLen = 0L;

        // if (pastKeyValues != null)
        // {
        //     // FIXME: use api to get historyLen.
        //     historyLen = pastKeyValues.CheckedShape[-2];
        // }
        var batchSize = inputsEmbeds.CheckedShape[0];
        var seqLen = inputsEmbeds.CheckedShape[1];
        var targetLength = historyLen + seqLen + 1L;
        if (attentionMask != null)
        {
            targetLength = attentionMask.CheckedShape[^1];
        }

        var dtype = inputsEmbeds.CheckedDataType;

        Expr casualMask = Prepare4dCausalAttentionMaskWithCachePosition(
                                            attentionMask,
                                            seqLen,
                                            targetLength,
                                            dtype,
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

    public virtual Call Embeding(Expr input, Tensor embedingWeight, long? paddingIdx = null)
    {
        var embedingDim = embedingWeight.Shape[^1];
        var gatherResult = IR.F.Tensors.Gather(embedingWeight, 0, input);
        if (paddingIdx == null)
        {
            return gatherResult;
        }
        else
        {
            var zeros = Tensor.Zeros(embedingWeight.ElementType, new long[] { embedingDim.FixedValue });
            var paddingMask = IR.F.Math.Equal(input, paddingIdx);
            paddingMask = IR.F.Tensors.Unsqueeze(paddingMask, new long[] { 1 });
            paddingMask = IR.F.Tensors.Expand(paddingMask, gatherResult.CheckedShape);
            var results = IR.F.Tensors.Where(paddingMask, zeros, gatherResult);
            return results;
        }
    }

    public virtual Tuple<Expr, Expr> LLMSelfAttention(
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
        var (queryStates, keyStates, valueStates) = QKVCompute(count, hiddenStates, seq_len, head_dim);

        var (cos, sin) = positionEmbeddings;

        // // apply_rotary_pos_emb
        (queryStates, keyStates) = ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);

        AttentionDimKind[] qSrcLayout = [AttentionDimKind.Head, AttentionDimKind.Seq, AttentionDimKind.Dim];
        AttentionDimKind[] kvSrcLayout = [AttentionDimKind.Head, AttentionDimKind.Seq, AttentionDimKind.Dim];
        {
            AttentionDimKind[] kvDestLayout = { AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq };
            var padedK = seq_len is DimVar ? IR.F.NN.Pad(keyStates, new(new(0, 0), new(0, ((long)seq_len.Metadata.Range!.Value.Max) - seq_len), new(0, 0)), PadMode.Constant, Tensor.Zero(DataTypes.Float32)) : keyStates;
            var kvPerms = ModelUtils.GetLayoutPerm(kvSrcLayout, kvDestLayout);
            var (kvLanes, kvPackedAxis) = ModelUtils.GetQKVPackParams(pagedAttentionConfig, kvDestLayout);
            var transK = IR.F.Tensors.Transpose(padedK, kvPerms);
            var castK = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(transK, pagedAttentionConfig.KVPrimType) : transK;
            var packedK = kvLanes.Length > 0 ? IR.F.Tensors.Pack(castK, kvLanes, kvPackedAxis) : castK;
            paskKeyValues = IR.F.NN.UpdatePagedAttentionKVCache(packedK, paskKeyValues, AttentionCacheKind.Key, count, kvDestLayout);

            var padedV = seq_len is DimVar ? IR.F.NN.Pad(valueStates, new(new(0, 0), new(0, ((long)seq_len.Metadata.Range!.Value.Max) - seq_len), new(0, 0)), PadMode.Constant, Tensor.Zero(DataTypes.Float32)) : valueStates;
            var transV = IR.F.Tensors.Transpose(padedV, kvPerms);
            var castV = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(transV, pagedAttentionConfig.KVPrimType) : transV;
            var packedV = kvLanes.Length > 0 ? IR.F.Tensors.Pack(castV, kvLanes, kvPackedAxis) : castV;
            paskKeyValues = IR.F.NN.UpdatePagedAttentionKVCache(packedV, paskKeyValues, AttentionCacheKind.Value, count, kvDestLayout);
        }

        var scaling = Tensor.FromScalar((float)(1.0f / System.Math.Sqrt((double)head_dim)));

        // var mergedKeyValue = MergeKV(keyStates, valueStates);
        AttentionDimKind[] qDestLayout = { AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq };
        var qPerm = ModelUtils.GetLayoutPerm(qSrcLayout, qDestLayout);
        var (qLanes, qPackedAxis) = ModelUtils.GetQKVPackParams(pagedAttentionConfig, qDestLayout);
        var padedQ = seq_len is DimVar ? IR.F.NN.Pad(queryStates, new(new(0, 0), new(0, ((long)seq_len.Metadata.Range!.Value.Max) - seq_len), new(0, 0)), PadMode.Constant, Tensor.Zero(DataTypes.Float32)) : queryStates;
        var transQ = IR.F.Tensors.Transpose(padedQ, qPerm);
        var castQ = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(transQ, pagedAttentionConfig.KVPrimType) : transQ;
        var packedQ = qLanes.Length > 0 ? IR.F.Tensors.Pack(castQ, qLanes, qPackedAxis) : castQ;

        // cpu : [q_head, max_query_len, max_seq_len + 1 ]<primtype>
        var extra_size = pagedAttentionConfig.KVPrimType.SizeInBytes * (long)Context.Config["num_attention_heads"] * Context.ImportOptions.HuggingFaceOptions.MaxModelLen * (Context.ImportOptions.HuggingFaceOptions.MaxModelLen + 1);

        // xpu : 10 mb.
        if (Context.CompileSession!.Target.Name == "xpu")
        {
            extra_size = 10 * 1024 * 1024;
        }

        var output = IR.F.NN.PagedAttention(
            packedQ,
            paskKeyValues,
            IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, [extra_size]),
            scaling.CastTo(pagedAttentionConfig.KVPrimType, CastMode.KDefault),
            count,
            qDestLayout);

        output = qLanes.Length > 0 ? IR.F.Tensors.Unpack(output, qLanes, qPackedAxis) : output;
        output = pagedAttentionConfig.KVPrimType != DataTypes.Float32 ? IR.F.Tensors.Cast(output, DataTypes.Float32) : output;
        output = IR.F.Tensors.Transpose(output, ModelUtils.GetLayoutPerm(qDestLayout, qSrcLayout));
        output = seq_len is DimVar ? IR.F.Tensors.Slice(output, new[] { 0 }, new Dimension[] { seq_len }, new[] { 1 }, new[] { 1 }) : output;
        output = IR.F.Tensors.Transpose(output, new[] { 1, 0, 2 });

        output = IR.F.Tensors.Reshape(output, new RankedShape(seq_len, -1L));
        var oProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.o_proj.weight"];

        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.o_proj.input_scale", out var ifScaleO);
        Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.o_proj.weight_scale", out var wScaleO);

        output = Linear(output, oProjW, null, ifScaleO, wScaleO, $"model.layers.{count}.self_attn.o_proj");
        return System.Tuple.Create(output, paskKeyValues);
    }

    public virtual Tuple<Expr, Expr?> LLMModel(
            Expr inputIds,
            Expr pastKeyValues)
    {
        /*
         * 1.1 embedding
         * self.padding_idx = config.pad_token_id
         * self.vocab_size = config.vocab_size
         * self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
         */
        var embedTokensWeight = Context!.ConstTensors!["model.embed_tokens.weight"];

        Expr? inputEmbeds;
        if (inputIds.CheckedShape.Rank > 2 && inputIds.CheckedDataType.IsFloat())
        {
            System.Console.WriteLine("inputIds rank >2 && dtype.isFloat()==true ,regard input_id as embedding...");
            inputEmbeds = inputIds;
        }
        else
        {
            long? padding_idx = null;
            if (Context.Config!.Keys.Contains("pad_token_id"))
            {
                padding_idx = (long)Context.Config["pad_token_id"];
            }

            inputEmbeds = Embeding(inputIds, embedTokensWeight, padding_idx);
        }

        var hiddenStates = inputEmbeds;

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
        //             inputEmbeds.CheckedShape[1].FixedValue;
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
        var (invFreq, attentionScaling) = ModelUtils.RoPEInit(Context!.Config!);
        var positionEmbeddings = RotaryEmbedding(hiddenStates, pastKeyValues, invFreq, attentionScaling);

        // var allHiddenStates = new List<Expr>();
        // var allSelfAttns = new List<Expr>();
        // var allKVcaches = new List<Expr>();
        // Expr? lastHiddenStates = null;
        // Expr? allSelfAttns = null;
        Expr? allHiddenStates = null;

        // Expr? allSelfAttns = null;
        // Expr? allKVcaches = null;
        // _ = new List<Tuple<Call, Call>>();
        for (int i = 0; i < (int)(long)Context!.Config!["num_hidden_layers"]; i++)
        {
            if (Context.ImportOptions!.HuggingFaceOptions.OutputHiddenStates)
            {
                // allHiddenStates.Add(IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 }));
                if (i == 0)
                {
                    allHiddenStates = IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 });
                }
                else
                {
                    allHiddenStates = IR.F.Tensors.Concat(new IR.Tuple(allHiddenStates!, IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 })), 0);
                }
            }

            // var (hiddenStatesTmp, selfAttenWeights) = DecodeLayer(i, hiddenStates, casualMask, positionIds,
            //     pastKeyValues, outputAttentions,
            //     useCache, cachePosition, positionEmbeddings);
            var (hiddenStatesTmp, pastKeyValuesTmp) = DecodeLayer(
                i,
                hiddenStates,
                pastKeyValues,
                positionEmbeddings);
            pastKeyValues = pastKeyValuesTmp;
            hiddenStates = hiddenStatesTmp;
        }

        // the last one
        Expr lastHiddenStates = LLMLayerNorm(hiddenStates, "model.norm.weight");

        if (Context.ImportOptions!.HuggingFaceOptions.OutputHiddenStates)
        {
            allHiddenStates = IR.F.Tensors.Concat(new IR.Tuple(allHiddenStates!, IR.F.Tensors.Unsqueeze(lastHiddenStates, new long[] { 0 })), 0);
        }

        return Tuple.Create(lastHiddenStates, allHiddenStates);

        // return Tuple.Create(lastHiddenStates, allSelfAttns, allKVcaches);
    }

    // private IR.Tuple SdpaAttention(
    //     Call query,
    //     Call key,
    //     Call value,
    //     Expr? attentionMask,
    //     float scaling,
    //     bool isCausal,
    //     Expr seqLen)
    // {
    //     /*
    //      * def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
    //            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    //            L, S = query.size(-2), key.size(-2)
    //            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    //            attn_bias = torch.zeros(L, S, dtype=query.dtype)
    //            if is_causal:
    //                assert attn_mask is None
    //                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    //                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    //                attn_bias.to(query.dtype)
    //            if attn_mask is not None:
    //                if attn_mask.dtype == torch.bool:
    //                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    //                else:
    //                    attn_bias += attn_mask
    //            if enable_gqa:
    //                key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    //                value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    //            attn_weight = query @ key.transpose(-2, -1) * scale_factor
    //            attn_weight += attn_bias
    //            attn_weight = torch.softmax(attn_weight, dim=-1)
    //            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    //            return attn_weight @ value
    //      */
    //     var numKVGroups = (long)Context!.Config!["num_attention_heads"] / (long)Context.Config!["num_key_value_heads"];
    //     var keyStates = RepeatKV(key, numKVGroups);
    //     var valueStates = RepeatKV(value, numKVGroups);
    //     // var scalingExpr = Cast(Tensor.FromScalar(scaling), query.CheckedDataType);
    //     var casualMask = attentionMask;
    //     if (attentionMask != null)
    //     {
    //         // casualMask
    //         casualMask = Slice(
    //            casualMask!,
    //            new[] { 0L },
    //            Stack(new IR.Tuple(ShapeOf(keyStates)[-2]), 0),
    //            new[] { -1L },
    //            new[] { 1L });
    //     }
    //     // if (isCausal == null)
    //     // {
    //     isCausal = if (casualMask == null && seqLen > 1) true else false
    //     // }
    //     var (l, s) = (ShapeOf(query)[-2], ShapeOf(key)[-2]);
    //     var scaleFactor = 1.0f / F.Math.Sqrt(Cast(ShapeOf(query)[-1], query.CheckedDataType));
    //     var attnBias = (Call)F.Tensors.Broadcast(Tensor.FromScalar(0f), F.Tensors.Stack(new IR.Tuple(l, s), 0L));
    //     // if (isCausal == true)
    //     // {
    //     //     var tempMask = (Call)Tensor.FromScalar(0f, new RankedShape(l, s));
    //     // }
    //     if (attentionMask != null)
    //     {
    //         attnBias = Binary(BinaryOp.Add, attnBias, attentionMask);
    //     }
    //     var attnWeight =
    //         IR.F.Math.MatMul(
    //             queryStates,
    //             Transpose(keyStates, ShapeExprUtility.GetPermutation(keyStates, [-2, -1]))) * scaleFactor;
    //     attnWeight += attnBias;
    //     attnWeight = Softmax(attnWeight, -1L);
    //     var attnOutput = F.Math.MatMul(attnWeight, valueStates);
    //     attnOutput = Transpose(attnOutput, ShapeExprUtility.GetPermutation(attnOutput, [1, 2]));
    //     return Tuple.Create(attnOutput, (Call)null);
    // }
    public virtual void VisitForCausalLM()
    {
        if (Context!.ConstTensors == null)
        {
            throw new ArgumentNullException(nameof(Context.ConstTensors));
        }

        Var input_ids = Context.Inputs![0]!;
        _ = Context.Inputs[1];
        _ = Context.Inputs[2];
        var pastKeyValues = Context.Inputs![3];

        var (lastHiddenStates, allHiddenStates) = LLMModel(
            input_ids,
            pastKeyValues!);

        var lmHeadWeights = Context.ConstTensors["model.embed_tokens.weight"];
        if (Context!.Config!.ContainsKey("tie_word_embeddings") && !Context!.Config!.GetNestedValue<bool>("tie_word_embeddings") && Context.ConstTensors.ContainsKey("lm_head.weight"))
        {
            lmHeadWeights = Context.ConstTensors["lm_head.weight"];
        }

        var lmHead = Linear(lastHiddenStates, lmHeadWeights, null, null, null, "lm_head");

        // FIXIT: this is work around for bfloat16
        if (Context.ImportOptions!.HuggingFaceOptions.OutputLogits)
        {
            Context.Outputs!.Add("logits", IR.F.Tensors.Cast(lmHead, DataTypes.Float32));
        }
        else
        {
            Context.Outputs!.Add("lastHiddenStates", IR.F.Tensors.Cast(lastHiddenStates, DataTypes.Float32));
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            // FIXIT: this is work around for bfloat16
            Context.Outputs!["hiddenStates"] = IR.F.Tensors.Cast(allHiddenStates!, DataTypes.Float32);
        }
    }
}

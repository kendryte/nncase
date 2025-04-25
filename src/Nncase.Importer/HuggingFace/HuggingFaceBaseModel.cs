// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Nncase.IR;
using Nncase.Utilities;

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

    public virtual (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        var hiddenSize = (long)Context!.Config!["hidden_size"];
        var numsHiddenLayers = (long)Context.Config!["num_hidden_layers"];
        var num_attention_heads = (long)Context.Config!["num_attention_heads"];
        var headDim = hiddenSize / num_attention_heads;
        if (Context.Config.ContainsKey("head_dim"))
        {
            headDim = (long)Context.Config["head_dim"];
        }

        var numKVHeads = (long)Context.Config!["num_key_value_heads"];

        Context.Inputs = [];
        Context.DynVarMap = new Dictionary<string, Var>();
        var varMap = new Dictionary<Var, Expr[]>();

        var bucketOptions = Context.CompileSession!.CompileOptions.ShapeBucketOptions;
        Context.FixVarMap = bucketOptions.FixVarMap;

        // local test set
        // _fixVarMap["sequence_length"] = 10;
        // _fixVarMap["history_len"] = 0;
        // TODO: control by config file
        if (!Context.FixVarMap.ContainsKey("sequence_length"))
        {
            Context.DynVarMap["sequence_length"] = Var.SizeVar("sequence_length");
            if (Context.CompileSession.CompileOptions.ShapeBucketOptions.RangeInfo.ContainsKey("sequence_length"))
            {
                Context.DynVarMap["sequence_length"].Metadata.Range = Context.CompileSession.CompileOptions.ShapeBucketOptions.RangeInfo["sequence_length"];
            }
            else
            {
                Context.DynVarMap["sequence_length"].Metadata.Range = new(1, 64);
            }
        }

        // if (!_fixVarMap.ContainsKey("history_len"))
        // {
        //     _dynVarMap["history_len"] = Var.SizeVar("history_len");
        //     _dynVarMap["history_len"].Metadata.Range=new (4096,8192);
        // }
        if (!Context.FixVarMap.ContainsKey("batch_size"))
        {
            Context.DynVarMap["batch_size"] = Var.SizeVar("batch_size");
            if (Context.CompileSession.CompileOptions.ShapeBucketOptions.RangeInfo.ContainsKey("batch_size"))
            {
                Context.DynVarMap["batch_size"].Metadata.Range = Context.CompileSession.CompileOptions.ShapeBucketOptions.RangeInfo["batch_size"];
            }
            else
            {
                Context.DynVarMap["batch_size"].Metadata.Range = new(1, 4);
            }
        }


        var inputIdsShapeExpr = new Expr[] { Context.DynVarMap["batch_size"],
                                               Context.DynVarMap["sequence_length"],
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
            new TensorType(DataTypes.Int64, new Shape(
                                                Context.DynVarMap["batch_size"],
                                                Context.DynVarMap["sequence_length"])));

        // var attentionMask = new Var(
        //     "attention_mask",
        //     new TensorType(
        //         DataTypes.Float32,
        //         new Shape(
        //             1L, // _dynVarMap["batch_size"],
        //             20L)));
        // var positionIds = new Var(
        //     "position_ids",
        //     new TensorType(DataTypes.Float32, new Shape(
        //                                     1L, // _dynVarMap["batch_size"],
        //                                     20L)));

        // // [decode_layers, k_or_v, batch_size, num_heads, past_seq_length, head_dim]
        // var pastKeyValue = new Var(
        //     "past_key_values",
        //     new TensorType(DataTypes.Float32, new Shape(
        //         numsHiddenLayers,
        //         2L,
        //         1L, // _dynVarMap["batch_size"],
        //         numKVHeads,
        //         0, // _dynVarMap["history_len"],
        //         headDim)));
        Context.Inputs.Add(inputIds);
        Context.Inputs.Add(null); // attentionMask
        Context.Inputs.Add(null); // positionIds
        Context.Inputs.Add(null); // pastKeyValue

        // _inputs.Add(attentionMask);
        // _inputs.Add(positionIds);
        // _inputs.Add(pastKeyValue);
        varMap[inputIds] = inputIdsShapeExpr;

        // varMap[attentionMask] = attentionMaskShapeExpr;
        // varMap[positionIds] = positionIdsShapeExpr;
        // varMap[pastKeyValue] = pastKeyValueShapeExpr;
        var inputs = new List<Var> { };

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
    public virtual Expr CreateOutputs()
    {
        // TODO: use self.config.output_attention to judge wether output kvache
        var logits = Context!.Outputs["logits"];
        Expr? outAttention = null;
        Expr? kvCache = null;
        Expr? hiddenStates = null;

        if (Context.ImportOptions!.HuggingFaceOptions.UseCache)
        {
            kvCache = Context.Outputs["kvCache"];
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputAttentions)
        {
            outAttention = Context.Outputs["outAttention"];
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            hiddenStates = Context.Outputs["hiddenStates"];
        }

        var output = new List<Expr?> { logits, kvCache, outAttention, hiddenStates,
                                        // Context.Outputs["debug"]
                                        };
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

        var hiddenStatesShape = IR.F.Tensors.ShapeOf(hiddenStates);
        var batch_size = hiddenStatesShape[0];
        var numKVHeads = hiddenStatesShape[1];
        var seqLen = hiddenStatesShape[2];
        var headDim = hiddenStatesShape[3];
        hiddenStates = IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 2 });

        var tmp = IR.F.Tensors.Stack(new IR.Tuple(batch_size, numKVHeads, nRep, seqLen, headDim), 0L);
        hiddenStates = IR.F.Tensors.Expand(hiddenStates, tmp);
        hiddenStates = IR.F.Tensors.Reshape(hiddenStates, IR.F.Tensors.Stack(new IR.Tuple(batch_size, numKVHeads * nRep, seqLen, headDim), 0L));
        return hiddenStates;
    }

    public virtual System.Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, long unSqueezeDim = 1)
    {
        cos = IR.F.Tensors.Unsqueeze(cos, Tensor.From<long>(new long[] { 1 }));
        sin = IR.F.Tensors.Unsqueeze(sin, Tensor.From<long>(new long[] { 1 }));

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
        var xS3 = IR.F.Tensors.ShapeOf(x)[3];
        var x1 = IR.F.Tensors.Slice(
            x,
            new[] { 0L },
            IR.F.Tensors.Stack(new IR.Tuple(xS3 / 2L), 0L),
            new[] { -1L },
            new[] { 1L });
        var x2 = IR.F.Tensors.Slice(
            x,
            IR.F.Tensors.Stack(new IR.Tuple(xS3 / 2L), 0L),
            IR.F.Tensors.Stack(new IR.Tuple(xS3), 0L),
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
        var bias = Tensor.FromScalar(0f, weight.CheckedShape);
        int axis = -1;

        float eps = 1e-6F;
        if (Context!.Config!.ContainsKey("rms_norm_eps"))
        {
            eps = (float)Context!.Config!.GetNestedValue<double>("rms_norm_eps");
        }

        return IR.F.Tensors.Cast(IR.F.NN.LayerNorm(axis, eps, hiddenStates, weight, bias, false), originDtype);
    }

    public virtual Call Linear(Expr expr, Tensor weight, Tensor? bias = null)
    {
        var transposed_weight = IR.F.Tensors.Transpose(weight, new long[] { 1, 0 });
        var result = IR.F.Math.MatMul(expr, transposed_weight);
        if (bias != null)
        {
            result = IR.F.Math.Add(result, bias);
        }

        return result;
    }

    public virtual Tuple<Expr, Expr, Expr> DecodeLayer(
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

        // if (outputAttentions == true && selfAttenKV is not null)
        // {
        return System.Tuple.Create<Expr, Expr, Expr>(output, outAttention, currentKV);

        // }

        // return Tuple.Create<Call, Call>(output, null);
    }

    public virtual Call LLMMlp(int count, Expr hiddenStates)
    {
        var gateProjW = Context!.ConstTensors![$"model.layers.{count}.mlp.gate_proj.weight"];
        var upProjW = Context.ConstTensors![$"model.layers.{count}.mlp.up_proj.weight"];
        var downProjW = Context.ConstTensors![$"model.layers.{count}.mlp.down_proj.weight"];

        var tmp = Linear(hiddenStates, gateProjW);
        if (Context!.Config!.ContainsKey("hidden_act"))
        {
            var actType = Context!.Config!.GetNestedValue<string>("hidden_act");
            tmp = ModelUtils.ActFunc(tmp, actType);
        }

        return Linear(tmp * Linear(hiddenStates, upProjW), downProjW);
    }

    public virtual Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Expr headDim)
    {
        var batchSize = Context!.DynVarMap!["batch_size"];
        var seqLen = Context.DynVarMap["sequence_length"];
        var hidden_shape1 = IR.F.Tensors.Stack(new IR.Tuple(batchSize, seqLen, (long)Context!.Config!["num_attention_heads"], headDim), 0L);

        var qProjW = Context!.ConstTensors![$"model.layers.{count}.self_attn.q_proj.weight"];
        Tensor? qProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.q_proj.bias"))
        {
            qProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.q_proj.bias"];
        }

        var queryStates = Linear(hiddenStates, qProjW, qProjB);
        queryStates = IR.F.Tensors.Reshape(queryStates, hidden_shape1);

        // batch_size, num_heads, seq_len, head_dim
        queryStates = IR.F.Tensors.Transpose(queryStates, new long[] { 0, 2, 1, 3 });

        var kProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.weight"];
        Tensor? kProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.k_proj.bias"))
        {
            kProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.bias"];
        }

        var keyStates = Linear(hiddenStates, kProjW, kProjB);
        var hidden_shape2 = IR.F.Tensors.Stack(new IR.Tuple(batchSize, seqLen, (long)Context!.Config!["num_key_value_heads"], headDim), 0L);
        keyStates = IR.F.Tensors.Reshape(keyStates, hidden_shape2);
        keyStates = IR.F.Tensors.Transpose(keyStates, new long[] { 0, 2, 1, 3 });

        var vProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.weight"];
        Tensor? vProjB = null;
        if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.v_proj.bias"))
        {
            vProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.bias"];
        }

        var valueStates = Linear(hiddenStates, vProjW, vProjB);
        valueStates = IR.F.Tensors.Reshape(valueStates, hidden_shape2);
        valueStates = IR.F.Tensors.Transpose(valueStates, new long[] { 0, 2, 1, 3 });
        return System.Tuple.Create(queryStates, keyStates, valueStates);
    }

    public virtual Tuple<Expr, Expr> EagerAttentionForward(Expr query, Expr key, Expr value, Expr? attentionMask, float scaling)
    {
        var numKVGroups = (long)Context!.Config!["num_attention_heads"] / (long)Context.Config!["num_key_value_heads"];
        var keyStates = RepeatKV(key, numKVGroups);
        var valueStates = RepeatKV(value, numKVGroups);
        var scalingExpr = IR.F.Tensors.Cast(Tensor.FromScalar(scaling), query.CheckedDataType);
        Expr attnWeights = IR.F.Math.MatMul(query, IR.F.Tensors.Transpose(keyStates, ShapeExprUtility.GetPermutation(keyStates, [2, 3]))) * scalingExpr;
        if (attentionMask is not null)
        {
            var causalMask = IR.F.Tensors.Slice(
                    attentionMask,
                    new[] { 0L },
                    IR.F.Tensors.Stack(new IR.Tuple(IR.F.Tensors.ShapeOf(keyStates)[2]), 0L),
                    new[] { 3L },
                    new[] { 1L });

            attnWeights += causalMask;
        }

        attnWeights = IR.F.Tensors.Cast(IR.F.NN.Softmax(IR.F.Tensors.Cast(attnWeights, DataTypes.Float32), 3L), valueStates.CheckedDataType);

        Expr attnOutput = IR.F.Math.MatMul(attnWeights, valueStates);
        attnOutput = IR.F.Tensors.Transpose(attnOutput, ShapeExprUtility.GetPermutation(attnOutput, [1, 2]));

        // TODO: base on config to decide output attnWeights or not
        return System.Tuple.Create(attnOutput, attnWeights);
    }

    public virtual Tuple<Expr, Expr> RotaryEmbedding(Expr x, Expr positionIds)
    {
        // rope type not in config, so it is default. :_compute_default_rope_parameters
        // if "dynamic" in self.rope_type:
        //      self._dynamic_frequency_update(position_ids, device=x.device)
        var (invFreq_, attentionScaling) = ModelUtils.RoPEInit(Context!.Config!);

        // var a = x.CheckedShape[0];
        var invFreq = Tensor.FromArray(invFreq_.ToArray()); // Unsqueeze(Unsqueeze(Tensor.FromArray(inv_freq.ToArray()), new[] { 0 }),new[] { -1 });

        // Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        var invFreq_float = IR.F.Tensors.Cast(invFreq, DataTypes.Float32);
        var invFreqExpanded = IR.F.Tensors.Unsqueeze(invFreq_float, Tensor.From<long>(new long[] { 0, 2 }));
        var batch_size = IR.F.Tensors.ShapeOf(positionIds)[0];
        var dim_div_2 = IR.F.Tensors.ShapeOf(invFreq)[0];
        var shape_tensor = IR.F.Tensors.Stack(new IR.Tuple(batch_size, dim_div_2, 1L), 0);

        invFreqExpanded = IR.F.Tensors.Expand(invFreqExpanded, shape_tensor);

        // var invFreqExpanded = Broadcast(
        //     inv_freq_tensor,
        //     new Dimension[] { x.CheckedShape[0], inv_freq.Count, 1 });
        var positionIdsExpanded = IR.F.Tensors.Unsqueeze(positionIds, Tensor.From<long>(new long[] { 1 }));
        positionIdsExpanded = IR.F.Tensors.Cast(positionIdsExpanded, DataTypes.Float32);

        var freqs = IR.F.Math.MatMul(invFreqExpanded, positionIdsExpanded);
        freqs = IR.F.Tensors.Transpose(freqs, new long[] { 0, 2, 1 });

        // F.Tensors.Transpose(F.Math.MatMul(invFreqExpanded, positionIdsExpanded),new Dimension[] { 0, 2, 1 });
        var emb = IR.F.Tensors.Concat(new IR.Tuple(freqs, freqs), -1);

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
                            Expr seqLen,
                            Expr targtLen,
                            DataType dtype,
                            Expr cachePosition,
                            Expr batchSize,
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
            var mask_shape = IR.F.Tensors.Stack(new IR.Tuple([seqLen, targtLen]), 0L);
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
            */
            // ;
            var diagonalAttendMaskShape = IR.F.Tensors.Stack(new IR.Tuple([seqLen, 1L]), 0L);
            var diagonalAttendMask = IR.F.Tensors.Range(0L, targtLen, 1L) > IR.F.Tensors.Reshape(cachePosition, diagonalAttendMaskShape);
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
            var expandShape = IR.F.Tensors.Stack(new IR.Tuple(batchSize, 1L, seqLen, targtLen), 0L);
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
                var maskLength = IR.F.Tensors.ShapeOf(attentionMask);
                var paddingMask = IR.F.Tensors.Slice(
                    casualMask,
                    new[] { 0L, 0L, 0L, 0L },
                    IR.F.Tensors.Stack(new IR.Tuple(maskLength), 0L),
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
                    IR.F.Tensors.Stack(new IR.Tuple(maskLength), 0L),
                    new[] { -1L },
                    new[] { 1L });

                var minDtypeMatrix = IR.F.Tensors.ConstantOfShape(IR.F.Tensors.ShapeOf(maskPart), minValue);

                maskPart = IR.F.Tensors.Where(paddingMask, minDtypeMatrix, maskPart);

                // TODO: for dynamic cache, maskLength== sequence length == target length
                //  just return maskPart
                var leftPart = IR.F.Tensors.Slice(
                    casualMask,
                    IR.F.Tensors.Stack(new IR.Tuple(maskLength), 0),
                    IR.F.Tensors.Stack(new IR.Tuple(IR.F.Tensors.ShapeOf(casualMask)[-1]), 0L),
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
        Expr historyLen = 0L;
        if (pastKeyValues != null)
        {
            historyLen = IR.F.Tensors.ShapeOf(pastKeyValues)[-2];
        }

        // var inputsEmbedsShape = IR.F.Tensors.ShapeOf(inputsEmbeds);
        var batchSize = (Expr)Context!.DynVarMap!["batch_size"];
        var seqLen = (Expr)Context.DynVarMap["sequence_length"];
        Expr targtLen = historyLen + seqLen + 1L;
        if (attentionMask != null)
        {
            targtLen = IR.F.Tensors.ShapeOf(attentionMask)[-1];
        }

        var dtype = inputsEmbeds.CheckedDataType;

        Expr casualMask = Prepare4dCausalAttentionMaskWithCachePosition(
                                            attentionMask,
                                            seqLen,
                                            targtLen,
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
        var embedingDim = embedingWeight.Shape[-1];
        var gatherResult = IR.F.Tensors.Gather(embedingWeight, 0, input);
        if (paddingIdx == null)
        {
            return gatherResult;
        }
        else
        {
            var zeros = Tensor.Zeros(embedingWeight.ElementType, new long[] { 1, embedingDim.FixedValue });
            var paddingMask = IR.F.Math.Equal(input, paddingIdx);
            paddingMask = IR.F.Tensors.Unsqueeze(paddingMask, new long[] { 2 });
            paddingMask = IR.F.Tensors.Expand(paddingMask, IR.F.Tensors.ShapeOf(gatherResult));
            var results = IR.F.Tensors.Where(paddingMask, zeros, gatherResult);
            return results;
        }
    }

    public virtual Tuple<Expr, Expr, Expr> LLMSelfAttention(
                int count,
                Expr hiddenStates,
                Expr? attentionMask,
                Expr? paskKeyValues,
                Expr cachePosition,
                Tuple<Expr, Expr> positionEmbeddings)
    {
        // if (Context!.Config!.GetNestedValue<bool>("output_attentions"))
        // {
        //     return EagerAttentionForward();
        // }
        var head_dim = (long)Context!.Config!["hidden_size"] / (long)Context.Config["num_attention_heads"];
        if (Context.Config!.Keys.Contains("head_dim"))
        {
            head_dim = (long)Context.Config["head_dim"];
        }

        var batchSize = Context.DynVarMap!["batch_size"];
        var seqLength = Context.DynVarMap["sequence_length"];
        var (queryStates, keyStates, valueStates) = QKVCompute(count, hiddenStates, head_dim);

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
        //     attentionMask,
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
        var tmpShape = IR.F.Tensors.ShapeOf(hiddenStates);
        var inputShape = IR.F.Tensors.Stack(new IR.Tuple(batchSize, seqLength, tmpShape[2] * tmpShape[3]), 0L);
        hiddenStates = IR.F.Tensors.Reshape(hiddenStates, inputShape);
        var oProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.o_proj.weight"];
        hiddenStates = Linear(hiddenStates, oProjW);

        var mergedKeyValue = MergeKV(keyStates, valueStates);

        return System.Tuple.Create(hiddenStates, selfAttenWeight, mergedKeyValue);
    }

    public virtual Tuple<Expr, List<Expr>, List<Expr>, List<Expr>> LLMModel(
            Expr input_ids,
            Expr? attentionMask,
            Expr? positionIds,
            Expr? pastKeyValues)
    {
        /*
         * 1.1 embedding
         * self.padding_idx = config.pad_token_id
         * self.vocab_size = config.vocab_size
         * self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
         */
        var embedTokensWeight = Context!.ConstTensors!["model.embed_tokens.weight"];

        Expr? inputEmbeds;
        if (input_ids.CheckedShape.Rank > 2 && input_ids.CheckedDataType.IsFloat())
        {
            System.Console.WriteLine("input_ids rank >2 && dtype.isFloat()==true ,regard input_id as embedding...");
            inputEmbeds = input_ids;
        }
        else
        {
            long? padding_idx = null;
            if (Context.Config!.Keys.Contains("pad_token_id"))
            {
                padding_idx = (long)Context.Config["pad_token_id"];
            }

            inputEmbeds = Embeding(input_ids, embedTokensWeight, padding_idx);
        }

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
        Expr historyLen = 0L;
        if (pastKeyValues != null)
        {
            // FIXME: load from pagedAttention::context_length
            historyLen = IR.F.Tensors.ShapeOf(pastKeyValues)[-2];
        }

        var seqLen = Context.DynVarMap!["sequence_length"];
        var cachePosition = IR.F.Tensors.Range(historyLen, historyLen + seqLen, 1L);
        var casualMask = UpdatecasualMask(attentionMask, inputEmbeds, cachePosition, pastKeyValues, outputAttentions: false);
        var hiddenStates = inputEmbeds;
        if (positionIds == null)
        {
            positionIds = IR.F.Tensors.Cast(IR.F.Tensors.Unsqueeze(cachePosition, 0), hiddenStates.CheckedDataType);
        }

        var positionEmbeddings = RotaryEmbedding(hiddenStates, positionIds);

        var allHiddenStates = new List<Expr>();
        var allSelfAttns = new List<Expr>();
        var allKVcaches = new List<Expr>();

        // _ = new List<Tuple<Call, Call>>();
        for (int i = 0; i < (int)(long)Context!.Config!["num_hidden_layers"]; i++)
        {
            if (Context.ImportOptions!.HuggingFaceOptions.OutputHiddenStates)
            {
                allHiddenStates.Add(IR.F.Tensors.Unsqueeze(hiddenStates, new long[] { 0 }));
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

            if (Context.ImportOptions.HuggingFaceOptions.OutputAttentions)
            {
                allSelfAttns.Add(IR.F.Tensors.Unsqueeze(outAttention, new[] { 0L }));
            }

            if (Context.ImportOptions.HuggingFaceOptions.UseCache)
            {
                allKVcaches.Add(currentKV);
            }
        }

        // the last one
        Expr lastHiddenStates = LLMLayerNorm(hiddenStates, "model.norm.weight");

        if (Context.ImportOptions!.HuggingFaceOptions.OutputHiddenStates)
        {
            allHiddenStates.Add(IR.F.Tensors.Unsqueeze(lastHiddenStates, new long[] { 0 }));
        }

        return System.Tuple.Create(lastHiddenStates, allKVcaches, allHiddenStates, allSelfAttns);

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
    //     //     var tempMask = (Call)Tensor.FromScalar(0f, new Shape(l, s));
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
        var attention_mask = Context.Inputs[1];
        var position_ids = Context.Inputs[2];
        var pastKeyValues = Context.Inputs[3];

        var (lastHiddenStates, allSelfKV, allHiddenStates, allSelfAttns) = LLMModel(
            input_ids,
            attention_mask,
            position_ids,
            pastKeyValues);

        var lmHeadWeights = Context.ConstTensors["model.embed_tokens.weight"];
        if (Context!.Config!.ContainsKey("tie_word_embeddings") && !Context!.Config!.GetNestedValue<bool>("tie_word_embeddings") && Context.ConstTensors.ContainsKey("lm_head.weight"))
        {
            lmHeadWeights = Context.ConstTensors["lm_head.weight"];
        }

        var lmHead = Linear(lastHiddenStates, lmHeadWeights);

        // FIXIT: this is work around for bfloat16
        Context.Outputs!.Add("logits", IR.F.Tensors.Cast(lmHead, DataTypes.Float32));
        if (Context.ImportOptions!.HuggingFaceOptions.OutputAttentions)
        {
            var outAttention = IR.F.Tensors.Concat(new IR.Tuple(allSelfAttns.ToArray()), 0);

            // FIXIT: this is work around for bfloat16
            Context.Outputs!["outAttention"] = IR.F.Tensors.Cast(outAttention, DataTypes.Float32);
        }

        if (Context.ImportOptions.HuggingFaceOptions.UseCache)
        {
            var kvCache = IR.F.Tensors.Concat(new IR.Tuple(allSelfKV.ToArray()), 0);

            // FIXIT: this is work around for bfloat16
            Context.Outputs!["kvCache"] = IR.F.Tensors.Cast(kvCache, DataTypes.Float32);
        }

        if (Context.ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            var hiddenStates = IR.F.Tensors.Concat(new IR.Tuple(allHiddenStates.ToArray()), 0);

            // FIXIT: this is work around for bfloat16
            Context.Outputs!["hiddenStates"] = IR.F.Tensors.Cast(hiddenStates, DataTypes.Float32);
        }
    }
}

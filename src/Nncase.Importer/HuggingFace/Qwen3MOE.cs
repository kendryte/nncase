// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.F;

namespace Nncase.Importer
{
    public class Qwen3A3B : Qwen3
    {
        public override Call LLMMlp(int count, Expr hiddenStates)
        {
            var expertNum = Config.GetNestedValue<long>("num_experts");

            var gateW = GetWeight($"model.layers.{count}.mlp.gate.weight")!;

            List<Expr> allGateInputScale = new();
            List<Expr> allGateProjW = new();
            List<Expr> allGateProjScale = new();
            List<Expr> allDownInputScale = new();
            List<Expr> allDownProjW = new();
            List<Expr> allDownProjScale = new();
            List<Expr> allUpInputScale = new();
            List<Expr> allUpProjW = new();
            List<Expr> allUpProjScale = new();
            for (long expertIndex = 0; expertIndex < expertNum; expertIndex++)
            {
                allGateInputScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.gate_proj.input_scale")!);
                allGateProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.gate_proj.weight")!);
                allGateProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.gate_proj.weight_scale")!);
                allDownInputScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.down_proj.input_scale")!);
                allDownProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.down_proj.weight")!);
                allDownProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.down_proj.weight_scale")!);
                allUpInputScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.up_proj.input_scale")!);
                allUpProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.up_proj.weight")!);
                allUpProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.up_proj.weight_scale")!);
            }

            var gateInputScale = Tensors.Concat(new IR.Tuple(allGateInputScale.ToArray()), 0).Evaluate();
            var gateProjW = Tensors.Concat(new IR.Tuple(allGateProjW.ToArray()), 0).Evaluate();
            var gateProjScale = Tensors.Concat(new IR.Tuple(allGateProjScale.ToArray()), 0).Evaluate();
            var downProjW = Tensors.Concat(new IR.Tuple(allDownProjW.ToArray()), 0).Evaluate();
            var downProjInputScale = Tensors.Concat(new IR.Tuple(allDownInputScale.ToArray()), 0).Evaluate();
            var downProjScale = Tensors.Concat(new IR.Tuple(allDownProjScale.ToArray()), 0).Evaluate();
            var upProjW = Tensors.Concat(new IR.Tuple(allUpProjW.ToArray()), 0).Evaluate();
            var upProjInputScale = Tensors.Concat(new IR.Tuple(allUpInputScale.ToArray()), 0).Evaluate();
            var upProjScale = Tensors.Concat(new IR.Tuple(allUpProjScale.ToArray()), 0).Evaluate();

            // var moeRes = IR.F.NN.Qwen3MoE(
            //     q: hiddenStates,
            //     moeGateW: gateW,
            //     moeExpertGateInputScale: gateInputScale.AsTensor(),
            //     moeExpertGateProjW: gateProjW.AsTensor(),
            //     moeExpertGateProjScale: gateProjScale.AsTensor(),
            //     moeExpertDownInputScale: downProjInputScale.AsTensor(),
            //     moeExpertDownProjW: downProjW.AsTensor(),
            //     moeExpertDownProjScale: downProjScale.AsTensor(),
            //     moeExpertUpInputScale: upProjInputScale.AsTensor(),
            //     moeExpertUpProjW: upProjW.AsTensor(),
            //     moeExpertUpProjScale: upProjScale.AsTensor(),
            //     layerId: count,
            //     hiddenSize: Config.GetNestedValue<long>("hidden_size"),
            //     intermediateSize: Config.GetNestedValue<long>("intermediate_size"),
            //     moeIntermediateSize: Config.GetNestedValue<long>("moe_intermediate_size"),
            //     numExpert: expertNum,
            //     numTopK: Config.GetNestedValue<long>("num_experts_per_tok"),
            //     isNormTopkProb: Config.GetNestedValue<bool>("norm_topk_prob") ? 1L : 0L);
            // return (Call)moeRes;

            // Split MoE logic.
            var routerLogits = Linear(hiddenStates, gateW, layerName: "router_logits");
            routerLogits = Tensors.Cast(routerLogits, DataTypes.Float32);
            var topkProbs = NN.Softmax(routerLogits, -1).With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.moe.softmax" } });
            var topKRes = Tensors.TopK(topkProbs, Tensor.FromScalar(DataTypes.Int64, Config.GetNestedValue<long>("num_experts_per_tok"), [1]), -1, true, true).With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.moe.topk" } });
            var routerWeights = topKRes[0];
            var selectedExperts = topKRes[1];

            if (Config.GetNestedValue<bool>("norm_topk_prob"))
            {
                routerWeights = IR.F.Math.Binary(
                    BinaryOp.Div,
                    routerWeights,
                    Tensors.Reduce(ReduceOp.Sum, routerWeights, new long[] { -1L }, 0f, true).With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.moe.norm_reduce_sum" } }))
                    .With(metadata: new IRMetadata() { OutputNames = new[] { $"model.layers.{count}.moe.norm_weights" } });
            }

            string nameSparseExperts = $"model.layers.{count}.moe.sparse_experts";
            var sparseExpertOutput = IR.F.NN.SparseExperts(
                hiddenStates,
                selectedExperts, // expert ids
                routerWeights, // expert weights
                moeExpertGateInputScale: gateInputScale.AsTensor(),
                moeExpertGateProjW: gateProjW.AsTensor(),
                moeExpertGateProjScale: gateProjScale.AsTensor(),
                moeExpertDownInputScale: downProjInputScale.AsTensor(),
                moeExpertDownProjW: downProjW.AsTensor(),
                moeExpertDownProjScale: downProjScale.AsTensor(),
                moeExpertUpInputScale: upProjInputScale.AsTensor(),
                moeExpertUpProjW: upProjW.AsTensor(),
                moeExpertUpProjScale: upProjScale.AsTensor(),
                hiddenSize: Config.GetNestedValue<long>("hidden_size"),
                moeIntermediateSize: Config.GetNestedValue<long>("moe_intermediate_size"),
                numExpert: expertNum,
                numTopK: Config.GetNestedValue<long>("num_experts_per_tok"),
                Context!.CompileSession!.CompileOptions.ShapeBucketOptions.RangeInfo["sequence_length"].Max).With(metadata: new IRMetadata() { OutputNames = new[] { nameSparseExperts } });

            return sparseExpertOutput;
        }

        private Call? GetWeightAndExpand(string name, long expertNum = 0)
        {
            var weight = GetWeight(name);
            if (weight == null)
            {
                // Create an empty tensor with shape [0] to indicate no input scaling
                weight = Tensor.FromScalar(1.0f).Reshape(new long[] { 1 });
            }

            var expandWeight = Tensors.Unsqueeze(weight, new[] { expertNum });

            return expandWeight;
        }
    }
}

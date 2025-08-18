// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Importer
{
    public class Qwen3A3B : Qwen3
    {
        public override Call LLMMlp(int count, Expr hiddenStates)
        {
            var expertNum = Config.GetNestedValue<long>("num_experts");

            var gateW = GetWeight($"model.layers.{count}.mlp.gate.weight")!;

            List<Expr> allGateProjW = new();
            List<Expr> allGateProjScale = new();
            List<Expr> allDownProjW = new();
            List<Expr> allDownProjScale = new();
            List<Expr> allUpProjW = new();
            List<Expr> allUpProjScale = new();
            for (long expertIndex = 0; expertIndex < expertNum; expertIndex++)
            {
                allGateProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.gate_proj.weight")!);
                allGateProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.gate_proj.weight_scale")!);
                allDownProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.down_proj.weight")!);
                allDownProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.down_proj.weight_scale")!);
                allUpProjW.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.up_proj.weight")!);
                allUpProjScale.Add(GetWeightAndExpand($"model.layers.{count}.mlp.experts.{expertIndex}.up_proj.weight_scale")!);
            }

            var gateProjW = IR.F.Tensors.Concat(new IR.Tuple(allGateProjW.ToArray()), 0);
            var gateProjScale = IR.F.Tensors.Concat(new IR.Tuple(allGateProjScale.ToArray()), 0);
            var downProjW = IR.F.Tensors.Concat(new IR.Tuple(allDownProjW.ToArray()), 0);
            var downProjScale = IR.F.Tensors.Concat(new IR.Tuple(allDownProjScale.ToArray()), 0);
            var upProjW = IR.F.Tensors.Concat(new IR.Tuple(allUpProjW.ToArray()), 0);
            var upProjScale = IR.F.Tensors.Concat(new IR.Tuple(allUpProjScale.ToArray()), 0);

            var moeRes = IR.F.NN.Qwen3MoE(
                q: hiddenStates,
                moeGateW: gateW,
                moeExpertGateProjW: gateProjW,
                moeExpertGateProjScale: gateProjScale,
                moeExpertDownProjW: downProjW,
                moeExpertDownProjScale: downProjScale,
                moeExpertUpProjW: upProjW,
                moeExpertUpProjScale: upProjScale,
                layerId: count,
                hiddenSize: Config.GetNestedValue<long>("hidden_size"),
                intermediateSize: Config.GetNestedValue<long>("intermediate_size"),
                moeIntermediateSize: Config.GetNestedValue<long>("moe_intermediate_size"),
                numExpert: expertNum,
                numTopK: Config.GetNestedValue<long>("num_experts_per_tok"),
                isNormTopkProb: Config.GetNestedValue<bool>("norm_topk_prob") ? 1L : 0L);
            return (Call)moeRes;
        }

        private Call GetWeightAndExpand(string name, long expertNum = 0)
        {
            var weight = (Expr)GetWeight(name)!;
            var expandWeight = IR.F.Tensors.Unsqueeze(weight, new[] { expertNum });

            return expandWeight;
        }
    }
}

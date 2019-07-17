using System;
using System.Collections.Generic;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.Targets.CPU.CodeGen.Operators;
using NnCase.Targets.CPU.Evaluation.Operators;
using NnCase.Targets.CPU.Transforms;
using NnCase.Transforms;

namespace NnCase.Targets.CPU
{
    public class CPUTarget : Target
    {
        protected override void AddOptimize2Transforms(List<Transform> transforms)
        {
            transforms.AddRange(new Transform[]
            {
                new CPUDepthwiseConv2DTransform(),
                new CPUConv2DTransform(),
                new CPUReduceWindow2DTransform()
            });
        }

        public override void RegisterEvaluators(EvaluatorRegistry registry)
        {
            CPUEvaulators.Register(registry);
        }

        public override void RegisterEmitters(CodeGenRegistry registry)
        {
            CPUEmitters.Register(registry);
        }

        protected override void AddQuantizationCheckpointsTransforms(List<Transform> transforms)
        {
            transforms.AddRange(new Transform[]
            {
                new AddCPUQuantizeCheckpointTransform()
            });
        }

        protected override void AddQuantizeTransforms(List<Transform> transforms, Quantizer quantizer)
        {
            transforms.AddRange(new Transform[]
            {
                new CPUQuantizedConv2DTransform(quantizer),
                new CPUQuantizedDepthwiseConv2DTransform(quantizer)
            });
        }
    }
}

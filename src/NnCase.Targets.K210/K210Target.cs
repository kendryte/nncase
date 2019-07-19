using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.Targets.CPU;
using NnCase.Targets.K210.Evaluation;
using NnCase.Targets.K210.Evaluation.Operators;
using NnCase.Targets.K210.Transforms;
using NnCase.Transforms;

namespace NnCase.Targets.K210
{
    public class K210Target : CPUTarget
    {
        public override void AddAllocators(Dictionary<MemoryType, MemoryAllocator> allocators)
        {
            base.AddAllocators(allocators);

            allocators.Add(MemoryType.K210KPU, new KPUMemoryAllocator());
        }

        protected override void AddOptimize2Transforms(List<Transform> transforms)
        {
            transforms.AddRange(new Transform[]
            {
                new KPUFakeConv2DTransform()
            });

            base.AddOptimize2Transforms(transforms);
        }

        public override void RegisterEvaluators(EvaluatorRegistry registry)
        {
            K210Evaulators.Register(registry);
            base.RegisterEvaluators(registry);
        }

        protected override void AddQuantizationCheckpointsTransforms(List<Transform> transforms)
        {
            transforms.AddRange(new Transform[]
            {
                new AddKPUQuantizeCheckpointTransform()
            });
        }

        protected override void AddQuantizeTransforms(List<Transform> transforms, Quantizer quantizer)
        {
            transforms.AddRange(new Transform[]
            {
                new KPUConv2DTransform(quantizer),
                new FoldKPUUploadTransform(),
                new FuseKPUDownloadTransform()
            });
        }
    }
}

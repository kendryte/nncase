using System;
using System.Collections.Generic;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.Transforms;

namespace NnCase.Targets
{
    public abstract class Target
    {
        public void OptimizePass2(Graph graph)
        {
            var transforms = GetDefaultTransforms();

            AddOptimize2Transforms(transforms);
            Transform.TransformGraph(graph, transforms);
        }

        public void AddQuantizationCheckpoints(Graph graph)
        {
            var transforms = GetDefaultTransforms();

            AddQuantizationCheckpointsTransforms(transforms);
            Transform.TransformGraph(graph, transforms);
        }

        public void QuantizeGraph(Graph graph, Quantizer quantizer)
        {
            var transforms = GetDefaultTransforms();

            AddQuantizeTransforms(transforms, quantizer);
            Transform.TransformGraph(graph, transforms);
        }

        private List<Transform> GetDefaultTransforms()
        {
            var transforms = new List<Transform>
            {
                new FoldTransposeTransform(),
                new FoldNopTransposeTransform(),
                new FoldNopReshapeTransform(),
                new FoldNopPadTransform(),
                new ConstantFoldingTransform(),
                new TransposeBinaryMotionTransform(),
                new TransposeConcatMotionTransform(),
                new TransposeReduceMotionTransform(),
                new TransposeQuantizeMotionTransform(),
                new FoldQuantizeTransform(),
                new FoldInputAndQuantizeTransform(),
            };

            AddDefaultTransforms(transforms);
            return transforms;
        }

        public virtual void RegisterEvaluators(EvaluatorRegistry registry)
        {
        }

        public virtual void AddAllocators(Dictionary<MemoryType, MemoryAllocator> allocators)
        {
            allocators.Add(MemoryType.Constant, new MemoryAllocator());
            allocators.Add(MemoryType.Main, new MemoryAllocator());
        }

        public virtual void RegisterEmitters(CodeGenRegistry registry)
        {
        }

        protected virtual void AddDefaultTransforms(List<Transform> transforms)
        {
        }

        protected virtual void AddOptimize2Transforms(List<Transform> transforms)
        {
        }

        protected virtual void AddQuantizationCheckpointsTransforms(List<Transform> transforms)
        {
        }

        protected virtual void AddQuantizeTransforms(List<Transform> transforms, Quantizer quantizer)
        {
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.Transforms;

namespace NnCase.Targets
{
    public abstract class Target
    {
        public void OptimizePass2(Graph graph)
        {
            var transforms = new List<Transform>
            {
                new FoldTransposeTransform(),
                new FoldNopTransposeTransform(),
                new FoldNopReshapeTransform(),
                new ConstantFoldingTransform(),
                new TransposeBinaryMotionTransform(),
                new TransposeConcatMotionTransform(),
                new TransposeReduceMotionTransform()
            };

            AddOptimize2Transforms(transforms);
            Transform.TransformGraph(graph, transforms);
        }

        public virtual void RegisterEvaluators(EvaluatorRegistry registry)
        {
        }

        protected virtual void AddOptimize2Transforms(List<Transform> transforms)
        {
        }
    }
}

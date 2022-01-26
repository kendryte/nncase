using Autofac;

namespace Nncase.Evaluator
{
    /// <summary>
    /// The Autofac.Module `EvaluatorModule` will auto load when the program startup, see the `ConfigureContainer(ContainerBuilder builder){ builder.RegisterAssemblyModules(assemblies)}`
    /// </summary>
    public class EvaluatorModule : Module
    {
        /// <inheritdoc/>
        protected override void Load(ContainerBuilder builder)
        {
            var evaluators = Evaluator.GetAllEvaluator(typeof(Evaluator));
            foreach (var evaluator in evaluators)
            {
                builder.RegisterType<Ops.CeluEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.EluEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.HardSwishEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.LeakyReluEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReluEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SeluEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SigmoidEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.BinaryEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.CastEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ClampEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ConcatEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.Conv2DEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.Conv2DTransposeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.CumSumEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ExpandEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.FlattenEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.GatherEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.GatherNDEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.MatMulEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.BatchNormalizationEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.InstanceNormalizationEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.LRNEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.OneHotEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.PadEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ProdEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.RandomEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.RangeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReduceEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReduceArgEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReduceWindow2DEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReshapeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ReverseSequenceEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.ShapeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SizeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SliceEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.LogSoftMaxEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SoftMaxEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SoftPlusEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SoftSignEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.SqueezeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.StackEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.TransposeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.UnaryEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.UnSqueezeEvaluator>().AsImplementedInterfaces();
                builder.RegisterType<Ops.BroadcastEvaluator>().AsImplementedInterfaces();
            }
        }
    }
}
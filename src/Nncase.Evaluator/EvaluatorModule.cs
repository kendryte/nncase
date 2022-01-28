using Autofac;

namespace Nncase.Evaluator
{
    public class MathModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            var evaluators = Evaluator.GetAllEvaluator(typeof(Evaluator));
            foreach (var evaluator in evaluators)
            {
                builder.RegisterType(evaluator).AsImplementedInterfaces();
            }
        }
    }
}
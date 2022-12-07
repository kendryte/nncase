using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

internal sealed class FusionMergeRule : IRewriteRule
{

    public IPattern Pattern { get; } =
      IsCall(
        "caller",
        IsFusion("caller_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
        IsCall(
          "callee",
          IsFusion("callee_fusion", Callable.StackVMModuleKind, IsWildcard(), IsVArgs(IsVar())),
          IsWildcard("callee_input")
          )
      );

    public static Fusion MergeSingleInputFusion(Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion, RunPassOptions passOptions)
    {
        if (callee_fusion.Parameters.Count != 1 || caller_fusion.Parameters.Count != 1)
            throw new NotSupportedException("Not Support Multi Inputs Fusion Merge");

        // 1. replace the caller_fusion input_var with the callee_fusion body
        var new_fusion_body = Transform.Mutator.Substitute(e => object.ReferenceEquals(e, caller_fusion.Parameters[0]) ? callee_fusion.Body : null)().Visit(caller_fusion.Body);
        // 2. fold the store load
        // new_fusion_body = CompilerServices.Rewrite(new_fusion_body, new[] { new Transform.Rules.K510.FoldStoreLoad() }, passOptions.IndentDir("MergeSingleInputFusion"));
        var new_fusion = new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", Callable.StackVMModuleKind, new_fusion_body, callee_fusion.Parameters);

        return new_fusion;
    }

    private Dictionary<int, Call> MergedCache = new();

    public Expr? GetReplace(IMatchResult result, RunPassOptions passOptions)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var caller_fusion = (Fusion)result["caller_fusion"];

        // note each patter will generator the new expr. so need cache it.
        var hashcode = HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(caller), ReferenceEqualityComparer.Instance.GetHashCode(callee));
        if (!MergedCache.TryGetValue(hashcode, out var new_call))
        {
            // 1. merge new fusion
            var merged_fusion = MergeSingleInputFusion(caller, callee, caller_fusion, callee_fusion, passOptions);

            // if (true)
            // {
            new_call = new Call(merged_fusion, ImmutableArray.Create((Expr)result["callee_input"]));
            MergedCache.Add(hashcode, new_call);
        }
        else
        {
            System.Console.WriteLine("Re Add Merged Fusion Call");
        }
        return new_call;
        //     merged_calls.Add(caller);
        //     merged_calls.Add(callee);
        //     return true;
        // }
    }
}

public class UnitTestEGraphFusion : TestFixture.UnitTestFixtrue
{
    [Fact]
    public async void TestResNet18Fusion()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        var target = CompilerServices.GetTarget(compileOptions.Target);

        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, ImmutableArray.Create(input));
        IRModule module = new(main);

        CompilerServices.InferenceType(main);
        CompilerServices.DumpIR(main, "", passOptions.DumpDir);

        var pass = new EGraphPass("AutoMergeFusion"){
          new FusionMergeRule()
        };
        await pass.RunAsync(main, passOptions);
    }

}
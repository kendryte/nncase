// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDistributeSchema : TestClassBase
{
    [Fact]
    public void TestExportScheme()
    {
        var scheme = new DistributedSchema("1", "llama", new DistributedSchema.Node[] { new("hidden_in", new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.S(new[] { 1 }), SBP.B }, new[] { 8, 4, 4 }, "cbt") });
        var except = @"{
  ""Version"": ""1"",
  ""Model"": ""llama"",
  ""Outputs"": [
    {
      ""Name"": ""hidden_in"",
      ""NdSBP"": [
        {
          ""$type"": ""B""
        },
        {
          ""$type"": ""S"",
          ""Axes"": [
            0
          ]
        },
        {
          ""$type"": ""S"",
          ""Axes"": [
            1
          ]
        },
        {
          ""$type"": ""B""
        }
      ],
      ""Hierarchy"": [
        8,
        4,
        4
      ],
      ""HierarchyName"": ""cbt""
    }
  ]
}";

        var options = new JsonSerializerOptions();
        options.Converters.Add(new SBPConverter());
        options.WriteIndented = true;
        var export = JsonSerializer.Serialize(scheme, options);
#if DEBUG
        System.Console.WriteLine(export);
#endif
        Assert.Equal(except, export);

        var obj = JsonSerializer.Deserialize<DistributedSchema>(export, options);
    }

    [Fact]
    public async Task TestLoadScheme()
    {
        var path = Path.Join(SolutionDirectory, "src/Nncase.Tests/Distributed/hidden_in.json");
        var options = new Nncase.Targets.CpuTargetOptions()
        {
            Hierarchies = new[] { new[] { 8, 8, 4 } },
            HierarchyNames = "cbt",
            DistributedScheme = path,
        };

        CompileOptions.TargetOptions = options;

        Function func;
        {
            var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 512, 8192 }));
            input.Metadata.OutputNames = new string[] { "hidden_in" };
            var leaky = IR.F.Math.Unary(UnaryOp.Cos, input);
            var output = leaky;
            func = new("main", "cpu", output, default);
        }

        var pass = new AutoDistributedPass(true, "cpu", CompileOptions);

        var result = await pass.RunAsync(func, new());

        Dumpper.DumpIR(result, "result");

        Assert.True(result is Function { Body: Call { Target: IR.Distributed.Boxing } boxing } && boxing.Arguments[0] is Call { Target: IR.Math.Unary { UnaryOp: UnaryOp.Cos } } unary && unary.CheckedType is DistributedType dt && dt == new DistributedType(new(DataTypes.Float32, new[] { 1, 512, 8192 }), new[] { (SBP)SBP.B, SBP.S(new[] { 0 }), SBP.S(new[] { 1, 2 }) }, new(new[] { 8, 8, 4 }, "cbt")));
    }
}

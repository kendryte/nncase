using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.Targets;

public class UnitTestCPUTarget
{
    [Fact]
    public void TestCreateCPUTarget()
    {
        var target = CompilerServices.GetTarget("cpu");
        Assert.NotNull(target);
    }

    [Fact]
    public void TestCreateStackVMModuleBuilder()
    {
        var target = CompilerServices.GetTarget("cpu");
        var moduleBuilder = target.CreateModuleBuilder("stackvm");
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestSimpleCodeGen()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget("cpu");
        var modelBuilder = new ModelBuilder(target);
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open("testSimpleCodegen.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
    }
}

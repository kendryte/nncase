using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
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
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenUseVarMultiTimes()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenTuple()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        TestCodeGen(new IR.Tuple(y), new[] { x });
    }

    private void TestCodeGen(Expr body, Var[] vars, [CallerMemberName] string name = null)
    {
        var main = new Function("main", body, vars);
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget("cpu");
        var modelBuilder = new ModelBuilder(target);
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open($"{name}.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
    }
}

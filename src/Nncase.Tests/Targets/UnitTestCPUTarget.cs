using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
}

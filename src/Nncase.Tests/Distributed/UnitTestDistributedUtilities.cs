// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.DistributedTest;

public sealed class UnitTestDistributedUtilities
{
    [Fact]
    public void TestGenerateReduceGroups()
    {
        Assert.Single(Utilities.LinqUtility.Combination(1));
        Assert.Equal(3, Utilities.LinqUtility.Combination(2).Count());
    }
}

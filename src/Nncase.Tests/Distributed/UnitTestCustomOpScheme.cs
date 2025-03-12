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
public class UnitTestCustomOpScheme : TestClassBase
{
    [Fact]
    public void TestExportScheme()
    {
        var scheme = new CustomOpScheme("1", "matmul", new CustomOpScheme.Node[] { new CustomOpScheme.Node(string.Empty, "Matmul", [[32, 32], [32, 32]], new[] { new SBP[] { SBP.B, SBP.B, SBP.B }, new SBP[] { SBP.B, SBP.B, SBP.S(1) } }, 1, string.Empty) });
        var except = @"{
  ""Version"": ""1"",
  ""Model"": ""matmul"",
  ""Outputs"": [
    {
      ""Name"": """",
      ""Op"": ""Matmul"",
      ""Shape"": [
        [
          32,
          32
        ],
        [
          32,
          32
        ]
      ],
      ""SBP"": [
        [
          {
            ""$type"": ""B""
          },
          {
            ""$type"": ""B""
          },
          {
            ""$type"": ""B""
          }
        ],
        [
          {
            ""$type"": ""B""
          },
          {
            ""$type"": ""B""
          },
          {
            ""$type"": ""S"",
            ""Axis"": 1
          }
        ]
      ],
      ""Cost"": 1,
      ""CSourcePath"": """"
    }
  ]
}";

        var export = JsonSerializer.Serialize(scheme, new JsonSerializerOptions() { WriteIndented = true });
#if DEBUG
        System.Console.WriteLine(export);
#endif
        Assert.Equal(except, export);

        var obj = JsonSerializer.Deserialize<CustomOpScheme>(export);
    }
}

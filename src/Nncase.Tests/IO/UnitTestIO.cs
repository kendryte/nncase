// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Text;
using Nncase.IO;
using Xunit;

namespace Nncase.Tests.IOTest
{
    public class UnitTestIO
    {
        [Fact]
        public void TestWriter()
        {
            var buf = new byte[10];
            var bw = new BitWriter(buf);
            bw.Write(10, 4); // 1010
            bw.Write(3, 3); // 011
            bw.Write(5, 3); // 10 | 1 011 1010
            bw.Flush();
            var bin = Convert.ToString(buf[0], 2);
            Console.WriteLine(bin);
            Assert.Equal("10111010", bin);

            bin = Convert.ToString(buf[1], 2);
            Assert.Equal("10", bin);

            // TestReader
            var br = new BitReader(buf);
            Assert.Equal(10, br.Read<int>(4));
            Assert.Equal(3, br.Read<int>(3));
            Assert.Equal(5, br.Read<int>(3));
        }
    }
}

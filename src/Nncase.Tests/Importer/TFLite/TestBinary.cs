using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;
using System.Collections.Generic;
using Python.Runtime;
using System;
using System.IO;

namespace Nncase.Tests.Importer.TFLite
{

    public class TestBasic
    {
        [Fact]
        public void TestBinary()
        {
            using (Py.GIL())
            {
                // var path = Environment.GetEnvironmentVariable("PYTHONPATH");
                // Environment.SetEnvironmentVariable("PYTHONPATH", $"/Users/lisa/Documents/nncase/tests:{path}");
                Console.WriteLine(Environment.GetEnvironmentVariable("PYTHONPATH"));
                dynamic basic = Py.Import("importer.tflite.basic");
                // dynamic binary = Py.Import("test_binary");
                basic.test_binary.generate();

                // Py.Import("importer");
                // Py.Import("tests");
                // Py.Import("importer");
                // Py.Import("tflite");
                // Py.Import("basic");
                // dynamic obj = Py.Import("tests");
                // obj.test_binary.generate();
                // Py.Import("test_binary");
                //Py.Import("test_runner");
            }
        }
    }


}
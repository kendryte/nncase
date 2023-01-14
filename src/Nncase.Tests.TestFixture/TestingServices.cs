// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Transform;

namespace Nncase.Tests;

public static class Testing
{
    /// <summary>
    /// the fixed rand generator, maybe need impl by each module.
    /// </summary>
    public static readonly Random RandGenerator = new System.Random(123);

    /// <summary>
    /// fixup the seq rand tensor into gived range.
    /// </summary>
    /// <param name="range"></param>
    /// <param name="symmetric"></param>
    /// <returns></returns>
    public static ValueRange<float> FixupRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = Math.Max(Math.Max(Math.Abs(range.Min), Math.Abs(range.Max)), 0.01f);
            return new() { Min = -r, Max = r };
        }
        else
        {
            if (range.Min < -1e3f)
            {
                range.Min = -1e3f;
            }

            if (range.Max > 1e3f)
            {
                range.Max = 1e3f;
            }

            var r = range.Max - range.Min;
            if (r == 0)
            {
                r = 0.1f;
            }
            else if (r < 0.01f)
            {
                r = 0.01f;
            }

            range.Max = range.Min + r;

            if (range.Max < 0)
            {
                range.Max = 0;
            }

            if (range.Min > 0)
            {
                range.Min = 0;
            }
        }

        return range;
    }

    /// <summary>
    /// create the rand value by gived datatype.
    /// </summary>
    /// <param name="dataType"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor Rand(DataType dataType, params int[] shape)
    {
        return IR.F.Random.Normal(dataType, 0, 1, 1, shape).Evaluate().AsTensor();
    }

    /// <summary>
    /// create the rand value by gived datatype.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor<T> Rand<T>(params int[] shape)
        where T : unmanaged, IEquatable<T>
    {
        return IR.F.Random.Normal(DataType.FromType<T>(), 0, 1, 1, shape).Evaluate().AsTensor().Cast<T>();
    }

    /// <summary>
    /// create the seq value by gived datatype.
    /// </summary>
    /// <param name="dataType"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor Seq(DataType dataType, params int[] shape)
    {
        return (Tensor)typeof(Testing).GetMethod("Seq", new[] { typeof(int[]) })!.MakeGenericMethod(dataType.CLRType).Invoke(null, new object[] { shape })!;
    }

    /// <summary>
    /// create the seq value by gived datatype.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor<T> Seq<T>(params int[] shape)
        where T : unmanaged, IEquatable<T>
    {
        return Tensor.FromArray(Enumerable.Range(0, (int)TensorUtilities.GetProduct(shape)).ToArray())
            .Cast<T>(CastMode.KDefault).Reshape(shape);
    }

    /// <summary>
    /// NOTE 映射一个sequence到新的range.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="t"></param>
    /// <param name="range"></param>
    /// <returns></returns>
    public static Tensor<T> ReArangeSeq<T>(Tensor<T> t, ValueRange<float> range)
      where T : unmanaged, System.IEquatable<T>
    {
        var scale = (range.Max - range.Min) / t.Length;
        return Tensor.FromArray(t.Cast<float>(CastMode.KDefault).Select(i => (i * scale) + range.Min).ToArray())
                .Cast<T>()
                .Reshape(t.Shape);
    }

    /// <summary>
    /// check all value close.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="tol"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static int AllClose(Tensor a, Tensor b, float tol = .003f)
    {
        if (a.Shape != b.Shape)
        {
            throw new InvalidOperationException();
        }

        if (a.ElementType != b.ElementType)
        {
            throw new InvalidOperationException();
        }

        int err_count = 0;

        // int offset = 0;
        foreach (var p in a.Cast<float>().Zip(b.Cast<float>()))
        {
            if (Math.Abs(p.Item1 - p.Item2) > tol)
            {
                err_count++;
            }
        }

        return err_count;
    }

    /// <summary>
    /// dump value.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="writer"></param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static void DumpValue(IValue v, StreamWriter writer)
    {
        switch (v)
        {
            case TensorValue t:
                writer.WriteLine(t.AsTensor().GetArrayString());
                break;
            case TupleValue tp:
                foreach (var f in tp)
                {
                    DumpValue(f, writer);
                }

                break;
            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    /// <summary>
    /// dump value.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="path"></param>
    public static void DumpValue(IValue v, string path)
    {
        using (var sw = new StreamWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
        {
            DumpValue(v, sw);
        }
    }

    /// <summary>
    /// build kmodel.
    /// </summary>
    /// <param name="name">the dumped kmodel name.</param>
    /// <param name="module"></param>
    /// <param name="compileSession"></param>
    /// <returns>kmodel_path and kmodel bytes.</returns>
    public static (string, byte[]) BuildKModel(string name, IR.IRModule module, CompileSession compileSession)
    {
        var modelBuilder = compileSession.GetRequiredService<IModelBuilder>();
        var linkedModel = modelBuilder.Build(module);

        Directory.CreateDirectory(compileSession.CompileOptions.DumpDir);
        var kmodel_path = Path.Combine(compileSession.CompileOptions.DumpDir, $"{name}.kmodel");
        using (var output = System.IO.File.Open(kmodel_path, System.IO.FileMode.Create))
        {
            linkedModel.Serialize(output);
        }

        return (kmodel_path, File.ReadAllBytes(kmodel_path));
    }

    /// <summary>
    /// dump kmodel args and bin for cli interp.
    /// </summary>
    /// <param name="kmodel_path"></param>
    /// <param name="input_tensors"></param>
    /// <param name="dumpper"></param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static void DumpInterpModel(string kmodel_path, Tensor[] input_tensors, IDumpper dumpper)
    {
        string input_pool_path = Path.Join(dumpper.Directory, "input_pool.bin");
        var output_pool_path = Path.Join(dumpper.Directory, "output_pool.bin");
        using var args_writer = new StreamWriter(dumpper.OpenFile("args.txt"));
        args_writer.WriteLine(kmodel_path);
        args_writer.WriteLine(input_pool_path);
        args_writer.WriteLine(output_pool_path);

        uint start = 0;
        uint size = 0;
        args_writer.WriteLine(input_tensors.Length);
        using (var pool_writer = new BinaryWriter(File.OpenWrite(input_pool_path)))
        {
            foreach (var in_tensor in input_tensors)
            {
                pool_writer.Write(in_tensor.BytesBuffer);
                size = checked((uint)in_tensor.BytesBuffer.Length);
                byte dt_code = in_tensor.ElementType switch
                {
                    var x when x == DataTypes.Boolean => 0x00,
                    var x when x == DataTypes.Int8 => 0x02,
                    var x when x == DataTypes.Int16 => 0x03,
                    var x when x == DataTypes.Int32 => 0x04,
                    var x when x == DataTypes.Int64 => 0x05,
                    var x when x == DataTypes.UInt8 => 0x06,
                    var x when x == DataTypes.UInt16 => 0x07,
                    var x when x == DataTypes.UInt32 => 0x08,
                    var x when x == DataTypes.UInt64 => 0x09,
                    var x when x == DataTypes.Float16 => 0x0A,
                    var x when x == DataTypes.Float32 => 0x0B,
                    var x when x == DataTypes.Float64 => 0x0C,
                    var x when x == DataTypes.BFloat16 => 0x0D,
                    _ => throw new ArgumentOutOfRangeException(),
                };
                args_writer.WriteLine($"{dt_code}");
                args_writer.WriteLine(in_tensor.Shape.Count);
                args_writer.WriteLine($"{string.Join(' ', in_tensor.Shape)}");
                args_writer.WriteLine($"{start} {size}");
                start += size;
            }
        }
    }

    public static IValue RunKModel(byte[] kmodel, string dump_path, Tensor[] input_tensors)
    {
        using (var interp = Nncase.Runtime.Interop.RTInterpreter.Create())
        {
            interp.SetDumpRoot(dump_path);
            interp.LoadModel(kmodel);
            var entry = interp.Entry!;

            var rtInputs = input_tensors.Select(Nncase.Runtime.Interop.RTTensor.FromTensor).ToArray();
            return entry.Invoke(rtInputs).ToValue();
        }
    }

    public static IValue RunKModel(byte[] kmodel, string dump_path, Runtime.Interop.RTTensor[] input_tensors)
    {
        using (var interp = Nncase.Runtime.Interop.RTInterpreter.Create())
        {
            interp.SetDumpRoot(dump_path);
            interp.LoadModel(kmodel);
            var entry = interp.Entry!;
            return entry.Invoke(input_tensors).ToValue();
        }
    }
}

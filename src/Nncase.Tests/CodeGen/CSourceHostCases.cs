// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
public abstract class ICodeGenCase
{
    /// <summary>
    /// get the entry function
    /// </summary>
    /// <returns></returns>
    public abstract PrimFunction GetEntry();

    /// <summary>
    /// custom equal compare method
    /// </summary>
    /// <param name="rtmod"></param>
    public abstract void CompareEqual(IRTModel rtmod);

    public virtual FunctionPass GetPass()
    {
        return new EmptyPass();
    }
}

public class SubCase : ICodeGenCase
{
    public override PrimFunction GetEntry()
    {
        var func = T.PrimFunc("sub",
                  T.Buffer(TensorType.Scalar(DataTypes.Float32), MemoryLocation.Input, out var x),
                  T.Buffer(TensorType.Scalar(DataTypes.Float32), MemoryLocation.Input, out var y)).Body(
          x - y
        );
        return func;
    }

    public override void CompareEqual(IRTModel rtmod)
    {
        Assert.Equal(2.3f - 2.1f, rtmod.Invoke(2.3f, 2.1f));
    }
}

public class ForCase : ICodeGenCase
{
    void RefFunc(int[] A, int n)
    {
        for (int i = 0; i < n; i++)
        {
            A[i] = A[i] + 1;
            for (int j = 0; j < n; j++)
            {
                A[i] = A[i] + j;
            }
        }
    }

    /// <inheritdoc/>
    public override void CompareEqual(IRTModel rtmod)
    {
        var rand = new Random();
        int n = 12;
        var A1 = Enumerable.Range(0, n).Select(i => rand.Next(456)).ToArray();
        var A2 = new int[n];
        A1.CopyTo(A2, 0);

        RefFunc(A1, n);
        rtmod.Invoke(A2, n);
        Assert.True(Enumerable.Range(0, n).All(i => A1[i] == A2[i]));
    }

    public override PrimFunction GetEntry()
    {
        return T.PrimFunc("for_loop",
               T.Buffer(new(DataTypes.Int32, new[] { 100 }), MemoryLocation.Input, out var A),
               T.Buffer(TensorType.Scalar(DataTypes.Int32), MemoryLocation.Input, out var n)
               ).Body(
          T.Serial(out var i, n).Body(
            T.Store(A[i], A[i] + 1),
            T.Serial(out var j, n).Body(
              T.Store(A[i], A[i] + j)
            )
          )
        );
    }
}

public class ForGridCase : ICodeGenCase
{

    void RefFunc(int[] A, int n, int m)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                A[i * n + j] = i + j;
            }
        }
    }

    public override void CompareEqual(IRTModel rtmod)
    {
        int n = 10, m = 20;
        var A1 = new Tensor<int>(new[] { n, m }).ToArray();
        var A2 = new Tensor<int>(new[] { n, m }).ToArray();
        RefFunc(A1, n, m);
        rtmod.Invoke(A2, n, m);
        Assert.True(Enumerable.Range(0, n * m).All(i => A1[i] == A2[i]));
    }

    public override PrimFunction GetEntry()
    {
        var n = T.SizeVar("n");
        var m = T.SizeVar("m");
        // T.Buffer( DataTypes.Int32, out var A);
        var func = T.PrimFunc("main", A.Handle, n, m).Body(
          T.Grid(out var i, out var j, (n, m)).Body(
            T.Store(A[i * n + j], i + j)
          )
        );
        return func;
    }
}

public class BlockCase : ICodeGenCase
{

    public override PrimFunction GetEntry()
    {
        var n = T.SizeVar("n");
        var m = T.SizeVar("m");
        var A = T.DeclBuffer((n, m), name: "A");
        var func = T.PrimFunc("func", A.Handle, n, m).Body(
        T.Grid(out var i, out var j, (n, m), out var lp).Body(
          T.Block("init").Remap(out var vi, out var vj, (lp.i, lp.j), "SS").
          Init(
            T.Store(A[vi, vj], 1.0f)
          ).Body(
            T.Store(A[vi, vj], IR.F.Tensors.Cast(vi + vj, DataTypes.Float32))
          )
        ),
        n + m
        );
        return func;
    }

    public int RefFunc(float[] A, int n, int m)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                A[i * m + j] = i + j;
            }
        }
        return n + m;
    }

    public override void CompareEqual(IRTModel rtmod)
    {
        int n = 10, m = 12;
        var A1 = new float[n * m];
        var A2 = new float[n * m];
        var r1 = RefFunc(A1, n, m);
        var r2 = rtmod.Invoke(A1, n, m);
        Assert.Equal(r1, r2);
    }

    public override FunctionPass GetPass()
    {
        var pass = new TIRPass("TIRPass"){
                 new Transform.Mutator.LowerBlockInit(),
                 new Transform.Mutator.ConvertBlocksToOpaque(),
                 new Transform.Mutator.FlattenBuffer()
             };
        return pass;
    }
}
 }
#endif

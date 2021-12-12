using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// Tensor Range Define
    /// </summary>
    /// <param name="Min">beginning of the nodes</param>
    /// <param name="Extent">the extend of range</param>
    public sealed record TRange(Expr Min, Expr Extent)
    {
        /// <summary>
        /// <see cref="TRange"/>
        /// </summary>
        /// <param name="tuple"> value tuple </param>
        public TRange(ValueTuple<int, int> tuple) : this(tuple.Item1, tuple.Item2) { }

        /// <summary>
        /// <see cref="TRange"/>
        /// </summary>
        /// <param name="tuple"> value tuple </param>
        public static implicit operator TRange(ValueTuple<int, int> tuple) => new TRange(tuple.Item1, tuple.Item2);
    }

    /// <summary>
    ///  Iteration Variable,
    ///  represents an iteration over an integer interval.
    /// </summary>
    /// <param name="Dom">the domain of iteration, if known, can be None
    ///  For the intermediate schedule node, before schedule.</param>
    /// <param name="Var">The looping variable </param>
    /// <param name="IterType">The type of the IterVar </param>
    /// <param name="ThreadTag"> additional tag on the iteration variable, set this if this is binded already to a known thread tag. </param>
    public sealed record IterVar(TRange Dom, Var Var, IterMode IterMode, string ThreadTag = "")
    {
        public override string ToString()
        {
            return $"iter_var({Var.Name},{Dom},{ThreadTag})";
        }
    }


    /// <summary>
    /// a named variable represents a tensor index size
    /// </summary>
    /// <param name="Name"></param>
    /// <param name="DType"></param>
    public sealed record SizeVar(string Name, DataType DType) : Var(Name, new TensorType(DType, Shape.Scalar))
    {
        /// <summary>
        /// <see cref="SizeVar"/>
        /// </summary>
        public SizeVar(string Name = "int32", ElemType DType = ElemType.Int32) : this(Name, new DataType(DType, 1)) { }

        public override string ToString()
        {
            return $"{{{Name}>=0}}";
        }
    }
}
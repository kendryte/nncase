using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// Tensor Range Define
    /// </summary>
    /// <param name="Min">beginning of the nodes</param>
    /// <param name="Extent">the extend of range</param>
    public sealed record Range(Expr Min, Expr Extent)
    {
        /// <summary>
        /// <see cref="Range"/>
        /// </summary>
        /// <param name="tuple"> value tuple </param>
        public Range(ValueTuple<int, int> tuple) : this(tuple.Item1, tuple.Item2) { }

        /// <summary>
        /// <see cref="Range"/>
        /// </summary>
        /// <param name="tuple"> value tuple </param>
        public static implicit operator Range(ValueTuple<int, int> tuple) => new Range(tuple.Item1, tuple.Item2);
    }

    /// <summary>
    ///  Iteration Variable,
    ///  represents an iteration over an integer interval.
    /// </summary>
    /// <param name="Dom">
    ///  the domain of iteration, if known, can be None For the intermediate schedule node, before schedule.
    /// </param>
    /// <param name="Var">The looping variable </param>
    /// <param name="IterMode">The type of the IterVar </param>
    /// <param name="ThreadTag"> additional tag on the iteration variable, set this if this is binded already to a known thread tag. </param>
    public sealed record IterVar(Range Dom, Var Var, IterMode IterMode, string ThreadTag = "")
    {
        public override string ToString()
        {
            return $"iter_var({Var.Name},{Dom},{ThreadTag})";
        }
    }


    /// <summary>
    /// <see cref="T.SizeVar(string, ElemType)"/>
    /// </summary>
    public sealed record SizeVar(string Name, DataType DType) : Var(Name, new TensorType(DType, Shape.Scalar))
    {
        public override string ToString()
        {
            return $"{{{Name}>=0}}";
        }
    }
}
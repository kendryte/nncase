using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR
{

    internal static class Util
    {
        public static string PrintList<T>(IRArray<T> Exprs)
        {
            string ret = "";
            foreach (var (i, expr) in Enumerable.Range(0, Exprs.Count).Zip(Exprs))
            {
                ret += $"{expr}";
                if (i < Exprs.Count - 1)
                {
                    ret += ", ";
                }
            }
            return ret;
        }
    }

    public sealed record Select(Expr Condition, Expr TrueValue, Expr FalseValue) : Expr { }

    public sealed record BufferLoad(Buffer Buffer, IRArray<Expr> Indices) : Expr
    {
        public override string ToString()
        {
            return $"{Buffer.Name}[{Util.PrintList(Indices)}]";
        }
    }

    /// <summary>
    /// Load value from the result produced by the producer.
    /// </summary>
    /// <remarks>
    /// This node only appears in high-level DSLs that are built on top of the TIR.
    /// It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
    /// this node before TIR transformations.
    /// </remarks>
    /// <seealso cref="ProducerLoad"/>
    /// <param name="Producer">The buffer producer.</param>
    /// <param name="Indices">The location arguments.</param>
    public sealed record ProducerLoad(DataProducer Producer, IRArray<Expr> Indices) : Expr
    {
        public override string ToString()
        {
            return $"{Producer.GetNameHint()}[{Util.PrintList(Indices)}]";
        }
    }

    /// <summary>
    /// Create a vector where all the elements are value.
    /// </summary>
    /// <param name="Value"> The base value. </param>
    /// <param name="lanes"> The number of lanes. </param>
    public sealed record Broadcast(Expr Value, int lanes) : Expr
    {
    }

    /// <summary>
    /// Let binding. Bind var to value then evaluate body.
    /// </summary>
    /// <param name="Var"> The variable. </param>
    /// <param name="Value"> The value to be binded. </param>
    /// <param name="Body"> The result expression. </param>
    public sealed record Let(Var Var, Expr Value, Expr Body) : Expr
    {
        public override string ToString()
        {
            return $"(let {Var.Name} = {Value} in {Body})";
        }
    }

    /// <summary>
    /// Shuffle instruction.
    ///  vec = concat(vectors)
    ///  result = (vec[indices[0]], vec[indices[1]] ...)
    /// </summary>
    /// <param name="Vectors">the input vectors. </param>
    /// <param name="Indices">The indices of each element. </param>
    public sealed record Shuffle(IRArray<Expr> Vectors, IRArray<Expr> Indices) : Expr
    {
        public override string ToString()
        {
            return $"shuffle({Util.PrintList(Vectors)},{Util.PrintList(Indices)})";
        }
    }


    // Reduce operator
    /*!
     * \brief A commutative reducer node to represent a commutative
     *  binary operator with identity element
     */
    /// <summary>
    /// 
    /// </summary>
    /// <param name="Lhs">The left argument of reducer </param>
    /// <param name="Rhs">The right argument of reducer </param>
    /// <param name="Result">The result of reducer </param>
    /// <param name="IdentityElement">The identity element of reducer, which leaves other
    ///  elements unchanged when combined with it, with respect to
    ///  the binary operation of this reducer uses. </param>
    public sealed record CommReducer(IRArray<Var> Lhs, IRArray<Var> Rhs, IRArray<Expr> Result, IRArray<Expr> IdentityElement)
    {
        /// <summary>
        /// Function call operator to combine a and b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public IRArray<Expr> Combine(IRArray<Expr> a, IRArray<Expr> b) { return a; }

        public override string ToString()
        {
            return $"comm_reducer(result={Result}, lhs= {Lhs}, rhs= {Rhs}, identity_element= {IdentityElement})";
        }
    }


    /// <summary>
    /// Reduction operator 
    /// </summary>
    public sealed record Reduction : Expr
    {
        /// <summary>
        /// The source operand. 
        /// </summary>
        public CommReducer? Combiner;

        /// <summary>
        /// The reduction axis. 
        /// </summary>
        public IRArray<Expr> Source;

        /// <summary>
        /// Predicate on the reduction Only add the body to reduction if condition is true.
        /// </summary>
        public Expr Condition;

        /// <summary>
        /// The init operand. 
        /// </summary>
        public IRArray<Expr>? Init;

        /// <summary>
        /// the index of this reduce node
        /// </summary>
        public IRArray<IterVar> Axis;

        /// <summary>
        /// The commutative combiner. 
        /// </summary>
        public int ValueIndex;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="combiner"></param>
        /// <param name="source"></param>
        /// <param name="axis"></param>
        /// <param name="condition"></param>
        /// <param name="value_index"></param>
        /// <param name="init"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public Reduction(CommReducer? combiner, IRArray<Expr> source, IRArray<IterVar> axis,
               Expr? condition = null, int value_index = 0, IRArray<Expr>? init = null)
        {
            if (!axis.All(x => x.IterMode == IterMode.CommReduce))
                throw new InvalidOperationException("Can only take axis created by reduce_axis");
            if (condition is null)
                condition = (Const)1;

            if (init is not null)
            {
                if (source.Count != init?.Count) throw new InvalidOperationException("");
                if (!init.Value.All(x => (
                  x is ProducerLoad) ||
                   ((x is Const con) &&
                    (con.IsIntImm() || con.IsFloatImm()))))
                {
                    throw new InvalidOperationException("init can only be a IntImm, FloatImm or ProducerLoad");
                }
            }
            Combiner = combiner;
            Source = source;
            Axis = axis;
            Condition = condition;
            ValueIndex = value_index;
            Init = init;
        }

        public override string ToString()
        {
            return $"reduction(combiner= {Combiner}, source= {Source}, init= {Init}, axis= {Axis}, where= {Condition}, value_index= {ValueIndex})";
        }
    }


    /// <summary>
    /// Any shape.
    /// </summary>
    public sealed record Any() : Expr
    {
        /// <summary>
        /// Convert to var.
        /// </summary>
        /// <returns></returns>
        public Var ToVar() { return new Var("any_dim", new TensorType(DataType.Int32, Shape.Scalar)); }

        /// <summary>
        /// Convert to SizeVar.
        /// </summary>
        /// <returns></returns>
        public SizeVar ToSizeVar() { return new SizeVar("any_dim", DataType.Int32); }

        public override string ToString()
        {
            return "?";
        }
    }
}
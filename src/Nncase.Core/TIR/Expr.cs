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


    /// <summary>
    /// The container of Exprs.
    /// Represent a sequence of Expr.
    /// </summary>
    /// <param name="Bodys">internal sequence content.</param>
    public sealed record Sequential(IRArray<Expr> Bodys) : Expr
    {
        public int Count => Bodys.Count;
        public Expr this[int index] => Bodys[index];
    }


    /// <summary>
    /// select the value and return it, the true and false must have same type!
    /// </summary>
    /// <param name="Condition"></param>
    /// <param name="TrueValue"></param>
    /// <param name="FalseValue"></param>
    public sealed record Select(Expr Condition, Expr TrueValue, Expr FalseValue) : Expr { }

    public sealed record BufferLoad(Buffer Buffer, IRArray<Expr> Indices) : Expr
    {
        public override string ToString()
        {
            return $"{Buffer.Name}[{Util.PrintList(Indices)}]";
        }
    }


    /// <summary>
    /// Store, return unit
    /// </summary>
    public sealed record Store : Expr
    {
        /// <summary>
        ///The buffer variable.
        /// </summary>
        public Var BufferHandle;
        /// <summary>
        ///The value to be stored.
        /// </summary>
        public Expr Value;
        /// <summary>
        ///The index locations to be stored.
        /// </summary>
        public Expr Index;
        /// <summary>
        ///The predicate to mask which lanes would be stored.
        /// </summary>
        public Expr Predicate;

        /// <summary>
        /// Store value to the buffer.
        /// Equivalent to ((DType*)buffer_var)[index] = value.
        /// where DType is the type specified by type().element_of().
        /// <example>
        /// if type = float32x3, then the store will corresponds to
        /// <code>
        ///  auto buffer = static_cast<float*>(buffer_var);
        ///  buffer[index.v0] = value.v0;
        ///  buffer[index.v1] = value.v1;
        ///  buffer[index.v2] = value.v2;
        /// </code>
        /// </example>
        /// </summary>
        /// <param name="buffer_handle">The buffer Variable.</param>
        /// <param name="value">The value we want to store.</param>
        /// <param name="index">he index in the store expression.</param>
        /// <param name="predicate">The store predicate.</param>
        public Store(Var buffer_handle, Expr value, Expr index, Expr? predicate = null)
        {
            predicate ??= F.TOps.MakeConst<int>(1, F.TOps.LanesOp(buffer_handle));
            BufferHandle = buffer_handle;
            Value = value;
            Index = index;
            Predicate = predicate;
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
    /// Let binding. Bind var to value then evaluate body. return unit
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
    /// A While loop
    /// <code>
    ///   while (condition) { body }
    /// </code>
    /// </summary>
    /// <param name="Condition">The termination condition.</param>
    /// <param name="Body">The body of the while loop.</param>
    public sealed record While(Expr Condition, Sequential Body) : Expr
    { }

    /// <summary>
    /// A for loop, with poissible type annotations.
    /// <example>
    /// <code>
    ///   for (loop_var = min; loop_var < min + extent; ++loop_var) {
    ///     body
    ///    }
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="LoopVar">The loop variable.</param>
    /// <param name="Min">The minimum value of iteration.</param>
    /// <param name="Extent">The extent of the iteration.</param>
    /// <param name="Kind">The kind of the for loop.</param>
    /// <param name="Body">The body of the for loop.</param>
    /// <param name="ThreadBinding"> Only valid when kind == ForKind::kThreadBinding The context thread that this loop variable bounds to.</param>
    public sealed record For(Var LoopVar, Expr Min, Expr Extent, ForMode Kind, Sequential Body, IterVar? ThreadBinding = null) : Expr
    {

        /// <summary>
        ///   These annotations can be used as auxiliary hint
        ///  to future transformations. An annotation should
        ///  not change the control flow semantics of the loop
        ///  and can be ignored in most passes.
        /// </summary>
        public readonly Dictionary<string, object> Annotations = new();
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
}
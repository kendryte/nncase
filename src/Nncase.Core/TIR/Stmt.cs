using Nncase.IR;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System;
using static Nncase.IR.Utility;

namespace Nncase.TIR
{
    /// <summary>
    /// Assert condition, if an error occurs, return the error message.
    /// </summary>
    /// <param name="Condition">Condition to be checked.</param>
    /// <param name="Message">Error message when assertion failed.</param>
    /// <param name="Body">Body which this assertion holds true. Will be executed after the assertion.</param>
    // public sealed record AssertStmt(Expr Condition, Expr Message, Stmt Body) : Stmt { }



    /// <summary>
    /// Store value to the high dimension buffer.
    /// <code>
    /// buffer[i, j] = value;
    /// </code>
    /// </summary>
    /// <param name="Buffer">The buffer variable.</param>
    /// <param name="Value">The value to be stored.</param>
    /// <param name="Indices">The indices location to be stored.</param>
    // public sealed record BufferStore(Buffer Buffer, Expr Value, IRArray<Expr> Indices) : Stmt { }

    /// <summary>
    /// Annotate the region where the buffer need to
    ///  be read and write in the body.
    ///  We only need to allocate the space for the corresponding region.
    ///  <remarks> 
    ///  There should be at most one BufferRealize for each buffer. 
    ///  BufferRealize is not necessary for external buffers,       
    ///  since they are assumed to be fully allocated. 
    ///  </remarks>
    /// </summary>
    /// <param name="Buffer">The buffer variable.</param>
    /// <param name="Bounds">Bounds to be realized</param>
    /// <param name="Condition">Only realize if condition holds.</param>
    /// <param name="Body">The body of realization.</param>
    // public sealed record BufferRealize(Buffer Buffer, IRArray<Range> Bounds, Expr Condition, Stmt Body) : Stmt
    // { }
    /// <summary>
    /// Store value into mult-dimensional array that will be read by the consumer
    /// of the producer.
    /// <remarks>
    ///   This node only appears in high-level DSLs that are built on top of the TIR.
    ///   It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
    ///   this node before TIR transformations.
    /// </remarks>
    /// </summary>
    /// <param name="Producer">The producer to store the results into.</param>
    /// <param name="Value">The value to be stored.</param>
    /// <param name="Indices">The index arguments of the function.</param>
    // public sealed record ProducerStore(DataProducer Producer, Expr Value, IRArray<Expr> Indices) : Stmt
    // { }
    /// <summary>
    /// Annotate the bounds where the data produced by the producer
    ///  need to be written and read in body.
    ///  We will need to allocate space for the corresponding regions.
    ///  <remarks>
    ///   This node only appears in high-level DSLs that are built on top of the TIR.
    ///   It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
    ///   this node before TIR transformations.
    /// </remarks>
    /// </summary>
    /// <param name="Producer">The producer that produces the data.</param>
    /// <param name="Bounds">Bounds to be realized.</param>
    /// <param name="Condition">Only realize if condition holds.</param>
    /// <param name="Body">The body of realization.</param>
    /// <param name="StorageScope">The storage scope associated with this realization.</param>
    // public sealed record ProducerRealizeNode(DataProducer Producer, IRArray<Range> Bounds, Expr Condition, Stmt Body, string StorageScope) : Stmt
    // { }

    /// <summary>
    /// Allocate a buffer that can be used in body.
    /// </summary>
    // public sealed record Allocate : Stmt
    // {
    //     /// <summary>
    //     /// The buffer variable.
    //     /// </summary>
    //     public Var BufferVar;

    //     /// <summary>
    //     /// The extents of the buffer.
    //     /// </summary>
    //     public IRArray<Expr> Extents;

    //     /// <summary>
    //     /// Only allocate buffer when condition is satisfied.
    //     /// </summary>
    //     public Expr Condition;

    //     /// <summary>
    //     /// The body to be executed.
    //     /// </summary>
    //     public Stmt Body;

    //     /// <summary>
    //     /// Additional annotations about the allocation.
    //     /// These annotations can be used as auxiliary hint
    //     ///  to future transformations.
    //     /// </summary>
    //     public Dictionary<string, object> Annotations;

    //     /// <summary>
    //     /// <see cref="Allocate"/>
    //     /// </summary>
    //     /// <param name="buffer_var">The buffer variable.</param>
    //     /// <param name="extents">The extents of the allocate</param>
    //     /// <param name="condition">The condition.</param>
    //     /// <param name="body">The body statement.</param>
    //     /// <param name="annotations">Additional annotation hints</param>
    //     public Allocate(Var buffer_var, IRArray<Expr> extents, Expr condition,
    //                        Stmt body, Dictionary<string, object>? annotations = null)
    //     {
    //         foreach (var x in extents) { IsScalar().Check(x.CheckedType); }
    //         (IsBool() & IsScalar()).Check(condition.CheckedType);
    //         BufferVar = buffer_var;
    //         Extents = extents;
    //         Condition = condition;
    //         Body = body;
    //         Annotations = annotations ?? new();
    //     }
    // }

    /// <summary>
    /// IfThenElse statment.
    /// </summary>
    public sealed record IfThenElse : Expr
    {
        /// <summary>
        /// The condition.
        /// </summary>
        public Expr Condition;
        /// <summary>
        /// The branch to be executed when condition is true.
        /// </summary>
        public Sequential Then;
        /// <summary>
        /// The branch to be executed when condition is false, can be null.
        /// </summary>
        public Sequential Else;

        /// <summary>
        /// <see cref="IfThenElse"/>
        /// </summary>
        /// <param name="condition">The expression</param>
        /// <param name="then_case">The statement to execute if condition is true.</param>
        /// <param name="else_case">The statement to execute if condition is false.</param>
        public IfThenElse(Expr condition, Sequential then_case, Sequential else_case)
        {
            (IsBool() & IsScalar()).Check(condition.CheckedType);
            Condition = condition;
            Then = then_case;
            Else = else_case;
        }
    }

    // /// <summary>
    // /// A prefetch hint for a buffer
    // /// </summary>
    // /// <param name="Buffer">The function to be prefetched.</param>
    // /// <param name="Bounds">Bounds to be prefetched.</param>
    // public sealed record Prefetch(Buffer Buffer, IRArray<Range> Bounds) : Stmt
    // { }

    /// <summary>
    /// Define certain auxiliary attribute for the body to be a symbolic value.
    /// This provide auxiliary information for IR passes that transforms body.
    ///  In terms of effect, this is equivalent to Block(Evaluate(value), body).
    /// <example>
    ///     Examples of possible usage:
    ///     - Bound of function, variables.
    ///     - Hint which block corresponds to a parallel region.
    /// </example> 
    /// </summary>
    /// <param name="Node">this is attribute about certain node.</param>
    /// <param name="Key">the type key of the attribute.</param>
    /// <param name="Value">The attribute value, value is well defined at current scope..</param>
    /// <param name="Body">The body statement to be executed.</param>
    // public sealed record AttrStmt(object Node, string Key, Expr Value, Stmt Body) : Stmt { }

}
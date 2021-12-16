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
    /// The container of seq statement.
    /// Represent a sequence of statements.
    /// </summary>
    /// <param name="Seq">internal sequence content.</param>
    // public sealed record SeqStmt(IRArray<Stmt> Seq) : Stmt
    // {
    //     public int Count => Seq.Count;
    //     public Stmt this[int index] => Seq[index];
    // }

    /// <summary>
    /// IfThenElse statment.
    /// </summary>
    // public sealed record IfThenElse : Stmt
    // {
    //     /// <summary>
    //     /// The condition.
    //     /// </summary>
    //     public Expr Condition;
    //     /// <summary>
    //     /// The branch to be executed when condition is true.
    //     /// </summary>
    //     public Stmt Then;
    //     /// <summary>
    //     /// The branch to be executed when condition is false, can be null.
    //     /// </summary>
    //     public Stmt Else;

    //     /// <summary>
    //     /// <see cref="IfThenElse"/>
    //     /// </summary>
    //     /// <param name="condition">The expression</param>
    //     /// <param name="then_case">The statement to execute if condition is true.</param>
    //     /// <param name="else_case">The statement to execute if condition is false.</param>
    //     public IfThenElse(Expr condition, Stmt then_case, Stmt else_case)
    //     {
    //         (IsBool() & IsScalar()).Check(condition.CheckedType);
    //         Condition = condition;
    //         Then = then_case;
    //         Else = else_case;
    //     }
    // }
    /// <summary>
    /// Evaluates an expression.
    /// This is mostly used for putting a Call node into Stmt.
    /// If value do not have side-effect, this node can be safely removed.
    /// </summary>
    /// <param name="Value">The expression to be evaluated.</param>
    // public sealed record EvalExpr(Expr Value) : Stmt { }


  

    /// <summary>
    /// A prefetch hint for a buffer
    /// </summary>
    /// <param name="Buffer">The function to be prefetched.</param>
    /// <param name="Bounds">Bounds to be prefetched.</param>
    // public sealed record Prefetch(Buffer Buffer, IRArray<Range> Bounds) : Stmt
    // { }

    /// <summary>
    /// Representing the region of multi-dimensional buffer access.
    /// </summary>
    /// <param name="Buffer">The buffer of the buffer region.</param>
    /// <param name="Region">The region array of the buffer region.</param>
    // public sealed record BufferRegion(Buffer Buffer, IRArray<Range> Region) : Stmt
    // {
    //     /// <summary>
    //     /// Create a BufferRegion which is full region of the given buffer.
    //     /// </summary>
    //     /// <param name="Buf">The buffer to generate full BufferRegion.</param>
    //     /// <returns>The BufferRegion which covers all region of the given buffer</returns>
    //     public static BufferRegion Full(Buffer Buf) => new BufferRegion(Buf, new(Buf.Shape.Select(extent => new Range(0, extent))));

    //     /// <summary>
    //     /// Create a BufferRegion which is a single point of the given buffer.
    //     /// </summary>
    //     /// <param name="Buf">The buffer to generate single point BufferRegion.</param>
    //     /// <param name="Indices">The access point indices of the buffer</param>
    //     /// <returns>The BufferRegion which is the single point of the given buffer.</returns>
    //     public static BufferRegion FromPoint(Buffer Buf, IRArray<Expr> Indices) => new BufferRegion(Buf, new(Indices.Select(index => new Range(index, 1))));
    // }

    /// <summary>
    /// Match introduces a constraint that the source buffer region can be remapped to the data
    /// layout specified by the buffer field. The constraint can be checked in later part of lowering (or
    /// optionally during runtime).
    ///
    /// MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
    /// low-level hardware primitives in the IR and defer the check after the sequence of
    /// transformations.
    /// </summary> 
    /// <param name="Buffer">The target buffer.</param>
    /// <param name="Source">The source buffer region.</param>
    // public sealed record MatchBufferRegion(Buffer Buffer, BufferRegion Source) : Stmt
    // { }

    /// <summary>
    /// A block is a basic schedule unit in TIR.
    /// <remarks>
    /// Block's body is parameterized by iter vars.
    /// </remarks>
    /// <code>
    ///   with T.block(name):
    ///   v0 = T.axis.S(domain, value0)
    ///   v1 = T.axis.R(domain, value1)
    ///   ...
    ///   T.reads([buffer0[start:end, ...], ...])
    ///   T.writes([buffer1[start:end, ...], ...])
    ///   T.where(predicate)
    ///   buffer2 = T.alloc_buffer(shape, dtype)
    ///   buffer3 = T.match_buffer(source_buffer[start:end, ...])
    ///   T.attr({attr_key: attr_value, ...})
    ///   with T.init():
    ///      init body
    ///    body
    /// </code>
    /// </summary>
    /// <param name="Iter_Vars"> The variables of the block.</param>
    /// <param name="Reads"> The read buffer regions of the block.</param>
    /// <param name="Writes"> The write buffer regions of the block.</param>
    /// <param name="Name_Hint"> The name_hint of the block.</param>
    /// <param name="Body"> The body of the block.</param>
    /// <param name="Init"> The init statement is executed during the first iteration of reduction loops in a
    ///  reduction block. The optional init field allows us to represent initialization and
    ///  reduction update in a single block and transform them collectively.
    ///  We also provide primitives to decompose the init into a separate block during scheduling.
    ///  Init field is `NullOpt` if there is no reduction iter_vars </param>
    /// <param name="Alloc_Buffers"> The buffer allocated in the block. </param>
    /// <param name="Match_Buffers"> The match buffer regions. </param>
    /// <param name="Annotations">The annotation of the block. </param>
    // public sealed record Block(IRArray<IterVar> Iter_Vars, IRArray<BufferRegion> Reads, IRArray<BufferRegion> Writes, string Name_Hint, Stmt Body, Stmt? Init, IRArray<Buffer> Alloc_Buffers, IRArray<MatchBufferRegion> Match_Buffers, Dictionary<string, object> Annotations) : Stmt { }

    /// <summary>
    /// A block realization node represents execution of the block at the binding values.
    /// </summary>
    /// <param name="IterValues">The corresponding values of the iter vars.</param>
    /// <param name="Predicate">The predicate of the block realization, the block will only be executed when the
    /// predicate is true.</param>
    /// <param name="Block">The block to be realized.</param>
    // public sealed record BlockRealize(IRArray<Expr> IterValues, Expr Predicate, Block Block) : Stmt
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
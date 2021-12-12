using Nncase.IR;
using System.Collections.Generic;

namespace Nncase.TIR
{

    /// <summary>
    /// Primitive functions that contains TIR statements.
    /// The PrimFunc provides low-level code representation does not
    /// automatically manage
    /// </summary>
    /// <param name="Params">Function parameters.</param>
    /// <param name="Body">The body of the function.</param>
    /// <param name="BufferMap">
    /// Maps some parameters to specific Buffer data structures.
    ///
    ///  buffer_map provides a way to express data structure's field and shape
    ///  constraints. The provided information is used in the program analysis
    ///  and the code generation.
    ///
    ///  - It defines the vars in the Buffer (m, n) in the cases below when
    ///    they appears in the buffer_map for the first time.
    ///  - When a var appears multiple times, they translate into runtime
    ///    assertion to check the field constraint.
    /// <code>
    ///   # The corresponding fields of f are as follows
    ///   #
    ///   # - f.params = [a, b]
    ///   # - f.buffer_map = {a: A, b: B}
    ///   # - A = decl_buffer(shape=[m, n])
    ///   # - B = decl_buffer(shape=[m, n])
    ///   def f(a, b):
    ///       m, n = var(), var()
    ///       A = bind_buffer(a, shape=[m, n])
    ///       B = bind_buffer(b, shape=[m, n])
    ///       # body
    /// </code>
    ///  buffer_map is a sugar to express:
    ///  - Parameter unpacking: e.g. I can load a.shape[0] to get value of m
    ///  - Constraint checking: a.shape[0] must equal b.shape[0] because they
    ///    both corresponds to m.
    ///
    ///  While we could have express parameter unpacking and constraint using
    ///  normal statements, making buffer_map as first class citizen of PrimFunc
    ///  will make program analysis much easier.
    /// </param>
    public sealed record TFunc(IRArray<Var> Params, Stmt Body, Dictionary<Var, TBuffer> BufferMap) : Expr
    {
        public TFunc(IRArray<Var> Params, Stmt Body) : this(Params, Body, new()) { }
    }

    /// <summary>
    /// Describes one parameter that should be linked into the generated module.
    ///
    /// When parameters are to be linked in with generated code (i.e. on target_host-compatible
    /// backends), Relay attaches instances of this object to a global TIR function. Code-generators
    /// use the information contained in this node to include the parameter data in the generated
    /// module.
    /// </summary>
    /// <param name="Id">Unique numeric identifier used by runtimes to lookup this parameter.</param>
    /// <param name="Param">Parameter data which should get linked into the final module.</param>
    public sealed record LinkedParam(int Id, object Param)
    {
    }

}
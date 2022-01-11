namespace Nncase.TIR
{
    public enum ForMode
    {

        /// <summary>
        /// default semantics -- serial execution. 
        /// </summary>
        Serial,
        /// <summary>
        /// Parallel execution on CPU. 
        /// </summary>
        Parallel,

        /// <summary>
        /// Vector SIMD loop.
        ///  The loop body will be vectorized.
        /// </summary>
        Vectorized,

        /// <summary>
        /// The loop body must be unrolled. 
        /// </summary>
        Unrolled,

        /// <summary>
        /// The loop variable is bound to a thread in
        /// an environment. In the final stage of lowering,
        /// the loop is simply removed and the loop variable is
        /// mapped to the corresponding context thread.
        /// </summary>
        ThreadBinding
    }

}
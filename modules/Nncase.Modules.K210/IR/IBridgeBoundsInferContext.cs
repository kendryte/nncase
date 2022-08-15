namespace Nncase.IR;



public interface IBridgeBoundsInferContext
{

    /// <summary>
    /// get the expr
    /// </summary>
    /// <param name="op"></param>
    /// <param name="parameter"></param>
    /// <returns></returns>
    Expr GetArgument(Op op, ParameterInfo parameter);

    /// <summary>
    /// Get argument expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument expression.</returns>
    Tensor GetArgumentTensor(Op op, ParameterInfo parameter);

    /// <summary>
    /// Get argument type.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument type.</returns>
    IR.Shape GetArgumentShape(Op op, ParameterInfo parameter);

    /// <summary>
    /// get argument tile step.
    /// </summary>
    /// <param name="op"></param>
    /// <param name="parameter"></param>
    /// <returns></returns>
    ReadOnlySpan<IR.Segment> GetArgumentTileStep(Op op, ParameterInfo parameter);

    /// <summary>
    /// Add the Inferenced bounds infomation into the Context.
    /// </summary>
    /// <param name="op"></param>
    /// <param name="parameter"> parameter. </param>
    /// <param name="segments"> segment. </param>
    void SetArgumentBounds(Op op, ParameterInfo parameter, IRArray<TIR.Range> segments, int cache_level);

    /// <summary>
    /// get current call tile step.
    /// </summary>
    /// <param name="tile_step"></param>/
    void SetTileStep(ReadOnlySpan<IR.Segment> tile_step);

    /// <summary>
    /// Get the current call shape.
    /// </summary>
    /// <returns></returns>
    IR.Shape CurrentCallShape { get; }

    /// <summary>
    /// Get the env
    /// todo 后续这里可能是从一个运行的scope中获取到. 
    /// </summary>
    //public Transform.Rules.K210.GNNEEnv Env { get; }


    public int GetGreatestCommonDivisor(int a, int b)
    {
        int Remainder;
        while (b != 0)
        {
            Remainder = a % b;
            a = b;
            b = Remainder;
        }
        return a;
    }

    public int GetMinimumCommonMultiple(int a, int b)
    {
        return (a * b) / GetGreatestCommonDivisor(a, b);
    }

    public bool IsFullTileStep(Op target, ParameterInfo parameter, int index = -1)
      => index switch
      {
          -1 => GetArgumentShape(target, parameter).Zip(GetArgumentTileStep(target, parameter).ToArray()).All(
            t => t.Item1.FixedValue == t.Item2.Stop &&
                t.Item2.Start == 1 &&
                t.Item2.Step == 1
          ),
          _ => GetArgumentTileStep(target, parameter)[index] is
          {
              Start: 0,
              Step: 1,
              Stop: var stop
          } && stop == GetArgumentShape(target, parameter)[index]
      };
}
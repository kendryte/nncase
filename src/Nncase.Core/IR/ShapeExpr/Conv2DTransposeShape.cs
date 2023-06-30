namespace Nncase.IR.ShapeExpr;

public class Conv2DTransposeShape : ShapeExprOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Conv2DTransposeShape), 0, "input");

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(Conv2DTransposeShape), 1, "weights");

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(Conv2DTransposeShape), 2, "stride");

    /// <summary>
    /// Gets Dilation.
    /// </summary>
    public static readonly ParameterInfo Dilation = new(typeof(Conv2DTransposeShape), 3, "dilation");

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(Conv2DTransposeShape), 4, "padding");

    /// <summary>
    /// Gets Output Padding.
    /// </summary>
    public static readonly ParameterInfo OutputPadding = new(typeof(Conv2DTransposeShape), 5, "output_padding");

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public static readonly ParameterInfo Groups = new(typeof(Conv2DTransposeShape), 6, "groups");
}

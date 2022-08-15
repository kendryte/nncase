using System.Numerics.Tensors;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.K210;
using Nncase.TIR;
using OrtKISharp;
using Nncase.Evaluator;
using Nncase.IR.Math;
using Range = Nncase.TIR.Range;

namespace Nncase.Evaluator.K210;

public class KPUConv2DEvaluator:IEvaluator<KPUConv2D>, ITypeInferencer<KPUConv2D>, IBoundInferencer<KPUConv2D>
{
    public IValue Visit(IEvaluateContext context, KPUConv2D target)
    {
        var input = context.GetOrtArgumentValue(target, KPUConv2D.Input);
        var weights = context.GetOrtArgumentValue(target, KPUConv2D.Weights);
        var BatchNorms = context.GetOrtArgumentValue(target, KPUConv2D.BatchNorms).ToValue();
        var activition = context.GetOrtArgumentValue(target, KPUConv2D.OutputQuantParam).ToTensor();
        var stride = new long[] { 1, 1 };
        var pad = Enumerable.Repeat((long)KPUUtility.GetKPUPadding(target.FilterType), 4).ToArray();
        var dilation = new long[] { 1, 1 };
        var clamp = new long[] {1, 1};
        var groups = 1L;
        var kernelShape = weights.Shape;
        var result = OrtKI.Conv(
            input.ToType(OrtDataType.Float), weights.ToType(OrtDataType.Float), EvaluatorUtil.DefaultBias(BatchNorms, weights.Shape[0]).ToType(OrtDataType.Float),
            "NOTSET", dilation.ToArray(),
            groups, new long[] { weights.Shape[2], weights.Shape[3] }, EvaluatorUtil.ToOnnxPadFormat(pad), stride.ToArray());
        if (BatchNorms != Value.None)
        {
            result = result + BatchNorms.AsTensor().ToOrtTensor();
        }

        return result.ToValue();
    }

    public IRType Visit(ITypeInferenceContext context, KPUConv2D target)
    {
        var input = context.GetArgument(target, KPUConv2D.Input).CheckedDataType;
        var weights = context.GetArgument(target, KPUConv2D.Weights).CheckedDataType;
        var stride = new[] { 1, 1 };
        var pad = Tensor.FromScalar(KPUUtility.GetKPUPadding(target.FilterType), new[] { 2, 2 });
        var dilation = new[] { 1, 1 };
        var groups = 1;
        return TypeInference.Conv2DType((TensorType) input, (TensorType)weights, stride, pad, dilation, groups);
    }

    public void Visit(IBridgeBoundsInferContext context, KPUConv2D target, IRArray<Range> output_bounds)
    {
        var output_shape = context.CurrentCallShape.ToValueArray();
        var input_shape = context.GetArgumentShape(target, KPUConv2D.Input).ToValueArray();
        var weights_shape = context.GetArgumentShape(target, KPUConv2D.Weights).ToValueArray();

        var filter_h = weights_shape[2];
        var filter_w = weights_shape[3];
        var padding = Enumerable.Repeat((long)KPUUtility.GetKPUPadding(target.FilterType), 4).ToArray();
        var padding_h = padding[0];
        var padding_w = padding[1];
        var dilation = new long[] { 1, 1 };
        var dilation_h = dilation[0];
        var dilation_w = dilation[1];
        var stride = new long[] { 1, 1 };
        var stride_h = stride[0];
        var stride_w = stride[1];
        var groups = 1L;
        var out_groups = output_shape[1] / groups;
        var in_groups = input_shape[1] / groups;

        (Nncase.TIR.Range[], Nncase.TIR.Range[]) infer_index_range(Expr ob, Expr oc, Expr oh, Expr ow)
        {
            var in_y_origin = (oh * stride_h) - padding_h;
            var in_x_origin = (ow * stride_w) - padding_w;

            var filter_y_start = IR.F.Math.Max(0, IR.F.Tensors.Cast(IR.F.Tensors.Cast(-in_y_origin + dilation_h - 1, DataTypes.Float32) / (float)dilation_h, DataTypes.Int32));
            var filter_y_end = IR.F.Math.Min(filter_h, IR.F.Tensors.Cast(IR.F.Tensors.Cast(input_shape![2] - in_y_origin + dilation_h - 1, DataTypes.Float32) / (float)dilation_h, DataTypes.Int32));
            var filter_x_start = IR.F.Math.Max(0, IR.F.Tensors.Cast(IR.F.Tensors.Cast(-in_x_origin + dilation_w - 1, DataTypes.Float32) / (float)dilation_w, DataTypes.Int32));
            var filter_x_end = IR.F.Math.Min(filter_w, IR.F.Tensors.Cast(IR.F.Tensors.Cast(input_shape[3] - in_x_origin + dilation_w - 1, DataTypes.Float32) / (float)dilation_w, DataTypes.Int32));

            var in_y_start = in_y_origin + dilation_h * filter_y_start;
            var in_y_end = in_y_origin + dilation_h * filter_y_end;
            var in_x_start = in_x_origin + dilation_w * filter_x_start;
            var in_x_end = in_x_origin + dilation_w * filter_x_end;
            var in_c_start = (oc / out_groups) * in_groups;
            var in_c_end = in_c_start + in_groups;
            // NOTE 这里不能用这个, 因为我需要负的index来给后续做padding处理
            // var input_range = new Nncase.TIR.Range[] { new(ob, ob + 1, 1),
            //                                            new(in_c_start, in_c_end, 1),
            //                                            new(in_y_start, in_y_end, dilation_h),
            //                                            new(in_x_start, in_x_end, dilation_w) };
            // var w_range = new Nncase.TIR.Range[] {     new(oc, oc + 1, 1),
            //                                            new(0, in_groups, 1),
            //                                            new(filter_y_start,filter_y_end,1),
            //                                            new(filter_x_start,filter_x_end,1) };
            var input_range = new Nncase.TIR.Range[] { new(ob, ob+1, 1),
                                                       new(in_c_start, in_c_end, 1),
                                                       new(in_y_origin, in_y_origin + dilation_h*filter_h, dilation_h),
                                                       new(in_x_origin, in_x_origin + dilation_w*filter_w, dilation_w) };
            var w_range = new Nncase.TIR.Range[] { new(oc, oc+1, 1),
                                                  new(0, in_groups, 1),
                                                  new(0, filter_h,1),
                                                  new(0, filter_w,1) };
            return (input_range, w_range);
        }

        var (input_start, w_start) = infer_index_range(output_bounds[0].Start, output_bounds[1].Start, output_bounds[2].Start, output_bounds[3].Start);
        var (input_end, w_end) = infer_index_range(output_bounds[0].Stop - 1, output_bounds[1].Stop - 1, output_bounds[2].Stop - 1, output_bounds[3].Stop - 1);

        // calc input range
        var input_bounds = new Nncase.TIR.Range[] { new(input_start[0].Start, input_end[0].Stop, 1),
                                                    new(input_start[1].Start, input_end[1].Stop, 1),
                                                    new(input_start[2].Start, input_end[2].Stop, dilation_h),
                                                    new(input_start[3].Start, input_end[3].Stop, dilation_w)};

        context.SetArgumentBounds(target, KPUConv2D.Input, input_bounds, 1);

        // calc weights range
        var weight_bounds = new Nncase.TIR.Range[] { new(w_start[0].Start, w_end[0].Stop, 1),
                                                      new(w_start[1].Start, w_end[1].Stop, 1),
                                                      new(w_start[2].Start, w_end[2].Stop, 1),
                                                      new(w_start[3].Start, w_end[3].Stop, 1) };
        context.SetArgumentBounds(target, KPUConv2D.Weights, weight_bounds, 1);

        // calc psum
        context.SetArgumentBounds(target, KPUConv2D.BatchNorms, output_bounds, 1);

        // calc act
        context.SetArgumentBounds(target, KPUConv2D.OutputQuantParam, new[] { output_bounds[1], new(0, 5, 1) }, 1);

    }

    public void VisitTileStep(IBridgeBoundsInferContext ctx, KPUConv2D target)
    {
        if ((ctx.GetArgument(target, KPUConv2D.BatchNorms) is var BatchNorms &&
             (BatchNorms is None ||
              ctx.IsFullTileStep(target, KPUConv2D.BatchNorms))) &&
            ctx.IsFullTileStep(target, KPUConv2D.OutputQuantParam) &&
            ctx.IsFullTileStep(target, KPUConv2D.Weights))
        {
            if (ctx.IsFullTileStep(target, KPUConv2D.Input))
            {
                var out_shape = ctx.CurrentCallShape.ToValueArray();
                ctx.SetTileStep(new IR.Segment[] {
                    new (1, 1, 1 ), // batch 限制为1.
                    //new (1,System.Math.Min(out_shape[1], ctx.Env.pu_width * ctx.Env.tcu_act_num),1 ), // out channel 限制
                    new(out_shape[2], out_shape[2], out_shape[2]),
                    new(out_shape[3], out_shape[3], out_shape[3])
                });
            }
            else
            {
                var in_shape = ctx.GetArgumentShape(target, KPUConv2D.Input).ToValueArray();
                var in_steps = ctx.GetArgumentTileStep(target, KPUConv2D.Input);
                if (
                    in_steps[1].Start == 1 && in_steps[1].Stop == in_shape[1] && in_steps[1].Step == 1 &&
                    in_steps[2].Start == in_shape[2] && in_steps[2].Stop == in_shape[2] && in_steps[2].Step == in_shape[2] &&
                    in_steps[3].Start == in_shape[3] && in_steps[3].Stop == in_shape[3] && in_steps[3].Step == in_shape[3])
                {
                    var out_shape = ctx.CurrentCallShape.ToValueArray();
                    ctx.SetTileStep(new IR.Segment[] {
                        in_steps[0],
                        new(1,out_shape[1],1),
                        new(out_shape[2], out_shape[2], out_shape[2]),
                        new(out_shape[3], out_shape[3], out_shape[3])
                    });
                }
            }
        }
        else
            throw new NotSupportedException();
    }
}
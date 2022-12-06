using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;


internal interface IForwardable
{
    Expr Forward(params Expr[] inputs);
}

internal sealed record ForwardFusion(Func<Expr[], Fusion> Creator) : IForwardable
{
    public Expr Forward(params Expr[] inputs)
    {
        var fusion = Creator(inputs);
        return new Call(fusion, inputs);
    }

    public static ForwardFusion binary(BinaryOp binaryOp)
    {
        var creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 2)
                throw new NotSupportedException();
            var a = inputs[0];
            var b = inputs[1];
            if (a.CheckedType is null)
                CompilerServices.InferenceType(a);
            if (b.CheckedType is null)
                CompilerServices.InferenceType(b);
            var in_a = new Var(a.CheckedType!);
            var in_b = new Var(b.CheckedType!);
            return new Fusion("BinaryFusion", "stackvm", IR.F.Math.Binary(BinaryOp.Add, in_a, in_b), new[] { in_a, in_b });
        };
        return new(creator);
    }

    public static ForwardFusion conv3x3(int in_planes, int out_planes, int stride = 1, int groups = 1, int dilation = 1)
    {
        var creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
                throw new NotSupportedException();
            var input = inputs[0];
            if (input.CheckedType is null)
                CompilerServices.InferenceType(input);
            var v_input = new Var(input.CheckedType!);
            var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes, v_input.CheckedShape[1].FixedValue, 3, 3 }).Evaluate().AsTensor();
            var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes }).Evaluate().AsTensor();

            return new Fusion("Conv3x3Fusion", "stackvm", IR.F.NN.Conv2D(v_input, weights, bias, new[] { stride, stride }, new[,] { { dilation, dilation }, { dilation, dilation } }, new[] { dilation, dilation }, PadMode.Constant, groups), new[] { v_input });
        };
        return new(creator);
    }


    public static ForwardFusion conv1x1(int in_planes, int out_planes, int stride = 1)
    {
        var creator = (Expr[] inputs) =>
          {
              if (inputs.Length != 1)
                  throw new NotSupportedException();
              var input = inputs[0];
              if (input.CheckedType is null)
                  CompilerServices.InferenceType(input);
              var v_input = new Var(input.CheckedType!);
              var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes, v_input.CheckedShape[1].FixedValue, 1, 1 }).Evaluate().AsTensor();
              var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes }).Evaluate().AsTensor();
              return new Fusion("Conv1x1Fusion", "stackvm", IR.F.NN.Conv2D(v_input, weights, bias, new[] { stride, stride }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1), new[] { v_input });
          };
        return new(creator);
    }
}


internal sealed record ForwardSequential(IForwardable[] array) : IForwardable
{
    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
            throw new NotSupportedException();
        var x = inputs[0];
        Expr last = x;
        foreach (var f in array)
        {
            last = f.Forward(last);
        }
        return last;
    }
}


public class ExpansionAttribute : Attribute
{
    private int _expansion;

    public ExpansionAttribute(int expansion)
    {
        _expansion = expansion;
    }

    public int Expansion => _expansion;
}

[Expansion(1)]
internal sealed class BasicBlock : IForwardable
{
    private ForwardFusion conv1;
    private ForwardFusion conv2;
    private ForwardFusion? downsample;
    private ForwardFusion binary_add;
    private int stride;

    public BasicBlock(int inplanes, int planes, int stride = 1, ForwardFusion? downsample = null, int groups = 1, int base_width = 64, int dilation = 1)
    {
        if (groups != 1 || base_width != 64)
            throw new System.NotSupportedException();
        // raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if (dilation > 1)
            // raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            throw new System.NotSupportedException();
        // # Both this.conv1 and this.downsample layers downsample the input when stride != 1
        conv1 = ForwardFusion.conv3x3(inplanes, planes, stride);
        conv2 = ForwardFusion.conv3x3(planes, planes);
        this.downsample = downsample;
        this.stride = stride;
        this.binary_add = ForwardFusion.binary(BinaryOp.Add);
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
            throw new NotSupportedException();
        var x = inputs[0];
        var identity = x;
        var y = conv1.Forward(x);
        y = conv2.Forward(y);
        if (downsample is not null)
            identity = downsample.Forward(x);

        y = binary_add.Forward(y, identity);
        return y;
    }
}

[Expansion(4)]
internal sealed class Bottleneck : IForwardable
{

    // # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(this.conv2)
    // # while original implementation places the stride at the first 1x1 convolution(this.conv1)
    // # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    // # This variant is also known as ResNet V1.5 and improves accuracy according to
    // # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    private ForwardFusion conv1;
    private ForwardFusion conv2;
    private ForwardFusion conv3;
    private ForwardFusion? downsample;
    private ForwardFusion binary_add;
    private int stride;

    public Bottleneck(
        int inplanes,
        int planes,
        int stride = 1,
        ForwardFusion? downsample = null,
        int groups = 1,
        int base_width = 64,
        int dilation = 1
    )
    {
        var width = (int)(planes * (base_width / 64.0)) * groups;
        //  Both this.conv2 and this.downsample layers downsample the input when stride != 1
        this.conv1 = ForwardFusion.conv1x1(inplanes, width);
        this.conv2 = ForwardFusion.conv3x3(width, width, stride, groups, dilation);
        var expansion = (ExpansionAttribute)Attribute.GetCustomAttribute(typeof(Bottleneck), typeof(ExpansionAttribute))!;
        this.conv3 = ForwardFusion.conv1x1(width, planes * expansion.Expansion);
        this.downsample = downsample;
        this.stride = stride;
        this.binary_add = ForwardFusion.binary(BinaryOp.Add);
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
            throw new NotSupportedException();
        var x = inputs[0];
        var identity = x;
        var y = this.conv1.Forward(x);
        y = this.conv2.Forward(y);
        y = this.conv3.Forward(y);

        if (this.downsample is not null)
            identity = this.downsample.Forward(x);

        y = binary_add.Forward(y, identity);
        return y;
    }

}

internal sealed class ResNet
{
    int inplanes;
    int dilation;
    int groups;
    int base_width;
    ForwardFusion conv1;
    ForwardFusion maxpool;
    IForwardable layer1, layer2, layer3, layer4;// avgpool, fc;

    public ResNet(
        Type block,
        int[] layers,
        int num_classes = 1000,
        bool zero_init_residual = false,
        int groups = 1,
        int width_per_group = 64,
        bool[]? replace_stride_with_dilation = null
    )
    {
        this.inplanes = 64;
        this.dilation = 1;
        if (replace_stride_with_dilation is null)
        {
            // # each element in the tuple indicates if we should replace
            // # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = new[] { false, false, false };
        }
        this.groups = groups;
        this.base_width = width_per_group;
        var conv1_creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
                throw new NotSupportedException();
            var input = inputs[0];
            if (input.CheckedType is null)
                CompilerServices.InferenceType(input);

            var v_input = new Var(input.CheckedType!);
            var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, v_input.CheckedShape[1].FixedValue, 7, 7 }).Evaluate().AsTensor();
            var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor();
            return new Fusion("Conv3x3Fusion", "stackvm", IR.F.NN.Conv2D(v_input, weights, bias, new[] { 2, 2 }, new[,] { { 3, 3 }, { 3, 3 } }, new[] { 1, 1 }, PadMode.Constant, groups), new[] { v_input });
        };
        this.conv1 = new(conv1_creator);

        var maxpool_creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
                throw new NotSupportedException();
            var input = inputs[0];
            if (input.CheckedType is null)
                CompilerServices.InferenceType(input);

            var v_input = new Var(input.CheckedType!);

            return new Fusion("ReduceWindowFusion", "stackvm", IR.F.NN.ReduceWindow2D(ReduceOp.Max, v_input, 0.0f, new[] { 3, 3 }, new[] { 2, 2 }, new[,] { { 1, 1 }, { 1, 1 } }, new[] { 1, 1 }, false, false), new[] { v_input });
        };
        this.maxpool = new(maxpool_creator);
        this.layer1 = this._make_layer(block, 64, layers[0]);
        this.layer2 = this._make_layer(block, 128, layers[1], stride: 2, dilate: replace_stride_with_dilation[0]);
        this.layer3 = this._make_layer(block, 256, layers[2], stride: 2, dilate: replace_stride_with_dilation[1]);
        this.layer4 = this._make_layer(block, 512, layers[3], stride: 2, dilate: replace_stride_with_dilation[2]);
        // this.avgpool = nn.AdaptiveAvgPool2d((1, 1));
        // this.fc = nn.Linear(512 * block.expansion, num_classes);
    }

    private IForwardable _make_layer(Type block, int planes, int blocks, int stride = 1, bool dilate = false)
    {

        IForwardable? downsample = null;
        var previous_dilation = this.dilation;

        if (dilate)
        {
            this.dilation *= stride;
            stride = 1;
        }
        // block.GetField("expansion").GetValue()
        var expansion = ((ExpansionAttribute)Attribute.GetCustomAttribute(block, typeof(ExpansionAttribute))!).Expansion;
        if (stride != 1 || this.inplanes != planes * expansion)
        {
            downsample = ForwardFusion.conv1x1(this.inplanes, planes * expansion, stride);
        }

        var layers = new List<IForwardable>();

        layers.Add(
          (IForwardable)Activator.CreateInstance(block,
            new object?[] { this.inplanes, planes, stride, downsample, this.groups, this.base_width, previous_dilation })!
        );
        this.inplanes = planes * expansion;
        for (int i = 1; i < blocks; i++)
        {
            layers.Add(
              (IForwardable)Activator.CreateInstance(block,
            new object?[] { this.inplanes, planes, 1, null, this.groups, this.base_width, this.dilation })!
            );
        }
        return new ForwardSequential(layers.ToArray());
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
            throw new NotSupportedException();
        var x = inputs[0];
        x = this.conv1.Forward(x);
        x = this.maxpool.Forward(x);
        x = this.layer1.Forward(x);
        x = this.layer2.Forward(x);
        x = this.layer3.Forward(x);
        x = this.layer4.Forward(x);
        return x;
    }

}
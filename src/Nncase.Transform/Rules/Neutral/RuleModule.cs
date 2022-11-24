using Autofac;
namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Evaluator module.
/// </summary>
internal class NeutralModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<AddRangeOfAndMarkerToBinary>().AsImplementedInterfaces();
        builder.RegisterType<AddRangeOfAndMarkerToConv2D>().AsImplementedInterfaces();
        builder.RegisterType<AddRangeOfAndMarkerToConv2DTranspose>().AsImplementedInterfaces();
        builder.RegisterType<AddRangeOfAndMarkerToMatMul>().AsImplementedInterfaces();
        builder.RegisterType<AddRangeOfAndMarkerToRedeceWindow2D>().AsImplementedInterfaces();
        builder.RegisterType<AddRangeOfAndMarkerToUnary>().AsImplementedInterfaces();
        builder.RegisterType<AddToConv2D>().AsImplementedInterfaces();
        builder.RegisterType<CombinePadUnary>().AsImplementedInterfaces();
        builder.RegisterType<CombineReshapeUnary>().AsImplementedInterfaces();
        builder.RegisterType<CombineSliceUnary>().AsImplementedInterfaces();
        builder.RegisterType<CombineTranposeUnary>().AsImplementedInterfaces();
        builder.RegisterType<CombineTransposeBinary>().AsImplementedInterfaces();
        builder.RegisterType<CombineTransposeConcat>().AsImplementedInterfaces();
        builder.RegisterType<CombineTransposePad>().AsImplementedInterfaces();
        builder.RegisterType<CombineTransposeReduce>().AsImplementedInterfaces();
        builder.RegisterType<CombineTransposeUnary>().AsImplementedInterfaces();
        builder.RegisterType<CommutateMul>().AsImplementedInterfaces();
        builder.RegisterType<FoldConstCall>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopBinary>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopCast>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopClamp>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopPad>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopReshape>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopSlice>().AsImplementedInterfaces();
        builder.RegisterType<FoldNopTranspose>().AsImplementedInterfaces();
        builder.RegisterType<FoldShapeOf>().AsImplementedInterfaces();
        builder.RegisterType<FoldTwoCasts>().AsImplementedInterfaces();
        builder.RegisterType<FoldTwoPads>().AsImplementedInterfaces();
        builder.RegisterType<FoldTwoReshapes>().AsImplementedInterfaces();
        builder.RegisterType<FoldTwoSlices>().AsImplementedInterfaces();
        builder.RegisterType<FoldTwoTransposes>().AsImplementedInterfaces();
        builder.RegisterType<FusePadConv2d>().AsImplementedInterfaces();
        builder.RegisterType<IntegralPromotion>().AsImplementedInterfaces();
        builder.RegisterType<MatMulToConv2D>().AsImplementedInterfaces();
        builder.RegisterType<ReassociateDiv>().AsImplementedInterfaces();
        builder.RegisterType<ReassociateMul>().AsImplementedInterfaces();
        builder.RegisterType<TransposeToReshape>().AsImplementedInterfaces();
        builder.RegisterType<XDivX>().AsImplementedInterfaces();
        builder.RegisterType<Xmul1>().AsImplementedInterfaces();
    }
}
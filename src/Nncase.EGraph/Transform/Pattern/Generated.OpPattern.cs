using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.Tensors;

namespace Nncase.Transform.Pattern
{
    public abstract record OpPattern : ExprPattern
    {
        public bool MatchLeaf(Op op) => (this, op)switch
        {
        (BinaryPattern binarypattern, Binary binary) => binarypattern.MatchLeaf(binary), (ClampPattern clamppattern, Clamp clamp) => clamppattern.MatchLeaf(clamp), (UnaryPattern unarypattern, Unary unary) => unarypattern.MatchLeaf(unary), (SigmoidPattern sigmoidpattern, Sigmoid sigmoid) => sigmoidpattern.MatchLeaf(sigmoid), (BatchToSpacePattern batchtospacepattern, BatchToSpace batchtospace) => batchtospacepattern.MatchLeaf(batchtospace), (BroadcastPattern broadcastpattern, Broadcast broadcast) => broadcastpattern.MatchLeaf(broadcast), (CastPattern castpattern, Cast cast) => castpattern.MatchLeaf(cast), (ConcatPattern concatpattern, Concat concat) => concatpattern.MatchLeaf(concat), (DeQuantizePattern dequantizepattern, DeQuantize dequantize) => dequantizepattern.MatchLeaf(dequantize), (GatherPattern gatherpattern, Gather gather) => gatherpattern.MatchLeaf(gather), (GatherNDPattern gatherndpattern, GatherND gathernd) => gatherndpattern.MatchLeaf(gathernd), (PadPattern padpattern, Pad pad) => padpattern.MatchLeaf(pad), (QuantizePattern quantizepattern, Quantize quantize) => quantizepattern.MatchLeaf(quantize), (ReducePattern reducepattern, Reduce reduce) => reducepattern.MatchLeaf(reduce), (ReshapePattern reshapepattern, Reshape reshape) => reshapepattern.MatchLeaf(reshape), (SlicePattern slicepattern, Slice slice) => slicepattern.MatchLeaf(slice), (SpaceToBatchPattern spacetobatchpattern, SpaceToBatch spacetobatch) => spacetobatchpattern.MatchLeaf(spacetobatch), (SplitPattern splitpattern, Split split) => splitpattern.MatchLeaf(split), (SqueezePattern squeezepattern, Squeeze squeeze) => squeezepattern.MatchLeaf(squeeze), (TransposePattern transposepattern, Transpose transpose) => transposepattern.MatchLeaf(transpose), (_, _) => false
        }

        ;
    }
}
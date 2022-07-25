using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.K210;

public partial class Emitter
{
    private readonly BinaryWriter _writer;

    public void Add()
    {
        Write((byte)57);
    }

    public void And()
    {
        Write((byte)64);
    }

    public void Br(int target)
    {
        Write((byte)91);
        Write(target);
    }

    public void Break()
    {
        Write((byte)99);
    }

    public void BrFalse(int target)
    {
        Write((byte)93);
        Write(target);
    }

    public void BrTrue(int target)
    {
        Write((byte)92);
        Write(target);
    }

    public void Call(ushort args, int target)
    {
        Write((byte)95);
        Write(args);
        Write(target);
    }

    public void Ceq()
    {
        Write((byte)75);
    }

    public void Cge()
    {
        Write((byte)76);
    }

    public void CgeU()
    {
        Write((byte)77);
    }

    public void Cgt()
    {
        Write((byte)78);
    }

    public void CgtU()
    {
        Write((byte)79);
    }

    public void Cle()
    {
        Write((byte)73);
    }

    public void CleU()
    {
        Write((byte)74);
    }

    public void Clt()
    {
        Write((byte)71);
    }

    public void CltU()
    {
        Write((byte)72);
    }

    public void Cne()
    {
        Write((byte)80);
    }

    public void ConvBR2()
    {
        Write((byte)89);
    }

    public void ConvI()
    {
        Write((byte)84);
    }

    public void ConvI1()
    {
        Write((byte)81);
    }

    public void ConvI2()
    {
        Write((byte)82);
    }

    public void ConvI4()
    {
        Write((byte)83);
    }

    public void ConvR4()
    {
        Write((byte)90);
    }

    public void ConvU()
    {
        Write((byte)88);
    }

    public void ConvU1()
    {
        Write((byte)85);
    }

    public void ConvU2()
    {
        Write((byte)86);
    }

    public void ConvU4()
    {
        Write((byte)87);
    }

    public void Div()
    {
        Write((byte)60);
    }

    public void DivU()
    {
        Write((byte)61);
    }

    public void Dup()
    {
        Write((byte)46);
    }

    public void ECall(ushort args)
    {
        Write((byte)96);
        Write(args);
    }

    public void ExtCall(ushort args)
    {
        Write((byte)97);
        Write(args);
    }

    public void Ldarg(ushort index)
    {
        Write((byte)39);
        Write(index);
    }

    public void Ldarg0()
    {
        Write((byte)40);
    }

    public void Ldarg1()
    {
        Write((byte)41);
    }

    public void Ldarg2()
    {
        Write((byte)42);
    }

    public void Ldarg3()
    {
        Write((byte)43);
    }

    public void Ldarg4()
    {
        Write((byte)44);
    }

    public void Ldarg5()
    {
        Write((byte)45);
    }

    public void LdcI4(int imm)
    {
        Write((byte)2);
        Write(imm);
    }

    public void LdcI4_0()
    {
        Write((byte)3);
    }

    public void LdcI4_1()
    {
        Write((byte)4);
    }

    public void LdcR4(float imm)
    {
        Write((byte)5);
        Write(imm);
    }

    public void LdDataType()
    {
        Write((byte)54);
    }

    public void LdelemBR2()
    {
        Write((byte)31);
    }

    public void LdelemI()
    {
        Write((byte)26);
    }

    public void LdelemI1()
    {
        Write((byte)23);
    }

    public void LdelemI2()
    {
        Write((byte)24);
    }

    public void LdelemI4()
    {
        Write((byte)25);
    }

    public void LdelemR4()
    {
        Write((byte)32);
    }

    public void LdelemU()
    {
        Write((byte)30);
    }

    public void LdelemU1()
    {
        Write((byte)27);
    }

    public void LdelemU2()
    {
        Write((byte)28);
    }

    public void LdelemU4()
    {
        Write((byte)29);
    }

    public void LdindBR2()
    {
        Write((byte)14);
    }

    public void LdindI()
    {
        Write((byte)9);
    }

    public void LdindI1()
    {
        Write((byte)6);
    }

    public void LdindI2()
    {
        Write((byte)7);
    }

    public void LdindI4()
    {
        Write((byte)8);
    }

    public void LdindR4()
    {
        Write((byte)15);
    }

    public void LdindU()
    {
        Write((byte)13);
    }

    public void LdindU1()
    {
        Write((byte)10);
    }

    public void LdindU2()
    {
        Write((byte)11);
    }

    public void LdindU4()
    {
        Write((byte)12);
    }

    public void Ldlocal(ushort index)
    {
        Write((byte)48);
        Write(index);
    }

    public void LdNull()
    {
        Write((byte)1);
    }

    public void LdShape()
    {
        Write((byte)50);
    }

    public void LdStrides()
    {
        Write((byte)51);
    }

    public void LdTensor()
    {
        Write((byte)55);
    }

    public void LdTuple()
    {
        Write((byte)53);
    }

    public void LdTupleElem()
    {
        Write((byte)52);
    }

    public void LeaGP(byte gpid, int offset)
    {
        Write((byte)22);
        Write(gpid);
        Write(offset);
    }

    public void Mul()
    {
        Write((byte)59);
    }

    public void Neg()
    {
        Write((byte)56);
    }

    public void Nop()
    {
        Write((byte)0);
    }

    public void Not()
    {
        Write((byte)67);
    }

    public void Or()
    {
        Write((byte)65);
    }

    public void Pop()
    {
        Write((byte)47);
    }

    public void Rem()
    {
        Write((byte)62);
    }

    public void RemU()
    {
        Write((byte)63);
    }

    public void Ret()
    {
        Write((byte)94);
    }

    public void Shl()
    {
        Write((byte)68);
    }

    public void Shr()
    {
        Write((byte)69);
    }

    public void ShrU()
    {
        Write((byte)70);
    }

    public void StelemBR2()
    {
        Write((byte)37);
    }

    public void StelemI()
    {
        Write((byte)36);
    }

    public void StelemI1()
    {
        Write((byte)33);
    }

    public void StelemI2()
    {
        Write((byte)34);
    }

    public void StelemI4()
    {
        Write((byte)35);
    }

    public void StelemR4()
    {
        Write((byte)38);
    }

    public void StindBR2()
    {
        Write((byte)20);
    }

    public void StindI()
    {
        Write((byte)19);
    }

    public void StindI1()
    {
        Write((byte)16);
    }

    public void StindI2()
    {
        Write((byte)17);
    }

    public void StindI4()
    {
        Write((byte)18);
    }

    public void StindR4()
    {
        Write((byte)21);
    }

    public void Stlocal(ushort index)
    {
        Write((byte)49);
        Write(index);
    }

    public void Sub()
    {
        Write((byte)58);
    }

    public void Throw()
    {
        Write((byte)98);
    }

    public void Xor()
    {
        Write((byte)66);
    }

    public partial class TensorEmitter
    {
        private readonly Emitter _emitter;

        public void BatchNormalization()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)0);
        }

        public void BatchToSpace()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)1);
        }

        public void Binary(BinaryOp binaryOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)2);
            _emitter.Write((byte)binaryOp);
        }

        public void Broadcast()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)3);
        }

        public void Cast(DataType newType)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)4);
            _emitter.Write(newType);
        }

        public void Celu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)5);
        }

        public void Clamp()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)6);
        }

        public void Compare(CompareOp compareOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)7);
            _emitter.Write((byte)compareOp);
        }

        public void Concat()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)8);
        }

        public void Conv2D(PadMode padMode)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)9);
            _emitter.Write((byte)padMode);
        }

        public void Conv2DTranspose(PadMode padMode)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)10);
            _emitter.Write((byte)padMode);
        }

        public void CumSum()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)11);
        }

        public void Dequantize(DataType targetType)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)12);
            _emitter.Write(targetType);
        }

        public void Elu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)13);
        }

        public void Expand()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)14);
        }

        public void Flatten()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)15);
        }

        public void Gather()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)16);
        }

        public void GatherND()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)17);
        }

        public void GetItem()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)18);
        }

        public void Hardmax()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)19);
        }

        public void HardSigmoid()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)20);
        }

        public void HardSwish()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)21);
        }

        public void InstanceNormalization()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)22);
        }

        public void L2Normalization()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)23);
        }

        public void LeakyRelu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)24);
        }

        public void LogSoftmax()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)25);
        }

        public void LpNormalization()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)26);
        }

        public void LRN()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)27);
        }

        public void LSTM(LSTMDirection direction, LSTMLayout layout, string[] activations)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)28);
            _emitter.Write((int)direction);
            _emitter.Write((int)layout);
            _emitter.Write(activations);
        }

        public void MatMul()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)29);
        }

        public void Normal(DataType type)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)30);
            _emitter.Write(type);
        }

        public void NormalLike(DataType type)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)31);
            _emitter.Write(type);
        }

        public void OneHot(OneHotMode oneHotMode)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)32);
            _emitter.Write((byte)oneHotMode);
        }

        public void Pad(PadMode padMode)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)33);
            _emitter.Write((byte)padMode);
        }

        public void PRelu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)34);
        }

        public void Prod()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)35);
        }

        public void Quantize(DataType targetType)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)36);
            _emitter.Write(targetType);
        }

        public void QuantParamOf(QuantMode quantMode)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)37);
            _emitter.Write((int)quantMode);
        }

        public void Range()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)38);
        }

        public void RangeOf()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)39);
        }

        public void Reduce(ReduceOp reduceOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)40);
            _emitter.Write((byte)reduceOp);
        }

        public void ReduceArg(ReduceArgOp reduceArgOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)41);
            _emitter.Write((byte)reduceArgOp);
        }

        public void ReduceWindow2D(ReduceOp reduceOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)42);
            _emitter.Write((byte)reduceOp);
        }

        public void Relu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)43);
        }

        public void Relu6()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)44);
        }

        public void Require(string message)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)45);
            _emitter.Write(message);
        }

        public void Reshape()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)46);
        }

        public void ResizeImage(ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode, bool isTFResize)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)47);
            _emitter.Write((byte)resizeMode);
            _emitter.Write((int)transformationMode);
            _emitter.Write((int)nearestMode);
            _emitter.Write(isTFResize);
        }

        public void ReverseSequence()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)48);
        }

        public void Select()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)49);
        }

        public void Selu()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)50);
        }

        public void ShapeOf()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)51);
        }

        public void Sigmoid()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)52);
        }

        public void SizeOf()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)53);
        }

        public void Slice()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)54);
        }

        public void Softmax()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)55);
        }

        public void Softplus()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)56);
        }

        public void Softsign()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)57);
        }

        public void SpaceToBatch()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)58);
        }

        public void Split()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)59);
        }

        public void Squeeze()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)60);
        }

        public void Stack()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)61);
        }

        public void Tile()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)62);
        }

        public void Transpose()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)63);
        }

        public void Unary(UnaryOp unaryOp)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)64);
            _emitter.Write((byte)unaryOp);
        }

        public void Uniform(DataType type)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)65);
            _emitter.Write(type);
        }

        public void UniformLike(DataType type)
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)66);
            _emitter.Write(type);
        }

        public void Unsqueeze()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)67);
        }

        public void Where()
        {
            _emitter.Write((byte)100);
            _emitter.Write((ushort)68);
        }
    }
}

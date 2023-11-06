using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitFields;

namespace IsaGen
{
	[System.AttributeUsage(AttributeTargets.Enum, Inherited = false, AllowMultiple = false)]
	public sealed class BitLengthAttribute : Attribute
	{
		public uint BitLength { get; }

		public BitLengthAttribute(uint bitLength)
		{
			BitLength = bitLength;
		}
	}

	[System.AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = false)]
	public sealed class EnumNameAttribute : Attribute
	{
		public string Name { get; }

		public EnumNameAttribute(string name)
		{
			Name = name;
		}
	}

	[BitLength(8)]
	[EnumName("opcode_t")]
	public enum OpCode
	{
		NOP,
		LDNULL,
		LDC_I4,
		LDC_I4_0,
		LDC_I4_1,
		LDC_R4,
		LDIND_I1,
		LDIND_I2,
		LDIND_I4,
		LDIND_I,
		LDIND_U1,
		LDIND_U2,
		LDIND_U4,
		LDIND_U,
		LDIND_BR2,
		LDIND_R4,
		STIND_I1,
		STIND_I2,
		STIND_I4,
		STIND_I,
		STIND_BR2,
		STIND_R4,
		LEA_GP,
		LEA_BUFFER,

		LDELEM_I1,
		LDELEM_I2,
		LDELEM_I4,
		LDELEM_I,
		LDELEM_U1,
		LDELEM_U2,
		LDELEM_U4,
		LDELEM_U,
		LDELEM_BR2,
		LDELEM_R4,
		STELEM_I1,
		STELEM_I2,
		STELEM_I4,
		STELEM_I,
		STELEM_BR2,
		STELEM_R4,

		LDARG,
		LDARG_0,
		LDARG_1,
		LDARG_2,
		LDARG_3,
		LDARG_4,
		LDARG_5,

		DUP,
		POP,

		STSHAPE,
		STPADDINGS,

		NEG,
		ADD,
		SUB,
		MUL,
		DIV,
		DIV_U,
		REM,
		REM_U,
		AND,
		OR,
		XOR,
		NOT,
		SHL,
		SHR,
		SHR_U,

		CLT,
		CLT_U,
		CLE,
		CLE_U,
		CEQ,
		CGE,
		CGE_U,
		CGT,
		CGT_U,
		CNE,

		CONV_I1,
		CONV_I2,
		CONV_I4,
		CONV_I,
		CONV_U1,
		CONV_U2,
		CONV_U4,
		CONV_U,
		CONV_BR2,
		CONV_R4,

		BR,
		BR_TRUE,
		BR_FALSE,
		RET,
		CALL,
		ECALL,
		THROW,
		BREAK,

		TENSOR,
	}

	[BitLength(16)]
	[EnumName("tensor_function_t")]
	public enum TensorFunction
	{
		BATCH_TO_SPACE,
		BINARY,
		BROADCAST,
		CALL,
		COMPARE,
		CLAMP,
		CONV2D,
		CONV2D_TRANSPOSE,
		CONVERT,
		COPY,
		CUMSUM,
		DEQUANTIZE,
		GATHER,
		GATHER_ND,
		HARDMAX,
		LOGISTIC,
		LUT1D,
		MATMUL,
		ONEHOT,
		PAD,
		QUANTIZE,
		RANDOM_NORMAL,
		RANDOM_UNIFORM,
		REDUCE,
		REDUCE_ARG,
		REDUCE_PROD,
		REDUCE_WINDOW2D,
		RESIZE_IMAGE,
		ROI_ALIGN,
		SIGMOID,
		SLICE,
		SOFTMAX,
		SPACE_TO_BATCH,
		TAKE,
		TERNARY,
		TOPK,
		TRANSPOSE,
		TRILU,
		UNARY,
		GRU,
		TFLITE_DETECTION_POSTPROCESS,
		LAYER_NORMALIZATION,
		COMPRESS,
		GATHER_ELEMENTS,
		INSTANCE_NORMALIZATION
	}

	[BitLength(8)]
	[EnumName("datatype_t")]
	[Browsable(false)]
	public enum DataType
	{
	}

	[BitLength(8)]
	[EnumName("onehot_mode_t")]
	[Browsable(false)]
	public enum OneHotMode
	{
	}

	[BitLength(8)]
	[EnumName("pad_mode_t")]
	[Browsable(false)]
	public enum PadMode
	{
	}

	[BitLength(8)]
	[EnumName("memory_location_t")]
	[Browsable(false)]
	public enum MemoryLocation
	{
	}

	[BitLength(8)]
	[EnumName("reduce_op_t")]
	[Browsable(false)]
	public enum ReduceOp
	{
	}

	[BitLength(8)]
	[EnumName("reduce_arg_op_t")]
	[Browsable(false)]
	public enum ReduceArgOp
	{
	}

	[BitLength(8)]
	[EnumName("image_resize_mode_t")]
	[Browsable(false)]
	public enum ImageResizeMode
	{
	}

	[BitLength(8)]
	[EnumName("binary_op_t")]
	[Browsable(false)]
	public enum BinaryOp
	{
	}

	[BitLength(8)]
	[EnumName("unary_op_t")]
	[Browsable(false)]
	public enum UnaryOp
	{
	}

	[BitLength(8)]
	[EnumName("compare_op_t")]
	[Browsable(false)]
	public enum CompareOp
	{
	}

	[BitLength(8)]
	[EnumName("roi_align_mode_t")]
	[Browsable(false)]
	public enum RoiAlignMode
	{
	}

	public abstract class Instruction
	{
		[DisplayName("opcode")]
		[Description("OpCode")]
		public abstract OpCode OpCode { get; }
	}

	[DisplayName("NOP")]
	[Category("Control and Status Instructions")]
	[Description("No operation")]
	public class NopInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.NOP;
	}

	[DisplayName("LDC_I4")]
	[Category("Immediate Instructions")]
	[Description("Load immedidate I4 to stack")]
	public class LdcI4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDC_I4;

		[DisplayName("imm")]
		[Description("Immedidate I4")]
		public int Imm { get; set; }
	}

	[DisplayName("LDNULL")]
	[Category("Immediate Instructions")]
	[Description("Load immedidate nullptr as I to stack")]
	public class LdNullInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDNULL;
	}

	[DisplayName("LDC_I4_0")]
	[Category("Immediate Instructions")]
	[Description("Load immedidate 0 as I4 to stack")]
	public class LdcI4_0Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDC_I4_0;
	}

	[DisplayName("LDC_I4_1")]
	[Category("Immediate Instructions")]
	[Description("Load immedidate 1 as I4 to stack")]
	public class LdcI4_1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDC_I4_1;
	}

	[DisplayName("LDC_R4")]
	[Category("Immediate Instructions")]
	[Description("Load immedidate R4 to stack")]
	public class LdcR4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDC_R4;

		[DisplayName("imm")]
		[Description("Immedidate R4")]
		public float Imm { get; set; }
	}

	[Category("Load Store Instructions")]
	public abstract class LdStindInstruction : Instruction
	{
	}

	[DisplayName("LDIND_I1")]
	[Description("Load indirect I1 to stack")]
	public class LdindI1Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_I1;
	}

	[DisplayName("LDIND_I2")]
	[Description("Load indirect I2 to stack")]
	public class LdindI2Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_I2;
	}

	[DisplayName("LDIND_I4")]
	[Description("Load indirect I4 to stack")]
	public class LdindI4Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_I4;
	}

	[DisplayName("LDIND_I")]
	[Description("Load indirect I to stack")]
	public class LdindIInstruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_I;
	}

	[DisplayName("LDIND_U1")]
	[Description("Load indirect U1 to stack")]
	public class LdindU1Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_U1;
	}

	[DisplayName("LDIND_U2")]
	[Description("Load indirect U2 to stack")]
	public class LdindU2Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_U2;
	}

	[DisplayName("LDIND_U4")]
	[Description("Load indirect U4 to stack")]
	public class LdindU4Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_U4;
	}

	[DisplayName("LDIND_U")]
	[Description("Load indirect U to stack")]
	public class LdindUInstruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_U;
	}

	[DisplayName("LDIND_BR2")]
	[Description("Load indirect BR2 to stack")]
	public class LdindBR2Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_BR2;
	}

	[DisplayName("LDIND_R4")]
	[Description("Load indirect R4 to stack")]
	public class LdindR4Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.LDIND_R4;
	}

	[DisplayName("STIND_I1")]
	[Description("Store indirect I1 from stack")]
	public class StindI1Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_I1;
	}

	[DisplayName("STIND_I2")]
	[Description("Store indirect I2 from stack")]
	public class StindI2Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_I2;
	}

	[DisplayName("STIND_I4")]
	[Description("Store indirect I4 from stack")]
	public class StindI4Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_I4;
	}

	[DisplayName("STIND_I")]
	[Description("Store indirect I from stack")]
	public class StindIInstruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_I;
	}

	[DisplayName("STIND_BR2")]
	[Description("Store indirect BR2 from stack")]
	public class StindBR2Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_BR2;
	}

	[DisplayName("STIND_R4")]
	[Description("Store indirect R4 from stack")]
	public class StindR4Instruction : LdStindInstruction
	{
		public override OpCode OpCode => OpCode.STIND_R4;
	}

	[DisplayName("LEA_GP")]
	[Category("Load Store Instructions")]
	[Description("Load a global pointer with offset to stack")]
	public class LeaGPInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LEA_GP;

		[DisplayName("gpid")]
		[Description("Global pointer id")]
		public byte GpId { get; set; }

		[DisplayName("offset")]
		[Description("Signed immediate offset")]
		public int Offset { get; set; }
	}

	[DisplayName("LEA_BUFFER")]
	[Category("Load Store Instructions")]
	[Description("Load a buffer pointer with offset to stack")]
	public class LeaBufferInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LEA_BUFFER;

		[DisplayName("location")]
		[Description("Location")]
		public MemoryLocation Location { get; set; }

		[DisplayName("subres_id")]
		[Description("SubresourceId")]
		public byte SubresourceId { get; set; }

		[DisplayName("offset")]
		[Description("Unsigned immediate offset")]
		public uint Offset { get; set; }
	}

	[DisplayName("LDELEM_I1")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of I1 to stack")]
	public class LdelemI1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_I1;
	}

	[DisplayName("LDELEM_I2")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of I2 to stack")]
	public class LdelemI2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_I2;
	}

	[DisplayName("LDELEM_I4")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of I4 to stack")]
	public class LdelemI4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_I4;
	}

	[DisplayName("LDELEM_I")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of I to stack")]
	public class LdelemIInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_I;
	}

	[DisplayName("LDELEM_U1")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of U1 to stack")]
	public class LdelemU1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_U1;
	}

	[DisplayName("LDELEM_U2")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of U2 to stack")]
	public class LdelemU2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_U2;
	}

	[DisplayName("LDELEM_U4")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of U4 to stack")]
	public class LdelemU4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_U4;
	}

	[DisplayName("LDELEM_U")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of U to stack")]
	public class LdelemUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_U;
	}

	[DisplayName("LDELEM_BR2")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of BR2 to stack")]
	public class LdelemBR2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_BR2;
	}

	[DisplayName("LDELEM_R4")]
	[Category("Load Store Instructions")]
	[Description("Load an array element of R4 to stack")]
	public class LdelemR4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDELEM_R4;
	}

	[DisplayName("STELEM_I1")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of I1 from stack")]
	public class StelemI1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_I1;
	}

	[DisplayName("STELEM_I2")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of I2 from stack")]
	public class StelemI2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_I2;
	}

	[DisplayName("STELEM_I4")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of I4 from stack")]
	public class StelemI4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_I4;
	}

	[DisplayName("STELEM_I")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of I from stack")]
	public class StelemIInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_I;
	}

	[DisplayName("STELEM_BR2")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of BR2 from stack")]
	public class StelemBR2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_BR2;
	}

	[DisplayName("STELEM_R4")]
	[Category("Load Store Instructions")]
	[Description("Store an array element of R4 from stack")]
	public class StelemR4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.STELEM_R4;
	}

	[DisplayName("LDARG")]
	[Category("Load Store Instructions")]
	[Description("Load an argument to stack")]
	public class LdargInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG;

		[DisplayName("index")]
		[Description("Argument index")]
		public uint Index { get; set; }
	}

	[DisplayName("LDARG_0")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 0 to stack")]
	public class Ldarg0Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_0;
	}

	[DisplayName("LDARG_1")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 1 to stack")]
	public class Ldarg1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_1;
	}

	[DisplayName("LDARG_2")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 2 to stack")]
	public class Ldarg2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_2;
	}

	[DisplayName("LDARG_3")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 1 to stack")]
	public class Ldarg3Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_3;
	}

	[DisplayName("LDARG_4")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 4 to stack")]
	public class Ldarg4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_4;
	}

	[DisplayName("LDARG_5")]
	[Category("Load Store Instructions")]
	[Description("Load an argument with index of 5 to stack")]
	public class Ldarg5Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.LDARG_5;
	}

	[DisplayName("STSHAPE")]
	[Category("Load Store Instructions")]
	[Description("Store a shape from stack")]
	public class StShapeInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.STSHAPE;

		[DisplayName("rshape")]
		[Description("Shape register index")]
		public byte Rshape { get; set; }

		[DisplayName("rank")]
		[Description("Shape's rank")]
		public byte Rank { get; set; }
	}

	[DisplayName("STPADDINGS")]
	[Category("Load Store Instructions")]
	[Description("Store paddings from stack")]
	public class StPaddingsInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.STPADDINGS;

		[DisplayName("rpaddings")]
		[Description("Paddings register index")]
		public byte Rpaddings { get; set; }

		[DisplayName("rank")]
		[Description("Paddings' rank")]
		public byte Rank { get; set; }
	}

	[DisplayName("DUP")]
	[Category("Stack Instructions")]
	[Description("Duplicate the top item of stack")]
	public class DupInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.DUP;
	}

	[DisplayName("POP")]
	[Category("Stack Instructions")]
	[Description("Pop the top item of stack")]
	public class PopInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.POP;
	}

	[DisplayName("NEG")]
	[Category("Computational Instructions")]
	[Description("Negates a value and pushes the result onto the evaluation stack")]
	public class NegInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.NEG;
	}

	[DisplayName("ADD")]
	[Category("Computational Instructions")]
	[Description("Adds two values and pushes the result onto the evaluation stack")]
	public class AddInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.ADD;
	}

	[DisplayName("SUB")]
	[Category("Computational Instructions")]
	[Description("Subtracts one value from another and pushes the result onto the evaluation stack")]
	public class SubInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.SUB;
	}

	[DisplayName("MUL")]
	[Category("Computational Instructions")]
	[Description("Multiplies two values and pushes the result on the evaluation stack")]
	public class MulInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.MUL;
	}

	[DisplayName("DIV")]
	[Category("Computational Instructions")]
	[Description("Divides two values and pushes the result as a floating-point (type F) or quotient (type int32) onto the evaluation stack")]
	public class DivInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.DIV;
	}

	[DisplayName("DIV_U")]
	[Category("Computational Instructions")]
	[Description("Divides two unsigned integer values and pushes the result (int32) onto the evaluation stack")]
	public class DivUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.DIV_U;
	}

	[DisplayName("REM")]
	[Category("Computational Instructions")]
	[Description("Divides two values and pushes the remainder onto the evaluation stack")]
	public class RemInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.REM;
	}

	[DisplayName("REM_U")]
	[Category("Computational Instructions")]
	[Description("Divides two unsigned values and pushes the remainder onto the evaluation stack")]
	public class RemUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.REM_U;
	}

	[DisplayName("AND")]
	[Category("Computational Instructions")]
	[Description("Computes the bitwise AND of two values and pushes the result onto the evaluation stack")]
	public class AndInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.AND;
	}

	[DisplayName("OR")]
	[Category("Computational Instructions")]
	[Description("Compute the bitwise complement of the two integer values on top of the stack and pushes the result onto the evaluation stack")]
	public class OrInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.OR;
	}

	[DisplayName("XOR")]
	[Category("Computational Instructions")]
	[Description("Computes the bitwise XOR of the top two values on the evaluation stack, pushing the result onto the evaluation stack")]
	public class XorInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.XOR;
	}

	[DisplayName("NOT")]
	[Category("Computational Instructions")]
	[Description("Computes the bitwise complement of the integer value on top of the stack and pushes the result onto the evaluation stack as the same type")]
	public class NotInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.NOT;
	}

	[DisplayName("SHL")]
	[Category("Computational Instructions")]
	[Description("Shifts an integer value to the left (in zeroes) by a specified number of bits, pushing the result onto the evaluation stack")]
	public class ShlInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.SHL;
	}

	[DisplayName("SHR")]
	[Category("Computational Instructions")]
	[Description("Shifts an integer value (in sign) to the right by a specified number of bits, pushing the result onto the evaluation stack")]
	public class ShrInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.SHR;
	}

	[DisplayName("SHR_U")]
	[Category("Computational Instructions")]
	[Description("Shifts an unsigned integer value (in zeroes) to the right by a specified number of bits, pushing the result onto the evaluation stack")]
	public class ShrUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.SHR_U;
	}

	[DisplayName("CLT")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If the first value is less than the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CltInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CLT;
	}

	[DisplayName("CLT_U")]
	[Category("Computational Instructions")]
	[Description("Compares the unsigned or unordered values value1 and value2. If value1 is less than value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CltUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CLT_U;
	}

	[DisplayName("CLE")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If the first value is less than or equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CleInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CLE;
	}

	[DisplayName("CLE_U")]
	[Category("Computational Instructions")]
	[Description("Compares the unsigned or unordered values value1 and value2. If value1 is less than or equal to value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CleUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CLE_U;
	}

	[DisplayName("CEQ")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If they are equal, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CeqInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CEQ;
	}

	[DisplayName("CGE")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If the first value is greater than or equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CgeInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CGE;
	}

	[DisplayName("CGE_U")]
	[Category("Computational Instructions")]
	[Description("Compares the unsigned or unordered values value1 and value2. If value1 is greater than or equal to value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CgeUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CGE_U;
	}

	[DisplayName("CGT")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If the first value is greater than the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CgtInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CGT;
	}

	[DisplayName("CGT_U")]
	[Category("Computational Instructions")]
	[Description("Compares the unsigned or unordered values value1 and value2. If value1 is greater than value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CgtUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CGT_U;
	}

	[DisplayName("CNE")]
	[Category("Computational Instructions")]
	[Description("Compares two values. If the first value is not equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack")]
	public class CneInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CNE;
	}

	[DisplayName("CONV_I1")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to int8, and extends it to int32")]
	public class ConvI1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_I1;
	}

	[DisplayName("CONV_I2")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to int16, and extends it to int32")]
	public class ConvI2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_I2;
	}

	[DisplayName("CONV_I4")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to int32, and extends it to int32")]
	public class ConvI4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_I4;
	}

	[DisplayName("CONV_I")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to native int, and extends it to int32")]
	public class ConvIInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_I;
	}

	[DisplayName("CONV_U1")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to unsigned int8, and extends it to int32")]
	public class ConvU1Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_U1;
	}

	[DisplayName("CONV_U2")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to unsigned int16, and extends it to int32")]
	public class ConvU2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_U2;
	}

	[DisplayName("CONV_U4")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to unsigned int32, and extends it to int32")]
	public class ConvU4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_U4;
	}

	[DisplayName("CONV_U")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to unsigned native int, and extends it to int32")]
	public class ConvUInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_U;
	}

	[DisplayName("CONV_BR2")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to bfloat16")]
	public class ConvBR2Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_BR2;
	}

	[DisplayName("CONV_R4")]
	[Category("Conversion Instructions")]
	[Description("Converts the value on top of the evaluation stack to float32")]
	public class ConvR4Instruction : Instruction
	{
		public override OpCode OpCode => OpCode.CONV_R4;
	}

	[DisplayName("BR")]
	[Category("Control and Status Instructions")]
	[Description("Unconditionally transfers control to a target instruction")]
	public class BrInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.BR;

		[DisplayName("target")]
		[Description("Branches to a target instruction at the specified offset")]
		public int Target { get; set; }
	}

	[DisplayName("BR_TRUE")]
	[Category("Control and Status Instructions")]
	[Description("Transfers control to a target instruction if value is true, not null, or non-zero")]
	public class BrTrueInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.BR_TRUE;

		[DisplayName("target")]
		[Description("Branches to a target instruction at the specified offset")]
		public int Target { get; set; }
	}

	[DisplayName("BR_FALSE")]
	[Category("Control and Status Instructions")]
	[Description("Transfers control to a target instruction if value is false, null, or zero")]
	public class BrFalseInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.BR_FALSE;

		[DisplayName("target")]
		[Description("Branches to a target instruction at the specified offset")]
		public int Target { get; set; }
	}

	[DisplayName("RET")]
	[Category("Control and Status Instructions")]
	[Description("Return")]
	public class RetInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.RET;
	}

	[DisplayName("CALL")]
	[Category("Control and Status Instructions")]
	[Description("Call a target method")]
	public class CallInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.CALL;

		[DisplayName("args")]
		[Description("Arguments count")]
		public byte ArgsCount { get; set; }

		[DisplayName("target")]
		[Description("Call a target method at the specified offset")]
		public int Target { get; set; }
	}

	[DisplayName("ECALL")]
	[Category("Control and Status Instructions")]
	[Description("Call a environment method")]
	public class ECallInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.ECALL;

		[DisplayName("args")]
		[Description("Arguments count")]
		public byte ArgsCount { get; set; }
	}

	[DisplayName("THROW")]
	[Category("Control and Status Instructions")]
	[Description("Throw a error code currently on the evaluation stack")]
	public class ThrowInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.THROW;
	}

	[DisplayName("BREAK")]
	[Category("Control and Status Instructions")]
	[Description("Inform the debugger that a break point has been tripped")]
	public class BreakInstruction : Instruction
	{
		public override OpCode OpCode => OpCode.BREAK;
	}

	public static class TensorCalls
	{
		public abstract class TensorInstruction : Instruction
		{
			public sealed override OpCode OpCode => OpCode.TENSOR;

			[DisplayName("funct")]
			[Description("Tensor call function")]
			public abstract TensorFunction Function { get; }
		}

		[DisplayName("TENSOR.BATCH_TO_SPACE")]
		[Category("Tensor Instructions")]
		[Description("BatchToSpace")]
		public class BatchToSpaceInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.BATCH_TO_SPACE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_block")]
			[Description("Block shape register")]
			public byte RshapeBlock { get; set; }

			[DisplayName("rpad_crops")]
			[Description("Crops paddings register")]
			public byte RpadCrops { get; set; }
		}

		[DisplayName("TENSOR.BROADCAST")]
		[Category("Tensor Instructions")]
		[Description("Broadcast")]
		public class BroadcastInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.BROADCAST;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.BINARY")]
		[Category("Tensor Instructions")]
		[Description("Binary")]
		public class BinaryInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.BINARY;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src1")]
			[Description("Source1 shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("rstride_src1")]
			[Description("Source1 stride register")]
			public byte RstrideSrc1 { get; set; }

			[DisplayName("rshape_src2")]
			[Description("Source2 shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("rstride_src2")]
			[Description("Source2 stride register")]
			public byte RstrideSrc2 { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("binary_op")]
			[Description("Binary operator")]
			public BinaryOp BinaryOp { get; set; }

			[DisplayName("fused_clamp_low")]
			[Description("FusedClampLow")]
			public float FusedClampLow { get; set; }

			[DisplayName("fused_clamp_high")]
			[Description("FusedClampHigh")]
			public float FusedClampHigh { get; set; }
		}

		[DisplayName("TENSOR.CALL")]
		[Category("Tensor Instructions")]
		[Description("Call")]
		public class CallInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.CALL;

			[DisplayName("function_id")]
			[Description("Function Id")]
			public uint FunctionId { get; set; }

			[DisplayName("module_id")]
			[Description("Module Id")]
			public ushort ModuleId { get; set; }

			[DisplayName("num_src")]
			[Description("Source count")]
			public byte SrcCount { get; set; }

			[DisplayName("num_dst")]
			[Description("Dest count")]
			public byte DstCount { get; set; }
		}

		[DisplayName("TENSOR.COMPARE")]
		[Category("Tensor Instructions")]
		[Description("Compare")]
		public class CompareInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.COMPARE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src1")]
			[Description("Source1 shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("rstride_src1")]
			[Description("Source1 stride register")]
			public byte RstrideSrc1 { get; set; }

			[DisplayName("rshape_src2")]
			[Description("Source2 shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("rstride_src2")]
			[Description("Source2 stride register")]
			public byte RstrideSrc2 { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("compare_op")]
			[Description("Compare operator")]
			public CompareOp CompareOp { get; set; }
		}
		[DisplayName("TENSOR.CONV2D")]
		[Category("Tensor Instructions")]
		[Description("Conv2D")]
		public class Conv2DInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.CONV2D;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rshape_kernel")]
			[Description("Kernel shape register")]
			public byte RshapeKernel { get; set; }

			[DisplayName("rstride_kernel")]
			[Description("Kernel stride register")]
			public byte RstrideKernel { get; set; }

			[DisplayName("rstride_bias")]
			[Description("Bias stride register")]
			public byte RstrideBias { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("groups")]
			[Description("Groups")]
			public ushort Groups { get; set; }

			[DisplayName("stride_h")]
			[Description("StrideH")]
			public ushort StrideH { get; set; }

			[DisplayName("stride_w")]
			[Description("StrideW")]
			public ushort StrideW { get; set; }

			[DisplayName("dilation_h")]
			[Description("DilationH")]
			public ushort DilationH { get; set; }

			[DisplayName("dilation_w")]
			[Description("DilationW")]
			public ushort DilationW { get; set; }

			[DisplayName("fused_clamp_low")]
			[Description("FusedClampLow")]
			public float FusedClampLow { get; set; }

			[DisplayName("fused_clamp_high")]
			[Description("FusedClampHigh")]
			public float FusedClampHigh { get; set; }
		}

		[DisplayName("TENSOR.COPY")]
		[Category("Tensor Instructions")]
		[Description("Copy")]
		public class CopyInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.COPY;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape")]
			[Description("Shape register")]
			public byte Rshape { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.CONVERT")]
		[Category("Tensor Instructions")]
		[Description("Convert")]
		public class ConvertInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.CONVERT;

			[DisplayName("in_datatype")]
			[Description("Source Datatype")]
			public DataType SrcDataType { get; set; }

			[DisplayName("dst_datatype")]
			[Description("Dest Datatype")]
			public DataType DestDataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source1 shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.CUMSUM")]
		[Category("Tensor Instructions")]
		[Description("CumSum")]
		public class CumSumInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.CUMSUM;

			[DisplayName("datatype")]
			[Description("Input/Output datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public int Axis { get; set; }

			[DisplayName("exclusive")]
			[Description("Exclusive")]
			public bool Exclusive { get; set; }

			[DisplayName("reverse")]
			[Description("Reverse")]
			public bool Reverse { get; set; }
		}

		[DisplayName("TENSOR.DEQUANTIZE")]
		[Category("Tensor Instructions")]
		[Description("Dequantize")]
		public class DequantizeInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.DEQUANTIZE;

			[DisplayName("in_datatype")]
			[Description("Source Datatype")]
			public DataType SrcDataType { get; set; }

			[DisplayName("dst_datatype")]
			[Description("Dest Datatype")]
			public DataType DestDataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.GATHER")]
		[Category("Tensor Instructions")]
		[Description("Gather")]
		public class GatherInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.GATHER;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_indices")]
			[Description("Indices shape register")]
			public byte RshapeIndices { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public byte Axis { get; set; }
		}

		[DisplayName("TENSOR.GATHER_ND")]
		[Category("Tensor Instructions")]
		[Description("GatherND")]
		public class GatherNDInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.GATHER_ND;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_indices")]
			[Description("Indices shape register")]
			public byte RshapeIndices { get; set; }

			[DisplayName("batch_dims")]
			[Description("Batch Dims")]
			public byte Batchdims { get; set; }
		}

		[DisplayName("TENSOR.HARDMAX")]
		[Category("Tensor Instructions")]
		[Description("Hardmax")]
		public class HardmaxInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.HARDMAX;

			[DisplayName("datatype")]
			[Description("Input/Output datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public int Axis { get; set; }
		}

		[DisplayName("TENSOR.LUT1D")]
		[Category("Tensor Instructions")]
		[Description("Lut1D")]
		public class LUT1DInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.LUT1D;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("table_len")]
			[Description("Table length")]
			public ushort TableLength { get; set; }
		}

		[DisplayName("TENSOR.MATMUL")]
		[Category("Tensor Instructions")]
		[Description("Matmul")]
		public class MatmulInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.MATMUL;

			[DisplayName("rshape_src1")]
			[Description("Source1 shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("rstride_src1")]
			[Description("Source1 stride register")]
			public byte RstrideSrc1 { get; set; }

			[DisplayName("rshape_src2")]
			[Description("Source2 shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("rstride_src2")]
			[Description("Source2 stride register")]
			public byte RstrideSrc2 { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("fused_clamp_low")]
			[Description("FusedClampLow")]
			public float FusedClampLow { get; set; }

			[DisplayName("fused_clamp_high")]
			[Description("FusedClampHigh")]
			public float FusedClampHigh { get; set; }
		}

		[DisplayName("TENSOR.ONEHOT")]
		[Category("Tensor Instructions")]
		[Description("OneHot")]
		public class OneHotInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.ONEHOT;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_indices")]
			[Description("Indices shape register")]
			public byte RshapeIndices { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public byte Axis { get; set; }

			[DisplayName("onehot_mode")]
			[Description("OneHot Mode")]
			public OneHotMode OneHotMode { get; set; }
		}

		[DisplayName("TENSOR.PAD")]
		[Category("Tensor Instructions")]
		[Description("Pad")]
		public class PadInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.PAD;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rpaddings")]
			[Description("Paddings register")]
			public byte Rpaddings { get; set; }

			[DisplayName("pad_mode")]
			[Description("Pad mode")]
			public PadMode PadMode { get; set; }
		}

		[DisplayName("TENSOR.QUANTIZE")]
		[Category("Tensor Instructions")]
		[Description("Quantize")]
		public class QuantizeInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.QUANTIZE;

			[DisplayName("in_datatype")]
			[Description("Source Datatype")]
			public DataType SrcDataType { get; set; }

			[DisplayName("dst_datatype")]
			[Description("Dest Datatype")]
			public DataType DestDataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.RANDOM_NORMAL")]
		[Category("Tensor Instructions")]
		[Description("RandomNormal")]
		public class RandomNormalInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.RANDOM_NORMAL;

			[DisplayName("datatype_dest")]
			[Description("Output datatype")]
			public DataType DataTypeDest { get; set; }

			[DisplayName("rshape_dest")]
			[Description("output shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("mean")]
			[Description("Mean")]
			public float Mean { get; set; }

			[DisplayName("std")]
			[Description("Std")]
			public float Std { get; set; }

			[DisplayName("seed")]
			[Description("Seed")]
			public float Seed { get; set; }
		}

		[DisplayName("TENSOR.RANDOM_UNIFORM")]
		[Category("Tensor Instructions")]
		[Description("RandomUniform")]
		public class RandomUniformInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.RANDOM_UNIFORM;

			[DisplayName("datatype_dest")]
			[Description("Output datatype")]
			public DataType DataTypeDest { get; set; }

			[DisplayName("rshape_dest")]
			[Description("output shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("low")]
			[Description("Low")]
			public float Low { get; set; }

			[DisplayName("high")]
			[Description("High")]
			public float High { get; set; }

			[DisplayName("seed")]
			[Description("Seed")]
			public float Seed { get; set; }
		}

		[DisplayName("TENSOR.REDUCE")]
		[Category("Tensor Instructions")]
		[Description("Reduce")]
		public class ReduceInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.REDUCE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("reduce_op")]
			[Description("Reduce operator")]
			public ReduceOp ReduceOp { get; set; }

			[DisplayName("rshape_axis")]
			[Description("Axis shape register")]
			public byte RshapeAxis { get; set; }

			[DisplayName("keep_dims")]
			[Description("Keep dimensions")]
			public bool KeepDims { get; set; }
		}

		[DisplayName("TENSOR.REDUCE_ARG")]
		[Category("Tensor Instructions")]
		[Description("ReduceArg")]
		public class ReduceArgInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.REDUCE_ARG;

			[DisplayName("datatype_src")]
			[Description("Input datatype")]
			public DataType DataTypeSrc { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("datatype_dest")]
			[Description("Output datatype")]
			public DataType DataTypeDest { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("reduce_arg_op")]
			[Description("Reduce arg operator")]
			public ReduceArgOp ReduceArgOp { get; set; }

			[DisplayName("rshape_axis")]
			[Description("Axis shape register")]
			public byte RshapeAxis { get; set; }

			[DisplayName("keep_dims")]
			[Description("Keep dimensions")]
			public bool KeepDims { get; set; }

			[DisplayName("select_last_idx")]
			[Description("select last index")]
			public bool SelectLastIdx { get; set; }
		}

		[DisplayName("TENSOR.REDUCE_PROD")]
		[Category("Tensor Instructions")]
		[Description("ReduceProd")]
		public class ReduceProdInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.REDUCE_PROD;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_axes")]
			[Description("Axes shape register")]
			public byte RshapeAxes { get; set; }

			[DisplayName("keep_dims")]
			[Description("Keep dimensions")]
			public bool KeepDims { get; set; }
		}

		[DisplayName("TENSOR.REDUCE_WINDOW2D")]
		[Category("Tensor Instructions")]
		[Description("REDUCE_WINDOW2D")]
		public class ReduceWindow2DInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.REDUCE_WINDOW2D;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("reduce_op")]
			[Description("Reduce operator")]
			public ReduceOp ReduceOp { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("filter_h")]
			[Description("FilterH")]
			public ushort FilterH { get; set; }

			[DisplayName("filter_w")]
			[Description("FilterW")]
			public ushort FilterW { get; set; }

			[DisplayName("stride_h")]
			[Description("StrideH")]
			public ushort StrideH { get; set; }

			[DisplayName("stride_w")]
			[Description("StrideW")]
			public ushort StrideW { get; set; }

			[DisplayName("dilation_h")]
			[Description("DilationH")]
			public ushort DilationH { get; set; }

			[DisplayName("dilation_w")]
			[Description("DilationW")]
			public ushort DilationW { get; set; }

			[DisplayName("fused_clamp_low")]
			[Description("FusedClampLow")]
			public float FusedClampLow { get; set; }

			[DisplayName("fused_clamp_high")]
			[Description("FusedClampHigh")]
			public float FusedClampHigh { get; set; }
		}

		[DisplayName("TENSOR.RESIZE_IMAGE")]
		[Category("Tensor Instructions")]
		[Description("RESIZE_IMAGE")]
		public class ResizeImageInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.RESIZE_IMAGE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("align_corners")]
			[Description("Align Corners")]
			public bool AlignCorners { get; set; }

			[DisplayName("half_pixel_centers")]
			[Description("Half Pixel Centers")]
			public bool HalfPixelCenters { get; set; }

			[DisplayName("image_resize_mode")]
			[Description("Image Resize Mode")]
			public ImageResizeMode ImageResizeMode { get; set; }
		}

		[DisplayName("TENSOR.ROI_ALIGN")]
		[Category("Tensor Instructions")]
		[Description("RoiAlign")]
		public class RoiAlignInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.ROI_ALIGN;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rshape_dest")]
			[Description("Dest shape register")]
			public byte RshapeDest { get; set; }

			[DisplayName("mode")]
			[Description("Mode")]
			public RoiAlignMode mode { get; set; }

			[DisplayName("spatial_scale")]
			[Description("Spatial Scale")]
			public float SpatialScale { get; set; }

			[DisplayName("sampling_ratio")]
			[Description("Sampling Ratio")]
			public long SamplingRatio { get; set; }
		}

		[DisplayName("TENSOR.SIGMOID")]
		[Category("Tensor Instructions")]
		[Description("Sigmoid")]
		public class SigmoidInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.SIGMOID;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.SLICE")]
		[Category("Tensor Instructions")]
		[Description("Slice")]
		public class SliceInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.SLICE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rbegins")]
			[Description("Begins shape register")]
			public byte Rbegins { get; set; }

			[DisplayName("rends")]
			[Description("Ends shape register")]
			public byte Rends { get; set; }

			[DisplayName("rstrides")]
			[Description("Strides shape register")]
			public byte Strides { get; set; }
		}

		[DisplayName("TENSOR.SOFTMAX")]
		[Category("Tensor Instructions")]
		[Description("Softmax")]
		public class SoftmaxInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.SOFTMAX;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public int Axis { get; set; }

			[DisplayName("beta")]
			[Description("Beta")]
			public float Beta { get; set; }
		}

		[DisplayName("TENSOR.SPACE_TO_BATCH")]
		[Category("Tensor Instructions")]
		[Description("SpaceToBatch")]
		public class SpaceToBatchInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.SPACE_TO_BATCH;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_block")]
			[Description("Block shape register")]
			public byte RshapeBlock { get; set; }

			[DisplayName("rpad_crops")]
			[Description("Crops paddings register")]
			public byte RpadCrops { get; set; }
		}

		[DisplayName("TENSOR.TERNARY")]
		[Category("Tensor Instructions")]
		[Description("Ternary")]
		public class TernaryInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.TERNARY;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src1")]
			[Description("Source1 shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("rstride_src1")]
			[Description("Source1 stride register")]
			public byte RstrideSrc1 { get; set; }

			[DisplayName("rshape_src2")]
			[Description("Source2 shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("rstride_src2")]
			[Description("Source2 stride register")]
			public byte RstrideSrc2 { get; set; }

			[DisplayName("rshape_src3")]
			[Description("Source3 shape register")]
			public byte RshapeSrc3 { get; set; }

			[DisplayName("rstride_src3")]
			[Description("Source3 stride register")]
			public byte RstrideSrc3 { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }
		}

		[DisplayName("TENSOR.TOPK")]
		[Category("Tensor Instructions")]
		[Description("Topk")]
		public class TopKInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.TOPK;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rshape_dest1")]
			[Description("Dest1 shape register")]
			public byte RshapeDest1 { get; set; }

			[DisplayName("rstride_dest1")]
			[Description("Dest1 stride register")]
			public byte RstrideDest1 { get; set; }

			[DisplayName("rshape_dest2")]
			[Description("Dest2 shape register")]
			public byte RshapeDest2 { get; set; }

			[DisplayName("rstride_dest2")]
			[Description("Dest2 stride register")]
			public byte RstrideDest2 { get; set; }

			[DisplayName("k")]
			[Description("K")]
			public long K { get; set; }

			[DisplayName("axis")]
			[Description("Axis")]
			public int Axis { get; set; }

			[DisplayName("largest")]
			[Description("Largest")]
			public bool Largest { get; set; }

			[DisplayName("sorted")]
			[Description("Sorted")]
			public bool Sorted { get; set; }
		}

		[DisplayName("TENSOR.TRILU")]
		[Category("Tensor Instructions")]
		[Description("Trilu")]
		public class TriluInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.TRILU;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("upper")]
			[Description("Upper")]
			public bool Upper { get; set; }

			[DisplayName("k")]
			[Description("K")]
			public long K { get; set; }
		}

		[DisplayName("TENSOR.UNARY")]
		[Category("Tensor Instructions")]
		[Description("Unary")]
		public class UnaryInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.UNARY;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source1 shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("unary_op")]
			[Description("Unary operator")]
			public UnaryOp UnaryOp { get; set; }
		}

		[DisplayName("TENSOR.TRANSPOSE")]
		[Category("Tensor Instructions")]
		[Description("Transpose")]
		public class TransposeInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.TRANSPOSE;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("rshape_src")]
			[Description("Source shape register")]
			public byte RshapeSrc { get; set; }

			[DisplayName("rstride_src")]
			[Description("Source stride register")]
			public byte RstrideSrc { get; set; }

			[DisplayName("rstride_dest")]
			[Description("Dest stride register")]
			public byte RstrideDest { get; set; }

			[DisplayName("rshape_perm")]
			[Description("Perm shape register")]
			public byte RshapePerm { get; set; }
		}
		[DisplayName("TENSOR.GRU")]
		[Category("Tensor Instructions")]
		[Description("Gru")]
		public class GruInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.GRU;

			[DisplayName("input_shape_src")]
			[Description("Input shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("w_shape_src")]
			[Description("W shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("direction")]
			[Description("direction register")]
			public byte Direction { get; set; }

			[DisplayName("linear_before_reset")]
			[Description("LBR register")]
			public bool LinearBeforeReset { get; set; }

		}
		[DisplayName("TENSOR.TFLITE_DETECTION_POSTPROCESS")]
		[Category("Tensor Instructions")]
		[Description("Tflite_Detection_Postprocess")]
		public class TfliteDetectionPostprocessInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.TFLITE_DETECTION_POSTPROCESS;

			[DisplayName("box_shape_src")]
			[Description("Box shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("score_shape_src")]
			[Description("Score shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("anchor_shape_src")]
			[Description("Anchor shape register")]
			public byte RshapeSrc3 { get; set; }

			[DisplayName("max_detections")]
			[Description("max_detections register")]
			public int MaxDetections { get; set; }

			[DisplayName("max_classes_per_detection")]
			[Description("max_classes_per_detection register")]
			public int MaxClassesPerDetection { get; set; }

			[DisplayName("detections_per_class")]
			[Description("detections_per_class register")]
			public int DetectionsPerClass { get; set; }

			[DisplayName("use_regular_non_max_suppression")]
			[Description("use_regular_non_max_suppression register")]
			public bool UseRegularNonMaxSuppression { get; set; }

			[DisplayName("nms_score_threshold")]
			[Description("nms_score_threshold register")]
			public float NmsScoreThreshold { get; set; }

			[DisplayName("nms_iou_threshold")]
			[Description("nms_iou_threshold register")]
			public float NmsIouThreshold { get; set; }

			[DisplayName("num_classes")]
			[Description("num_classes register")]
			public int NumClasses { get; set; }

			[DisplayName("y_scale")]
			[Description("y_scale register")]
			public float YScale { get; set; }

			[DisplayName("x_scale")]
			[Description("x_scale register")]
			public float XScale { get; set; }

			[DisplayName("h_scale")]
			[Description("h_scale register")]
			public float HScale { get; set; }

			[DisplayName("w_scale")]
			[Description("w_scale register")]
			public float WScale { get; set; }
		}

		[DisplayName("TENSOR.LAYER_NORMALIZATION")]
		[Category("Tensor Instructions")]
		[Description("LAYER_NORMALIZATION")]
		public class LayerNormInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.LAYER_NORMALIZATION;

			[DisplayName("datatype")]
			[Description("Datatype")]
			public DataType DataType { get; set; }

			[DisplayName("input_shape")]
			[Description("input_shape")]
			public byte input_shape { get; set; }
			[DisplayName("axis")]
			[Description("axis")]
			public int axis { get; set; }

			[DisplayName("epsilon")]
			[Description("epsilon")]
			public float epsilon { get; set; }
		}

		[DisplayName("TENSOR.COMPRESS")]
		[Category("Tensor Instructions")]
		[Description("Compress")]
		public class CompressInstruction : TensorInstruction
		{
			public override TensorFunction Function => TensorFunction.COMPRESS;

			[DisplayName("input_shape_src")]
			[Description("Input shape register")]
			public byte RshapeSrc1 { get; set; }

			[DisplayName("condition_shape_src")]
			[Description("Condition shape register")]
			public byte RshapeSrc2 { get; set; }

			[DisplayName("axis")]
			[Description("axis register")]
			public float axis { get; set; }
		}

		[DisplayName("TENSOR.GATHER_ELEMENTS")]
        [Category("Tensor Instructions")]
        [Description("Gather_Elements")]
        public class Gather_ElementsInstruction : TensorInstruction
        {
            public override TensorFunction Function => TensorFunction.GATHER_ELEMENTS;

            [DisplayName("input_shape_src")]
            [Description("Input shape register")]
            public byte RshapeSrc1 { get; set; }

            [DisplayName("indices_shape_src")]
            [Description("Indices shape register")]
            public byte RshapeSrc2 { get; set; }

            [DisplayName("axis")]
            [Description("Axis")]
            public int Axis { get; set; }
        }

        [DisplayName("TENSOR.INSTANCE_NORMALIZATION")]
        [Category("Tensor Instructions")]
        [Description("INSTANCE_NORMALIZATION")]
        public class InstanceNormInstruction : TensorInstruction
        {
            public override TensorFunction Function => TensorFunction.INSTANCE_NORMALIZATION;

            [DisplayName("datatype")]
            [Description("Datatype")]
            public DataType DataType { get; set; }

            [DisplayName("input_shape")]
            [Description("input_shape")]
            public byte input_shape { get; set; }

            [DisplayName("epsilon")]
            [Description("epsilon")]
            public float epsilon { get; set; }
        }
	}
}

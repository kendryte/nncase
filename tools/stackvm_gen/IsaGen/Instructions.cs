// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitFields;

namespace IsaGen
{
    [EnumName("opcode_t")]
    public enum OpCode : byte
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

        LDLOCAL,
        STLOCAL,

        LDTUPLE_ELEM,
        LDTUPLE,

        LDDATATYPE,
        LDTENSOR,
        LDSCALAR,

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
        EXTCALL,
        CUSCALL,
        THROW,
        BREAK,

        TENSOR,
    }

    [System.AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = false)]
    public sealed class EnumNameAttribute : Attribute
    {
        public EnumNameAttribute(string name)
        {
            Name = name;
        }

        public string Name { get; }
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
        public ushort Index { get; set; }
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

    [DisplayName("LDTUPLE_ELEM")]
    [Category("Load Store Instructions")]
    [Description("Load an element of tuple to stack")]
    public class LdTupleElemInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDTUPLE_ELEM;
    }

    [DisplayName("LDTUPLE")]
    [Category("Load Store Instructions")]
    [Description("Load a tuple to stack")]
    public class LdTupleInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDTUPLE;
    }

    [DisplayName("LDDATATYPE")]
    [Category("Load Store Instructions")]
    [Description("Load a datatype to stack")]
    public class LdDataTypeInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDDATATYPE;
    }

    [DisplayName("LDTENSOR")]
    [Category("Load Store Instructions")]
    [Description("Load a tensor to stack")]
    public class LdTensorInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDTENSOR;
    }

    [DisplayName("LDSCALAR")]
    [Category("Load scalar Instructions")]
    [Description("Load a local object to scalar from stack")]
    public class LdScalarInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDSCALAR;
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

    [DisplayName("LDLOCAL")]
    [Category("Load Store Instructions")]
    [Description("Load a local to stack")]
    public class LdlocalInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.LDLOCAL;

        [DisplayName("index")]
        [Description("Local index")]
        public ushort Index { get; set; }
    }

    [DisplayName("STLOCAL")]
    [Category("Load Store Instructions")]
    [Description("Store a local from stack")]
    public class StlocalInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.STLOCAL;

        [DisplayName("index")]
        [Description("Local index")]
        public ushort Index { get; set; }
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
        public ushort ArgsCount { get; set; }

        [DisplayName("target")]
        [Description("Call a target method at the specified offset")]
        public int Target { get; set; }
    }

    [DisplayName("ECALL")]
    [Category("Control and Status Instructions")]
    [Description("Call an environment method")]
    public class ECallInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.ECALL;

        [DisplayName("args")]
        [Description("Arguments count")]
        public ushort ArgsCount { get; set; }
    }

    [DisplayName("EXTCALL")]
    [Category("Control and Status Instructions")]
    [Description("Call an external method")]
    public class ExtCallInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.EXTCALL;

        [DisplayName("args")]
        [Description("Arguments count")]
        public ushort ArgsCount { get; set; }

        [DisplayName("is_prim_func")]
        [Description("Is prim function")]
        public bool IsPrimFunc { get; set; }
    }

    [DisplayName("CUSCALL")]
    [Category("Control and Status Instructions")]
    [Description("Custom Call an User customed method")]
    public class CusCallInstruction : Instruction
    {
        public override OpCode OpCode => OpCode.CUSCALL;

        [DisplayName("registered_name")]
        [Description("Global Registered Name")]
        public string RegisteredName { get; set; } = string.Empty;

        [DisplayName("fields_span")]
        [Description("Fields Span")]
        public byte[] FieldsSpan { get; set; } = Array.Empty<byte>();

        [DisplayName("args")]
        [Description("Arguments count")]
        public ushort ArgsCount { get; set; }
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
}

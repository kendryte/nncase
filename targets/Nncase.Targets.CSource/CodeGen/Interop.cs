using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// <see cref="DynamicAssemble(string)"/>
/// </summary>
internal class DynamicAssemble
{
    /// <summary>
    /// name
    /// </summary>
    AssemblyName assemblyName;
    /// <summary>
    /// asm builder for whole module
    /// </summary>
    AssemblyBuilder asmBuilder;
    /// <summary>
    /// module buidler
    /// </summary>
    ModuleBuilder modBuilder;
    /// <summary>
    /// save the func name <=> func delegate type
    /// </summary>
    readonly Dictionary<string, Type> delegateTypes = new();

    /// <summary>
    /// a DynamicAssemble instance, it's contains one rtmodule's all functions defination.
    /// </summary>
    /// <param name="Name"> asmble name </param>
    public DynamicAssemble(string Name)
    {
        assemblyName = new AssemblyName(Name);
        asmBuilder = AssemblyBuilder.DefineDynamicAssembly(assemblyName, AssemblyBuilderAccess.RunAndCollect);
        modBuilder = asmBuilder.DefineDynamicModule(assemblyName.Name!);

    }

    /// <summary>
    /// <see cref="CreateDelegateType(string, Type, Type[]?)"/>
    /// </summary>
    /// <param name="function"></param>
    /// <returns> func delegate type</returns>
    public Type BuildDelegateType(Callable function)
    {
        Type deleType;
        if (function.CheckedType is CallableType ctype)
        {
            deleType = CreateDelegateType(function.Name, ctype.ReturnType.ToType(), ctype.Parameters.Select(Interop.ToType).ToArray());
        }
        else { throw new NotSupportedException(function.CheckedType?.ToString()); }
        return deleType;
    }

    /// <summary>
    /// dynamic create delegate type for function.
    /// </summary>
    /// <param name="funcName"></param>
    /// <param name="returnType"></param>
    /// <param name="ParamTypes"></param>
    /// <returns></returns>
    /// <exception cref="InvalidProgramException"></exception>
    public Type CreateDelegateType(string funcName, Type returnType, params Type[]? ParamTypes)
    {
        if (!delegateTypes.TryGetValue(funcName, out var ret))
        {
            ParamTypes ??= new Type[] { };
            TypeBuilder tb = modBuilder.DefineType(funcName, TypeAttributes.Public | TypeAttributes.Sealed, typeof(MulticastDelegate));
            tb.DefineConstructor(MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, CallingConventions.Standard | CallingConventions.HasThis, new[] { typeof(object), typeof(IntPtr) }).SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);
            tb.DefineMethod("Invoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, returnType, ParamTypes).SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);
            tb.DefineMethod("BeginInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(IAsyncResult), ParamTypes.Concat(new[] { typeof(IAsyncResult), typeof(object) }).ToArray()).SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);
            tb.DefineMethod("EndInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, returnType, new[] { typeof(IAsyncResult) }).SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);
            ret = tb.CreateType();
            if (ret is null) { throw new InvalidProgramException($"Can't Create The Func {funcName}'s delegate Type!"); }
            delegateTypes.Add(funcName, ret);
        }
        return ret;
    }
}

/// <summary>
/// Interop helper
/// </summary>
public static class Interop
{
    /// <summary>
    /// collect the all dynamic asmbs
    /// </summary>
    private static readonly Dictionary<string, DynamicAssemble> _definedAsms = new();

    /// <summary>
    /// convert the ir type to the system type
    /// </summary>
    /// <param name="iRType"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public static Type ToType(this IRType iRType) => iRType switch
    {
        TensorType { IsScalar: true, DType: PrimType { } primType } => primType.CLRType,
        TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType primType } } => primType.CLRType.MakeArrayType(),
        TupleType ttype => (ttype == TupleType.Void) switch
        {
            true => typeof(void),
            false => throw new NotSupportedException($"Can't Support the {ttype}!")
        },
        _ => throw new NotSupportedException($"IRType is {iRType}!")
    };


    /// <summary>
    /// convrt function to delegate type
    /// </summary>
    /// <param name="function"> input function </param>
    /// <param name="libName"> the dynamic lib name </param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public static Type ToDelegateType(this Callable function, string libName)
    {
        if (!_definedAsms.TryGetValue(libName, out var dyasm))
        {
            dyasm = new DynamicAssemble(libName);
            _definedAsms.Add(libName, dyasm);
        }
        return dyasm.BuildDelegateType(function); ;
    }

    /// <summary>
    /// bind the delegate to funcptr.
    /// </summary>
    /// <param name="funcPtr"></param>
    /// <param name="funcType"></param>
    /// <returns></returns>
    public static Delegate BindDelegate(this IntPtr funcPtr, Type funcType) => Marshal.GetDelegateForFunctionPointer(funcPtr, funcType);
}

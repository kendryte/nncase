// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.IO;
// using System.Text;
// using Nncase.TIR;
// using Nncase.IR;

// namespace Nncase.CodeGen.Builder
// {

//     public static class RTSymbol
//     {
//         /// <summary>
//         /// Global variable to store module context.
//         /// </summary>
//         public static string module_ctx = "__module_ctx";
//         /// <summary>
//         /// Global variable to store device module blob
//         /// </summary>
//         public static string dev_mblob = "__dev_mblob";
//         /// <summary>
//         /// Number of bytes of device module blob.
//         /// </summary>
//         public static string dev_mblob_nbytes = "__dev_mblob_nbytes";
//         /// <summary>
//         /// global function to set device
//         /// </summary>
//         public static string set_device = "__set_device";
//         /// <summary>
//         /// Auxiliary counter to global barrier.
//         /// </summary>
//         public static string global_barrier_state = "__global_barrier_state";
//         /// <summary>
//         /// Prepare the global barrier before kernels that uses global barrier.
//         /// </summary>
//         public static string prepare_global_barrier = "__prepare_global_barrier";
//         /// <summary>
//         /// Placeholder for the module's entry function.
//         /// </summary>
//         public static string module_main = "__main__";
//         /// <summary>
//         /// Prefix for parameter symbols emitted into the main program.
//         /// </summary>
//         public static string param_prefix = "__param__";
//         /// <summary>
//         /// A PackedFunc that looks up linked parameters by storage_id.
//         /// </summary>
//         public static string lookup_linked_param = "_lookup_linked_param";
//         /// <summary>
//         /// The main AOT executor function generated from TIR
//         /// </summary>
//         public static string run_func_suffix = "run_model";
//         /// <summary>
//         /// Model entrypoint generated as an interface to the AOT function outside of TIR
//         /// </summary>
//         public static string entrypoint_suffix = "run";
//     }



//     public class CSourceBuilder : SourceBuilder
//     {
//         /// <summary>
//         /// Initialize the code generator.
//         /// </summary>
//         /// <param name="output_ssa">output_ssa Whether output SSA.</param>
//         public void Init(bool output_ssa)
//         {
//             print_ssa_form_ = output_ssa;
//         }

//         /// <summary>
//         /// Add the function to the generated module.
//         /// </summary>
//         /// <param name="f">The function to be compiled.</param>
//         public virtual void AddFunction(PrimFunction f)
//         {
//             // clear previous generated state.
//             InitFuncState(f);
//             // reserve keywords
//             ReserveKeywordsAsUnique();

//             bool no_alias = f.Attrs.HasFlag(Attr.NoAlias);

//             PrintFuncPrefix();
//             PrintExtraAttrs(f);
//             SrcWriter.Write($" {f.Name}(");
//             foreach (var (v, i) in f.Params.Select((p, i) => ((Var)p, i)))
//             {
//                 var vid = AllocVarID(v);
//                 if (i != 0) { SrcWriter.Write(", "); }
//                 if (v.CheckedType is PointerType ptype)
//                 {
//                     if (alloc_storage_scope_.TryGetValue(v, out var scope))
//                     {
//                         PrintStorageScope(scope);
//                     }
//                     PrintType(v.CheckedDataType);
//                     RegisterHandleType(v);
//                     if (no_alias)
//                     {
//                         PrintRestrict(v);
//                     }
//                 }
//                 else
//                 {
//                     PrintType(v.CheckedDataType);
//                 }
//                 SrcWriter.Write($" {vid}");
//             }
//             SrcWriter.Write(") {\n");
//             PreFunctionBody(f);
//             int func_scope = BeginScope();
//             PrintStmt(f.Body);
//             PrintFinalReturn();
//             EndScope(func_scope);
//             PrintIndent();
//             SrcWriter.Write("}\n\n");
//         }


//         /// <summary>
//         /// Finalize the compilation and return the code.
//         /// </summary>
//         /// <returns>The code.</returns>
//         public string Finish()
//         {
//             var decl = DeclWriter.Pop();
//             var src = SrcWriter.Pop();
//             return decl + src;
//         }

//         /// <summary>
//         /// Print the Stmt n to CodeGenC->stream
//         /// </summary>
//         /// <param name="n">The statement to be printed.</param>
//         public virtual void PrintStmt(Stmt n)
//         {
//             VisitStmt(n);
//         }

//         /// <summary>
//         /// Print the expression n(or its ssa id if in ssa mode) into os
//         /// </summary>
//         /// <param name="n">The expression to be printed.</param>
//         public virtual void PrintExpr(Expr n) { }

//         // The following parts are overloadable print operations.
//         /// <summary>
//         /// Print the function header before the argument list
//         /// <example>
//         /// stream << "void";
//         /// </example>
//         /// </summary>
//         public virtual void PrintFuncPrefix() { }

//         /// <summary>
//         /// Print extra function attributes
//         /// <example>
//         ///  __launch_bounds__(256) for CUDA functions
//         /// </example>
//         /// </summary>
//         /// <param name="f"> func </param>
//         public virtual void PrintExtraAttrs(PrimFunction f) { }

//         /// <summary>
//         /// Print the final return at the end the function.
//         /// </summary>
//         public virtual void PrintFinalReturn() { }


//         /// <summary>
//         /// Insert statement before function body.
//         /// </summary>
//         /// <param name="f">f The function to be compiled.</param>
//         public virtual void PreFunctionBody(PrimFunction f) { }

//         /// <summary>
//         /// Initialize codegen state for generating f.
//         /// </summary>
//         /// <param name="f">The function to be compiled.</param>
//         public virtual void InitFuncState(PrimFunction f)
//         {
//             alloc_storage_scope_.Clear();
//             handle_data_type_.Clear();
//             base.ClearFuncState();
//         }

//         /// <summary>
//         /// Print Type represetnation of type t.
//         /// </summary>
//         /// <param name="t">The type representation.</param>
//         /// <exception cref="NotImplementedException"></exception>
//         /// <exception cref="NotSupportedException"></exception>
//         public virtual void PrintType(DataType t)
//         {
//             if (t.Lanes != 1) throw new NotImplementedException("do not yet support vector types!");
//             var name = t.ElemType switch
//             {
//                 ElemType.Int8 => "int8_t",
//                 ElemType.Int16 => "int16_t",
//                 ElemType.Int32 => "int32_t",
//                 ElemType.Int64 => "int64_t",
//                 ElemType.UInt8 => "uint8_t",
//                 ElemType.UInt16 => "uint16_t",
//                 ElemType.UInt32 => "uint32_t",
//                 ElemType.UInt64 => "uint64_t",
//                 // ElemType.Float16 => "float16_t",
//                 ElemType.Float32 => "float",
//                 ElemType.Float64 => "double",
//                 // ElemType.BFloat16 => "bfloat16_t",
//                 _ => throw new NotSupportedException($"Current ElemType {t.ElemType} To C Type!")
//             };
//             SrcWriter.Write(name);
//         }

//         /// <summary>
//         /// Print Type represetnation of type type.
//         /// </summary>
//         /// <param name="type">The type representation.</param>
//         /// <exception cref="NotSupportedException"></exception>
//         public virtual void PrintType(IRType type)
//         {
//             void CheckLanes(DataType t) { }
//             switch (type)
//             {
//                 case TensorType t:
//                     CheckLanes(t.DType);
//                     break;
//                 case PointerType t:
//                     PrintType(t.DType);
//                     SrcWriter.Write(" *");
//                     break;
//                 default:
//                     throw new NotSupportedException($"The Type {type.GetType().Name} To C Type!");
//             }
//         }

//         /// <summary>
//         /// Print expr representing the thread tag
//         /// </summary>
//         /// <param name="iv">IterVar iv The thread index to be binded;</param>
//         /// <exception cref="NotImplementedException"></exception>
//         public virtual void BindThreadIndex(IterVar iv)
//         {
//             throw new NotImplementedException("Not Impl BindThreadIndex");
//         }
//         public virtual void PrintStorageScope(string scope)
//         {
//             if (scope != "global") { throw new InvalidOperationException("Scope Is Not Global!"); }
//         }

//         /// <summary>
//         /// print storage sync
//         /// </summary>
//         /// <param name="op"></param>
//         public virtual void PrintStorageSync(Call op) { }
//         // Binary vector op.
//         public virtual void PrintVecBinaryOp(string op, DataType op_type, Expr lhs, Expr rhs) { }

//         // print vector load
//         public virtual string GetVecLoad(DataType t, Var buffer, Expr base_offset)
//         {
//             return GetBufferRef(t, buffer, base_offset);
//         }

//         // print vector store
//         /// <summary>
//         /// 
//         /// </summary>
//         /// <param name="buffer"></param>
//         /// <param name="t"></param>
//         /// <param name="base_offset"></param>
//         /// <param name="value"></param>
//         public virtual void PrintVecStore(Var buffer, DataType t, Expr base_offset,
//                                    string value)
//         {
//             var str = GetBufferRef(t, buffer, base_offset);
//             PrintIndent();
//             SrcWriter.Write($"{str} = {value} ;\n");
//         }

//         /// <summary>
//         /// load vec from elem
//         /// </summary>
//         /// <param name="vec"></param>
//         /// <param name="t"></param>
//         /// <param name="i"></param>
//         public virtual void PrintVecElemLoad(string vec, DataType t, int i)
//         {
//             var hex = i.ToString("X");
//             SrcWriter.Write($"{vec}.s{hex}");
//         }
//         /// <summary>
//         /// print store of single element.
//         /// </summary>
//         /// <param name="vec"></param>
//         /// <param name="t"></param>
//         /// <param name="i"></param>
//         /// <param name="value"></param>
//         public virtual void PrintVecElemStore(string vec, DataType t, int i,
//                                         string value)
//         {
//             PrintIndent();
//             var hex = i.ToString("X");
//             SrcWriter.Write($"{vec}.s{hex} = {value} ;\n");
//         }

//         // Get a cast type from to
//         public virtual string CastFromTo(string value, DataType from, DataType target)
//         {
//             if (from == target)
//             {
//                 return value;
//             }
//             SrcWriter.Push();
//             SrcWriter.Write("((");
//             PrintType(target);
//             SrcWriter.Write($"){value})");
//             return SrcWriter.Pop();
//         }

//         /// <summary>
//         /// Get load of single element with expression 
//         /// </summary>
//         /// <param name="t"></param>
//         /// <param name="i"></param>
//         /// <param name="value"></param>
//         public virtual void PrintVecElemLoadExpr(DataType t, int i, string value)
//         {
//             if (t.Lanes != 1)
//             {
//                 throw new NotSupportedException("Lanes != 1");
//             }
//             if (t == DataType.Int8 || t == DataType.UInt8)
//             {
//                 if (i != 0)
//                 {
//                     SrcWriter.Write("|");
//                 }
//                 SrcWriter.Write($"((0x000000ff << {i * 8}) & ({value} << {i * 8}))");

//             }

//         }
//         // Print restrict keyword for a given Var if applicable
//         public virtual void PrintRestrict(Var v)
//         {
//             if (restrict_keyword_.Length != 0)
//             {
//                 SrcWriter.Write($" {restrict_keyword_}");
//             }
//         }

//         /// <summary>
//         /// Print reference to struct location 
//         /// </summary>
//         /// <param name="t"></param>
//         /// <param name="buffer"></param>
//         /// <param name="index"></param>
//         /// <param name="kind"></param>
//         /// <returns></returns>
//         protected string GetStructRef(DataType t, Expr buffer, Expr index, int kind)
//         {
//             throw new NotSupportedException("GetStructRef");
//         }

//         // Print reference to a buffer as type t in index.
//         protected string GetBufferRef(DataType t, Var buffer, Expr index)
//         {
//             SrcWriter.Push();
//             string vid = GetVarID(buffer);
//             alloc_storage_scope_.TryGetValue(buffer, out var scope);
//             bool is_vol = IsVolatile(buffer);
//             if (t.Lanes == 1)
//             {
//                 if (!HandleTypeMatch(buffer, t) || is_vol)
//                 {
//                     SrcWriter.Write("((");
//                     if (is_vol) { SrcWriter.Write("volatile "); }
//                     if (!(scope.Length != 0) && IsScopePartOfType())
//                     {
//                         PrintStorageScope(scope);
//                     }
//                     PrintType(t);
//                     SrcWriter.Write($"*) {vid})");
//                 }
//                 else
//                 {
//                     SrcWriter.Write(vid);
//                 }
//                 SrcWriter.Write("[(");
//                 PrintExpr(index);
//                 SrcWriter.Write(")");
//                 // if (DataTypes.GetLength(t) == 4)
//                 // {
//                 // }
//                 SrcWriter.Write("]");
//             }
//             else
//             {
//                 // not support Buffer declared as vector type.
//                 throw new NotSupportedException("The Vector Type!");
//             }
//             return SrcWriter.Pop();
//         }

//         /*!
//          * Handle volatile loads.
//          *
//          * This is to workaround a bug in CUDA cuda_fp16.h. Volatile accesses
//          * to shared memory are required for reductions. However, __half class
//          * does not implement volatile member functions. CUDA codegen will cast
//          * away volatile qualifier from CUDA __half types.
//          */
//         protected virtual void HandleVolatileLoads(string value, Load op)
//         {
//             // By default, do nothing but print the loaded value.
//             SrcWriter.Write(value);
//         }

//         /*!
//          * Check if scope is part of type in the target language.
//          *
//          * **NOTE** In OpenCL, __local is part of type, so "__local int *"
//          * is legal. This is not the case for CUDA, where "__shared__"
//          * or "__ant__" is not part of type but a storage class (like
//          * C/C++ static).
//          */
//         protected virtual bool IsScopePartOfType() { return true; }

//         /*!
//          * Print external function call.
//          * \param ret_type The return type.
//          * \param global_symbol The symbolc of the target function.
//          * \param args The arguments to the function.
//          * \param skip_first_arg Whether to skip the first arguments.
//          * \param os The output stream.
//          */
//         protected virtual void PrintCallExtern(Type ret_type, String global_symbol, IRArray<Expr> args, bool skip_first_arg) { }
//         /*!
//          * If buffer is allocated as type t.
//          * \param buf_var The buffer variable.
//          * \param t The type to be checked.
//          */
//         protected bool HandleTypeMatch(Var buf_var, DataType t)
//         {
//             if (!handle_data_type_.TryGetValue(buf_var, out var var_dtype))
//             {
//                 return false;
//             }
//             return var_dtype == t;
//         }

//         /// <summary>
//         /// Register the data type of buf_var
//         /// </summary>
//         /// <param name="buf_var">The buffer variable.</param>
//         protected void RegisterHandleType(Var buf_var)
//         {
//             if (!handle_data_type_.TryGetValue(buf_var, out var it))
//             {
//                 handle_data_type_[buf_var] = buf_var.CheckedDataType;
//             }
//             else
//             {
//                 if (it != handle_data_type_[buf_var])
//                 {
//                     throw new InvalidOperationException("The Var Dtype Has Been Regisiter!");
//                 }
//             }
//         }

//         /// <summary>
//         /// check the bracked
//         /// </summary>
//         /// <param name="s"></param>
//         /// <returns></returns>
//         private static bool CheckOutermostBracketMatch(string s)
//         {
//             if (s.Length != 0 && s.First() == '(' && s.Last() == ')')
//             {
//                 var len = s.Length;
//                 int n_unmatched = 0;
//                 for (int i = 0; i < len; ++i)
//                 {
//                     switch (s[i])
//                     {
//                         case '(':
//                             n_unmatched++;
//                             break;
//                         case ')':
//                             n_unmatched--;
//                             break;
//                         default: break;
//                     }
//                     if (n_unmatched == 0) { return i == len - 1; }
//                 }
//             }
//             return false;
//         }

//         protected void PrintSSAAssign(string target, string src, DataType t)
//         {
//             PrintType(t);
//             SrcWriter.Write($" {target} = ");
//             if (CheckOutermostBracketMatch(src))
//             {
//                 SrcWriter.Write(src.Substring(1, src.Length - 2));
//             }
//             else
//             {
//                 SrcWriter.Write(src);
//             }
//             SrcWriter.Write(";\n");
//         }
//         /*! reserves common C keywords */
//         protected void ReserveKeywordsAsUnique()
//         {

//             // skip the first underscore, so SSA variable starts from _1
//             GetUniqueName("_");
//             GetUniqueName("extern");
//             GetUniqueName("void");
//             GetUniqueName("int");
//             GetUniqueName("float");
//             GetUniqueName("double");
//             GetUniqueName("char");
//             GetUniqueName("unsigned");
//             GetUniqueName("short");
//             GetUniqueName("long");
//             GetUniqueName("if");
//             GetUniqueName("else");
//             GetUniqueName("switch");
//             GetUniqueName("case");
//             GetUniqueName("default");
//             GetUniqueName("for");
//             GetUniqueName("do");
//             GetUniqueName("while");
//             GetUniqueName("goto");
//             GetUniqueName("register");
//             GetUniqueName("continue");
//             GetUniqueName("break");
//             GetUniqueName("typedef");
//             GetUniqueName("struct");
//             GetUniqueName("enum");
//             GetUniqueName("union");
//             GetUniqueName("return");
//         }
//         /// <summary>
//         /// Check if buf_var is volatile or not.
//         /// </summary>
//         /// <param name="buf_var"></param>
//         /// <returns></returns>
//         protected bool IsVolatile(Var buf_var)
//         {
//             return volatile_buf_.Contains(buf_var);
//         }

//         /*! restrict keyword */
//         protected virtual string restrict_keyword_ { get => ""; }
//         /*! the storage scope of allocation */
//         protected Dictionary<Var, string> alloc_storage_scope_;
//         /*! the data type of allocated buffers */
//         protected Dictionary<Var, DataType> handle_data_type_;
//         /*! Record of ops that have pre-defined global symbol. */
//         // protected OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
//         // cache commonly used ops
//         // protected Op builtin_call_extern_ = builtin::call_extern();
//         // protected Op builtin_call_pure_extern_ = builtin::call_pure_extern();


//         /*! whether to print in SSA form */
//         private bool print_ssa_form_ = false;
//         /*! set of volatile buf access */
//         private HashSet<Var> volatile_buf_;
//         // deep comparison of Expr
//         // private ExprDeepEqual deep_equal_;
//         // binding of let variables. Enables duplicate var defs that map to same value
//         private Dictionary<Var, Let> let_binding_;

//         // /// <inheritdoc/>
//         // public RTModule Build(PrimModule mod, Target target)
//         // {
//         //     Init(output_ssa: false, emit_asserts: false, target.ToString());
//         //     foreach (var func in mod.Functions)
//         //     {
//         //         var func_name = func.Name;

//         //     }
//         // }

//     }
// }
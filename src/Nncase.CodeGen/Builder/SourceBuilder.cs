// using System;
// using System.Text;
// using System.IO;
// using Nncase.TIR;
// using Nncase.IR;
// using System.Collections.Generic;


// namespace Nncase.CodeGen.Builder
// {

//     /// <summary>
//     /// stacked write source
//     /// </summary>
//     public class StackedWriter
//     {
//         /// <summary>
//         /// string builer stack
//         /// </summary>
//         readonly Stack<StringBuilder> _streams = new();
//         /// <summary>
//         /// stack of writer
//         /// </summary>
//         readonly Stack<TextWriter> _writers = new();
//         /// <summary>
//         /// current stream
//         /// </summary>
//         StringBuilder _cur_stream;
//         /// <summary>
//         /// current writer
//         /// </summary>
//         TextWriter _cur_writer;

//         /// <summary>
//         /// write string
//         /// </summary>
//         /// <param name="s"></param>
//         public void Write(string s)
//         {
//             _cur_writer.Write(s);
//         }

//         /// <summary>
//         /// pushed the write and stream
//         /// </summary>
//         public void Push()
//         {
//             _cur_stream = new StringBuilder();
//             _cur_writer = new StringWriter(_cur_stream);
//             _streams.Push(_cur_stream);
//             _writers.Push(_cur_writer);
//         }
//         public StackedWriter()
//         {
//             Push();
//         }

//         /// <summary>
//         /// pop the write stack
//         /// </summary>
//         /// <returns> current scope writed string </returns>
//         public string Pop()
//         {
//             _cur_writer.Flush();
//             _cur_writer.Close();
//             var s = _cur_stream.ToString();
//             _writers.Pop();
//             _streams.Pop();
//             if (_writers.Count >= 1)
//             {
//                 _cur_writer = _writers.Peek();
//                 _cur_stream = _streams.Peek();
//             }
//             return s;
//         }

//     }

//     // public class SourceGenContext
//     // {
//     //     public SourceGenContext(Dictionary<Expr, string> exprMemo, Dictionary<Stmt, string> stmtMemo)
//     //     {
//     //         SrcWriter.Push(srcWriter);
//     //         DeclWriter.Push(declWriter);
//     //     }
//     // }

//     /// <summary>
//     /// base source builder class
//     /// </summary>
//     public class SourceBuilder : StmtFunctor<string, string, string>
//     {

//         public StackedWriter SrcWriter, DeclWriter;
//         /// <summary>
//         /// constructor
//         /// </summary>
//         public SourceBuilder()
//         {
//             SrcWriter = new();
//             DeclWriter = new();
//         }

//         /// <summary>
//         /// Register constant value appeared in expresion tree
//         ///  This avoid generated a ssa id for each appearance of the value
//         /// </summary>
//         /// <param name="value">The constant value.</param>
//         public void MarkConst(string value) { }

//         /// <summary>
//         /// entry in ssa assign map
//         /// </summary>
//         struct SSAEntry
//         {
//             /// <summary>
//             /// The value id
//             /// </summary>
//             public string VId;
//             /// <summary>
//             /// The scope id, used to check if this entry is invalid.
//             /// </summary>
//             public int ScopeId;
//         }

//         /// <summary>
//         /// Clear the states that might relates to function generation
//         /// </summary>
//         protected virtual void ClearFuncState()
//         {
//             _name_alloc_map.Clear();
//             _ssa_assign_map.Clear();
//             _var_idmap.Clear();
//             _scope_mark.Clear();
//         }

//         /// <summary>
//         /// Clear the states that might relates to function generation
//         /// </summary>
//         protected void PrintIndent()
//         {
//             SrcWriter.Write(new string(' ', _indent));
//         }

//         /// <summary>
//         /// Allocate a variable name for a newly defined var.
//         /// </summary>
//         /// <param name="var">The variable.</param>
//         /// <returns>the variable name.</returns>
//         protected string AllocVarID(Var @var)
//         {
//             if (_var_idmap.ContainsKey(@var))
//             {
//                 throw new InvalidOperationException($"Need input to be in SSA form dup {@var.Name}");
//             }
//             var vid = GetUniqueName(@var.Name);
//             _var_idmap[@var.Name] = vid;
//             return vid;
//         }

//         /// <summary>
//         /// Get a variable name.
//         /// </summary>
//         /// <param name="var">The variable.</param>
//         /// <returns>the variable name.</returns>
//         protected string GetVarID(Var @var)
//         {
//             if (!_var_idmap.TryGetValue(@var, out var it))
//             {
//                 throw new InvalidOperationException($"Find undefined Variable {@var.Name}");
//             }
//             return it;
//         }

//         /// <summary>
//         /// Get the SSA ID corresponds to src If necessary, generate new assignment
//         /// </summary>
//         /// <param name="src">The source expression</param>
//         /// <param name="dtype">The type of the expression</param>
//         /// <returns>ssa id string</returns>
//         protected string SSAGetID(string src, DataType dtype)
//         {
//             if (_name_alloc_map.ContainsKey(src)) { return src; }
//             if (_ssa_assign_map.TryGetValue(src, out var it))
//             {
//                 if (_scope_mark[it.ScopeId])
//                 {
//                     return it.VId;
//                 }
//             }
//             SSAEntry e = new();
//             e.VId = GetUniqueName("_");
//             e.ScopeId = _scope_mark.Count - 1;
//             _ssa_assign_map[src] = e;
//             PrintIndent();
//             PrintSSAAssign(e.VId, src, dtype);
//             return e.VId;
//         }

//         /// <summary>
//         /// get a unique name with the corresponding prefix
//         /// </summary>
//         /// <param name="prefix">prefix The prefix of the name.</param>
//         /// <returns>The returned name.</returns>
//         protected string GetUniqueName(string prefix)
//         {
//             prefix = prefix.Replace(".", "_");
//             if (_name_alloc_map.TryGetValue(prefix, out var it))
//             {
//                 StringBuilder sb = new();
//                 while (true)
//                 {
//                     sb.Append(prefix);
//                     sb.Append(++it);
//                     var name = sb.ToString();
//                     if (!_name_alloc_map.ContainsKey(name))
//                     {
//                         prefix = name;
//                         break;
//                     }
//                 }
//             }
//             _name_alloc_map[prefix] = 0;
//             return prefix;
//         }

//         /// <summary>
//         /// mark the beginning of a new scope
//         /// </summary>
//         /// <returns>The scope id.</returns>
//         protected int BeginScope()
//         {
//             int sid = _scope_mark.Count;
//             _scope_mark.Add(true);
//             _indent += 2;
//             return sid;
//         }

//         /// <summary>
//         /// mark the end of an old scope.
//         /// </summary>
//         /// <param name="scope_id">The scope id to be ended.</param>
//         protected void EndScope(int scope_id)
//         {
//             _scope_mark[scope_id] = false;
//             _indent -= 2;
//         }
//         /*!
//          * \brief Print assignment of src to the id in ssa entry.
//          * \param 
//          * \param src 
//          * \param t   
//          */

//         /// <summary>
//         /// Print assignment of src to the id in ssa entry.
//         /// </summary>
//         /// <param name="target">id of target variable.</param>
//         /// <param name="src">The source expression.</param>
//         /// <param name="t">The type of target.</param>
//         protected virtual void PrintSSAAssign(string target, string src, DataType t) { }

//         /// <summary>
//         /// name of each variable
//         /// </summary>
//         private readonly Dictionary<Var, string> _var_idmap = new();

//         /// <summary>
//         /// assignment map of ssa
//         /// </summary>
//         private readonly Dictionary<string, SSAEntry> _ssa_assign_map = new();
//         /// <summary>
//         /// name allocation map
//         /// </summary>
//         private readonly Dictionary<string, int> _name_alloc_map = new();
//         /// <summary>
//         /// array to check whether we are inside certain scope
//         /// </summary>
//         private List<bool> _scope_mark;
//         /// <summary>
//         /// The current indentation value
//         /// </summary>
//         private int _indent = 0;

//     }
// }
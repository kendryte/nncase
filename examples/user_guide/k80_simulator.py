import nncase
import numpy as np
from nncase_base_func import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
print("pid: ", os.getpid())


def to_np_type(t: str):
    '''
    string to np.type
    '''
    if t == torch.float32:
        return np.float32
    elif t == torch.float16:
        return np.float16
    else:
        return None


def compile_kmodel(model_path, dump_path, calib_data, config):
    """
    Set compile options and ptq options.
    Compile kmodel.
    Dump the compile-time result to 'compile_options.dump_dir'
    """
    print("\n----------   compile    ----------")
    print("Simplify...")
    model_file = model_simplify(model_path)
    # model_file = model_path

    print("Set options...")
    # import_options
    import_options = nncase.ImportOptions()
    import_options.huggingface_options.use_cache = True
    import_options.huggingface_options.output_attentions = True
    import_options.huggingface_options.output_hidden_states = False
    import_options.huggingface_options.config = config

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = "cpu"  # "cpu"
    compile_options.dump_ir = True  # if False, will not dump the compile-time result.
    compile_options.dump_asm = True
    compile_options.dump_dir = dump_path
    # compile_options.input_file = "/mnt/workspace/onnxruntime-inference-examples/quantization/language_model/llama/smooth_quant/new_model/onnx/llm.onnx.data"
    # compile_options.input_file = "/mnt/model/pipeline/llm-export/model24/onnx/llm.onnx.data"

    # TODO: fix this! need change nncase/python/nncase/__init__.py:174L
    target_options = nncase.NTTTargetOptions()
    target_options.Packing = False
    target_options.Hierarchies = [[1]]
    target_options.HierarchyNames = 't'
    target_options.HierarchySizes = [603979776]
    target_options.MemoryCapacities = [262144]
    target_options.MemoryBandWidths = [64]
    target_options.UnifiedMemoryArch = True

    compile_options.shape_bucket_enable = True
    compile_options.shape_bucket_segments_count = 2
    compile_options.shape_bucket_range_info = {"sequence_length": [1, 512]}
    compile_options.shape_bucket_fix_var_map = {"batch_size": 1}  # {"seq_len": 9, "history_len": 0}

    # quant
    # ptq_options = nncase.PTQTensorOptions()

    # ptq_options.quant_type = "uint8"  # datatype : "float32", "int8", "int16"
    # ptq_options.w_quant_type = "uint8"  # datatype : "float32", "int8", "int16"
    # ptq_options.calibrate_method = "NoClip"  # "Kld"
    # ptq_options.finetune_weights_method = "NoFineTuneWeights"
    # ptq_options.dump_quant_error = False
    # ptq_options.dump_quant_error_symmetric_for_signed = False

    # # detail in docs/MixQuant.md
    # ptq_options.quant_scheme = ""# "qwen_24_apply_all/try.json"
    # ptq_options.export_quant_scheme = False
    # ptq_options.export_weight_range_by_channel = False

    # ptq_options.samples_count = len(calib_data[0])
    # ptq_options.set_tensor_data(calib_data)

    print("Compiling...")
    compiler = nncase.Compiler(compile_options)
    # import
    # model_content = read_model_file(model_file)
    compiler.import_huggingface(model_file, import_options)
    evaluator = compiler.create_evaluator(3)

    return evaluator
    # compiler.use_ptq(ptq_options)

    # # compile
    # compiler.compile()
    # kmodel = compiler.gencode_tobytes()
    # print("Write kmodel...")
    # kmodel_path = os.path.join(dump_path, "test.kmodel")
    # with open(kmodel_path, 'wb') as f:
    #     f.write(kmodel)
    # print("----------------end-----------------")
    return kmodel_path


def main(model_path, dump_path, input_file):

    # calib_file = "/media/curio/500_disk/model/issue_model/DF/calib.jpg"
    input_data = [nncase.RuntimeTensor.from_numpy(
        np.fromfile(input_file, dtype=np.int64).reshape(1, 256))]

    calib_data = [input_data]

    config = AutoConfig.from_pretrained(model_path + "/config.json")
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    head_dim = config.head_dim if hasattr(
        config, "head_dim") else config.hidden_size // config.num_attention_heads
    kv_type = np.dtype(to_np_type(config.torch_dtype))
    block_size = 256
    num_blocks = 16
    max_sessions = 16
    max_model_len = (block_size * num_blocks) // max_sessions
    kv_cache_config = nncase.PagedAttentionConfig(
        num_layers,
        num_kv_heads,
        head_dim,
        kv_type,
        block_size,
        [nncase.PagedKVCacheDimKind.NumBlocks,
            nncase.PagedKVCacheDimKind.NumLayers,
            nncase.PagedKVCacheDimKind.KV,
            nncase.PagedKVCacheDimKind.BlockSize,
            nncase.PagedKVCacheDimKind.NumKVHeads,
            nncase.PagedKVCacheDimKind.HeadDim],
        [nncase.PagedKVCacheDimKind.HeadDim],
        [128 // 2],
        [nncase.PagedKVCacheDimKind.NumBlocks],
        [[0]],  # [[1], [2, 3]]
    )

    evaluator = compile_kmodel(model_path, dump_path, calib_data, kv_cache_config)

    # seq_length = np.array(256).astype(kv_type)
    scheduler = nncase._nncase.RefPagedAttentionScheduler(
        kv_cache_config, num_blocks, max_model_len, [1])
    kv_cache_obj = scheduler.schedule([0], [256])
    input_data.append(kv_cache_obj.as_ivalue())

    for idx, i in enumerate(input_data):
        if isinstance(i, nncase.RuntimeTensor):
            value = nncase._nncase.RTValue.from_runtime_tensor(i)
        elif isinstance(i, nncase._nncase.IValue):
            value = i
        evaluator.set_input_tensor(idx, value)
        # print(i.shape)
        # i.tofile(os.path.join(dump_path, "input_{}_0.bin".format(idx)))

    evaluator.run()

    for i in range(evaluator.outputs_size):
        result = evaluator.get_output_tensor(i).to_numpy()
        print(result.shape)
        result.tofile(os.path.join(dump_path, "nncase_result_{}.bin".format(i)))
        np.save(os.path.join(dump_path, "nncase_result_{}.npy".format(i)), result)


if __name__ == "__main__":
    model_path = "/Users/curio/Canaan/nncase/tests/llm/Qwen/Qwen2.5-0.5B-Instruct"
    dump_path = "/Users/curio/Canaan/nncase/tests_output/qwen/"
    input_file = "/Users/curio/Canaan/nncase/tests_output/test_qwen2/input/input_0.bin"
    main(model_path, dump_path, input_file)

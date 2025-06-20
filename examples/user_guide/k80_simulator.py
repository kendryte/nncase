import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os

from sklearn.metrics.pairwise import cosine_similarity

import nncase


def get_cosine(vec1, vec2):
    """
    result compare
    """
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))


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

def from_text(path):
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), "..", path)
        data = []
        with open(path, "r") as f:
            for i in f.readlines():
                data.append(i.strip("\n").strip("\""))
        return data

def run_eval(interpreter, input_data):
    for idx, i in enumerate(input_data):
        if isinstance(i, nncase._nncase.IValue):
            value = i
        else:
            value = nncase._nncase.RTValue.from_runtime_tensor(nncase.RuntimeTensor.from_numpy(i))
        interpreter.set_input_tensor(idx, value)

    interpreter.run()
    results = []
    for i in range(interpreter.outputs_size):
        result = interpreter.get_output_tensor(i).to_numpy()
        # print(result.shape)
        result.tofile(os.path.join(dump_path, "nncase_result_{}.bin".format(i)))
        np.save(os.path.join(dump_path, "nncase_result_{}.npy".format(i)), result)
        results.append(result)

    return results

def run_kmodel(kmodel_path, input_data):
    print("\n---------start run kmodel---------")
    print("Load kmodel...")
    model_sim = nncase.Simulator()
    with open(kmodel_path, 'rb') as f:
        model_sim.load_model(f.read())

    print("Set input data...")
    for i, data in enumerate(input_data):
        if isinstance(data, nncase._nncase.IValue):
            value = data
        else:
            value = nncase._nncase.RTValue.from_runtime_tensor(nncase.RuntimeTensor.from_numpy(data))
        model_sim.set_input_tensor(i, value)

    print("Run...")
    model_sim.run()

    print("Get output result...")
    all_result = []
    for i in range(model_sim.outputs_size):
        result = model_sim.get_output_tensor(i).to_numpy()
        all_result.append(result)
    print("----------------end-----------------")
    return all_result

def compile_kmodel(model_path, dump_path, config, is_eval: bool):
    """
    Set compile options and ptq options.
    Compile kmodel.
    Dump the compile-time result to 'compile_options.dump_dir'
    """
    print("\n----------   compile    ----------")
    # model_file = model_path

    print("Set options...")
    # import_options
    import_options = nncase.ImportOptions()

    import_options.huggingface_options.output_logits = True
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


    print("Compiling...")
    compiler = nncase.Compiler(compile_options)

    compiler.import_huggingface(model_path, import_options)

    if is_eval:
        evaluator = compiler.create_evaluator(3)
        return evaluator

    # # compile
    compiler.compile()
    kmodel = compiler.gencode_tobytes()
    print("Write kmodel...")
    kmodel_path = os.path.join(dump_path, "test.kmodel")
    with open(kmodel_path, 'wb') as f:
        f.write(kmodel)
    print("----------------end-----------------")
    return kmodel_path

def prepare_tokens(tokenizer, input_file):
    data = from_text(input_file)

    # messages = [
    #     {"role": "system", "content": "You are a assistant!"},
    #     {"role": "user", "content": data[0]}
    # ]
    text = tokenizer.apply_chat_template(
        data[0],
        tokenize=False,
        add_generation_prompt=True
    )
    data = tokenizer([text], return_tensors="np").input_ids[0]

    return data

    evaluator = compile_kmodel(model_path, dump_path, calib_data, kv_cache_config)

    # seq_length = np.array(256).astype(kv_type)
    scheduler = nncase._nncase.RefPagedAttentionScheduler(
        kv_cache_config, num_blocks, max_model_len, [1])
    kv_cache_obj = scheduler.schedule([0], [256])
    input_data.append(kv_cache_obj.as_ivalue())

def main(model_path, dump_path, input_file, is_eval):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    '''
    convert txt to tokens
    '''
    tokens = prepare_tokens(tokenizer, input_file)
    input_data = []

    '''
    set paged attention config.
    '''
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



    ''' TODO:
    use cpp paged attention schedule instead of refpaged.
    '''


    if is_eval:
        interpreter = compile_kmodel(model_path, dump_path, kv_cache_config, is_eval)
        while(1):
            input_data.append(tokens)
            scheduler = nncase._nncase.RefPagedAttentionScheduler(
            kv_cache_config, num_blocks, max_model_len, [1])
            kv_cache_obj = scheduler.schedule([0], [input_data[0].shape[0]])

            input_data.append(kv_cache_obj.as_ivalue())
            result = run_eval(interpreter, input_data)
            # print(result[0].shape)
            new_tokens_index = torch.argmax(torch.from_numpy(result[0]), dim=-1, keepdim=True)
            new_tokens = tokenizer.batch_decode(new_tokens_index, skip_special_tokens=False)
            print(*new_tokens, end=' ')
            tokens = new_tokens_index.cpu().detach().numpy()[-1]
            input_data = []
            # input_data.append()
    else:
        kmodel_path = compile_kmodel(model_path, dump_path, kv_cache_config, False)
        result = run_kmodel(kmodel_path, input_data)


if __name__ == "__main__":
    model_path = "/Users/curio/Canaan/nncase/tests/llm/Qwen/Qwen2.5-0.5B-Instruct"
    dump_path = "/Users/curio/Canaan/nncase/tests_output/qwen/"
    prompt_file = "/Users/curio/Canaan/nncase/tests/importer/huggingface_/prompt.txt"
    is_eval = True
    main(model_path, dump_path, prompt_file, is_eval)

import os

def download_from_huggingface(model_api, tokenizer_api, model_name):
    print(f" Downloading \033[32m\033[1m {model_name} \033[0m from huggingface ... ")
    model_dir = os.path.join(os.path.dirname(__file__), model_name)
    
    try:
        model = model_api.from_pretrained(model_name)
        tokenizer = tokenizer_api.from_pretrained(model_name)
    except Exception as e:
        raise os.error(f"\033[31m Download {model_name} has error. Make sure it's a valid repository. Or check your network!\033[0m")
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"\033[32m\033[1m {model_name} \033[0m has been downloaded into \033[34m\033[5m {model_dir} \033[0m")
    return model_dir
    
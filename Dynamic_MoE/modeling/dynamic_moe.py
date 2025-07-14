from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='/mnt/data/models/dynamic_moe',
        tokenizer_path='/mnt/data/models/dynamic_moe',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype='auto'
        ),
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
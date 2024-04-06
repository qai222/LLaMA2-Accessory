## ORD Structured data extraction

All scripts here were taken from the [LLaMA-Accessory project](https://github.com/Alpha-VLLM/LLaMA2-Accessory), either
directly or after minor modifications.

### Setting up
1. install `apex` follow the instruction
   1. if encountering CUDA mismatch problem, see potential solutions from [this issue](https://github.com/NVIDIA/apex/pull/323#issuecomment-652382641).
   2. try to reverse to an earlier version when the compilation fails, see this [issue](https://github.com/NVIDIA/apex/issues/1735#issuecomment-1743179030).
2. include the accessory folder in the `PYTHONPATH`.
3. download the USPTO-ORD-100K dataset into this folder.

### Finetuning
1. The main script used is [finetune_adapter.sh](finetune_adapter.sh)
which passes arguments to [finetune_adapter.py](finetune_adapter.py).
2. Parameters for LLaMA Adapter is defined in [llamaAdapter.json](llamaAdapter.json).
3. Data path is defined in [ord_data_config_exp1.yaml](ord_data_config_exp1.yaml).

### Inference
The main script is [infer.sh](infer.sh), which uses either
[infer_batch.py](infer_batch.py) to run batch inferences, or open a
Gradio app using [infer_app.py](infer_app.py).

### Misc
1. To monitor gpu state, try `pip install gpustat` then
   ```shell
   tmux new -d -s 'gpu' 'watch -t -n 30 "gpustat | tee -a gpu.log"'
   ```
2. For 13B model additional params need to be specified in [llamaAdapter.json](llamaAdapter.json)
   ```json
   {
   "lora_rank": -1,
   "bias_tuning": true,
   "use_prefix_new_gate": false,
   "prefix_layers": 30,
   "prefix_len": 10,
   "trainable_mode": "sg",
   "dim": 5120,
   "multiple_of": 256,
   "n_heads": 40,
   "n_layers": 40,
   "norm_eps": 1e-05,
   "vocab_size": -1
   }
   ```
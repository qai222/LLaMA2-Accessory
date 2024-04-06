import argparse
import json
import random
from itertools import zip_longest

import numpy as np
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init
from tqdm import tqdm

from accessory.model.meta import MetaModel
from accessory.util import misc


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pretrained checkpoints')
    parser.add_argument('--llama_type', default=None, type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default=None, type=str, nargs="*",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='path to tokenizer.model')
    parser.add_argument('--test_json_path', type=str, default=None,
                        help='path to test.json')
    parser.add_argument('--output_path', type=str, default=None,
                        help='path to export inferences')
    parser.add_argument('--infer_batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="The dtype used for model weights and inference.")

    parser.add_argument('--dist_on_itp', action='store_true')
    return parser


args = get_args_parser().parse_args()

# define the model
random.seed(0)
torch.random.manual_seed(0)
np.random.seed(0)
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(dist.get_world_size())
target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]

model = MetaModel.from_pretrained(args.pretrained_path, args.llama_type, args.llama_config, args.tokenizer_path,
                                  with_visual=False, max_seq_len=2048,
                                  mp_group=fs_init.get_model_parallel_group(),
                                  dtype=target_dtype, device="cuda", )

print("Model = %s" % str(model))
model.bfloat16().cuda()


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


@torch.inference_mode()
def generate_one_batch(
        prompts: list[str],
        max_gen_len=2048,
        gen_t=0, top_p=1
):
    dist.barrier()
    with torch.cuda.amp.autocast(dtype=target_dtype):
        results = model.generate(prompts, images=None, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    return results


def load_test_json(batch_size=2):
    with open(args.test_json_path, "r") as f:
        test_data = json.load(f)
    batches = [*grouper(batch_size, test_data)]
    print("# of batches:", len(batches))
    return batches


def generate_batches():
    for ibatch, batch in tqdm(enumerate(load_test_json(args.infer_batch_size))):
        prompts = [d['instruction'] for d in batch if d is not None]
        outputs = generate_one_batch(prompts)
        output_json = str(ibatch).rjust(6, '0') + ".json"
        with open(args.output_path + "/" + output_json, "w") as f:
            json.dump(outputs, f)


if __name__ == '__main__':
    generate_batches()

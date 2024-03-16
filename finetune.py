import argparse
import copy
import torch
import torch.distributed as torch_distributed
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from tqdm import tqdm
from transformers import set_seed, default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from deepspeed.accelerator import get_accelerator
from accelerate.utils import DeepSpeedPlugin
import wandb
from megatron_lm.megatron.global_vars import set_global_variables
from src.utils.train_utils import (
    clear_gpu_cache,
    setup_environ_flags,
    train,
)
from src.utils.checkpoint import (
    load_model_state_dict,
    load_optimizer_state_dict,
    load_scheduler_state_dict,
    load_rng_state_dict,
    get_latest_iteration,
)


from src.optimizer import WarmupCosineAnnealingLR
from src.utils.distributed import (
    print_rank_0,
    is_rank_0,
    set_mpi_env,
    get_rank,
    get_local_rank
)



from src.arguments import parse_args

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def main():
    args = parse_args()

    set_global_variables(args=args)

    # Set the seeds for reproducibility
    set_seed(seed=args.seed)

    # Distributed args.
    if args.use_mpi:
        set_mpi_env()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.rank = rank
    args.world_size = world_size
    args.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    get_accelerator().set_device(get_local_rank())  # type: ignore

    # torch_distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    deepPlugin = DeepSpeedPlugin(
        hf_ds_config=args.zero_config,
        zero3_init_flag=True if args.zero_stage == 3 else False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.grad_clip_norm,
        zero_stage=args.zero_stage,
    )

    accelerator = Accelerator(
        mixed_precision='bf16' if args.bf16 else 'fp16',
        deepspeed_plugin=deepPlugin,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        even_batches=False,
    )

    # wandb setting
    if args.wandb_name and is_rank_0():
        import datetime

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(get_local_rank())  # type: ignore
        clear_gpu_cache(get_local_rank())  # type: ignore
        setup_environ_flags(get_rank())  # type: ignore
    

    iteration: int = get_latest_iteration(args.load)
    args.iteration = iteration
    torch_distributed.barrier()

    # random seed
    if args.load:
        load_rng_state_dict(args.load)
        torch_distributed.barrier()


    if args.architecture == "llama":
        from scaled_rope.modeling_llama_yarn import LlamaForCausalLM
        from scaled_rope.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        original_max_position_embeddings = args.original_max_position_embeddings if args.original_max_position_embeddings else 4096
    elif args.architecture == "mistral":
        from scaled_rope.modeling_mistral_yarn import MistralForCausalLM
        from scaled_rope.configuration_mistral import MistralConfig
        config_cls = MistralConfig
        model_cls = MistralForCausalLM
        original_max_position_embeddings = args.original_max_position_embeddings if args.original_max_position_embeddings else 8192

    config = config_cls.from_pretrained(args.base_model)
    config.rope_scaling = {
        "type": args.scaling_type,
        "factor": args.scaling_factor,
        "original_max_position_embeddings": original_max_position_embeddings
    }
    config.rope_theta = args.rope_theta
    config.max_position_embeddings = int(args.scaling_factor * original_max_position_embeddings) \
        if not args.max_position_embeddings else args.max_position_embeddings

    sliding_window_attention_schedule = [int(x) for x in args.sliding_window_attention_schedule.split(",")] \
        if args.sliding_window_attention_schedule else None
    if sliding_window_attention_schedule is not None and len(sliding_window_attention_schedule) == 1:
        config.sliding_window = sliding_window_attention_schedule[0]
        if is_rank_0():
            print(
             f"Sliding attention window set to {config.sliding_window}")


    model = model_cls.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        config=config,
        use_flash_attention_2=True
    )
    # dataset
    from src.datasets.pretrain_dataset import build_train_valid_test_datasets
    from megatron_lm.megatron.data.data_samplers import build_pretraining_data_loader
    train_dataset, validation_dataset, test_dataset = build_train_valid_test_datasets()
    
    args.consumed_train_samples = args.global_batch_size * args.iteration
    args.consumed_valid_samples = args.global_batch_size * (
        args.iteration // args.eval_interval) * args.eval_iters

    train_dataloader = build_pretraining_data_loader(
        dataset=train_dataset,
        consumed_samples=args.consumed_train_samples,
    )
    validation_dataloader = build_pretraining_data_loader(
        dataset=validation_dataset,
        consumed_samples=args.consumed_valid_samples,
    )
    torch_distributed.barrier()

    model.config.use_cache = False

    model.gradient_checkpointing_enable(  # type: ignore
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()  # type: ignore
    print_rank_0("Gradient checkpointing enable")


    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.bf16:
        model.to(torch.bfloat16)  # type: ignore
    elif args.fp16:
        model.to(torch.float16)  # type: ignore
    model.train()  # type: ignore

    optimizer = optim.AdamW(
        model.parameters(),  # type: ignore
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if args.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=args.lr_warmup_iters,
            decay_iterations=args.lr_decay_iters,
            max_iterations=args.train_iters,
            eta_min=args.min_lr,
        )
    else:   
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    if args.load:
        load_scheduler_state_dict(scheduler, args.load)  # type: ignore
    
    # ref: https://github.com/microsoft/DeepSpeed/pull/5008#issuecomment-1910607845
    model, optimizer, _, _, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
        scheduler,
    )
    if args.load:
        load_model_state_dict(model, args.load)  # type: ignore

    # Start the training process
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        accelerator=accelerator,
        local_rank=get_local_rank(),
        rank=get_rank(),
    )


if __name__ == "__main__":
    main()

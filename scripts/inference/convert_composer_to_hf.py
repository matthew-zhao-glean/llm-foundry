# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.distributed._shard.checkpoint import (
    load_state_dict,
    FileSystemReader,
)
import transformers
from composer.models.huggingface import get_hf_config_from_composer_state_dict
from composer.utils import (
    get_file,
    maybe_create_object_store_from_uri,
    parse_uri,
    safe_torch_load,
)
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase, LlamaForCausalLM, LlamaConfig
import yaml

from llmfoundry import MPTConfig, MPTForCausalLM
from llmfoundry.utils import get_hf_tokenizer_from_composer_state_dict
from llmfoundry.utils.checkpoint_conversion_helpers import load_tokenizer
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config=model_config)
    return model


def load_sharded_model_single_gpu(model,model_path):
    state_dict = {
        "model": model.state_dict()
    }

    load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )

    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def load_sharded_ckpt(
    fsdp_checkpoint_path: str="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path: str="", # Path to save the HF converted model checkpoints
    # Path/ name of the HF model that include config.json and tokenizer_config.json
    # (e.g. meta-llama/Llama-2-7b-chat-hf)
    hf_model_path_or_name: str="",
):
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file with specified encoding
        with open(train_params_path, 'r', encoding='utf-8') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            hf_model_path_or_name = data.get('model_name')

            print(f"Model name: {hf_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        hf_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {hf_model_path_or_name}")

    #load the HF model definition from config
    model_def = load_llama_from_config(hf_model_path_or_name)
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
    return model, tokenizer


def write_huggingface_pretrained_from_composer_checkpoint(
    checkpoint_path: Union[Path, str],
    output_path: Union[Path, str],
    trust_remote_code: bool,
    output_precision: str = 'fp32',
    local_checkpoint_save_location: Optional[Union[Path, str]] = None,
    is_sharded_checkpoint: bool = False,
    hf_model_path_or_name: str="",
) -> Tuple[PretrainedConfig, Optional[PreTrainedTokenizerBase]]:
    """Convert a Composer checkpoint to a pretrained HF checkpoint folder.

    Write a ``config.json`` and ``pytorch_model.bin``, like
    :meth:`transformers.PreTrainedModel.from_pretrained` expects, from a
    composer checkpoint.

    .. note:: This function will not work properly if you used surgery algorithms when you trained your model. In that case you will want to
        load the model weights using the Composer :class:`~composer.Trainer` with the ``load_path`` argument.
    .. testsetup::
        import torch
        dataset = RandomTextClassificationDataset(size=16, use_keys=True)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        import transformers
        from composer.models import HuggingFaceModel
        from composer.trainer import Trainer
        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        composer_model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=[], use_logits=True)
        trainer = Trainer(model=composer_model,
                            train_dataloader=train_dataloader,
                            save_filename='composer-hf-checkpoint.pt',
                            max_duration='1ep',
                            save_folder='./')
        trainer.fit()
        trainer.close()

    Example:
    .. testcode::
        from composer.models import write_huggingface_pretrained_from_composer_checkpoint
        write_huggingface_pretrained_from_composer_checkpoint('composer-hf-checkpoint.pt', './hf-save-pretrained-output')
        loaded_model = transformers.AutoModelForSequenceClassification.from_pretrained('./hf-save-pretrained-output')

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        output_path (Union[Path, str]): Path to the folder to write the output to.
        trust_remote_code (bool): Whether or not to use code outside of the transformers module.
        output_precision (str, optional): The precision of the output weights saved to `pytorch_model.bin`. Can be one of ``fp32``, ``fp16``, or ``bf16``.
        local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                Defaults to None, which will use a temporary file.
    """
    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[output_precision]

    tmp_dir = None
    # default local path to a tempfile if path is not provided
    if local_checkpoint_save_location is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_checkpoint_save_location = Path(
            tmp_dir.name,
        ) / 'local-composer-checkpoint.pt'

    # create folder
    os.makedirs(output_path)

    # download the checkpoint file
    print(
        f'Downloading checkpoint from {checkpoint_path} -> {local_checkpoint_save_location}',
    )
    if is_sharded_checkpoint:
        if not tmp_dir:
            tmp_dir = tempfile.TemporaryDirectory()
        checkpoint_metadata_path = checkpoint_path + '/.metadata'
        local_checkpoint_metadata_save_location = Path(tmp_dir.name) / '.metadata'
        get_file(checkpoint_metadata_path, local_checkpoint_metadata_save_location)
        for i in range(8):
            local_checkpoint_save_location = Path(tmp_dir.name) / f'__{i}_0.distcp'
            get_file(
                checkpoint_path + f'/__{i}_0.distcp',
                local_checkpoint_save_location,
            )
        return load_sharded_ckpt(tmp_dir.name, output_path, hf_model_path_or_name)

    get_file(str(checkpoint_path), str(local_checkpoint_save_location))

    # Load the Composer checkpoint state dict
    print('Loading checkpoint into CPU RAM...')
    composer_state_dict = safe_torch_load(local_checkpoint_save_location)

    if 'state' not in composer_state_dict:
        raise RuntimeError(
            f'"state" is not an available key in the provided composer checkpoint. Is {local_checkpoint_save_location} ill-formed?',
        )

    # Build and save HF Config
    print('#' * 30)
    print('Saving HF Model Config...')
    hf_config = get_hf_config_from_composer_state_dict(composer_state_dict)
    hf_config.torch_dtype = dtype
    hf_config.save_pretrained(output_path)
    print(hf_config)

    # Extract and save the HF tokenizer
    print('#' * 30)
    print('Saving HF Tokenizer...')
    hf_tokenizer = get_hf_tokenizer_from_composer_state_dict(
        composer_state_dict,
        trust_remote_code,
    )
    if hf_tokenizer is not None:
        hf_tokenizer.save_pretrained(output_path)
        print(hf_tokenizer)
    else:
        print('Warning! No HF Tokenizer found!')

    # Extract the HF model weights
    print('#' * 30)
    print('Saving HF Model Weights...')
    weights_state_dict = composer_state_dict
    if 'state' in weights_state_dict:
        weights_state_dict = weights_state_dict['state']['model']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        weights_state_dict,
        prefix='model.',
    )

    # Convert weights to desired dtype
    for k, v in weights_state_dict.items():
        if isinstance(v, torch.Tensor):
            weights_state_dict[k] = v.to(dtype=dtype)

    # Save weights
    torch.save(weights_state_dict, Path(output_path) / 'pytorch_model.bin')

    print('#' * 30)
    print(f'HF checkpoint folder successfully created at {output_path}.')

    return hf_config, hf_tokenizer


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert a HuggingFace causal LM in a Composer checkpoint into a standard HuggingFace checkpoint folder, and optionally upload to the hub.',
    )
    parser.add_argument('--composer_path', type=str, required=True)
    parser.add_argument('--hf_output_path', type=str, required=True)
    parser.add_argument(
        '--local_checkpoint_save_location',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--output_precision',
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
        default='fp32',
    )
    parser.add_argument('--hf_repo_for_upload', type=str, default=None)
    parser.add_argument('--test_uploaded_model', action='store_true')
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Whether or not to use code outside of transformers module.',
    )
    parser.add_argument('--is_sharded_checkpoint', action='store_true')
    parser.add_argument('--hf_model_path_or_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

    return parser.parse_args()


def _convert_composer_to_hf(args: Namespace) -> None:
    print()
    print('#' * 30)
    print('Converting Composer checkpoint to HuggingFace checkpoint format...')

    _, _, local_folder_path = parse_uri(args.hf_output_path)

    config, tokenizer = write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path=args.composer_path,
        output_path=local_folder_path,
        trust_remote_code=args.trust_remote_code,
        output_precision=args.output_precision,
        local_checkpoint_save_location=args.local_checkpoint_save_location,
        is_sharded_checkpoint=args.is_sharded_checkpoint,
        hf_model_path_or_name=args.hf_model_path_or_name,
    )

    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[args.output_precision]

    print(f'Loading model from {local_folder_path}')
    if config.model_type == 'mpt':
        config.attn_config['attn_impl'] = 'torch'
        config.init_device = 'cpu'

    if config.model_type == 'mpt':
        loaded_hf_model = MPTForCausalLM.from_pretrained(
            local_folder_path,
            config=config,
            torch_dtype=dtype,
        )
    else:
        loaded_hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            local_folder_path,
            config=config,
            torch_dtype=dtype,
        )

    delattr(loaded_hf_model.config, '_name_or_path')

    loaded_hf_model.save_pretrained(local_folder_path)

    print(f'Loading tokenizer from {local_folder_path}')

    tokenizer = load_tokenizer(
        local_folder_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(local_folder_path)

    # Only need to edit files for MPT because it has custom code
    if config.model_type == 'mpt':
        print('Editing files for HF compatibility...')
        edit_files_for_hf_compatibility(local_folder_path)

    object_store = maybe_create_object_store_from_uri(str(args.hf_output_path))

    if object_store is not None:
        print(
            f'Uploading HF checkpoint folder from {local_folder_path} -> {args.hf_output_path}',
        )
        for file in os.listdir(local_folder_path):
            remote_file = os.path.join(local_folder_path, file)
            local_file = os.path.join(local_folder_path, file)
            object_store.upload_object(remote_file, local_file)

    print(
        'Composer checkpoint successfully converted to HuggingFace checkpoint format.',
    )


def convert_composer_to_hf(args: Namespace) -> None:
    # Register MPT auto classes so that this script works with MPT
    # This script will not work without modification for other custom models,
    # but will work for other HuggingFace causal LMs
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
    MPTConfig.register_for_auto_class()
    MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

    try:
        _convert_composer_to_hf(args)
    except Exception as e:
        raise e
    finally:
        # Undo auto registration after running the script
        del CONFIG_MAPPING._extra_content['mpt']
        delattr(MPTConfig, '_auto_class')
        delattr(MPTForCausalLM, '_auto_class')


if __name__ == '__main__':
    convert_composer_to_hf(parse_args())

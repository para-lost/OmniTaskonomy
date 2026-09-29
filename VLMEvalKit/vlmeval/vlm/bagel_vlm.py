import os
import sys
import yaml
import torch
import random
import numpy as np
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class BAGEL(BaseModel):
    """BAGEL VLM implementation for VLMEvalKit.
    
    BAGEL is a unified multimodal foundation model that supports both visual understanding
    and generation tasks.
    """

    INSTALL_REQ = False
    INTERLEAVE = False  # BAGEL processes images separately from text

    def __init__(
        self, 
        model_path='hf/BAGEL-7B-MoT/', 
        use_cpu_offload=False,
        max_memory_per_gpu="40GiB",
        offload_folder="/tmp/bagel_offload",
        use_ema=True,
        **kwargs
    ):
        """Initialize BAGEL model.
        
        Args:
            model_path (str): Path to the model checkpoint directory.
            use_cpu_offload (bool): Whether to use CPU offload for multi-GPU inference.
                If True, uses accelerate's device_map to distribute model across GPUs and CPU.
            max_memory_per_gpu (str): Maximum memory per GPU when using CPU offload (e.g., "40GiB").
            offload_folder (str): Folder path for offloading weights to disk.
            use_ema (bool): Whether to use EMA weights. If True, loads 'ema.safetensors',
                otherwise loads 'model.safetensors'. Default is True.
            **kwargs: Additional generation config parameters.
        """
        assert model_path is not None, "model_path must be provided"
        self.model_path = model_path
        self.ori_model_path = os.environ.get('BAGEL_ORI_MODEL_PATH', model_path)
        self.use_cpu_offload = use_cpu_offload
        self.use_ema = use_ema
        # Add bagel directory to Python path for correct imports
        bagel_dir = os.path.join(os.path.dirname(__file__), 'bagel')
        if bagel_dir not in sys.path:
            sys.path.insert(0, bagel_dir)
        
        # Import BAGEL components
        from .bagel.modeling.bagel import (
            BagelConfig, 
            Bagel, 
            Qwen2Config, 
            Qwen2ForCausalLM, 
            SiglipVisionConfig, 
            SiglipVisionModel,
        )
        from .bagel.modeling.qwen2 import Qwen2Tokenizer
        from .bagel.data.data_utils import add_special_tokens
        from .bagel.data.transforms import ImageTransform
        from safetensors.torch import load_file
        
        # Load configurations
        llm_config = Qwen2Config.from_json_file(os.path.join(self.ori_model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.ori_model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            latent_patch_size=2,
            max_latent_size=64,
            connector_act='gelu_pytorch_tanh',
        )
        
        # Build model
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        
        if use_cpu_offload:
            # Use accelerate for CPU offload (multi-GPU inference with memory management)
            from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
            
            with init_empty_weights():
                language_model_empty = Qwen2ForCausalLM(llm_config)
                vit_model_empty = SiglipVisionModel(vit_config)
                self.model = Bagel(language_model_empty, vit_model_empty, config)
                self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
            
            # Infer device map for multi-GPU distribution
            # print(torch.cuda.device_count())
            device_map = infer_auto_device_map(
                self.model,
                max_memory={i: max_memory_per_gpu for i in range(torch.cuda.device_count())},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )
            
            # Ensure certain modules stay on the same device
            same_device_modules = [
                'language_model.model.embed_tokens',
                'time_embedder',
                'latent_pos_embed',
                'vae2llm',
                'llm2vae',
                'connector',
                'vit_pos_embed'
            ]
            
            if torch.cuda.device_count() == 1:
                first_device = device_map.get(same_device_modules[0], "cuda:0")
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device
                    else:
                        device_map[k] = "cuda:0"
            else:
                first_device = device_map.get(same_device_modules[0])
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device
            
            # Load checkpoint and dispatch to devices
            checkpoint_name = "ema.safetensors" if self.use_ema else "model.safetensors"
            model_state_dict_path = os.path.join(model_path, checkpoint_name)
            self.model = load_checkpoint_and_dispatch(
                self.model,
                checkpoint=model_state_dict_path,
                device_map=device_map,
                offload_buffers=True,
                dtype=torch.bfloat16,
                force_hooks=True,
                offload_folder=offload_folder
            )
            print(f"Model loaded with CPU offload using {checkpoint_name}. Device map: {device_map}")
        else:
            # Standard loading (original behavior)
            self.model = Bagel(language_model, vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
            
            # Load model weights
            checkpoint_name = "ema.safetensors" if self.use_ema else "model.safetensors"
            model_state_dict_path = os.path.join(model_path, checkpoint_name)
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            msg = self.model.load_state_dict(model_state_dict, strict=False)
            print(f"Model loaded from {checkpoint_name}: {msg}")
            del model_state_dict
            
            # Move model to GPU and set to eval mode
            self.model = self.model.cuda()
        
        # Set model to eval mode
        self.model = self.model.eval()

        # Load tokenizer and add special tokens
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.ori_model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # Build image transform
        self.image_transform = self._build_transform()
        
        # Default generation config
        self.default_kwargs = dict(
            max_length=100,
            do_sample=False,        # 改为True以支持采样
            temperature=1.0,       # 设置合理的temperature
        )
        self.default_kwargs.update(kwargs)
        
        # Seed for reproducibility (can be set per-inference)
        self.current_seed = None
        
        torch.cuda.empty_cache()
        print(f"BAGEL model initialized from {model_path}")
        print(f"Generation config: do_sample={self.default_kwargs['do_sample']}, "
              f"temperature={self.default_kwargs['temperature']}")

    def _build_transform(self):
        """Build image transformation pipeline."""
        from .bagel.data.transforms import ImageTransform
        
        # Find the config file
        config_paths = [
            os.path.join(os.path.dirname(__file__), "bagel/data/configs/example.yaml"),
            "./data/configs/example.yaml",
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            # Use default parameters if config file not found
            return ImageTransform(
                max_image_size=980,
                min_image_size=336,
                image_stride=14,
                max_pixels=980*980,
            )
        
        with open(config_path, "r") as f:
            data_config = yaml.safe_load(f)

        max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
        min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
        image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
        max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

        return ImageTransform(
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            image_stride=image_stride,
            max_pixels=max_pixels,
        )

    def set_seed(self, seed):
        """Set random seed for reproducible sampling.
        
        Args:
            seed (int): Random seed value.
        """
        self.current_seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Note: full determinism also requires:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # But this may impact performance
    
    def adjust_kwargs(self, dataset):
        """Adjust generation kwargs based on dataset type.
        
        Args:
            dataset (str): Dataset name.
            
        Returns:
            dict: Adjusted generation kwargs.
        """
        import copy
        kwargs = copy.deepcopy(self.default_kwargs)
        
        # Special case for CV-Bench-3D which requires long thinking
        if 'CV-Bench-3D' in dataset or 'CV_Bench_3D' in dataset:
            kwargs['max_length'] = 1000
            return kwargs
        
        dataset_type = DATASET_TYPE(dataset)
        
        if dataset_type in ['MCQ', 'Y/N']:
            kwargs['max_length'] = 32
        elif dataset_type == 'Caption':
            if 'COCO' in dataset:
                kwargs['max_length'] = 32
            else:
                kwargs['max_length'] = 100
        elif dataset_type == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_length'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_length'] = 10
            else:
                kwargs['max_length'] = 100
        
        return kwargs

    def generate_inner(self, message, dataset=None):
        """Generate response for the given message.
        
        Args:
            message (list[dict]): Multi-modal message with interleaved images and text.
                Each dict has 'type' (image/text) and 'value' (path/string).
            dataset (str, optional): Dataset name for adjusting generation parameters.
            
        Returns:
            str: Generated response.
        """
        from .bagel.data.data_utils import pil_img2rgb
        from PIL import Image
        
        # Extract images and text from message
        images = []
        text_parts = []
        
        for item in message:
            if item['type'] == 'image':
                # Load and convert image
                img_path = item['value']
                if isinstance(img_path, str):
                    img = Image.open(img_path).convert('RGB')
                else:
                    img = img_path
                img = pil_img2rgb(img)
                images.append(img)
            elif item['type'] == 'text':
                text_parts.append(item['value'])
        
        # Concatenate text prompt
        prompt = '\n'.join(text_parts)
        
        # Add dataset-specific prompt suffix
        if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            prompt += ' Answer:'
        
        # Adjust generation parameters based on dataset
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.default_kwargs
        
        # Generate response using BAGEL's chat method
        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
                image_transform=self.image_transform,
                images=images,
                prompt=prompt,
                max_length=kwargs['max_length'],
                do_sample=kwargs.get('do_sample', False),
                temperature=kwargs.get('temperature', 1.0),
            )
            return response.strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""

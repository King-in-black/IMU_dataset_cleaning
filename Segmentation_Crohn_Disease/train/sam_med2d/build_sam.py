# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import partial
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, ImageEncoderViTWithParallelBranch, MaskDecoderWithParallelBranch, ParallelCNNBranch,ResNetFirstStage
from torch.nn import functional as F
import torch.nn
def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter = args.encoder_adapter,
        reduce_ratio = args.reduce_ratio
        
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter = args.encoder_adapter,
        reduce_ratio = args.reduce_ratio
    )


def build_sam_vit_b(args):
    if not args.parallel_cnn:
        return _build_sam(
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            image_size=args.image_size,
            checkpoint=args.sam_checkpoint,
            encoder_adapter = args.encoder_adapter,
            reduce_ratio = args.reduce_ratio,
            adapter_mlp_ratio = args.adapter_mlp_ratio,
        )
    else:
        return _build_sam_parrallel_cnn(
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            image_size=args.image_size,
            checkpoint=args.sam_checkpoint,
            encoder_adapter = args.encoder_adapter,
            reduce_ratio = args.reduce_ratio,
        )
        # return _build_sam_parallel_cnn_mask_from_scratch(
        #     encoder_embed_dim=768,
        #     encoder_depth=12,
        #     encoder_num_heads=12,
        #     encoder_global_attn_indexes=[2, 5, 8, 11],
        #     image_size=args.image_size,
        #     checkpoint=args.sam_checkpoint,
        #     encoder_adapter = args.encoder_adapter,
        #     reduce_ratio = args.reduce_ratio,
        # )
        


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def initialize_weights(model):
    """
    Initialize weights for all layers in the model.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)

    print("All weights initialized using PyTorch defaults.")

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
    reduce_ratio,
    adapter_mlp_ratio,
):
    """
    Build the SAM model with optional checkpoint loading.
    """
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # 创建 SAM 模型
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train=encoder_adapter,
            reduce_ratio=reduce_ratio,
            adapter_mlp_ratio=adapter_mlp_ratio,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # 如果 checkpoint 存在，加载权重；否则初始化权重
    if checkpoint is not None:
        try:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict.keys():
                sam.load_state_dict(state_dict["model"], strict=False)
            else:
                sam.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint}.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        # 随机初始化模型权重
        initialize_weights(sam)
        print("Initialized SAM model from scratch (no checkpoint provided).")

    return sam



def _build_sam_parrallel_cnn(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
    reduce_ratio
):
    """
    Build the SAM model with a parallel CNN branch, supporting optional checkpoint loading.
    """
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # Parallel CNN branch
    parallel_cnn_branch = ResNetFirstStage(input_channels=3, output_channels=256)

    # Image encoder with parallel branch
    image_encoder = ImageEncoderViTWithParallelBranch(
        parallel_cnn_branch=parallel_cnn_branch,
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        adapter_train=encoder_adapter,
        reduce_ratio=reduce_ratio,
    )

    # Prompt encoder
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    # Mask decoder
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256
    )

    # Combine into the SAM model
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Checkpoint loading or weight initialization
    if checkpoint is not None:
        try:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
            if 'model' in state_dict.keys():
                print(f"Loading model weights from checkpoint with adapter: {encoder_adapter}")
                sam.load_state_dict(state_dict['model'], strict=False)
            else:
                sam.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Falling back to default weight initialization.")
            initialize_weights(sam)
    else:
        print("No checkpoint provided. Initializing weights from scratch.")
        initialize_weights(sam)

    return sam



def _build_sam_parallel_cnn_mask_from_scratch(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
    reduce_ratio
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    parallel_cnn_branch = ParallelCNNBranch(input_channels=3)
    image_encoder = ImageEncoderViTWithParallelBranch(
        parallel_cnn_branch=parallel_cnn_branch,
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        adapter_train=encoder_adapter,
        reduce_ratio=reduce_ratio,
    )
    
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    
    mask_parallel_branch = MaskDecoderWithParallelBranch(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256
    )
    
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_parallel_branch,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        try:
            if 'model' in state_dict.keys():
                print(encoder_adapter)
                state_dict = state_dict['model']
            
            # Remove the mask decoder parameters from the state_dict
            state_dict = {k: v for k, v in state_dict.items() if 'mask_decoder' not in k}
            
            if image_size == 1024 and encoder_adapter:
                sam.load_state_dict(state_dict, False)
            else:
                sam.load_state_dict(state_dict)
        except:
            print('*******interpolate')
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
        print(f"*******load {checkpoint}")
        
    return sam
def load_from(sam, state_dicts, image_size, vit_patch_size):

    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if 
                                                        '2' in k or 
                                                        '5' in k or 
                                                        '7' in k or 
                                                        '8' in k or 
                                                        '11' in k or 
                                                        '13' in k or
                                                        '15' in k or 
                                                        '23' in k or 
                                                        '31' in k] 
        # print(sam_dict)
        for k in global_rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict


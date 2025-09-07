import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from frozendict import frozendict

from sglang.srt.layers.moe.fused_moe_triton.kernel_config import KernelConfigs
from sglang.srt.utils import get_device_name, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()


class FusedMoeKernelConfig(KernelConfigs):
    kernel_name: str = "fused_moe_kernel"

    @classmethod
    def get_default_config(
        cls,
        M: int,
        E: int,
        N: int,
        K: int,
        topk: int,
        dtype: Optional[str],
        is_marlin: bool,
        block_shape: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        if dtype == "fp8_w8a8":
            if block_shape is None:
                config = {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 32,
                    "num_warps": 8,
                    "NUM_STAGE": 2 if _is_hip else 4,
                }
                if M <= E:
                    config = {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 128,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                        "NUM_STAGE": 2 if _is_hip else 4,
                    }
            else:
                # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": block_shape[0],
                    "BLOCK_SIZE_K": block_shape[1],
                    "GROUP_SIZE_M": 32,
                    "num_warps": 4,
                    "NUM_STAGE": 2 if _is_hip else 3,
                }
        else:
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "NUM_STAGE": 1,
            }
            # A heuristic: fused marlin works faster with this config for small M
            if M <= E or (is_marlin and M <= 32):
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "NUM_STAGE": 1,
                }
        return config

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_optimal_kernel_config(
        cls,
        M: int,
        N: int,
        K: int,
        topk_num: int,
        expert_num: int,
        mul_routed_weight: bool,
        use_fp8_w8a8: bool,
        out_dtype: str,
        is_marlin: bool,
        block_shape: Optional[List[int]] = None,
    ) -> dict:
        key_params = {
            "N": N,
            "K": K,
            "topk_num": topk_num,
            "expert_num": expert_num,
            "mul_routed_weight": mul_routed_weight,
            "use_fp8_w8a8": use_fp8_w8a8,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        found_config = cls.get_the_config(key_params)

        if found_config:
            config = found_config[
                min(found_config.keys(), key=lambda x: abs(int(x) - M))
            ]
            return config
        else:
            config = cls.get_default_config(
                M, expert_num, N, K, topk_num, out_dtype, is_marlin, block_shape
            )
        return config

    @classmethod
    def save_config(
        cls,
        N: int,
        K: int,
        topk_num: int,
        expert_num: int,
        mul_routed_weight: bool,
        use_fp8_w8a8: bool,
        out_dtype: str,
        config_json: dict,
    ):
        key_params = {
            "N": N,
            "K": K,
            "topk_num": topk_num,
            "expert_num": expert_num,
            "mul_routed_weight": mul_routed_weight,
            "use_fp8_w8a8": use_fp8_w8a8,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)

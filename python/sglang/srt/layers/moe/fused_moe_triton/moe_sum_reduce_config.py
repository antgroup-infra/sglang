from functools import lru_cache

from frozendict import frozendict

from sglang.srt.layers.moe.fused_moe_triton.kernel_config import KernelConfigs


class MoeSumReduceKernelConfig(KernelConfigs):
    kernel_name: str = "moe_sum_reduce_kernel"

    @staticmethod
    def _generate_key_params(
        topk_num: int, hidden_dim: int, out_dtype: str
    ) -> frozendict:
        """
        Generates a frozendict with key parameters used for configuration lookups.
        """
        return frozendict(
            {
                "topk_num": topk_num,
                "hidden_dim": hidden_dim,
                "out_dtype": str(out_dtype),
            }
        )

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_optimal_kernel_config(
        cls, M: int, topk_num: int, hidden_dim: int, out_dtype: str
    ) -> dict:
        """
        Attempts to retrieve the best configuration based on the parameters.
        """
        key_params = cls._generate_key_params(topk_num, hidden_dim, out_dtype)

        # Try to fetch the configuration from stored ones
        found_config = cls.get_the_config(key_params)

        if found_config:
            # Find the closest match to M
            closest_key = min(found_config.keys(), key=lambda x: abs(int(x) - M))
            return found_config[closest_key]
        else:
            # Default configuration if no match is found
            return {
                "BLOCK_M": 1,
                "BLOCK_DIM": 2048,
                "NUM_STAGES": 1,
                "NUM_WARPS": 16,
            }

    @classmethod
    def save_config(
        cls, topk_num: int, hidden_dim: int, out_dtype: str, config_json: dict
    ):
        """
        Saves the configuration with the provided parameters.
        """
        key_params = cls._generate_key_params(topk_num, hidden_dim, out_dtype)
        return cls.store_config(key_params, config_json)

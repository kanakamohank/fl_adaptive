import math
import hashlib
import torch
from typing import Dict

class EphemeralStructuredProjection:
    def __init__(
        self,
        target_k: int,
        model_blocks: Dict[str, int],  
        master_key: bytes = b'default_paper_key',
        k_min_per_block: int = 16,
    ):
        self.model_blocks = model_blocks  
        self.master_key = master_key      
        self.M = len(model_blocks)        

        # Every block gets at least k_min_per_block dimensions.
        #
        # The detector's statistic is dist^2 / sigma^2, which under a null of
        # honest clients behaves as chi^2(k_m)/k_m: its dispersion is set
        # ENTIRELY by k_m, with sd = sqrt(2/k_m). At k_m = 1 that is a single
        # squared Gaussian -- sd 1.41, and it exceeds five times its own mean
        # 2.5% of the time by chance alone. Since a client is flagged when ANY
        # block fires, ten blocks at k_m = 1 give a false-positive rate near
        # 1 - (1 - 0.025)^10 ~ 22% with no adversary present.
        #
        # The previous allocation was proportional to block size with a floor of
        # 1, which gave k_m = 1 to six of ten blocks on the evaluated model. No
        # choice of tau_z fixes that: the problem is the number of measurements,
        # not where the line is drawn on them. At k_m = 16, sd falls to 0.35 and
        # the per-block false-positive rate becomes negligible.
        self.k_min_per_block = max(1, k_min_per_block)
        self.target_k = max(target_k, self.M * self.k_min_per_block)
        self.k_m_allocations = self._allocate_k_m()
        
        # ---> CACHING FIX: Store matrices to prevent 234 GB RAM explosion <---
        self._cached_matrices = {}
        self._cached_round = -1

    def _allocate_k_m(self) -> Dict[str, int]:
        total_d = sum(self.model_blocks.values())
        # Floor first, then distribute what remains in proportion to block size.
        allocations = {block_name: self.k_min_per_block for block_name in self.model_blocks}

        leftover_k = self.target_k - self.M * self.k_min_per_block
        remaining_leftover = leftover_k

        # The leftover lands on the LARGEST block rather than the last one.
        # Previously it went to whichever block happened to be declared last,
        # so on the evaluated model a 10-parameter bias block received four
        # dimensions while a 5120-parameter weight block received one --
        # declaration order, not size, decided the resolution each block was
        # examined at.
        largest = max(self.model_blocks, key=self.model_blocks.get)
        for block_name, d_m in self.model_blocks.items():
            if block_name == largest:
                continue
            extra_k = int(leftover_k * (d_m / total_d))
            allocations[block_name] += extra_k
            remaining_leftover -= extra_k
        allocations[largest] += remaining_leftover

        return allocations

    def _get_secondary_seed(self, block_name: str, round_num: int) -> int:
        seed_data = f"{self.master_key.decode()}_{round_num}_{block_name}".encode()
        return int(hashlib.sha256(seed_data).hexdigest()[:8], 16)

    def _get_or_create_r_m(self, block_name: str, round_num: int, device: torch.device) -> torch.Tensor:
        """Generates the massive JL matrix ONCE per round and caches it."""
        if self._cached_round != round_num:
            # New round! Clear the old matrices from RAM
            self._cached_matrices.clear()
            self._cached_round = round_num
            
        if block_name not in self._cached_matrices:
            d_m = self.model_blocks[block_name]
            k_m = self.k_m_allocations[block_name]
            
            seed = self._get_secondary_seed(block_name, round_num)
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            
            std_dev = 1.0 / math.sqrt(k_m)
            # Generate the matrix just one time
            r_m = torch.normal(mean=0.0, std=std_dev, size=(k_m, d_m), generator=gen, device=device)
            self._cached_matrices[block_name] = r_m
            
        return self._cached_matrices[block_name]

    def generate_and_project_block(self, block_name: str, g_i_m: torch.Tensor, round_num: int) -> torch.Tensor:
        # Fetch the cached matrix instead of building it from scratch
        r_m = self._get_or_create_r_m(block_name, round_num, g_i_m.device)
        return torch.matmul(r_m, g_i_m.view(-1))

    def project_client_update(self, client_update: Dict[str, torch.Tensor], round_num: int) -> Dict[str, torch.Tensor]:
        projected_update = {}
        for block_name, g_i_m in client_update.items():
            if block_name in self.model_blocks:
                projected_update[block_name] = self.generate_and_project_block(block_name, g_i_m, round_num)
        return projected_update
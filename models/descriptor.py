from typing import Literal, Optional
import torch
class LimeDesc:
    def __init__(
            self,
            deltas_layers: list,
            lambda_kl: float,
            approach: Literal['opt', 'vanila'],
            modality_bos_idx: int,
            modality_eos_idx: int,
            prompt_len: int,
):
        self.deltas_layers = deltas_layers
        self.lambda_kl = lambda_kl
        self.approach = approach
        self.modality_bos_idx = modality_bos_idx
        self.modality_eos_idx = modality_eos_idx
        self.prompt_len = prompt_len
        self.kv_deltas: dict
        self.reference_logits: Optional[torch.Tensor]
        self.kl_loss: Optional[float]
        
    def set_kv_deltas(self, deltas: torch.tensor):
        self.kv_deltas = deltas

    def set_reference_logits(self, logits: Optional[torch.Tensor]):
        self.reference_logits = logits
    
    def set_kl_loss(self, loss: float):
        self.kl_loss = loss
    
    def kv_deltas_cleanup(self):
        for layer_idx in self.deltas_layers:
            dk, dv = self.kv_deltas[layer_idx]

            dk.grad = None
            dv.grad = None

            del dk
            del dv

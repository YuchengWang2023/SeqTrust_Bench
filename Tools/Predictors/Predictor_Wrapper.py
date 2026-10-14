from enformer_pytorch import str_to_one_hot
import sys
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch


def get_weight_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

class BatchedPredictor(pl.LightningModule):
    def __init__(self, predictors, batch_size: int = 512, config_name: str = "alphagenome", use_no_grad: bool = True):
        super().__init__()
        self.models = torch.nn.ModuleList(predictors)
        self.batch_size = batch_size
        self.config_name = config_name  # Add config_name to handle different input shapes
        self.use_no_grad = use_no_grad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor, shape (N, 4, L)
        return: torch.Tensor, shape (N, n_task), keep computational graph
        """

        # Predictor Uniformly accept input (N, L, 4)
        x = x.permute(0, 2, 1)

        n = x.shape[0]
        all_outputs = []
        for model in self.models:
            outputs = []

            if self.use_no_grad:
                with torch.no_grad():
                    for i in range(0, n, self.batch_size):
                        xb = x[i : i + self.batch_size]
                        pred = model(xb)
                        outputs.append(pred)
            else:
                for i in range(0, n, self.batch_size):
                    xb = x[i : i + self.batch_size]
                    pred = model(xb)
                    outputs.append(pred)

            pred_full = torch.cat(outputs, dim=0)  # (N, 1 or T)
            all_outputs.append(pred_full)

        out = torch.stack(all_outputs, dim=-1)  # (N, T, K) or (N, K)
        if out.dim() == 3 and out.shape[1] == 1:
            out = out.squeeze(1)  # (N, K)

        return out

def build_batched_predictor(
    model: str = "alphagenome",
    model_config: str = "./",
    model_ckpt: str = "./",
    length: int = 200,
    device_id: int = 0,
    batch_size: Optional[int] = None,
    use_no_grad: bool = True, 
    whether_softplus: bool = False, 
) -> Tuple[BatchedPredictor, object, torch.device]:
    

    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    )

    from Seq2Func import Predictor

    if isinstance(model_ckpt, str):
        model_ckpt = [model_ckpt]
    if isinstance(model_config, str):
        model_config = [model_config]
    assert len(model_ckpt) == len(model_config)
    predictors = []

    for ckpt, config in zip(model_ckpt, model_config):
        predictor = Predictor(
            model_type=model,
            pretrained_path=config,
            length=length,
            device=device, 
            whether_softplus=whether_softplus
        ).to(device)
        before = get_weight_vector(predictor).clone()
        predictor.load_ckpt(ckpt)
        after = get_weight_vector(predictor)
        diff = torch.mean(torch.abs(before - after)).item()
        print(f"[CKPT LOAD CHECK] diff = {diff:.6e}")
        if diff < 1e-6:
            print("WARNING: ckpt may NOT be loaded (weights unchanged)")
        else:
            print(f"ckpt loaded successfully from {model_ckpt}")
        predictor.eval()
        predictors.append(predictor)

    batched_predictor = BatchedPredictor(
        predictors,
        batch_size=batch_size,
        config_name=model, 
        use_no_grad=use_no_grad
    ).to(device)

    return batched_predictor
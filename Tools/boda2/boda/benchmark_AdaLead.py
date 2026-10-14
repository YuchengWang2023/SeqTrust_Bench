import os
import sys
import argparse
import pandas as pd
import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--length", type=int, default=200)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--predict_model", type=str, required=True)
    parser.add_argument("--predict_model_config", type=str, nargs="+", default=None)
    parser.add_argument("--predict_model_ckpt", type=str, nargs="+", default=None)
    parser.add_argument("--target_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)

    # AdaLead params
    parser.add_argument("--n_proposals", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--rho", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--energy_threshold", type=float, default=-2)
    parser.add_argument("--recomb_rate", type=float, default=0.1)
    parser.add_argument("--max_attempts", type=int, default=1000)

    parser.add_argument("--whether_softplus", type=bool, default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Unknown args ignored: {unknown}")
    return args


def main():
    args = get_args()

    print("\n===== CODA AdaLead Config =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("================================\n")
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    print(f"[Device] Using device: {args.device}")


    # =====================
    # PATH setup
    # =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    boda_path = os.path.normpath(os.path.join(script_dir, "..", "..", "..","Tools", "boda2"))
    sys.path.insert(0, boda_path)

    import boda

    predictor_path = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "Tools", "Predictors"))
    sys.path.insert(0, predictor_path)

    from Predictor_Wrapper import build_batched_predictor

    utils_path = os.path.normpath(os.path.join(script_dir, "..", "..", "..","Tools", "Utils"))
    sys.path.insert(0, utils_path)

    from utils import onehot_to_seq
    
    # Output paths derived from save_dir:
    #   <save_dir>/model.pt          (torch checkpoint)
    #   <save_dir>/generate_seqs.csv (generated sequences)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "model.pt")
    output_path = os.path.join(args.save_dir, "generate_seqs.csv")

    # =====================
    # Running
    # =====================
    print("Running AdaLead...")

    logits = torch.randn(args.batch_size, 4, args.length).to(args.device)
    # sampler
    params = boda.generator.parameters.PassThroughParameters(data=logits)
    params.to(args.device)
    
    predictor = build_batched_predictor(
        model=args.predict_model,
        model_config=args.predict_model_config,
        model_ckpt=args.predict_model_ckpt,
        length=args.length,
        device_id=0,
        batch_size=args.batch_size,
        use_no_grad=True,
        whether_softplus=args.whether_softplus,
    )

    # objective
    energy = boda.generator.MinGapEnergy(predictor, target_feature=args.target_index)

    # generator
    AL = boda.generator.AdaLead(energy, params)

    results = AL.generate(
        n_proposals=args.n_proposals,
        n_steps=args.n_steps,
        rho=args.rho,
        threshold=args.threshold,
        mu=args.mu,
        energy_threshold=args.energy_threshold,
        recomb_rate=args.recomb_rate,
        n_top_seqs_per_batch=1,
        model_queries_per_batch=10 * args.batch_size,
        max_attempts=args.max_attempts,
    )

    print("Design finished!")

    # =====================
    # SAVE CKPT
    # =====================
    torch.save(results, save_path)
    print(f"Ckpt saved: {save_path}")

    final_seqs = onehot_to_seq(results["proposals"].cpu().numpy())

    df_sequences = pd.DataFrame({"sequence": final_seqs})
    df_sequences.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Shape: {df_sequences.shape}")


if __name__ == "__main__":
    main()
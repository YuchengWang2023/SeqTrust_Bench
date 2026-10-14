import os
import argparse
import pandas as pd
import numpy as np
import torch
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--label_len", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="3", help="Prefix prompt used for generation.")

    parser.add_argument("--max_num", type=int, default=2000000)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--val_check_interval", type=int, default=10)


    parser.add_argument("--sample_number", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=2)
    

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Unknown args ignored: {unknown}")
    return args


def main():
    args = get_args()

    print("\n===== RegLM Benchmark Config =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==================================\n")

    # =====================
    # PATH setup
    # =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reglm_path = os.path.normpath(os.path.join(script_dir, "..", "..", "Tools", "regLM", "src"))
    sys.path.insert(0, reglm_path)

    import reglm.dataset
    import reglm.lightning


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output paths derived from save_dir:
    #   <save_dir>/model.pt          (torch checkpoint)
    #   <save_dir>/generate_seqs.csv (generated sequences)
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, "generate_seqs.csv")

    # =====================
    # Running
    # =====================
    print("Loading data...")

    data = pd.read_csv(args.data_path, dtype={"label": str}, sep='\t')
    if args.max_num < len(data):
        data = data.sample(n=args.max_num, random_state=42)
    else:
        print(f"max_num is above Data num {len(data)}, no sampling")


    data = data[['Sequence', 'label']].copy()
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(data)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = data.iloc[:n_train]
    val = data.iloc[n_train:n_train + n_val]
    test = data.iloc[n_train + n_val:]
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ===== dataset =====
    train_ds = reglm.dataset.CharDataset(
        train.Sequence.tolist(),
        labels=train.label.tolist()
    )

    val_ds = reglm.dataset.CharDataset(
        seqs=val.Sequence.tolist(),
        labels=val.label.tolist()
    )

    print("Dataset ready")

    # ===== config =====
    config = {
        'd_model': 32,
        'n_layer': 2,
        'd_inner': 32,
        'vocab_size': 12,
        'pad_vocab_size_multiple': 8,
        'return_hidden_state': True,
        'layer': {
            'emb_dim': 5,
            'filter_order': 64,
            'l_max': train_ds.seq_len + train_ds.label_len + 1,
            '_name_': 'hyena'
        }
    }

    print("Running regLM...")

    model = reglm.lightning.LightningModel(
        config=config,
        ckpt_dir="/home/wangyucheng/projects/SeqEval_Bench_ExternalSource/huggingface/LongSafari/hyenadna-medium-160k-seqlen",
        hyenadna_path="/home/wangyucheng/projects/SeqEval_Bench_ExternalSource/github/hyena-dna",
        save_dir=args.save_dir,
        lr=1e-3,
        label_len=args.label_len
    ).to(device)

    model.train_on_dataset(
        train_ds,
        val_ds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=8,
        device=0,
        val_check_interval=args.val_check_interval
    )

    print("Training finished")

    
    print("Generating sequences...")

    ckpt_path = os.path.join(
        args.save_dir,
        "lightning_logs/version_0/checkpoints"
    )

    #  checkpoint
    ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")]
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")

    ckpt_files.sort()
    ckpt_file = os.path.join(ckpt_path, ckpt_files[-1])

    print(f"Using checkpoint: {ckpt_file}")

    # model = reglm.lightning.LightningModel.load_from_checkpoint(
    #     ckpt_dir=ckpt_file,
    #     model="hyenadna-medium-160k-seqlen",
    #     hyenadna_path="/home/wangyucheng/projects/SeqEval_Bench_ExternalSource/github/hyena-dna",
    #     )

    # Do not use LightningModel.load_from_checkpoint:
    # torch>=2.6 defaults to weights_only=True, and checkpoints saved by the old
    # pytorch_lightning contain numpy scalars, which are rejected by the weights-only
    # unpickler. The old PL load_from_checkpoint cannot pass weights_only=False through.
    # Since the checkpoint is produced by this very script (trusted source), load it
    # manually and rebuild the model + load_state_dict instead.
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    # Note: must match training time -- build the backbone from the pretrained
    # hyenadna weights via ckpt_dir (in __init__, ckpt_dir takes precedence over
    # config, so the backbone is hyenadna-medium, not the small config)
    model = reglm.lightning.LightningModel(
        ckpt_dir=hp["ckpt_dir"],
        hyenadna_path=hp["hyenadna_path"],
        label_len=hp.get("label_len", args.label_len),
    )
    model.load_state_dict(ckpt["state_dict"])

    model = model.to(device)

    prompts = [args.prompt for _ in range(args.sample_number)]

    final_seqs = model.generate(
        prompts,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k
    )

    df_sequences = pd.DataFrame({'sequence': final_seqs})
    df_sequences.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Shape: {df_sequences.shape}")


if __name__ == "__main__":
    main()
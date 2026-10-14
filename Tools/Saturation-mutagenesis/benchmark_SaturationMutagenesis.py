import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
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
    parser.add_argument("--n_mutation", type=int, default=20)

    parser.add_argument("--whether_softplus", type=bool, default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Unknown args ignored: {unknown}")
    return args


def main():
    args = get_args()

    print("\n===== Saturation mutagenesis Config =====")
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
    predictor_path = os.path.normpath(os.path.join(script_dir, "..", "..", "Tools", "Predictors"))
    sys.path.insert(0, predictor_path)

    from Predictor_Wrapper import build_batched_predictor

    from scripts.utils import insilico_evolution

    utils_path = os.path.normpath(os.path.join(script_dir, "..", "..", "Tools", "Utils"))
    sys.path.insert(0, utils_path)

    from utils import onehot_to_seq

    # Output paths derived from save_dir:
    #   <save_dir>/model.pkl          (torch checkpoint)
    #   <save_dir>/generate_seqs.csv (generated sequences)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "model.pkl")
    output_path = os.path.join(args.save_dir, "generate_seqs.csv")


    # =====================
    # Running
    # =====================
    print("Running In Silico Evolution...")

    evolved_seq_dict = {}
    evolved_seq_dict["X"] = np.eye(4, dtype=np.int8)[
        np.random.randint(0, 4, size=(args.batch_size, args.length))
    ]

    # predictor
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

    # generator
    evolved_seq_dict["mut_pred"], evolved_seq_dict["mut_loc"] = insilico_evolution(
        regions = evolved_seq_dict["X"], 
        model = predictor, 
        class_no = args.target_index,
        n_mutation = args.n_mutation,
        device_id = 0
    )

    print("Design finished!")

    # =====================
    # SAVE CKPT
    # =====================
    f = open(save_path, "wb")
    pickle.dump(evolved_seq_dict,f)
    f.close()
    print(f"Ckpt saved: {save_path}")

    with open(save_path, "rb") as f:
        evolved_seq_dict = pickle.load(f)
    n_sequences = evolved_seq_dict["X"].shape[0]
    nuc_to_onehot = {"A":[1, 0, 0, 0],"C":[0, 1, 0, 0],"G":[0, 0, 1, 0],"T":[0, 0, 0, 1]}
    final_seqs = []
    for id_ in range(n_sequences):
        start_x = np.copy(evolved_seq_dict["X"][id_:id_+1])
        for i, mut_ in enumerate(evolved_seq_dict["mut_loc"][id_]):
            start_x[0][int(mut_.split("_")[0]),:] = np.array(nuc_to_onehot[mut_.split("_")[1]], dtype='int8')
        seq = onehot_to_seq(start_x)
        final_seqs.extend(seq)
    df_sequences = pd.DataFrame({'sequence': final_seqs})
    df_sequences.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Shape: {df_sequences.shape}")


if __name__ == "__main__":
    main()
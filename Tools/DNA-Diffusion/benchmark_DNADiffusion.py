import argparse
import os
import subprocess
import random


def get_args():

    parser = argparse.ArgumentParser()


    # =========================
    # dataset
    # =========================

    parser.add_argument(
        "--data_path",
        type=str,
        required=True
    )


    parser.add_argument(
        "--saved_data_path",
        type=str,
        required=True
    )


    parser.add_argument(
        "--save_dir",
        type=str,
        required=True
    )


    # =========================
    # GPU
    # =========================

    parser.add_argument(
        "--device_id",
        type=str,
        default="0"
    )


    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1
    )


    parser.add_argument(
        "--train_script",
        type=str,
        default="train.py"
    )


    args, _ = parser.parse_known_args()

    return args





def run_cmd(cmd, env):

    print("\nCMD:")
    print(" ".join(cmd))

    print(
        "\nCUDA_VISIBLE_DEVICES:",
        env["CUDA_VISIBLE_DEVICES"]
    )


    subprocess.run(
        cmd,
        env=env,
        check=True
    )
    # subprocess.run(
    #     cmd,
    #     env=env,
    #     check=True,
    #     start_new_session=True
    # )





def main():


    args = get_args()


    env = os.environ.copy()



    # =========================
    # GPU setting
    # =========================

    gpu_list = args.device_id.split(",")


    if len(gpu_list) != args.num_gpus:

        raise ValueError(
            f"""
            GPU configuration error:

            device_id:
            {args.device_id}

            num_gpus:
            {args.num_gpus}

            Number of GPUs mismatch!
            """
        )


    env["CUDA_VISIBLE_DEVICES"] = args.device_id



    print(
        "\n===== DNADiffusion ====="
    )


    for k,v in vars(args).items():

        print(
            f"{k}: {v}"
        )


    print(
        "========================\n"
    )



    if not os.path.exists(
        args.data_path
    ):

        raise FileNotFoundError(
            args.data_path
        )

    master_port = random.randint(
        20000,
        60000
    )

    # =========================
    # build command
    # =========================

    cmd = [

        "uv",
        "run",

    ]


    if args.num_gpus > 1:


        cmd += [

            "torchrun",

            "--nproc_per_node",

            str(args.num_gpus),

            "--master_port",
            str(master_port),

        ]


    else:

        cmd += [

            "python",

        ]



    cmd += [

        args.train_script,


        f"data.data_path={args.data_path}",


        "data.load_saved_data=False",


        f"data.saved_data_path={args.saved_data_path}",


        f"training.checkpoint_dir={args.save_dir}",


        
        "training.distributed=False"


    ]



    run_cmd(
        cmd,
        env
    )


    print(
        "Training finished!"
    )





if __name__ == "__main__":

    main()
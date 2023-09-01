import argparse
import os
import time


def main(args):
    import torch
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    from PIL import Image
    from zoedepth.utils.misc import pil_to_batched_tensor, get_image_from_url, save_raw_16bit, colorize

    system_type = os.name
    if args.demo:
        # Display input image
        if system_type == 'nt':
            os.system(f"start {args.image_path}")
        elif system_type == 'posix':
            os.system(f"open {args.image_path}")
        else:
            print("unknown system type for input image")

    start_time = time.time()
    model_path = 'compiled_model.pth'
    if not os.path.exists(model_path):
        print("Building and caching the model...")
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
        conf = get_config(args.model_name, "infer", config_version=args.config_version)
        model = build_model(conf)
        torch.save(model.state_dict(), model_path)
    else:
        print(f"Loading cached model from {model_path}")
        conf = get_config(args.model_name, "infer", config_version=args.config_version)
        model = build_model(conf)  # Initialize the model architecture
        model.load_state_dict(torch.load(model_path))  # Load the saved state

    model.to(args.device)

    end_load_time = time.time()
    print(f"Model loaded in {end_load_time - start_time} seconds.")

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        depth = model.infer_pil(image)

    if args.url:
        image = get_image_from_url(args.url)
        depth = model.infer_pil(image)

    # save_raw_16bit(depth, args.output_path)

    colored = colorize(depth)
    if not args.output_colored_path:
        args.output_colored_path = args.output_path.split(".")[0] + "_colored.png"
    Image.fromarray(colored).save(args.output_colored_path)
    print(f"inference time: {time.time() - end_load_time} seconds")
    print(f"total time: {time.time() - start_time} seconds")
    print(f"output path: {args.output_colored_path}")
    # use the default program to execute the output file

    if args.demo:
        system_type = os.name
        if system_type == 'nt':
            os.system(f"start {args.output_colored_path}")
        elif system_type == 'posix':
            os.system(f"open {args.output_colored_path}")
        else:
            print("unknown system type")


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description='Depth Inference.')
    parser.add_argument('--model_name', type=str, default="zoedepth", help='Name of the model to use.')
    parser.add_argument('--config_version', type=str, default=None, help='Configuration version.')
    parser.add_argument('--image_path', type=str, default=None, help='Path to the input image.')
    parser.add_argument('--url', type=str, default=None, help='Image URL for inference.')
    parser.add_argument('--output_path', type=str, default="/path/to/output.png", help='Output path for raw depth.')
    parser.add_argument('--output_colored_path', type=str, default=None, help='Output path for colored depth.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use.')
    # flag for running demo
    parser.add_argument('--demo', action='store_true', help='Run demo.')

    # if we are in demo mode then we display the input image and the output image
    # otherwise we do not

    args = parser.parse_args()
    main(args)

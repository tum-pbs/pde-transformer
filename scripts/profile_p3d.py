import time
import torch
from torch import autocast
from torchprofile import profile_macs
import warnings

from tqdm import tqdm
from pdetransformer.core.models.three_dimensional import RegionModel, ContextModel, P3D_S
from pdetransformer.core.models.three_dimensional.context_model import ContextEncoderConfig


def profile_model(model_class, model_args, data_size: int = 128, **kwargs):

    batch_size = 1
    channel_size = 1

    print(f"***** DATA SIZE {data_size} *****")

    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    torch.cuda.empty_cache()  # Clear any cached memory
    start_mem = torch.cuda.memory_allocated()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(*model_args, **kwargs).to(device)

    large_input_tensor = torch.randn(batch_size, channel_size,
                                     data_size, data_size, data_size,
                                     device=device)

    total_params = sum(p.numel() for p in model.parameters())
    start_time = time.time()

    with torch.no_grad():
        # run the model a few times to warm up the GPU
        for _ in range(10):
            _ = model(large_input_tensor).reconstructed

        NUM_SAMPLES = 25
        start_time_1 = time.time()
        for _ in tqdm(range(NUM_SAMPLES)):
            _ = model(large_input_tensor).reconstructed

        end_time_1 = time.time()

        print(f'Throughput: {NUM_SAMPLES * batch_size / (end_time_1 - start_time_1)}/s')
        peak_mem = torch.cuda.max_memory_allocated()
        print(f'Peak Memory Allocated (inference): {peak_mem / 1e6:.2f} MB')

    print(f'Total Params: {total_params / 1e6:.2f} M')

    flops = profile_macs(model, (large_input_tensor))
    print(f'G FLOPS: {flops / 1e9:.2f}')

    print(f'Computing backward pass...')

    out_tensor = model(large_input_tensor).reconstructed

    # loss and backward pass
    out_tensor.sum().backward()

    print(f'Output tensor shape: {out_tensor.shape}')

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated()

    print(f'Time Taken: {end_time - start_time:.4f} s')
    print(f'Memory Used: {(peak_mem - start_mem) / 1e6:.2f} MB')
    print(f'Peak Memory Allocated: {peak_mem / 1e6:.2f} MB')

    model_summary = {
        "in_channels": channel_size,
        "batch_size": batch_size,
        "data_size": data_size,
        "total_params": total_params / 1e6,
        "flops": flops / 1e9,
        "time_taken": end_time - start_time,
        "memory_used": (peak_mem - start_mem) / 1e6,
        "peak_memory_allocated": peak_mem / 1e6,
        "name: ": model_class.__name__
    }

    return model_summary


def profile_region_model(
        model,
        data_size: int = 192,
        crop_size: int = 64,
        grad_ratio_encoder: float = 0.0,
        grad_ratio_decoder: float = 0.1,
):

    batch_size = 1
    channel_size = 1

    print(f"***** CROP SIZE {crop_size} | DATA SIZE {data_size} *****")

    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    torch.cuda.empty_cache()  # Clear any cached memory
    start_mem = torch.cuda.memory_allocated()

    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.cuda()

        context_network_config = ContextEncoderConfig(
            n_layer=3,
            attention_method='hyper',
            num_heads=8,
            hidden_size=512,
        )

        region_model = RegionModel(model,
                         context_network_config,
                         crop_size = crop_size,
                         num_regions= 8,
                         emb_dims = [64, 128],
                         ).to(device)

        large_input_tensor = torch.randn(batch_size, channel_size,
                                         data_size, data_size, data_size,
                                         device=device)

        total_params = sum(p.numel() for p in region_model.parameters())
        start_time = time.time()

        with torch.no_grad():
            # run the model a few times to warm up the GPU
            for _ in range(10):
                _ = region_model(large_input_tensor,
                                 grad_ratio_encoder=grad_ratio_encoder,
                                 grad_ratio_decoder=grad_ratio_decoder
                                 ).reconstructed

            NUM_SAMPLES = 25
            start_time_1 = time.time()
            for _ in tqdm(range(NUM_SAMPLES)):
                _ = region_model(large_input_tensor,
                                 grad_ratio_encoder=grad_ratio_encoder,
                                 grad_ratio_decoder=grad_ratio_decoder
                                 ).reconstructed

            end_time_1 = time.time()

            print(f'Throughput: {NUM_SAMPLES * batch_size / (end_time_1 - start_time_1)}/s')
            peak_mem = torch.cuda.max_memory_allocated()
            print(f'Peak Memory Allocated (inference): {peak_mem / 1e6:.2f} MB')

        print(f'Total Params: {total_params / 1e6:.2f} M')

        flops = profile_macs(region_model, (large_input_tensor, None, None, None,
                                            grad_ratio_encoder,
                                            grad_ratio_decoder
                                            ))
        print(f'G FLOPS: {flops / 1e9:.2f}')

        print(f'Computing backward pass...')

        out_tensor = region_model(large_input_tensor,
                                   grad_ratio_encoder=grad_ratio_encoder,
                                   grad_ratio_decoder=grad_ratio_decoder
                                  ).reconstructed

        # loss and backward pass
        out_tensor.sum().backward()

        print(f'Output tensor shape: {out_tensor.shape}')

        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated()

        print(f'Time Taken: {end_time - start_time:.4f} s')
        print(f'Memory Used: {(peak_mem - start_mem) / 1e6:.2f} MB')
        print(f'Peak Memory Allocated: {peak_mem / 1e6:.2f} MB')

        model_summary = {
            "in_channels": channel_size,
            "batch_size": batch_size,
            "data_size": data_size,
            "total_params": total_params / 1e6,
            "flops": flops / 1e9,
            "time_taken": end_time - start_time,
            "memory_used": (peak_mem - start_mem) / 1e6,
            "peak_memory_allocated": peak_mem / 1e6,
            "name: ": region_model.__class__
        }

        return model_summary

    except Exception as e:

        print(f"Error: {e}")

if __name__ == "__main__":

    in_channels = 1
    data_size = 128
    crop_size = 64

    models = {
        'P3D_S': P3D_S,
    }

    models_args = {
        'P3D_S': (in_channels, None,
                  False, 1, False, True, [True, True, True],
                  4, True),
    }

    summary_list = []

    warnings.filterwarnings("ignore")

    # with autocast("cuda"):

    for model_desc in models:

        print(f"************ MODEL {model_desc} ************")

        profile_model(models[model_desc],
                      models_args[model_desc],
                      data_size,
        )

        print(f"************ REGION MODEL {model_desc} ************")

        model = models[model_desc](*models_args[model_desc])

        profile_region_model(model,
                             data_size=data_size,
                             crop_size=crop_size,
                             grad_ratio_encoder=0.0,
                             grad_ratio_decoder=0.1,
                            )


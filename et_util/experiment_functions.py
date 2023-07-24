import os
import subprocess as sp


def mask_unused_gpus(leave_unmasked=1):
    """
    Masks all unused GPUs, except for the amount specified.
    """

    acceptable_available_memory = 1024
    command = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)
                          if x > acceptable_available_memory]

        if len(available_gpus) < leave_unmasked:
            raise ValueError(f'Found only {len(available_gpus)} usable GPUs in the system')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
    except Exception as exception:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', exception)


def set_wandb_key(key: str):
    """
    Sets the API key to connect with the Weights and Biases server.
    """

    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

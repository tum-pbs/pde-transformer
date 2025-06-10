from ..utils import instantiate_from_config


def get_callbacks(config):

    callbacks = []

    for key, value in config.items():
        try:
            callbacks.append(instantiate_from_config(value))
        except Exception as e:
            print(f"Could not create callback: {key}")
            print(e)

    return callbacks
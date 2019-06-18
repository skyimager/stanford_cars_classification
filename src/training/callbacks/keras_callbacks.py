import os
import keras
from src.utils.utils import makedirs
from src.training.callbacks import RedirectModel
from src.training.callbacks.eval import Evaluate


def create_callbacks(model, training_model, prediction_model, validation_generator, config):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if config["tensorboard-dir"]:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = config["tensorboard-dir"],
            histogram_freq         = 0,
            batch_size             = config["batch-size"],
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if validation_generator:
        evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=config["weighted-average"])
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    makedirs(config["snapshot-path"])
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            config["snapshot-path"],
            '{backbone}_{{epoch:02d}}.h5'.format(backbone=config["backbone"])
        ),
        verbose=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks
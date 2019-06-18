
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.append(os.path.abspath('./src/networks'))

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def get_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.8,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

from config import retinanet as config
from src import networks
from src.training.csv_generator import create_generators

from src.networks.retinanet import retinanet_bbox
from src.utils.anchors import make_shapes_callback
from src.training.callbacks.keras_callbacks import create_callbacks
from src.networks.get_model import create_models


if __name__ == "__main__":

    # create object that stores backbone information
    backbone = networks.backbone(config["backbone"])

    # create the generators
    train_generator, validation_generator = create_generators(config,
                                                              backbone.preprocess_image)


    # create the model
    if config["resume-training"]:
        print('Loading model, this may take a second...')
        model = networks.load_model(config["snapshot"], backbone_name=config["backbone"])
        training_model   = model
        anchor_params    = None
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    else:
        weights = config["weights"]

        # default to imagenet if nothing else is specified
        if weights is None and config["imagenet_weights"]:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(backbone_retinanet=backbone.retinanet,
                                                                num_classes=train_generator.num_classes(),
                                                                weights=weights,
                                                                multi_gpu=config["multi-gpu"],
                                                                freeze_backbone=config["freeze-backbone"],
                                                                lr=config["lr"])

    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in config["backbone"] or 'densenet' in config["backbone"]:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        config,
    )

    # Use multiprocessing if workers > 0
    if config["workers"] > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    if not config["compute-val-loss"]:
        validation_generator = None

    training_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=config["steps"],
                                 epochs=config["epochs"],
                                 verbose=1,
                                 callbacks=callbacks,
                                 workers=config["workers"],
                                 use_multiprocessing=use_multiprocessing,
                                 max_queue_size=config["max-queue-size"],
                                 validation_data=validation_generator)

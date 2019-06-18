preprocessing = {
    "labels":"data/devkit/cars_meta.mat",
    "train_annotations":"data/devkit/cars_train_annos.mat",
    "test_annotations":"data/devkit/cars_test_annos.mat",
    "raw_train_images":"data/raw/cars_train",
    "raw_test_images":"data/raw/cars_train",
    "extracted_train_images":"data/processed/cars_train"
}

retinanet = {
    'annotations': "/data/stanford/data/raw/cars_train/train_annotations.csv",   #help='Path to CSV file containing annotations for training.'
    'classes': "/data/stanford/data/raw/cars_train/class_mapping.csv",           #help='Path to a CSV file containing class label mapping.'
    'val-annotations':"/data/stanford/data/raw/cars_train/val_annotations.csv",  #help='Path to CSV file containing annotations for validation.'

    'resume-training':False,
    'snapshot':"",                                                      #help='Resume training from a snapshot.'
    'imagenet-weights':True,                                            #help='Initialize the model with pretrained imagenet weights.
    'weights':"/data/stanford/data/pretrained/resnet50_coco_best_v2.0.1.h5",#help='Initialize the model with weights from a file.'

    'backbone':"resnet50",                                              #help='Backbone model used by retinanet.'
    'batch-size':16,                                                    #help='Size of the batches.'
    'multi-gpu':1,                                                      #help='Number of GPUs to use for parallel processing.'
    'multi-gpu-force': False,                                           #help='Extra flag needed to enable (experimental) multi-gpu support.'
    'epochs':20,                                                        #help='Number of epochs to train.'
    'steps':300,                                                        #help='Number of steps per epoch.'
    'lr':0.001,                                                         #help='Learning rate.'
    'snapshot-path':"./snapshots",                                      #help='Path to store snapshots of models during training
    'tensorboard-dir':"./logs",                                         #help='Log directory for Tensorboard output'

    'freeze-backbone':"",                                               #help='Freeze training of backbone layers.'
    'random-transform':True,
    'image-min-side':128,                                               #help='Rescale the image so the smallest side is min_side.'
    'image-max-side':512,                                              #help='Rescale the image if the largest side is larger than max_side.'

    'weighted-average': True,                                           #help='Compute the mAP using the weighted average of precisions among classes.'
    'compute-val-loss': True,                                           #help='Compute validation loss during training', dest='compute_val_loss'
    'workers':3,                                                        #help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0'
    'max-queue-size':10                                                 #help='Queue length for multiprocessing workers in fit generator.'
}

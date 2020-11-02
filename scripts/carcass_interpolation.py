""" Interpolate sparse carcass. """
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch

sys.path.insert(0, '..')
from seismiqb import Interpolator, Enhancer, Extender
from seismiqb import MODEL_CONFIG_DETECTION, MODEL_CONFIG_EXTENSION, MODEL_CONFIG_ENHANCE
from utils import make_config



# Global parameters
N_REPS = 1
SUPPORTS = 100
OVERLAP_FACTOR = 2.
ITERATIONS = 1
FREQUENCIES = [200, 200]


# Detection parameters
DETECTION_CROP_SHAPE = (1, 256, 256)       # shape of sampled 3D crops
DETECTION_ITERS = 500                      # number of train iterations
DETECTION_BATCH_SIZE = 64                  # number of crops inside one batch


# Extension parameters
EXTENSION_CROP_SHAPE = (1, 64, 64)         # shape of sampled 3D crops
EXTENSION_ITERS = 500                      # number of train iterations
EXTENSION_BATCH_SIZE = 64                  # number of crops inside one batch
EXTENSION_STRIDE = 32                      # step size for extension
EXTENSION_STEPS = 50                       # number of boundary extensions


# Enhancing parameters
ENHANCE_CROP_SHAPE = (1, 256, 256)         # shape of sampled 3D crops
ENHANCE_ITERS = 500                        # number of train iterations
ENHANCE_BATCH_SIZE = 64                    # number of crops inside one batch



# Help message
MSG = """Interpolate carcass to the whole cube,
OR make a sparce carcass from a full horizon and re-create it.
"""

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('horizon-path', 'path to the horizon in a seismic cube in CHARISMA format', str, None),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('device', 'which device to use: physical number', int, None),
]



def interpolate(train_cube, horizon, savedir, device):
    # Get all the params from configs
    device = torch.cuda.device(device) if isinstance(device, int) else device

    # Directory to save results to
    results_dir = savedir

    short_name_cube = train_cube.split('/')[-1].split('.')[0]
    short_name_horizon = horizon.split('/')[-1].split('.')[0]
    alias = os.path.join(short_name_cube, short_name_horizon)
    save_dir = os.path.join(results_dir, alias)

    return_value = [[], [], [], []] # coverages, window ratios, support corrs, local corrs

    ###################################################################################
    ##################################   DETECTION   ##################################
    ###################################################################################
    # Create Detector instance
    detector = Interpolator(
        batch_size=DETECTION_BATCH_SIZE,
        crop_shape=DETECTION_CROP_SHAPE,
        model_config=MODEL_CONFIG_DETECTION,
        device=device,
        save_dir=save_dir, bar=False
    )

    train_dataset = detector.make_dataset(train_cube,
                                          {short_name_cube : [horizon]})

    # Train model
    last_loss = detector.train(dataset=train_dataset,
                               frequencies=FREQUENCIES,
                               n_iters=DETECTION_ITERS,
                               width=5, batch_size_multiplier=1,
                               rebatch_threshold=0.9)

    # Inference on the same cube to interpolate horizon on whole spatial range
    detector.inference(dataset=train_dataset,
                       batch_size_multiplier=0.1,
                       version=1, orientation='ix',
                       overlap_factor=OVERLAP_FACTOR)

    infos = detector.evaluate(n=1, add_prefix=False, dump=True, supports=SUPPORTS)
    info = infos[0]
    horizon = detector.predictions[0]

    return_value[0].append(horizon.coverage)
    return_value[1].append(info['window_rate'])
    return_value[2].append(info['corrs'])
    return_value[3].append(info['local_corrs'])


    for i in range(ITERATIONS):
        ###################################################################################
        ###################################   EXTEND   ####################################
        ###################################################################################
        torch.cuda.empty_cache()

        # Create instance of Enhancer
        extender = Extender(
            batch_size=EXTENSION_BATCH_SIZE,
            crop_shape=EXTENSION_CROP_SHAPE,
            model_config=MODEL_CONFIG_EXTENSION,
            device=device,
            save_dir=os.path.join(save_dir, f'extended_{i}'), bar=False
        )

        # Train model
        extender.train(horizon, n_iters=EXTENSION_ITERS, width=5)

        # Inference: fill the holes and exterior
        horizon = extender.inference(horizon,
                                     n_steps=EXTENSION_STEPS,
                                     stride=EXTENSION_STRIDE)

        # Evaluate results
        horizon = extender.predictions[0]
        extender.targets = detector.targets
        infos = extender.evaluate(n=1, add_prefix=False, dump=True, supports=SUPPORTS)
        info = infos[0]

        return_value[0].append(horizon.coverage)
        return_value[1].append(info['window_rate'])
        return_value[2].append(info['corrs'])
        return_value[3].append(info['local_corrs'])


        ###################################################################################
        ###################################   ENHANCE   ###################################
        ###################################################################################
        torch.cuda.empty_cache()

        # Create instance of Enhancer
        enhancer = Enhancer(
            batch_size=ENHANCE_BATCH_SIZE,
            crop_shape=ENHANCE_CROP_SHAPE,
            model_config=MODEL_CONFIG_ENHANCE,
            device=device,
            save_dir=os.path.join(save_dir, f'enhanced_{i}'), bar=False
        )

        # Train model
        enhancer.train(horizon, n_iters=ENHANCE_ITERS, width=5)

        # Inference: try to make every crop a touch better
        enhancer.inference(horizon,
                           batch_size_multiplier=0.1,
                           version=1, orientation='ix',
                           overlap_factor=OVERLAP_FACTOR)

        # Evaluate results
        enhancer.targets = detector.targets
        infos = enhancer.evaluate(n=1, add_prefix=False, dump=True, supports=SUPPORTS)
        info = infos[0]
        horizon = enhancer.predictions[0]

        return_value[0].append(horizon.coverage)
        return_value[1].append(info['window_rate'])
        return_value[2].append(info['corrs'])
        return_value[3].append(info['local_corrs'])

    ###################################################################################
    ##############################   SAVE NEXT TO CUBE   ##############################
    ###################################################################################
    # cube_dir = os.path.dirname(horizon.geometry.path)
    # savepath = os.path.join(cube_dir, 'HORIZONS_DUMP', DUMP_NAME)
    # os.makedirs(savepath, exist_ok=True)
    # horizon.name = '+' + horizon.name.replace('enhanced_', '').replace('extended_', '')
    # savepath = os.path.join(savepath, horizon.name)
    # horizon.dump(savepath, add_height=False)
    # detector.log(f'Dumped horizon to {savepath}')

    ###################################################################################
    ###################################   RETURNS   ###################################
    ###################################################################################

    msg = ''
    returned_values = [
        'coverages', 'window_rates', 'corrs', 'local_corrs',
    ]

    for name, value in zip(returned_values, return_value):
        msg += f'        {name} -> {value}\n'
    detector.log(msg)



if __name__ == '__main__':
    config = make_config(MSG, ARGS, os.path.basename(__file__).split('.')[0])

    interpolate(
        train_cube=config['cube-path'],
        horizon=config['horizon-path'],
        savedir=config['savedir'],
        device=config['device'],
    )

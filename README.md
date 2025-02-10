# Super Resolution for Satellite Imagery

This repository contains the code for super resolution of satellite imagery. The code is based on the paper [Image Super-Resolution via Iterative Refinement (SR3)](https://arxiv.org/abs/2104.07636) and the [official implementation](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). The code is developed in PyTorch.

## Establishing the Environment

**Step 1:** Create a new conda environment with Python 3.12 and install the gdal.

```bash
conda create -n sr python=3.12 gdal -c conda-forge
```

**Step 2:** Activate the environment and install the required packages.

```bash
conda activate sr
pip install -r requirements.txt
```

## Training the Model

To train the model, you will need to prepare two specific images:

1. **High-Resolution Raster Image (hr.tif)**: Ensure this image is prepared with high resolution for optimal training results.
2. **Cubic Up-Sampled Raster Image (sr.tif)**: This image should be generated using cubic interpolation and must be pixel-level aligned with the high-resolution image. It should also have the same dimensions.


### Configuration File

Before initiating training, create a configuration file within the `config` folder. Below is an example structure for the config file:

```
{
    "datasets": {
        "train": {
            "name": "PL_NAIP",
            "mode": "HR",
            "dataroot": "path/to/your_high_res_images.tif:1234;path/to/your_training_images.tif:1234",
            "l_resolution": 25,
            "r_resolution": 128,
            "datatype": "img",
            "split_train_ratio": 0.8,
            "batch_size": 32,
            "num_workers": 128,
            "use_shuffle": true,
            "data_len": -1
        },
        "val": {
            "name": "PL_NAIP",
            "mode": "HR",
            "dataroot": "path/to/your_high_res_images.tif:1234;path/to/your_validation_images.tif:1234",
            "l_resolution": 25,
            "r_resolution": 128,
            "datatype": "img",
            "split_train_ratio": 0.8,
            "data_len": 16
        }
    }
}
```

#### Explanation of Parameters:

- **dataroot**: This is a critical parameter that indicates the file paths to your dataset. You should replace `path/to/your_high_res_images.tif` and `path/to/your_training_images.tif` with the actual paths to your data files. The colon `:` followed by numbers (e.g., `:1234`) denotes the band order used in these images. Adjust these as needed to match your data specifics.

- **l_resolution**: This defines the lower resolution level for the dataset. 

- **r_resolution**: This parameter sets the reference resolution, typically higher, used for comparisons or evaluations.

- **datatype**: Specifies the type of data being used. For image datasets, this will be set to "img."

- **split_train_ratio**: Determines the ratio of the dataset used for training. Setting it to `1.00` implies utilizing the entire dataset for training.

- **batch_size**: Defines the number of samples per gradient update; adjust according to your memory capacity.

- **num_workers**: Specifies the number of subprocesses to use for data loading. A higher number can accelerate loading, contingent on CPU resources.

- **use_shuffle**: When set to `true`, this enables shuffling of the training data, which is often recommended for more generalized learning.

- **data_len**: Indicates the total number of data samples to be used. Set to `-1` to use all available data.

### Training Command

Once your configuration is in place, use the following command to initiate the training process:

```bash
python sr.py -p train -c config/config.json
```

Ensure that the `config.json` path corresponds to the name and location of your configuration file.

## Testing the Model

To test the trained model, you'll need to prepare an `infer.json` configuration file, similar to the one used for training the model. This configuration should point to the model checkpoint from which you wish to resume. Below is an example structure for this file, using placeholder values:

```
{
    "path": {
        "log": "logs",
        "tb_logger": "tb_logger",
        "results": "results",
        "checkpoint": "checkpoint",
        "resume_state": "path/to/your_checkpoint_directory/checkpoint/I560000_E16" // Replace with your actual checkpoint path
    },
    "datasets": {
        "val": {
            "name": "Sample_Dataset",
            "mode": "HR",
            "dataroot": "path/to/your_test_images.tif:1234;path/to/your_test_images.tif:1234", // Insert actual file path
            "l_resolution": 25,
            "r_resolution": 128,
            "datatype": "img",
            "split_train_ratio": 1.00,
            "data_len": 16
        }
    }
}
```

### Key Parameters:

- **resume_state**: Update this field with the path to the specific checkpoint directory you wish to use for resuming the model state during inference. This path should lead to the saved model from your training phase.

- **dataroot**: This should point to the high-resolution (if available) and up-sampled SR (super-resolution) images you want to test. If no high-resolution image is available, you can list the same file path before and after the semicolon (`;`). This way, the HR image won't be used during testing.

- **split_train_ratio**: It should always be set to `1.00` for testing purposes.

### Running the Inference

Execute the inference by using the following command:

```bash
python infer.py -c config/infer.json
```

Ensure the `infer.json` path corresponds to the location of your inference configuration file.

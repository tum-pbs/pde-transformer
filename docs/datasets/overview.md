# Dataset Overview

**Brief HDF5 Introduction:** The datasets used here follow the [PBDL Dataloader](https://github.com/tum-pbs/pbdl-dataset/wiki) specification using HDF5 files. Unlike saved numpy arrays, a HDF5 file allows for storing a hierarchical collection of arrays with different datatypes, adding metadata directly to each array, slice-wise writing of arrays (if entire datasets exceeds RAM size), and multithreaded, slice-wise reading of data for efficient data loading. Storing or reading datasets in HDF5 files is performed via a string path-like identifier (e.g. `sims/sim0`) consisting of *groups* as intermediate parts of the path (e.g. `sims`) and so-called *datasets* as leaves of the path (e.g. `sim0`). *Groups* work similar to Python dictionaries, and *datasets* work similar to numpy arrays, and both can additionally store metadata or attributes via the `.attrs` member variable. 

**Dataset Structure:** The general structure prescribed by the [PBDL Dataloader](https://github.com/tum-pbs/pbdl-dataset/wiki) is the following: A single HDF5 file stores an entire dataset, consisting of multiple simulations of one PDE with a temporal, a channel, and several spatial dimensions. The file contains a single group called `sims` (potentially more due to some automatically generated normalization buffers, see [Adding New Datasets](#adding-new-datasets) below). This group has different metadata that is identical for all simulations stored in its attributes. The individual simulations are datasets inside this group labeled with increasing numbers, e.g. `sims/sim0` or `sims/sim1`. These datasets are arrays in the shape of `(temporal, channel, spatial x, spatial y)` for 2D data or `(temporal, channel, spatial x, spatial y, spatial z)` for 3D data. Furthermore, they have metadata in their attributes which are different for each simulation in the dataset, e.g. a varying Reynolds number or advection speed.

**Metadata:** For `sims`, the following metadata is mandatory (either because it is prescribed by the PBDL Dataloader or because it is required for the task embeddings). Additional metadata can be added without issues for informational purposes. When adding new datasets ensure that chosen values for each metadata field are in line with existing choices, as described [below](#adding-new-datasets):

- `PDE`: a string containing the name of the PDE
- `Dimension`: an integer indicating the number of spatial dimensions, i.e., `2` or `3`
- `Fields`: a list of strings with names of each field or channel in the order the data is stored, e.g. `[Velocity X, Velocity Y, Density, Pressure]`
- `Fields Scheme`: a string that indicates how data should be normalized, as vector or scalar quantities, e.g. `VVdp`. Consecutive identical letters indicate that the physical field comprises the corresponding indices (e.g., velocity x and velocity y form a vector field due to two consecutive Vs, while density and pressure are scalar fields). See [PBDL Dataloader Documentation](https://github.com/tum-pbs/pbdl-dataset/wiki/Dataset-structure) for details.

- `Domain Extent`: float or list of floats indicating the physical, spatial domain size
- `Resolution`: integer or list of integers indicating the spatial domain discretization
- `Time Steps`: integer indicating the number of stored time steps per simulation
- `Dt`: float indicating the physical stored time step
- `Boundary Conditions`: list of strings with boundary conditions at each domain end, e.g. `[periodic, periodic, periodic, periodic, wall, open]`,
- `Boundary Conditions Order`: list of strings showing the ordering of the boundary conditions, e.g. `[x negative, x positive, y negative, y positive, z negative, z positive]`, negative indicates the side where indexing starts at `0` and positive where indexing ends at `resolution`
- `Constants`: list of strings indicating which parameters of the simulations are varying across simulations, e.g. `[Reynolds Number]`. Each simulation array in the dataset must have attached attributes with these names and corresponding values.

Exemplary Python dictionaries that are converted to this metadata can be found in [simulation_setups.py](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/simulations_apebench/simulation_setups_2d.py) for simulations with [Exponax](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/simulations_apebench/DataGenerationWithExponax.md).


# Data Module Structure, Data Loading, and Helper Scripts

**Main Data Modules:** The main entry classes to load data in a config file are derived from [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). They handle multi-gpu distribution, split in training/validation/test set, and corresponding multi-thread dataloaders. The three main DataModules that should accessed from the config files are:

- [PBDLDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_module.py): for individual [PBDLDatasets](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py), and as a base for multi-dataset modules
- [MultiDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/multi_module.py): to combine multiple PyTorch datasets and treat them as one dataset in a single DataModule. This also handles cropping/padding subset to the same number of channel and spatial resolution for batching.
- [JointDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/joint_module.py): to join multiple [PBDLDataModules](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_module.py) on Lightning level into a single [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). This means, batches consist of a list or dictionary with a separate samples from each sub dataset (see the [lightning documention](https://lightning.ai/docs/pytorch/stable/data/iterables.html)). This allows datasets with a different number of channels and spatiotemporal resolution.

**Basic Datasets Classes:** The [LightningDataModules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) above wrap multiple individual PyTorch datasets. The core of these are datasets is the [PBDLDataset Class](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py) from the [PBDL Dataloader](https://github.com/tum-pbs/pbdl-dataset), that handles reading data from HDF5 and basic dataset configuration (trimming, unrolling steps, simulation selection, data normalization). Additionally, [VariableDtDataset](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_datatypes/variable_dt_dataset.py) inherits from [PBDLDataset](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py), and adds randomized time step strides for each sample. Apart from this functionality, it behaves identical to [PBDLDataset](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py).

**Dataset Types:** The basic datasets are defined according to different dataset types (which are specified in the config files as well). Dataset types are typically used for a different data source, file structure, dataset dimension, or training/validation/test splits. Each dataset type has a separate Python file in the `pbdl_datatypes` directory, that provides functions to return training/validation/test datasets and data transformations, see e.g. the ["2D_APE_xxl" data type](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_datatypes/ape_2d_xxl.py) for additionally generated ApeBench data. 

**Advanced Dataset Functionality:** There is some additional functionality wrapped around the datasets. This is implemented as an additional layer of PyTorch datasets around the basic datasets. First, the [MetadataDataset](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/metadata_dataset.py) handles loading metadata from the HDF5 files and attaching that to the samples. In adidtion, it unpackes the data format from the [PBDLDataset](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py) or [VariableDtDataset](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_datatypes/variable_dt_dataset.py) into a dictionary format that makes it easier to process batches later on. The metadata is converted to integers or floats and normalized according to the conversion functions in [metadata_remapping.py](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/metadata_remapping.py). The MultiDataModule contains an additional [ConcatDatasetDifferentShapes Class](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/multi_module.py) that takes multiple [MetadataDatasets](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/metadata_dataset.py) and combines them while adjusting spatial and channel dimensions. Around these datasets, there may be an additional [CachedDataset](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/cached_dataset.py) if the caching functionality is enabled.

**Helper Scripts:** The data module also contains some simple helper scripts for common tasks when preparing datasets:

- [Adjust Dataset Attributes](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_adjust_dataset_attributes.py): Change or add metadata of one or more dataset files via the HDF5 attributes
- [Convert Existing Datasets](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_convert_2d_ape_bench.py): Convert scraped data from [ApeBench](https://github.com/tum-pbs/apebench) or [ACDM](https://github.com/tum-pbs/autoreg-pde-diffusion) in a numpy format to HDF5
- [Compute Dataset Norm Buffers Manually](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_precompute_dataset_norm.py): Compute dataset norm buffers with an online algorithm. Useful for large 3D dataset where one simulation does not fit into memory (see [below](#adding-new-datasets))
- [Test Dataset Loading](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_test_dataset_loading.py): Basic checks if an HDF5 dataset can be loaded by [PBDLDataset](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py) and if correct metadata exists

**Code and Instructions for Dataset Generation:** For details on the supported dataset generators, have a look at the documentation on simulating data with [Exponax](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/simulations_apebench/DataGenerationWithExponax.md), or on downloading data from the [JHTDB](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/download_jhtdb/DataDownloadFromJHTDB.md).



# Adding New Datasets

To create new datasets and use them for training or inference, follow these steps:
1. **Setup data source:** Set up a solver, repository to download data from, etc. and retrieve data from this data source.

2. **Create metadata:** Create the metadata for the new datasets according to the information outlined [above](#hdf5-dataset-format-and-metadata). This is most convenient via a Python dictionary that is later written to HDF5. Please ensure that the metadata is consistent with existing datasets. This is especially important for the names of PDEs, constants, fields, and boundary conditions which should be spelled identically to existing metadata to end up in the same location in the task embeddings. Full lists with names for each type of metadata can be found in the corresponding functions in [metadata_remapping.py](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/metadata_remapping.py). Only if the lists do not contain the corresponding element, add a new entry to the **end** of the list. This ensures that existing metadata remains in identical locations in the task embedding for already trained models.

3. **Write to HDF5:** Write the data tensors from the source and the metadata to the HDF5 format. Assuming you have a list of data arrays from each simulation `data` where the Reynolds number was varied, a corresponding list of Reynolds numbers `constants`, and a metadata dictionary `metadata`:
```python
import h5py

with h5py.File("dataset_output.hdf5", "w") as h5py_file:
    group = h5py_file.create_group("sims", track_order=True)
    for key in metadata:
        group.attrs[key] = metadata[key]

    for s in range(len(data)):
        dataset = h5py_file.create_dataset("sims/sim%d" % (s), data=data[s])
        dataset.attrs["Reynolds Number"] = constants[s]

    h5py_file.close()
```

If the dataset is very large and one simulation does not fit in the RAM, it is also possible to write the time steps of the simulation slice by slice. This only requires the datatype and target shape `size` in the beginning, before a data stream can be written via a straightforward slicing syntax:
```python
import h5py

with h5py.File("dataset_output.hdf5", "w") as h5py_file:
    group = h5py_file.create_group("sims", track_order=True)
    for key in metadata:
        group.attrs[key] = metadata[key]

    for s in range(1): # single simulation
        dataset = h5py_file.create_dataset("sims/sim%d" % (s), shape=size, type=np.float32)
        dataset.attrs["Reynolds Number"] = constants[s]

        for t in range(100): # hundred time steps
            # ... run expensive simulation and write time steps one by one
            dataset[t] = data

    h5py_file.close()
```

4. **Normalization buffers:** To efficiently compute data normalizations for training, different data statistics like data mean, standard deviation, minimum, and maximum are precomputed once, stored as normalization buffers in the HDF5 files, and loaded on demand. By default, these buffers are automatically computed and stored by the [PBDLDataset](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py) once the data is loaded for the first time. However, this does not work if individual simulations are larger than the size of the RAM, as each simulation is loaded at once. To precompute these buffer for such large datasets, the [norm precomputation script](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_precompute_dataset_norm.py) can be used instead. It loads the simulations time step by time step and computes data statistics with an online algorithm instead. Note that normalization buffers are computed separately for each dataset file for now (i.e. separately for training and test set if they are in different files, however this does not seem to make a substantial difference in the buffer contents).

5. **First test:** As a first test, if the dataset structure and metadata is in the correct format, the [PBDLDataset class](https://github.com/tum-pbs/pbdl-dataset/blob/main/pbdl/dataset.py) should be able to load it at this point. To test this, you can use the script [test dataset loading script](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/script_test_dataset_loading.py) with an adjusted path to the directory of the dataset file. Once this works, the dataset can be integrated in the Lightning DataModules in the following steps.

6. **Dataset type:** Create an additional dataset type if the new dataset requires a different handling (e.g. transformations, split in training/validation/test, cropping, unrolling, file location etc.) than existing dataset types. A different dataset type is typically necessary if a different data source, file structure, or dataset dimension is used. To create a new type, add create a new python file `XXX.py` in the `pbdl_datatypes` directory that defines a `XXX_datasets` function which returns a training, validation, and test PyTorch dataset, and a `XXX_transforms` function which returns a training, validation, and test transformation object. Add these functions to the `get_datasets` and `get_transforms` methods with the corresponding dataset type in [pbdl_module.py](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_module.py). If the dataset should be added to an existing type, add the file to the corresponding existing directory, and adjust the already existing `XXX_datasets` and `XXX_transforms` functions accordingly if necessary.

7. **Register dataset location:** Add the dataset type and path to the dataset file to the dataset path index in the environment config files in the `env` directory. If the new dataset should receive an existing type, simply add the dataset file to the corresponding existing directory specified in the environment config files. The identifier to load the dataset in the next step is the file name without the .hdf5 extension.

8. **Finalize and test:** Test that the dataset is now accessible from the main classes that handle the data loading: [PBDLDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/pbdl_module.py), [MultiDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/multi_module.py), and [JointDataModule](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/joint_module.py). This also ensure that the [MetadataDataset](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/data/metadata_dataset.py) can handle the dataset and load the required metadata. If errors occur in this step, ensure that the dataset follows the metadata requirements [above](#hdf5-dataset-format-and-metadata) and adjust/add metadata if necessary.
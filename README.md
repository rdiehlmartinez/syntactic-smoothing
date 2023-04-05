![cover-banner](./misc/cover.png)
# BabyLM

Cambridge University & Collaborator's submission to the [Baby LM Challenge](https://babylm.github.io/). 

## Setup 

To get setup create a hugging face account and ask @rdiehlmartinez to add you to the group's private hugging face hub. The hub is where we keep data, tokenization, model and other artifacts. During training, we pull in these values directly from the hub (and occasionally also push progamatically to the hub). 

In order to interact with the hub, you need to generate read and write [access tokens](https://huggingface.co/docs/hub/security-tokens) from your hugging face account. Once generated, store these values as environment variables with the names HF_READ_TOKEN, and HF_WRITE_TOKEN in a file called `.env`.

You will also need to ask @rdiehlmartinez to add you to the wandb (weights and biases) baby-lm project. We use wandb to log out metrics generated by our runs. Once you've joined the group, you will need to go to wandb to retrieve your [API key](https://wandb.ai/authorize). You will be prompted for this key calling the `./setup.sh` (see below).

Before running the code, make sure to run the setup script `./setup.sh`. This script sets up the requirements imports as well as git hooks for automatic code formatting. Additionally, this script makes sure you are logged into wandb and huggingface.

## Overview 

The entry point to the codebase is the `train.py` file. This file expects to receive a hydra-style config file that stores all relevant parameters for the dataset, data processing, tokenization, and model training. [Hydra](https://hydra.cc/docs/tutorials/structured_config/intro/) provides a system for structuring config files in a hierarchical, decomposable format.

Our BabyLM project is particularly interested in looking at curriculum learning and its use in low-resource, cognitively-plausible language modeling. We hope to train a transformer language model via a curriculum that resembles how humans learn language, and through this show that the resulting moodel can acquire both syntax and high-level language understanding even with limited data. 

We experiment with two distinct types of curriculum learning: data-driven curricula and objective-driven curricula. The idea of data-driven curriculum learning is to dynamically change the type of data that a model is exposed to over the course of training. Initially, we might want to show the model simpler data (note that how difficulty is defined is a parameter that we experiment with) and later in training ramp up the difficulty, as the model begins to master simpler concepts. Objective-driven curriculum learning, on the other hand, looks at adapting the objective function that is used to train the model over the course of training. As in the data-driven setting, the high-level goal is to bootstrap quick language learning of the model by initially training it to perform 'easier' tasks and later increase the difficulty of the objective. 

In the subsequent section, we outline the high-level structure of our code-base. 

### Config Files
Under `/src/config.py` you will find the general structure of the hydra config file that our program expects. The purpose of explicitly defining the structure of the config in this manner is two fold 1) to show the user the set of available configurable options 2) to run type-checking on passed in configs, ensuring that the parameters and their types match this pre-defined format. 

We run automatic type-checking on all the passed in config files, and also check that there are no missing required parameters of the config file. If there are, we raise an error.

The `/conf` directory stores all the default configs and subconfigs. The entry point to the default config we use is `conf/config.yaml`. Taking a look at the `conf` directory, you will notice that each sub-directory of `conf` (i.e. `conf/data_curriculum`) stores a sub-configuration. One feature that stands out is that we separate two types of curriculum learning sub-configurations (one for `data_curriculum` and one for `objective_curriculum`) -  the first revolves around ordering the data according to some difficulty function and continuously increasing the threshold of data avalaible to the model at some rate during the training loop. The second method revoles around changing the model's objective (i.e. masking) function at specified, discrete training steps.

#### Specifying a data-driven curriculum learning strategy 
In order to specify a data-driven training curriculum, you must set three arguments as part of the `conf/data_curriculum` sub-config: `scoring_fn`, `pacing_fn`, and `pacing_fn_kwargs`. The scoring (aka difficulty) function should be the name of a feature in your dataset by which you want to order your data. By default, the Trainer will sort data in an ascending fashion, i.e. in increasing difficulty w.r.t this feature. This could be n_gram_perplexity, sentence_length, etc. 

The pacing function is one of `['linear', 'quad', 'root', 'step', 'exp', 'log']` or `None`. This function controls the rate at which the training loop will increase the range of data available to be sampled by the model. The keyword arguments passed in the dictionary `pacing_fn_kwargs` control (1) `start_percent`: the percentage of the dataset which is seen at the first step (2) `num_steps`: the number of steps over which the pacing function will increase the threshold of data, thus defining the curriculum learning region. and (3) `end_percent`: the percentage of the `num_steps` at which to release all of the training data for use in training, regardless of the current threshold set by the pacing function. 

Using the default values as an example, the first 10% of the sorted dataset would be sampled during the first training step, increasing per training step as specified by the pacing function, until the training step is (`end_percent`=0.8 x `num_steps`=10_000) 8_000, where the model now samples the full dataset until the end of training, regardless of the threshold that would have been by the pacing function.

#### Specifying an objective-driven curriculum learning strategy 
Specifying an objective-driven curriculum is slightly more involved than specifying a data-driven curriculum. For objective-driven curricula you need to first specify a general strategy, by creating a new `<strategy_name>.yaml` in the `conf/curriculum` directory (cf. `conf/curriculum/mlm_to_pos.yaml` as a template). This strategy stores information relating to what objectives will be used over the course of training and when to switch the training objective. Each training objective - what we call objective units - that you plan to use over the course of training also needs to be specified. These could be "mlm", "pos", or other custom objectives specified by config files in the `curriculum/units` dir. The strategy config is where you define a dictionary `steps`, which maps integer training step numbers to the string name of the training objective specified in the units. Objectives can be reused, e.g. ```steps: {0: "mlm", 10_000: "pos", 20_000: "mlm"}```, or specified under defaults and not used in the steps dict. 

### DataLoading 

We define a CustomDataLoader in `/src/dataloader.py` that subclasses the normal hugging face Dataloader class. In the CustomDataLoader, unlike in the normal DataLoader, we are able to keep track of the global step number of training (i.e. how many batches of training data have already been trained on). This information is useful because it allows us to configure special behavior of the DataLoader for different parts of training. In particular, when the CustomDataLoader goes to yield a next batch of data, we enable the DataLoader to check whether at the current step it should apply a different collator function to preprocess the data for a given (perhaps new) objective function. 

Thus, the CustomDataLoader is where the main logic for objective-driven curricula is implemented.  

### DataSampling 

In addition to a DataLoader, we also implement a custom sampler, CurriculumSampler, under `/src/datasampler.py`. We mentioned in the previous section that it is in the DataLoader that takes care of most of the logic related to objective-driven curriculum learning. In turn, it is in the DataSampler where we implement the logic pertaining to the data-driven curriculum learning approaches. As is standard, when we initialize the CustomDataLoader we pass it an instance of the CurriculumSampler. Just like the CustomDataLoader has access to a global step so to the CurriculumSampler stores the current global step of training, in order to adapt its sampling behavior conditioned on the current training step. The CurriculumSampler uses this information in order to determine if it should sample the indices for the next batch of samples from a smaller subsample of the total dataset. The purpose for this is so that the CurriculumSampler is forced to sample early in training easier, easier training samples (and over the course of training progressively ramp up the difficulty of the samples). The CurriculumSampler has access to a pacing function (`pacing_fn`) that it uses to determine the overall maximal difficulty of samples that it can draw at any given global step number.

### The Objective Function 

We define anything to do with the objective function currently inside of the `src/objective.py` module. Currently, the main functionality of this module is to return a given DataCollator (a subclass of `transformer.DataCollator`), given the current global step and the objective curriculum learning strategy. The CustomDataLoader calls on this helper function, to receive at the current step the DataCollator to use. 

### Preprocessing and Tokenization

Data preprocessing and tokenization helper methods can be found under `/src/preprocessing.py` and `/src/tokenization.py`. 

### Model Architecture 

For most of our experiments, we use variants of Roberta language models. The architectures and the associated configurations are specified under `/src/models`. To associate a model name with a given huggingface model and an assocaited config, we store a registry inside of the `models` package. When we load a model we query this registry. 

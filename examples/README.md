# How to use Makefiles
## Introduction
This project utilizes two primary Makefiles to manage and execute examples and custom models with llmart. The main Makefile triggers all llmart examples with predefined arguments, while makefile_commands.mk enables the execution of specific custom models with accelerated launch in either CPU or GPU mode.
There are multiple sub-Makefiles located in each subfolder within the examples directory, which can be parameterized and invoked directly.

## Makefiles Overview

### examples\Makefile
- **Purpose**: This Makefile is designed to run all llmart examples simultaneously with predefined arguments.
- **Modes**: Supports both CPU and GPU modes.
- **Usage**: Use this Makefile to execute all examples with a single command.

### examples\makefile_commands.mk
- **Purpose**: This Makefile is used to execute specific custom models with llmart.
- **Modes**: Also supports both CPU and GPU modes for accelerated launch.
- **Usage**: Utilize this Makefile to run custom models with flexibility in choosing the model and execution mode.

### examples\ *\Makefile
- **Purpose**: These Makefiles are used for individual sequences
- **Modes**: Also supports both CPU and GPU modes for accelerated launch.
- **Usage**: Utilize these makefiles with custom arguments supplied in the top section for individual execution.


## Usage Instructions

### Running All Examples with Makefile with CPU mode  
```bash
 make -j{nproc} -C examples/ GPU=0
```
### Running All Examples with Makefile with GPU mode
```bash
 make -j{nproc} -C examples/ GPU=1
```
### Running individual Example Makefiles - autogcg
```bash
make -j{nproc}-C examples/autogcg MODE=cpu SUBSET=2 STEPS=2 GPU=1
```
### Running custom commands with makefile_commands.mk with CPU mode
```bash
make -j{nproc} DEVICE=cpu MODEL=llama3-8b-instruct DATA=basic LOSS=model STEPS=2 PER_DEVICE_BS=1000
```

### Running custom commands with makefile_commands.mk with GPU mode
```bash
make -j{nproc} DEVICE=cuda NUM_GPU=4 MODEL=llama3-8b-instruct DATA=basic LOSS=model STEPS=2 PER_DEVICE_BS=1000
```


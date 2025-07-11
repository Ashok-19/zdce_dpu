



# Vitis-AI workflow


## Prerequisites:

![Requirements](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/screenshots/Vitis_prerequisites.png)

It is recommended to use ubuntu 22.04 as CentOS versions already reached its EndOfLife.

* Clear Understanding of [Vitis_ai_guide](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai)


* DPU accelerated board such as Kria KV260, ZCU104 or similar ones. 


## Tutorials:

 [Vitis-AI-Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials.git)



## Installation:

Refer  [Vitis 3.0 docs](https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html) to install docker and docker image of Vitis-AI.

#### Note : 

If you are facing any difficulties in GPU docker build, pull this 3rd party docker image in your system.

        docker pull samcts2309/vitis-ai-pytorch-gpu

This works only for pytorch.


## Setup

Activate the docker and conda environment in Vitis AI. Then follow the steps below,

* Install JupyterLab for better experience

        conda install jupyterlab

        pip install chardet

* Push and Commit the docker to save the changes permanently.


* Create a directory for zerodce under Vitis-AI/

        cd Vitis-AI
        mkdir zdce_extension
    

In this directory, move the entire [compile_for_dpu](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/22482e414fae7766ecaf7d2c2031815155d0839c/zdce_extension_fpga/compile_for_dpu) folder.

* To access the resnet18 example, go to

  		 cd src/vai_quantizer/vai_q_pytorch/example
  

* If facing issues with huge storage occupied by docker containers, run this command

        #Make sure the required docker containers are running to prevent them from being deleted

        docker system prune -a


## Model Definition:

If you want to make changes in the model's structure, modify [model.py](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/model.py)


Refer [supported operations](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators) for DPU compatible operations.

Note that if you're making some drastic changes, you need to retrain the entire model for updated weights.




## Inspection

Inspect the model to confirm all the operators are mapped to DPU. If any of the operators were not mapped, either wrap them as a custom operator or approximate them using supported DPU operations.




Note that custom operators will never be executed on DPU (PL) using Vitis-AI


Target -> DPUCZDX8G_ISA1_B4096 (ZCU104)

Target is specified in the code itself, no need to mention it explicitly.

    python zdce.py --mode float --inspect

By running the above command, you should get a result that says "All operations are mapped to DPU".

A dot image will be created in the [/inspect](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/quantize_result/inspect) folder.


## Calibration

Calibrate the model by using calib data provided in data folder.

    python zdce.py --mode calib


This calibrates the model and creates [quant_info.json](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/quantize_result/quant_info.json)



## Testing and Deployment

    python zdce.py --mode test --deploy

This step quantizes the model and creates .pt ,onnx and xmodel files of ZeroDCE++.

Inspect the xmodel file in [netron](https://netron.app/) to see the structure and workflow of the model.


## Onnx to Custom IP (Optional)

If you want to create verilog code and custom IP then the generated onnx model file can be used to create a custom IP for the target board using [NNgen](https://github.com/NNgen/nngen.git)

## Compiling the xmodel

Compile the xmodel for the board specific architecture using the following command,


    vai_c_xir -x <xmodel_file_path> -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o <output_directory> -n <netname>


The xmodel file will be compiled to board specific architecture with the name provided as <netname> in <output_directory>


To create the graph of this compiled model, use the following command,

    xdputil xmodel <compiled_xmodel> -s <name_for_svg_file>


All the output and compiled files are already provided in the repository.


# Board Setup and working

MicroSD - 16 GB or Higher

Refer [board setup](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#setup-the-target)


Connect the board to the host through putty or any other terminal.



* Copy the [output](https://github.com/Ashok-19/zerodce_extension_FPGA/tree/774385801b62efd721d7b1eb4cbe51f2512aedd0/zdce_extension_fpga/compile_for_dpu/output) folder to the board.

* Write an inference script either in python or C++ for processing the videos.

* For C++ scripts, you have to build the source code.
        Refer [Inference scripts](https://github.com/Xilinx/Vitis-AI/tree/091f75041ed941b74361c1068bbcd6528e61cbc6/examples/vai_runtime)

        

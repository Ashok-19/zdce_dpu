
# ZeroDCE++ FPGA

Porting image enhancement model ZeroDCE++ on DPU accelerated FPGA devices such as Kria KV260, Zynq Ultrascale boards, Versal boards with help of Vitis-AI.

## Introduction:

This project is inspired from [FPGA-OwlsEye](https://github.com/Gaurav-Shah05/FPGA-OwlsEye.git).

In FPGA-Owlseye, the ZeroDCE model was originally mapped to ARM processor of Zynq Ultrascale zcu104 board. This project focuses on mapping the ZeroDCE++ model to DPU (Deep Learning Processing Units) of the zcu104 board.The reason why ZeroDCE++ is chosen instead of ZeroDCE is because the difference between number of params while providing same results.
    
        ZeroDCE   - ~80K params
        ZeroDCE++ - ~10K params

Some key changes are made in the original design of the model to map it to DPU.






## Modified ZeroDCE++

Original [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) model used tanh activation in final convolution layer and power operations in LE curve estimation formula. These two operations are not supported by DPU. So, zeroDCE++ was modified with approximating the unsupported operations with supported operations as follows,

        tanh = 2 * hardSigmoid(x) - 1

        torch.pow(x, 2) = torch.mul(x, x)


![Tanh_approximation](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/22482e414fae7766ecaf7d2c2031815155d0839c/screenshots/tanh_approx.png)

## Prerequisites:

* Ubuntu 20.04 or 22.04 / CentOS 7.8,7.9,8.1 . Do note that CentOS versions mentioned here already reached their EndOFLife.

* Clone [Vitis AI - 3.0](https://xilinx.github.io/Vitis-AI/3.0/html/index.html) with tag -v3.0. 
* [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) , if you want to train the model from scratch.
* Zynq® UltraScale+™ MPSoC ZCU104 / Kria KV260 or any other boards that support Vitis-AI 

    

## Training the model

Refer [readme_train_test](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/dfde5eec19556ce0ee52162b31928902e4a39a33/zdce_extension_fpga/train_test_host/README.md)


## Vitis AI Workflow

Refer [readme_compile_for_dpu](https://github.com/Ashok-19/zerodce_extension_FPGA/blob/dfde5eec19556ce0ee52162b31928902e4a39a33/zdce_extension_fpga/compile_for_dpu/README.md)
    










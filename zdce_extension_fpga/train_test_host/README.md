

# Prerequisites:

* Nvidia GPU support for faster training. 


# Dataset:

Initially, only the images provided by [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) were used to train the model.

After that, [LOL-v2 Dataset](https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset) is added to training data and trained for 200 Epochs

The trained model parameters were again used to retrain the model but now with more lowlight training samples from [ExDark dataset](https://www.kaggle.com/datasets/washingtongold/exdark-dataset) (5180 vs 12543).

The training data must not have too many daylight or normal photos. The ratio of low-light images to normal images should not exceed more than 4:1.


# Training the model


Note that due to modification in the model, I had to retrain the model. If you modify any parameters (say scale factor,or functions, etc), make sure to retrain the model with atleast 100 epochs. 

If you plan on retrain the model , dont train it from scratch. Instead, make the load_pretrain = True in low_light_train.py

                load_pretrain = True

In the model path, provide the pretrained model's path.

This introduces Transfer learning, effectively making the model much more robust.


## Current trained model flow :


        --LoL-v2 dataset + train data provided by default totalling 5180 images)
                                |
                                |
        Trained for 200 epochs (scale_factor = 4 and upsample mode = 'nearest')
                                |
                                |
        --Added more train data from ExDark Dataset totalling 12543 images)
                                |
                                |
        --Retrained the model with preloaded weights of previously trained one for 100 epochs (scale_factor = 1.6 and upsample mode = 'bilinear')
                                 
Refer the steps below to train the model.

        #clone the repo
        git clone https://github.com/Ashok-19/zerodce_extension_FPGA.git


        cd zerodce_extension_FPGA/zdce_extension_fpga/train_test_host


Train the model using,
    
    python lowlight_train.py

Default parameters for training : 
    
* batch_size = 20
*  num_epochs = 200

Refer [low_light_train.py]() for more parameters

## Testing the model

    python low_light_test.py

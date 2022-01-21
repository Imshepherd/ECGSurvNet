# ECGSurvNet

ECGSurvNet is a deep survival neural network for predicting mortality risk from electrocardiogram (ECG). This repository demonstrates how to train and test ECGSurvNet on the open ECG dataset. ECGSurvNet predicts the patient’s risk of death from the waveform of ECG, which trained using the equations of Cox proportional hazards model as the loss function. Please refer to our paper for more details:<br>
  * C Lin, "Mortality risk prediction of electrocardiogram via deep survival neural network as an extensive long-term cardiovascular outcome predictor", submitted to journal in 2022.
  
  
# Requirements

  * [R (version 3.4.4)](https://www.r-project.org/) and [Rstudio (not necessary)](https://www.rstudio.com/)
  * [Rtools (Rtools35.exe)](https://cran.r-project.org/bin/windows/Rtools/history.html)
  
  You may need to have `Rtools` installed to compile the package. Use the above link for the installation of `Rtools`.

  * [MXNetR (version 1.3.0)](https://mxnet.apache.org/versions/1.3.1/install/index.html?platform=Windows&language=R&processor=CPU)
  
  You need to have `MXNet` to train and inference the deep learning model. You can install CPU verions of `MXNet` by running the following line in your R console:
  
  ```R
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
  ```

  * [rhdf5 (version 2.22.0)](https://bioconductor.org/packages/release/bioc/html/rhdf5.html)
  
  You need to have `rhdf5` to decode the ECG data from SaMi-Trop dataset. You can install `rhdf5` by running the following line in your R console:
  
  ```R
  ## try http:// if https:// URLs are not supported
  source("https://bioconductor.org/biocLite.R")
  biocLite("rhdf5")
  ```
  You need to have `data.table` to read the ECG data from SaMi-Trop dataset. You can install `data.table` by running the following line in your R console:  
  ```R
  install.packages("data.table", version = "1.11.8")
  ```
  
# Data preparation
  
We use the [SaMi-Trop dataset](https://zenodo.org/record/4905618#.YdzpJ8lBxPY) as the example data. The SaMi-Trop cohort is an open dataset with annotations of mortality and the correspondent ECG traces. In this repository, we randomly divided the dataset into training (80%) and validation (20%) sets.  
You can use the code ['code/1. processing data/1. download Sami-Trop.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/1.%20processing%20data/1.%20download%20Sami-Trop.R) to download the SaMi-Trop dataset, and use the codes ['code/1. processing data/2. pre-processing data.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/1.%20processing%20data/2.%20pre-processing%20data.R) to pre-process the dataset for training and validating ECGSurvNet.

  
# Deep learning model: ECGSurvNet

The model can be trained using the script ['code/train.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/train.R) once the data is prepared by ['code/1. processing data/2. pre-processing data.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/1.%20processing%20data/2.%20pre-processing%20data.R). Alternatively, pre-trained weights of the ECGSurvNet is available at ['model/ECGSurvNet/ECGSurvNet-0000.params'](https://github.com/Imshepherd/ECGSurvNet/blob/main/model/ECGSurvNet/ECGSurvNet-0000.params).  

A modified residual net (ResNet) with 1D convolutional layer is used in this repository, which is described in the script ['code/train.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/train.R): 

  ```R
  model_symbol <- ECGSurvNet(indata = var_list[["data"]], start_filter = 32, inverted_coef = 4,
                             num_filters = c(32, 64, 64, 128), num_unit = c(3, 3, 6, 4), end_filters = c(512))
  ```
  
  * input: dimension = (2800, 1, 12, N). The input tensor contains the 2,800 sequence signals from each ECG leads. In the SaMi-Trop dataset, ECG was sampled at 400 Hz but some data was recorded with a duration of 10 seconds and others of 7 seconds. The ECG was fill with zeros on both size in order to make data have same size with a length of 4,096 points. For detail of ECG data, please ref to [SaMi-Trop dataset](https://zenodo.org/record/4905618#.YdzpJ8lBxPY). We crop a length of 2,800 points from the middle of original ECG for model training and validation. The final tensor consisted the sequence signals from 12 different ECG leads.
  
  * output: shape = (N). The predicted mortality risk from the ECG.


# Performance

You can evaluate its success on validation set. The traditional Cox regression model was used as the baseline comparison, which was fitted using covariate data including age and sex. An example script of validation can be found in ['code/3.  evaluation/evaluation_ECGSurvNet.R'](https://github.com/Imshepherd/ECGSurvNet/blob/main/code/3.%20evaluation/evaluation_ECGSurvNet.R), and the performance is summarized as following:

  ```R
  message("C-index of Cox model using age and sex as covariates: ", round(cox_age_sex[["concordance"]][6], digits = 4))
  >> C-index of Cox model using age and sex as covariates: 0.6344
  
  message("C-index of Cox model using the output of ECGSurvNet as covariates: ", round(cox_ecg[["concordance"]][6], digits = 4))
  >> C-index of Cox model using the output of ECGSurvNet as covariates: 0.6553
  
  message("C-index of Cox model using age, sex, and the output of ECGSurvNet as covariates: ", round(cox_age_sex_ecg[["concordance"]][6], digits = 4))
  >> C-index of Cox model using age, sex, and the output of ECGSurvNet as covariates: 0.6754
  ```

# How to cite

If you use this code in your work, please cite.



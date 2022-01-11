# ECGSurvNet

ECGSurvNet is a deep survival neural network for predicting mortality risk from electrocardiogram (ECG). This repository demonstrates how to train and test ECGSurvNet on the open ECG dataset. ECGSurvNet predicts the patient’s risk of death from the waveform of ECG, which trained using the equations of Cox proportional hazards model as the loss function. The SaMi-Trop cohort is used as the training and validation dataset, which is an open dataset with annotations of mortality and the correspondent ECG traces. Please refer to our paper for more details:<br>
  * C Lin, "Mortality risk prediction of electrocardiogram via deep survival neural network as an extensive long-term cardiovascular outcome predictor", submitted to journal in January 2022.
  
  
# Requirements

  * [R (version 3.4.4)](https://www.r-project.org/) and [Rstudio (not necessary)](https://www.rstudio.com/)
  * [Rtools (Rtools35.exe)](https://cran.r-project.org/bin/windows/Rtools/history.html)
  
  You may need to have `Rtools` installed to compile the package. Use the above link for the installation of `Rtools`.

  * [MXNetR (version 1.3.0)](https://mxnet.apache.org/versions/1.3.1/install/index.html?platform=Windows&language=R&processor=CPU)
  
  You need to have `mxnet` to train and inference the deep learning model. You can install CPU verions of `mxnet` by running the following line in your R console:
  
  ```R
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
  ```
  
# Data preparation
  
We use the [SaMi-Trop dataset](https://zenodo.org/record/4905618#.Yduo4MlBxPa) as the example data.
  
  
# How to train the ECGSurvNet

...


# Performance

...


# How to cite

If you use this code in your work, please cite.



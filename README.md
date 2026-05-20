# DeepIR-Pred
The code is the implementation of our method described in the paper “DeepIR-Pred: Accurate Prediction of Insulin Receptor using Metaheuristic-Optimized Multi-View Novel Features with Deep Recurrent Learning”.
## (I) 1_Data
There are two datasets in the data Folder:
### (1)	Train dataset
The benchmark training dataset "IR_train_P1238_N1238.fasta", which contains 1238 positive and 1238 negative protein samples.
### (2)	Independent datasets
The independent dataset "IR_test_P137_N137.fasta" contains a total of 274 protein sequences (137 positive and 137 negative samples). <br />
## (II) 2_FeatureExtractionCode
This folder contains the following files.
### (1)	lib
the folder "lib" contains all the features extraction related necessary codes used in this study.<br />
## (IV)	3_FeatureSelection
4_FeatureSelection folder includes the following files and folders.
### (1)	Whale-Optimization-Algorithm-for-Feature-Selection
the Whale-Optimization-Algorithm-for-Feature-Selection folder contains all required files related to the SBWOA feature selection.
## (V) 5_ClassificationCode
5_ClassificationCode folder includes all the models used in this study.
## (V)	PSSM_based_Feature_Extraction.m
"PSSM_based_Feature_Extraction.m" is the MATLAB code for extracting <br />
(1)	PSSM-RICLBP Features.<br />
## (VI)	extract_QLC.m
"extract_QLC.m" is the MATLAB code for extracting<br />
(1) 	QLC Features.<br />
## (VII)	ProtT5_ESM_Embeddings.py
"ProtT5_ESM_Embeddings.py" is the Python code for extracting<br />
(1) 	ESM2 and<br />
(2) 	ProtT5 pLMs embeddings.<br />
## (VIII)	fsSBWOA.m
"fsSBWOA.m" is the MATLAB code to obtain<br /> 
(1)	SBWOA optimal features.<br />
## (IX)	main.py
"main.py" is the Python code for model training using 10-fold CV.<br />
## (X)	Contact
If you are interested in our work or if you have any suggestions and questions about our research work, please contact us at: E-mail: khan_bcs2010@hotmail.com.

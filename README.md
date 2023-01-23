# SalinityMLWorkshop_DMS_UCD
This repository contains DSM2 Machine Learning scripts and data for the workshop

Colab_Train_ANN_on_Augmented_Dataset.ipynb: A jupyter notebook for use with Google Colab, which runs the ANN code using input files from this folder.<BR>
Local_Train_ANN_on_Augmented_Dataset.ipynb: A jupyter notebook for use locally on a personal computer, which runs the ANN code using input files from this folder.<BR>
Salinity_DWR.yml: A YAML file used when creating a conda environment to run the jupyter notebooks locally. <BR>
annutils.py: A python module containing ANN code which is used by all ANN scripts and notebooks.<BR>
dsm2_ann_inputs_base.xlsx: An Excel file containing historical ANN inputs. <BR>
dsm2_ann_inputs_dcc0.xlsx: An Excel file containing historical ANN inputs, with Delta Cross-Channel gates always closed. <BR>
dsm2_ann_inputs_dcc1.xlsx: An Excel file containing historical ANN inputs, with Delta Cross-Channel gates always open. <BR>
dsm2_ann_inputs_rsacminus15day.xlsx: An Excel file containing historical ANN inputs, with Sacramento River inflows shifted forward by 15 days.<BR>
dsm2_ann_inputs_rsacminus20pct.xlsx: An Excel file containing historical ANN inputs, with Sacramento River inflows scaled down by 20%.<BR>
dsm2_ann_inputs_rsacplus15day.xlsx: An Excel file containing historical ANN inputs, with Sacramento River inflows shifted backward by 15 days.<BR>
dsm2_ann_inputs_rsacplus20pct.xlsx: An Excel file containing historical ANN inputs, with Sacramento River inflows scaled up by 20%.<BR>
dsm2_ann_inputs_rsanminus15day.xlsx: An Excel file containing historical ANN inputs, with San Joaquin River inflows shifted forward by 15 days.<BR>
dsm2_ann_inputs_rsanminus20pct.xlsx: An Excel file containing historical ANN inputs, with San Joaquin River inflows scaled down by 20%.<BR>
dsm2_ann_inputs_rsanplus15day.xlsx: An Excel file containing historical ANN inputs, withSan Joaquin River inflows shifted backward by 15 days. <BR>
dsm2_ann_inputs_rsanplus20pct.xlsx: An Excel file containing historical ANN inputs, withSan Joaquin River inflows scaled up by 20%.<BR>
dsm2_ann_inputs_smscg0.xlsx: An Excel file containing historical ANN inputs, with the Suisun Marsh Salinity Control gates always closed. <BR>
dsm2_ann_inputs_smscg1.xlsx: An Excel file containing historical ANN inputs, with the Suisun Marsh Salinity Control gates always open.<BR>
observed_data_daily.xlsx: An Excel file containing observed data, used to train ANNs.<BR>

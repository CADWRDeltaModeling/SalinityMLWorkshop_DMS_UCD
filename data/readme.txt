This folder contains the inputs to the ANN, in an Excel workbook with multiple worksheets.

The following inputs are included:

1. Northern flow = Sum(Sac, Yolo, Moke, CSMR, Calaveras, -NBA)
2. San Joaquin River flow (the model input time series)
3. Exports: Sum(Banks, Jones, CCC plants(Rock Sl, Middle R (actually old river), Victoria))
4. DCC gate operation as daily percentage
5. Net Delta CU, daily Sum(DIV+SEEP-DRAIN) for DCD and SMCD, all nodes
6. Tidal Energy: daily max-daily min
7. SJR inflow salinity at Vernalis, daily
8. Sacramento River EC
9. EC Output for various locations

To create the Excel ANN input file, a script is used to read data from DSS, perform any necessary calculations, and
write the results to multiple csv files, which are manually merged into a single Excel workbook.

To run the script, we recommend that you create a conda environment using the following yml file:
https://github.com/CADWRDeltaModeling/pydelmod/environment.yml

Once you have the conda environment installed and activated, the following commands will run the script:
setlocal
call conda activate <your environment name>
<your environment name> exec-create-ann-inputs
endlocal

where <your environment name> is the name of the conda environment you created. We suggest "pydelmod".



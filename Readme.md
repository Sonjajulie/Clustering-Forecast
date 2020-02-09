---
Clustering Forecast
---

Python version: python 3.7


### Library Requirements:
 - numpy
 - netCDF4
 - scipy
 - matplotlib
 - xarrar
 - configparser
 - seaborn
 - pandas
 - logging
 - cartopy
 - os
 - cftime
 - pathlib
 - pickle
 
 
###  Create masks:
For the clustering forecast algorithm masks must be written in asci-code.
Examples for creating masks are provided the folder create\_masks:

1.) Using the mathematica notebook *create_mask.nb* . The function *CountryData* creates the mask for the specified region as for example:

```mathematica
eu = Graphics[{Black, CountryData["Europe", "Polygon"], 
   CountryData["Turkey", "Polygon"]}, Frame -> False, 
  PlotRange -> {{-12, 45}, {25.5, 72}}, ImageSize -> {77, 63}]

Export["Europe-mask.txt", Reverse[ImageData[eu][[;; , ;; , 1]]], "Table"]
```

2.) Using *cdo* and *mathematica*. In the bash-file  *create_mask.sh* all necessary cdo-commands are specified:


###  Input and mask files
The input and mask files should be provided in the folders **input\_cluster**, **input** and **mask**. **input\_cluster** includes model data, whereas **input** is for observational data.
However, you can also specifiy your own input folder.
**mask** has all masks-files.


###  Ini-files
The information of the variables as for example clusters or precursors are given in the ini-file in the folder **ini**. The name of the ini-file is arbitrary. I used cluster_{var}.ini for variables, where I would like to calculate the clusters and composite_{var}.ini for variables, where I would like to calculate composites.
An example of an ini-file would be:
```ini
[PRED:Temperature]
filepath =  input_cluster/input_cluster_anom_TS
coords = 20,75,190,310
mask = mask/North-America-mask.txt
var = TS

[PREC:SCE]
filepath = input_cluster/input_cluster_anom_fsno/
coords =  0,90,0,360
var = FSNO
name = FSNO
figsize = 4
aspect = 3
hashsize = 1


[Forecast-Parameters]
begin = 1967
end = 2010
plot = 1
forecastprecs=["ICEFRAC","FSNO"]

```
Cluster/Forecast variables should start with **PRED:** and stands for predictor, precursor or composite variables should start with
**PREC** and is an abreviation for precursor. 
The keyword **filepath** specifies the filepath for the variable inputfile. If there is no extension of ".nc" (currently only netCDF files are supported), the program assumes that all netCDF-files in the specified path should be used.
The next keyword **coords** must be only specified if **not** the total area of the input file should be used. 
The first two arguments represents minimal and maximal latitude, the second lines minimal and maximal longitudes.
In **mask** should be the path of the mask specified.
The keyword **var** is used to specify which variable should be used, because sometimes a netCDF file has several variables.

For the precursor variable also plotting options should be provided in the inifiles. The keyword **name** can be used to specify what variable name should be used for plotting.
Since sometimes the actual variable name is a little bit confusing, or if one uses multiple areas of the same variable, information get lost.
The keyword **figsize** determines the size of the plot-image of the precursor variable.
The keyword **aspect** determines the aspect ratio of the plot-image of the precursor variable.
The keyword **crosssize** represents the thickness of the hashing of the plot-image.

The section **Forecast-Parameters** has all information for the forecast algorithm. The keywords **begin** and **end** specifies the
start and end point for the forecasts. In addition, the keyword **plot** is used to signal whether the results should be also visualized ( **1** means true).
The precursors, which should be used for forecasting are specified in **forecastprecs**.

In short:

| keyword | Description |
| --- | --- |
| \[PRED:VARIABLE\]| Section for Cluster/Forecast variables, should start with PRED:|
| \[PREC:VARIABLE\]| Section for Composite/Precursor variables, should start with PREC:|
| \[Forecast-Parameters\]| Section for forecast parameters|
| filepath| Filepath for the variable inputfile.|
| coords| Minimal and maximal latitude, the second lines minimal and maximal longitudes|
| mask| Path of the mask|
| var| Selected variable of input-file|
| name | Variable name for program  |
| figsize | Size of figure |
| aspect | Aspect ratio of figure |
| hashsize | Thickness of hashing |
| begin | Start year for forecast|
| end | End year for forecast |
| plot | Boolean whether results of forecasts should be plotted. True is 1 |
| forecastprecs | Which precursor variables should be used |

```bash
#!/bin/bash

# create seamask with same resolution as model data
cdo -f nc2 setctomiss,0 -gtc,0 -remapcon,r288x192 -topo seamask.nc

# cut only american part 
cdo -f nc sellonlatbox,-170,-50,20,75 seamask.nc seamask_america.nc

# set miss to zero
cdo setmisstoc,0 seamask_america.nc sa.nc
```

The output can be saved in a txt-file using mathematica:

```mathematica
america = 
 Mod[# + 1, 2] & /@ 
  Import["/home/sonja/Documents/Cluster-calculation/Mathematica/sa.\
nc", {"Datasets", "topo"}]
Export["/home/sonja/Documents/Cluster-calculation/mask/North-America-\
mask.txt", america, "Table"];
```


### Logs
In this program the python logging-module is used.
More information can be found here:
https://docs.python.org/2/library/logging.html

The log-file parameters can be specified in the file config_dict.py. 
You can modify the level of information which should be plotted in the dictionary as well:
```python
...
{'NAME-OF-FILE': {  
    'handlers': ['console','file'],
    'level': 'DEBUG', # at what level shall call begin (still different for the handler)
    'propagate': False,
}}
```
The key of the dictionary shows the file for which the level of the logfile should be specified.
The element **'handlers': ['console','file'],** means that the logs will be printed in the terminal as well as in the file.
The element **'level': 'DEBUG'** means that everything from debug - level will be plotted. That can be changed to e.g. **INFO** or **WARNING** etc.
The element **propagate: False** means that **no** events will be passed to the handlers of higher level (ancestor) loggers.

You can also specify the handlers itself:
```python
...
{'handlers': {
    'console': {
        'level': 'DEBUG',
        'formatter': 'formatter_short',
        'class': 'logging.StreamHandler',
    },
    'file': {
        'level': 'DEBUG',
        'formatter': 'formatter_exact',
        'class': 'logging.FileHandler',
        'filename': 'logs/composites_sys.argv[2].log',
        'mode': 'a',
    },
}}
```
I specified two handlers, one for the terminal (*console*) and one for the file(*file*).
The keyword **level** specifies what kind of events you want to report (e.g., debug, info, warning, error, ...).
The keyword **formatter** determines the text-format. In my case:

```python

{'formatters': {
    'formatter_short': {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    },
    'formatter_exact': {
        'format': '%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s',
},},}
```
 
In the keyword **class** you specify the handler (whether it should be printed to the terminal or to a file). 
If file is selected you can specify the output file via "filename". 
The keyword **mode** can be used to specify in which mode the file should be opened.


### Classes
In classes all necessary classes for the forecast algorithm are provided:

* Clusters
* Composites
* ExportVarPlot
* Forecast
* MixtureModel
* Precursors
* Predictand
* PredictandToyModel


### Tests
Unit-tests for the classes Clusters and Composites are provided in the folder tests.

### Start Clustering Forecast
There are several programs to calculate just the clusters, composites are forecasts.
Go to Clustering-Forecast - folder:
For the forecast algothim start program with: 
```bash
./main_cluster_forecast.py FORECAST-VARIABLE
```

For calculating the clusters use:
```bash
./main_cluster_timeplot.py FORECAST-VARIABLE
```

For calculating the composites (first clusters should be calculated) use:
```bash
./start_composite_program.sh
```

##### References
Totz, S., Tziperman, E., Coumou, D., Pfeiffer, K., & Cohen, J. 
Winter precipitation forecast in the European and Mediterranean regions using cluster analysis. 
Geophysical Research Letters, 44. https://doi.org/10.1002/2017GL075674, 2017 

Cohen, J., Coumou, C., Hwang, J., Mackey, L., Orenstein, P., Totz, S. and Tziperman, E.,  
S2S reboot: An argument for greater inclusion of machine learning in subseasonal to seasonal forecasts, ##
WIREs Climate Change, https://doi.org/10.1002/wcc.567, 2018

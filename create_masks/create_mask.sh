#!/bin/bash

# create seamask with same resolution as model data
cdo -f nc2 setctomiss,0 -gtc,0 -remapcon,r288x192 -topo seamask.nc

# cut only american part 
cdo -f nc sellonlatbox,-170,-50,20,75 seamask.nc seamask_america.nc

# set miss to zero
cdo setmisstoc,0 seamask_america.nc sa.nc

# txt calculation is made by mathematica --> create_mask.nb

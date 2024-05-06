# Beam Health

## Preview 
https://github.com/Ram-lankada/beam_health/assets/91232198/b7d3e843-0bb2-4eaf-a7e9-410ac7752f11

## Note 
This repository is a comprehensive work & implementation of the Research paper named "DEVELOPMENT OF A MACHINE LEARNING ALGORITHM
BASED ON RANDOM FOREST MODEL FOR LOCALIZATION OF CRACK IN THE COMPOSITE BEAMS" published at the 2 day National Conference on Conditional Monitoring(NCCM) under Conditional Monitoring Society of India(CMSI) at Naval Science & Technological Laboratory(NSTL). 
Research paper : [NCCM-NSTL-RFA sub (1).pdf](https://github.com/Ram-lankada/beam_health/files/12196483/NCCM-NSTL-RFA.sub.1.pdf)


## Beam health application architecture 
![Web_application_architecture](https://github.com/Ram-lankada/beam_health/assets/91232198/c51413f2-ff19-445e-aab7-7594bff04c59)

All the three models used are Machine learning models implementing Random Forest regression. 

### Crack Localization Model 
These models takes in 3 Frequencies as input and outputs the length & the depth of Crack if detected. 
If the sensed frequencies are of ideal frequencies then it is predicted to ba a Healthy Beam & no prediction of length & depth of the crack will be done.

### Stress Intensity Factor Prediction

This model takes in the Stress value sensed by the Strain Guage which is attached to the beam while applying the load. And predicts the Stress intensity factor on a scale from 0.0 to 1.0. 
And based on the, Stress intensity factor the Severity of the damage will be calculated and will be categorized into : 
- Low Severity
- Medium Severity
- High Severity

### Residual life Estimation

This model takes in the Stress value sensed by the Strain Guage which is attached to the beam while applying the load and predicts the maxium possible time till which the beam can be used, which can be termd as Residual life. After which the composite beam has to be discarded to prevent unexpected accidents / mishaps. 



## Project Architecture 
![random_forest_architecture](https://github.com/Ram-lankada/beam_health/assets/91232198/b900dd84-3142-4caa-84c6-2e95cc4ba02d)

## Other Resources 
You can get the architecture diagrams listed above in the `diagrams` folder of this repo. 

You can also get the UI design of this web application here in this figma file :
https://www.figma.com/file/DqHxcS90Un4YgFiOBWbxiV/BEAM-HEALTH?type=design&node-id=0%3A1&mode=design&t=xSr8cEdbHQvboO9P-1


## Execution Instructions
```
git clone https://github.com/Ram-lankada/beam_health.git
cd beam_health
python manage.py runserver
```


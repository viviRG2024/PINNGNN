# Data Collection
- [read original data from large dataset](./utils/pre-process/readOriginalData.py)
- [extract city info - date, lon, lat](./utils/pre-process/collectAddress&Day.py)
- [get rain data from url](./utils/pre-process/getRainData.py)
- [get road networks from osm](./utils/pre-process/getOSMRoadData.py)
- [select road networks based on detectors' area](./utils/pre-process/selectRoads.py)
- [get traffic data from sensor value](./utils/pre-process/attachSensorOnRoads.py)
- [transfer to npz and csv data](./utils/pre-process/mergeRoadAndSensorValue.py)

# Trainning
- [trainning model](./utils/trainning/trainModel.py)
import os


PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/Users/wangkai'

CUDA=True

EPSILON = 1e-8

UCMerced_Label=['agricultural','airplane','baseballdiamond','beach','buildings','chaparral','denseresidential','forest','freeway','golfcourse','harbor','intersection','mediumresidential','mobilehomepark','overpass','parkinglot','river','runway','sparseresidential','storagetanks','tenniscourt']

UCMerced_Label_id = {'agricultural.agricultural': 0, 'airplane.airplane': 1, 'baseballdiamond.baseballdiamond': 2, 'beach.beach': 3, 'buildings.buildings': 4, 'chaparral.chaparral': 5, 'denseresidential.denseresidential': 6
    , 'forest.forest': 7, 'freeway.freeway': 8, 'golfcourse.golfcourse': 9, 'harbor.harbor': 10, 'intersection.intersection': 11, 'mediumresidential.mediumresidential': 12
    , 'mobilehomepark.mobilehomepark': 13, 'overpass.overpass': 14, 'parkinglot.parkinglot': 15,'river.river': 16, 'runway.runway': 17, 'sparseresidential.sparseresidential': 18, 'storagetanks.storagetanks': 19, 'tenniscourt.tenniscourt': 20}

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')

import numpy as np 

CANON = np.asarray(
    [
        [1,1,1,2,3,6,8,10],
        [1,1,2,3,4,8,9,8],
        [2,2,2,3,6,8,10,8],
        [2,2,3,4,7,12,11,9],
        [3,3,8,11,10,16,15,11],
        [3,5,8,10,12,15,16,13],
        [7,10,11,12,15,17,17,14],
        [14,13,13,15,15,14,14,14]
    ]
)

NIKON = np.asarray(
    [
        [2,1,1,2,3,5,6,7],
        [1,1,2,2,3,7,7,7],
        [2,2,2,3,5,7,8,7],
        [2,2,3,3,6,10,10,7],
        [2,3,4,7,8,13,12,9],
        [3,4,7,8,10,12,14,11],
        [6,8,9,10,12,15,14,12],
        [9,11,11,12,13,12,12,12]
    ]
)

JPEG = np.asarray(
    [
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ]
)

MATRACES = {
    'cannon': CANON,
    'nikon': NIKON,
    'jpeg': JPEG
}
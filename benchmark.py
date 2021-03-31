import numpy as np
from statistical import ConfusionMatrix
from draw import drawMatrix

y_t = [1, 3, 4, 5, 6, 7, 8, 9, 2, 1, 4, 5, 6, 3, 2, 1, 2, 5, 6, 1, 2, 6, 4, 6, 4, 5, 6]
y_p = [1, 2, 3, 5, 6, 7, 8, 9, 2, 1, 4, 5, 6, 3, 2, 1, 2, 5, 6, 1, 2, 6, 4, 6, 4, 5, 6]

conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p)

drawMatrix.ConfusionMatrix(conf, save=True)
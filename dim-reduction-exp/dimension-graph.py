import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

i = 0
zero = 0
if __name__ == '__main__':
    data = pd.read_csv("data\\41K_processed.csv")
    r = data.rows
    c = data.cols

    ##what i want: a graph with on the y-axis a percentage
    ## on the x-axis the resolution
    ## then a line how fast this goes up.
    ## how to do this: sort all the data.
    #  get rid of everything with size 0 (and note how much this is, maybe a problem with algorithm)
    #  divide into 1000 equal-sized bins (41979 -> +/- 42 per bin)
    # note the size of every first element in the bin down and plot this

    r = np.sort(r)

    oldlen = len(r)
    r[r != 0] ##this doesn't work but i'll figure it out sometime else.
    print(r)
    print("This is how much was removed:")
    print(oldlen - len(r))

    xarray = [0]*1000
    yarray=np.linspace(1, 100, 1000)
    bsize = len(r)//1000
    for j in range(1,1000):
        xarray[j] = r[j*bsize]

    for i in range(10,100):
        if (i%10 == 0):
            print(i,"% data reduction gives lowest resolution of ",xarray[10*i])

    plt.plot(xarray,yarray)
    plt.show()

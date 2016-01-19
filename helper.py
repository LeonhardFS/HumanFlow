import numpy as np
import pandas as pd

# converts time stamp as int in DHHMM into index for the table
def timestamptoindex(ts):
    mm = ts % 100
    hh = (ts % (100 * 100)) // 100
    d = ts // (100 * 100)
    
    index = mm + 60 * hh + 60 * 24 * (d - 1)
    return index

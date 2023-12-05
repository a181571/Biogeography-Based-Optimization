# -*- coding: utf-8 -*-
"""
Python code of Biogeography-Based Optimization (BBO)
Coded by: Raju Pal (emailid: raju3131.pal@gmail.com) and Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar to code given at link: https://github.com/himanshuRepo/CKGSA-in-Python 
 and matlab version of the BBO at link: http://embeddedlab.csuohio.edu/BBO/software/

Reference: D. Simon, Biogeography-Based Optimization, IEEE Transactions on Evolutionary Computation, in print (2008).
@author: Dan Simon (http://embeddedlab.csuohio.edu/BBO/software/)

-- Benchmark Function File: Defining the benchmark function along its range lower bound, upper bound and dimensions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from typing import Tuple
import pandas as pd
from ubiutils.filtering import ekf_filter, ekf_filter_hm
from pathlib import Path
from typing import Union
import json
import gzip
from ubiutils.filtering import valid_time
from tqdm import tqdm
from typing import Union
from pathlib import Path
# define the function blocks
    
def F1(x):
    s=numpy.sum(x**2);
    return s

def getFunctionDetails(a):
    lb = np.array([-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2])
    ub = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    lb1 = np.array([-3, -3, -3, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2])
    ub1 = np.array([3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    # [name, lb, ub, dim]
   # param = {  0: ["F1",-5,2,30],
    #           1: ["F2",-5,-2,30]}
    param = {  0: ["F1",-5,2,30],
               1: ["F2",lb1,ub1,14],
               2: ["F3",lb1,ub1,14],
               3: ["F4",lb1,ub1,14]}
    return param.get(a, "nothing")


#cache_path = f"data/cache/f2joindfv3-5.parquet"
#joindf = pd.read_parquet(cache_path)

def read_json_gz(filename: Union[str, Path]) -> dict:
    """read a json.gz file
    Args:
        filename (str): path to gzip file
    Returns:
        dict: trip dictionary
    """
    with gzip.open(str(filename)) as fp:
        return json.load(fp)

#trip = read_json_gz("data/camp5/trips/05a80f0ee0ac53a036f82829536e4780.json.gz")
#df1 = pd.read_parquet("data/v3mad-10HZ-1tmo/participant_id_key=131844da-fe8f-4e4e-be51-6dcfbd42ee3c/trip_id_key=05a80f0ee0ac53a036f82829536e4780-3250622546/ebfd5c661a104c64be84b1fc2210295b-0.parquet")

def F2(x):
    pd.options.mode.chained_assignment = None
    global trip, df1
    _, df2 = ekf_filter_hm(
        trip=trip,
        window_size=31,
        threshold=0.01,
        med_comb_ratio=5,
        smooth=True,
        q= x[0:6],
        r= x[6:14],
        run_id="time_varying_opt_1",
    )
    df1 = df1.set_index("time_stamp").resample("s").mean(numeric_only=True).reset_index()
    df1["obd_acc"] = df1.obd_speed.diff(1)
    dfe = df1.merge(df2,left_on=["time_stamp"],right_on=["time_unix_epoch"],how="inner").set_index("time_stamp")
    std = sum((dfe.obd_speed - dfe.mad_speed_ekf) ** 2) / len(dfe)
    std1 = sum((dfe.obd_acc[1:] - dfe.acc_long_ekf[1:]) ** 2) / len(dfe)
    s = std + 10 * std1
    return s

def F3(x):
    pd.options.mode.chained_assignment = None
    global trip, df1
    i = 210
    try:  
        _, df2 = ekf_filter_hm(
            trip=trip,
            window_size=31,
            threshold=0.01,
            med_comb_ratio=5,
            smooth=True,
            q= x[0:6],
            r= x[6:14],
            run_id="time_varying_opt_1",
        )
        df1 = df1.set_index("time_stamp").resample("s").mean(numeric_only=True).reset_index()
        df = df1.iloc[i:i+30]
        df["obd_acc"] = df.obd_speed.diff(1)
        dfe = df.merge(df2,left_on=["time_stamp"],right_on=["time_unix_epoch"],how="inner").set_index("time_stamp")
        std = sum((dfe.obd_speed - dfe.mad_speed_ekf) ** 2) / len(dfe)
        std1 = sum((dfe.obd_acc[1:] - dfe.acc_long_ekf[1:]) ** 2) / len(dfe)
        s =  std1
    except:  
        s = 10000000
    return s

def F4(x, df1, trip, i):
    pd.options.mode.chained_assignment = None
    try:
        _, df2 = ekf_filter_hm(
            trip=trip,
            window_size=31,
            threshold=0.01,
            med_comb_ratio=5,
            smooth=True,
            q= x[0:6],
            r= x[6:14],
            run_id="time_varying_opt_1",
        )
        df = df1.iloc[i:i+2]
        df["obd_acc"] = df.obd_speed.diff(1)
        dfe = df.merge(df2,left_on=["time_stamp"],right_on=["time_unix_epoch"],how="inner").set_index("time_stamp")
        std = sum((dfe.obd_speed - dfe.mad_speed_ekf) ** 2) / len(dfe)
        std1 = sum(dfe.obd_acc[1:] - dfe.acc_long_ekf[1:]) ** 2 / 1
    except:
        std1=1000
    return std1
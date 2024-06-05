# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from os.path import join

# In[]

punti_gdf = gpd.read_file(join('data','cop.shp'))

eps, min_pts = 4, 4
n = punti_gdf.shape[0]
c = -1
labels = ['None']*n

for i, p in punti_gdf.iterrows():
    if labels[i] != 'None':
        continue
    
    N = punti_gdf[ punti_gdf.distance(p['geometry']) < eps ]
    
    if len(N) < min_pts:
        labels[i] = 'Noise'
        continue

    c += 1
    labels[i] = str(i)
    
    S = set(list(N.index)) - set([i])
    
    while len(S) > 0:
        q = S.pop()
        if labels[q] == 'Noise':
            labels[q] = str(c)
        if labels[q] != 'None':
            continue
        q_geom =  punti_gdf.loc[q, 'geometry']
        N = punti_gdf[ punti_gdf.distance(q_geom) < eps ]
        labels[q] = str(c)
        if len(N) < min_pts:
            continue
        S = S.union(set(list(N.index)) - set([q]))
        
punti_gdf['label'] = labels
punti_gdf.plot('label')
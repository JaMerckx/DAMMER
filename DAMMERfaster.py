#%%
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import astra
import scipy.io
import matplotlib.tri as tri
import triangle as tr
import pylops
#from functionstr import systmatrixpar
from scipy.sparse import csr_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import eigs
from shapely.geometry import Polygon
import torch
import os
import time

#%%

def triangle_area_intersection(triangles1, triangles2, vertices):
    """
    Computes the intersection areas between two sets of triangles.

    Parameters:
        triangles1: (N, 3) array of triangle vertex indices (rows)
        triangles2: (M, 3) array of triangle vertex indices (columns)
        vertices: (P, 2) array of 2D vertex coordinates

    Returns:
        area_matrix: (N, M) matrix where element (i, j) is the intersection area
                     between triangle i from list1 and triangle j from list2.
    """
    N = len(triangles1)
    M = len(triangles2)
    area_matrix = np.zeros((N, M))  # Result matrix

    # Convert triangles to Shapely polygons
    polys1 = [Polygon(vertices[t]) for t in triangles1]
    polys2 = [Polygon(vertices[t]) for t in triangles2]
    bounds1 = [p.bounds for p in polys1]
    bounds2 = [p.bounds for p in polys2]
    # Compute intersection areas
    for i in range(N):
        for j in range(M):
            b1 = bounds1[i]
            b2 = bounds2[j]
            # Quick bounding box rejection
            if not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]):
                intersection = polys1[i].intersection(polys2[j])
                if intersection.is_valid and not intersection.is_empty:
                    area_matrix[i, j] = intersection.area

    return area_matrix

def circumcircle(t):
  
  a = np.sqrt ( ( t[:,0, 0] - t[:,1, 0] ) ** 2 + ( t[:,0, 1] - t[:,1, 1] ) ** 2 )
  b = np.sqrt ( ( t[:,1, 0] - t[:,2, 0] ) ** 2 + ( t[:,1, 1] - t[:,2, 1] ) ** 2 )
  c = np.sqrt ( ( t[:,2, 0] - t[:,0, 0] ) ** 2 + ( t[:,2, 1] - t[:,0, 1] ) ** 2 )

  bot = ( a + b + c ) * ( - a + b + c ) * (   a - b + c ) * (   a + b - c )

  r = a * b * c / np.sqrt ( bot )
  center = np.zeros((len(t), 2)) 
  f = np.zeros((len(t), 2))

  f[:, 0] = ( t[:,0, 0] - t[:,1, 0] ) ** 2 + ( t[:,0, 1] - t[:,1, 1] ) ** 2
  f[:, 1] = ( t[:,2, 0] - t[:,0, 0] ) ** 2 + (t[:,2, 1] - t[:,0, 1] ) ** 2

  top = np.zeros((len(t), 2))

  top[:,0] =    (t[:,2, 1] - t[:,0, 1]) * f[:, 0] - (t[:,1, 1] - t[:,0, 1]) * f[:, 1]
  top[:,1] =  - (t[:,2, 0] - t[:,0, 0]) * f[:, 0] + (t[:,1, 0] - t[:,0, 0]) * f[:, 1]

  det  =    ( t[:, 2, 1] - t[:, 0, 1] ) * ( t[:, 1, 0] - t[:, 0, 0] ) \
          - ( t[:, 1,1] - t[:, 0, 1] ) * ( t[:, 2, 0] - t[:, 0,0] ) 

  center[:, 0] = t[:, 0, 0] + 0.5 * top[:, 0] / det
  center[:, 1] = t[:, 0, 1] + 0.5 * top[:, 1] / det
  return r, center

def MSEPSNRnonumba(triangles, vertices, attenuation, pixelvals):
 dim = len(pixelvals)
 mse = 0
 for tindex in range(len(triangles)): 
  triangletot = vertices[triangles[tindex]]
  startx = int(np.min(triangletot[:, 0])*dim-10)
  if startx < np.min(vertices):
     startx = np.min(vertices)
  eindx = int(np.max(triangletot[:, 0])*dim+10)
  if eindx > dim:
     eindx = dim
  starty = int(np.min(triangletot[:, 1])*dim-10)
  if starty < np.min(vertices):
     starty = np.min(vertices)
  eindy = int(np.max(triangletot[:, 1])*dim+10)
  if eindy > dim:
     eindy = dim
  for triangleind in range(startx, eindx):
     for pixelind in range(starty, eindy):
      triangl = triangletot
      x0pix = (triangleind)/dim
      x1pix = (triangleind+1)/dim
      y0pix =  pixelind/dim
      y1pix =  (pixelind+1)/dim
      #coordpix = np.array([[x0pix, y0pix], [x1pix, y0pix], [x1pix, y1pix], [x0pix, y1pix]])
      coordpix = np.zeros((4, 2))
      coordpix[0][0] = x0pix 
      coordpix[0][1] = y0pix
      coordpix[1][0] = x1pix 
      coordpix[1][1] = y0pix
      coordpix[2][0] = x1pix 
      coordpix[2][1] = y1pix
      coordpix[3][0] = x0pix 
      coordpix[3][1] = y1pix
      maxx = triangl[0, 0] 
      minx = triangl[0, 0] 
      miny = triangl[0, 1] 
      maxy = triangl[0, 1] 
      for indd in range(2):
         if triangl[indd + 1][0] > maxx :
            maxx = triangl[indd + 1][0] 
         if triangl[indd + 1][1] > maxy :
            maxy =triangl[indd + 1][1] 
         if triangl[indd + 1][0] < minx :
            minx =triangl[indd + 1][0] 
         if triangl[indd + 1][1] < miny :
            miny =triangl[indd + 1][1] 
      if x0pix < maxx and x1pix > minx and \
         y0pix < maxy and y1pix > miny:
         telnums = 3
         for cind in range(4):
            cp1 = coordpix[cind]
            if cind < 3:
              cp2 = coordpix[cind+1]
            else:   
              cp2 = coordpix[0]
            inputlist =  np.zeros((6, 2))
            for tindd in range(telnums): 
               inputlist[tindd][0] = triangl[tindd][0]
               inputlist[tindd][1] = triangl[tindd][1]
            triangl = np.zeros((6, 2))
            telnumsprev = 0
            telnumsprev = telnumsprev + telnums
            telnums = 0
            for tind in range(telnumsprev):
               e = inputlist[tind] 
               if tind > 0:
                 s = inputlist[tind - 1] 
               else: 
                  s = inputlist[telnumsprev - 1]   
               if (cp2[0]-cp1[0])*(e[1]-cp1[1]) > (cp2[1]-cp1[1])*(e[0]-cp1[0])-10**(-10): 
                  if (cp2[0]-cp1[0])*(s[1]-cp1[1]) < (cp2[1]-cp1[1])*(s[0]-cp1[0])-10**(-10):  
                     dc0 =  cp1[0] - cp2[0]
                     dc1 =  cp1[1] - cp2[1] 
                     dp0 =  s[0] - e[0]
                     dp1 =  s[1] - e[1] 
                     n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                     n2 = s[0] * e[1] - s[1] * e[0] 
                     n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                     triangl[telnums][0] = (n1*dp0 - n2*dc0) * n3 
                     triangl[telnums][1] = (n1*dp1 - n2*dc1) * n3
                     telnums+=1 
                  triangl[telnums][0] = e[0]
                  triangl[telnums][1] = e[1]
                  telnums+=1
               elif (cp2[0]-cp1[0])*(s[1]-cp1[1]) > (cp2[1]-cp1[1])*(s[0]-cp1[0])-10**(-10): 
                     dc0 =  cp1[0] - cp2[0]
                     dc1 =  cp1[1] - cp2[1] 
                     dp0 =  s[0] - e[0]
                     dp1 =  s[1] - e[1] 
                     n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                     n2 = s[0] * e[1] - s[1] * e[0] 
                     n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                     triangl[telnums][0] = (n1*dp0 - n2*dc0) * n3 
                     triangl[telnums][1] = (n1*dp1 - n2*dc1) * n3
                     telnums+=1 
      if telnums > 0:
            intarea = 0
            for vind in range(telnums): 
               if vind < telnums - 1:
                 intarea += 0.5*(triangl[vind][1] + triangl[vind+1][1])* \
                  (triangl[vind][0] - triangl[vind+1][0])
               else: 
                 intarea += 0.5*(triangl[vind][1] + triangl[0][1])* \
                  (triangl[vind][0] - triangl[0][0])
            intarea = (intarea**2)**(1/2) 
            atdif = ((attenuation[tindex] - pixelvals[dim-1-pixelind][triangleind])**2)
            mse+= intarea*atdif
 
 psnr = 10*np.log((np.max(pixelvals.ravel())**2/mse))/np.log(10)
 return mse, psnr

import numba

@cuda.jit()
def PSNRcalc(triangletot, attenuation, pixelvals, mserrorabs, mserror):
   triangleind, pixelind = cuda.grid(2)
   dim = len(pixelvals)
   if triangleind < dim and pixelind < dim:
      triangl = triangletot
      x0pix = (triangleind)/dim
      x1pix = (triangleind+1)/dim
      y0pix =  pixelind/dim
      y1pix =  (pixelind+1)/dim
      #coordpix = np.array([[x0pix, y0pix], [x1pix, y0pix], [x1pix, y1pix], [x0pix, y1pix]])
      coordpix = cuda.local.array((4, 2), dtype=numba.float64)
      coordpix[0][0] = x0pix 
      coordpix[0][1] = y0pix
      coordpix[1][0] = x1pix 
      coordpix[1][1] = y0pix
      coordpix[2][0] = x1pix 
      coordpix[2][1] = y1pix
      coordpix[3][0] = x0pix 
      coordpix[3][1] = y1pix
      maxx = triangl[0, 0] 
      minx = triangl[0, 0] 
      miny = triangl[0, 1] 
      maxy = triangl[0, 1] 
      for indd in range(2):
         if triangl[indd + 1][0] > maxx :
            maxx = triangl[indd + 1][0] 
         if triangl[indd + 1][1] > maxy :
            maxy =triangl[indd + 1][1] 
         if triangl[indd + 1][0] < minx :
            minx =triangl[indd + 1][0] 
         if triangl[indd + 1][1] < miny :
            miny =triangl[indd + 1][1] 
      if x0pix < maxx and x1pix > minx and \
         y0pix < maxy and y1pix > miny:
         telnums = 3
         for cind in range(4):
            cp1 = coordpix[cind]
            if cind < 3:
              cp2 = coordpix[cind+1]
            else:   
              cp2 = coordpix[0]
            inputlist =  cuda.local.array((6, 2), dtype=numba.float64)
            for tindd in range(telnums): 
               inputlist[tindd][0] = triangl[tindd][0]
               inputlist[tindd][1] = triangl[tindd][1]
            triangl = cuda.local.array((6, 2), dtype=numba.float64)
            telnumsprev = 0
            telnumsprev = telnumsprev + telnums
            telnums = 0
            for tind in range(telnumsprev):
               e = inputlist[tind] 
               if tind > 0:
                 s = inputlist[tind - 1] 
               else: 
                  s = inputlist[telnumsprev - 1]   
               if (cp2[0]-cp1[0])*(e[1]-cp1[1]) > (cp2[1]-cp1[1])*(e[0]-cp1[0])-10**(-10): 
                  if (cp2[0]-cp1[0])*(s[1]-cp1[1]) < (cp2[1]-cp1[1])*(s[0]-cp1[0])-10**(-10):  
                     dc0 =  cp1[0] - cp2[0]
                     dc1 =  cp1[1] - cp2[1] 
                     dp0 =  s[0] - e[0]
                     dp1 =  s[1] - e[1] 
                     n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                     n2 = s[0] * e[1] - s[1] * e[0] 
                     n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                     triangl[telnums][0] = (n1*dp0 - n2*dc0) * n3 
                     triangl[telnums][1] = (n1*dp1 - n2*dc1) * n3
                     telnums+=1 
                  triangl[telnums][0] = e[0]
                  triangl[telnums][1] = e[1]
                  telnums+=1
               elif (cp2[0]-cp1[0])*(s[1]-cp1[1]) > (cp2[1]-cp1[1])*(s[0]-cp1[0])-10**(-10): 
                     dc0 =  cp1[0] - cp2[0]
                     dc1 =  cp1[1] - cp2[1] 
                     dp0 =  s[0] - e[0]
                     dp1 =  s[1] - e[1] 
                     n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                     n2 = s[0] * e[1] - s[1] * e[0] 
                     n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                     triangl[telnums][0] = (n1*dp0 - n2*dc0) * n3 
                     triangl[telnums][1] = (n1*dp1 - n2*dc1) * n3
                     telnums+=1 
         if telnums > 0:
            intarea = 0
            for vind in range(telnums): 
               if vind < telnums - 1:
                 intarea += 0.5*(triangl[vind][1] + triangl[vind+1][1])* \
                  (triangl[vind][0] - triangl[vind+1][0])
               else: 
                 intarea += 0.5*(triangl[vind][1] + triangl[0][1])* \
                  (triangl[vind][0] - triangl[0][0])
            intarea = (intarea**2)**(1/2) 
            atdif = ((attenuation - pixelvals[dim-1-triangleind][pixelind])**2)**(1/2)
            cuda.atomic.add(mserrorabs, 0, intarea)#intarea*atdif)
            cuda.atomic.add(mserror, 0, intarea)#intarea*atdif**2)

def  clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, neighbors, scalerseg, res):
    systemmat = systemmat.tocsc()
    connectionarray = np.arange(len(triangles))
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)
    numangles = int(len(sinogram)/numpix)
    startvalue = 1
    con_dict = {}
    for c in range(len(triangles)):
        con_dict[c] = [c]

    areas = [Polygon(vertices[t]).area for t in triangles]
    areas = np.array(areas)
    trianglist = np.zeros((len(np.where((neighbors) > -0.5)[0]), 2))
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    trianglist = trianglist.astype(int)
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])
    projdif0ns = systemmat@attenuation - sinogram  

    lengrad = len(gradoveredges)/2
    funpos0 = projdif0 + scalerseg*lengrad
    simplificdegree = 10
    nummax = 100
    attenuation = attenuation.copy()
    #while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
    maxgrad = np.max(gradoveredges) 
    gradorder = np.argsort(gradoveredges)
    compared = np.zeros((2, 2))
    for gind in range(len(gradoveredges)):
       mingrad = gradorder[gind]#np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       tmin = np.min(connectionarray[np.array([triangle1, triangle2])])
       tmax = np.max(connectionarray[np.array([triangle1, triangle2])])
       con1 = con_dict[connectionarray[triangle1]]
       con2 = con_dict[connectionarray[triangle2]]
       if len(con1) > 10 and len(con2) > 10:
           if np.abs(attenuation[triangle1] - attenuation[triangle2]) > 0.1:
               continue
       
       wheremin = np.where(compared[:, 0] == tmin)[0]
       if len(wheremin) > 0:
          wheremax = np.where(compared[wheremin, 1] == tmax)[0]
          if len(wheremax) > 0:
             continue
       atprev = attenuation[np.hstack((con1, con2))].copy()
       attenuation[np.hstack((con1, con2))] = np.sum(attenuation[np.hstack((con1, con2))]* \
                                   areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])

       lennext = 0
       for con in neighbors[con1].ravel():
          if con in con2:
              lennext += 1
            

       projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]- atprev)
       funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
       if funpos < funpos0:
           funpos0 = funpos
           lengrad -= lennext
           minval = np.min(connectionarray[np.array([triangle1, triangle2])])
           otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
           connectionarray[np.hstack((con1, con2))] = minval
           con_dict[minval].extend(con_dict[otherval]) 
           del con_dict[otherval]
           compared = np.zeros((2, 2))
       else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]\
            @(attenuation[np.hstack((con1, con2))]- atprev)
            attenuation[np.hstack((con1, con2))] = atprev
            compared = np.vstack((compared, np.array([tmin, tmax])))
       #gradoveredges[mingrad] = maxgrad
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])  
    gradorder = np.argsort(gradoveredges)  
    compared = np.zeros((2, 2))
    for gind in range(len(gradoveredges)):
       mingrad = gradorder[gind]#np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       tmin = np.min(connectionarray[np.array([triangle1, triangle2])])
       tmax = np.max(connectionarray[np.array([triangle1, triangle2])])
       radss, _ = circumcircle(vertices[triangles[np.array([triangle1, triangle2])]])
       if (radss[0] > res or len(con_dict[connectionarray[triangle1]]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle1]]), connectionarray[triangle1])) < 2) \
         and (radss[1] > res or len(con_dict[connectionarray[triangle2]]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle2]]), connectionarray[triangle2])) < 2):
         con1 = con_dict[connectionarray[triangle1]]
         con2 = con_dict[connectionarray[triangle2]]
         if len(con1) > 10 and len(con2) > 10:
            if np.abs(attenuation[triangle1] - attenuation[triangle2]) > 0.1:
                continue
         wheremin = np.where(compared[:, 0] == tmin)[0]
         if len(wheremin) > 0:
          wheremax = np.where(compared[wheremin, 1] == tmax)[0]
          if len(wheremax) > 0:
             continue

         atprev = attenuation[np.hstack((con1, con2))].copy()
         attenuation[np.hstack((con1, con2))] = np.sum(attenuation[np.hstack((con1, con2))]* \
                                    areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])

         lennext = 0
         for con in neighbors[con1].ravel():
            if con in con2:
                lennext += 1
            

         projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]-atprev)
         funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)

         if funpos < funpos0:
            funpos0 = funpos
            lengrad -= lennext
            minval = np.min(connectionarray[np.array([triangle1, triangle2])])
            otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
            connectionarray[np.hstack((con1, con2))] = minval
            con_dict[minval].extend(con_dict[otherval])   
            del con_dict[otherval]  
            compared = np.zeros((2, 2))     
         else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]-atprev)
            attenuation[np.hstack((con1, con2))] = atprev
            compared = np.vstack((compared, np.array([tmin, tmax])))

         gradoveredges[mingrad] = maxgrad
       else:
            con1 = con_dict[connectionarray[triangle1]]
            con2 = con_dict[connectionarray[triangle2]]
            atprev = attenuation[np.hstack((con1, con2))].copy()
            attenuation[np.hstack((con1, con2))] = np.sum(attenuation[np.hstack((con1, con2))]* \
                                       areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
            
            lennext = 0
            for con in neighbors[con1].ravel():
              if con in con2:
                  lennext += 1
            
            projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]-atprev)
            funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            minval = np.min(connectionarray[np.array([triangle1, triangle2])])
            otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
            connectionarray[np.hstack((con1, con2))] = minval
            con_dict[minval].extend(con_dict[otherval])   
            del con_dict[otherval]   
            compared = np.zeros((2, 2))    
            

    for tind in range(len(triangles)):
       if np.min(neighbors[tind]) < 0:
          continue
       if len(np.intersect1d(connectionarray[neighbors[tind]], connectionarray[tind])) > 0 and\
          len(np.where(connectionarray[neighbors[tind]] != connectionarray[tind])[0]) > \
           len(np.where(connectionarray[neighbors[tind]] == connectionarray[tind])[0]):
           atprev = attenuation[tind].copy()
           values, counts = np.unique(attenuation[neighbors[tind]], return_counts=True)
           attenuation[tind] = values[np.argmax(counts)]
           indn = np.where(attenuation[neighbors[tind]] == values[np.argmax(counts)])[0][0]
           lennext = 1
           funpos = np.sum((sinogram - systemmat@attenuation)**2) + scalerseg*(lengrad - lennext)
           if funpos < funpos0: 
              connectionarray[tind] = connectionarray[neighbors[tind][indn]]
              lengrad = lengrad - lennext
              funpos0 = funpos 
           else:   
              attenuation[tind] = atprev
       if len(np.intersect1d(connectionarray[neighbors[tind]], connectionarray[tind])) == 0: 
            values, counts = np.unique(attenuation[neighbors[tind]], return_counts=True)
            attenuation[tind] = values[np.argmin(np.abs(values - attenuation[tind]))]
            indn = np.where(attenuation[neighbors[tind]] == values[np.argmin(np.abs(values - attenuation[tind]))])[0][0]
            lennext = len(np.where(connectionarray[neighbors[tind]] ==\
                          connectionarray[neighbors[tind]][indn])[0])
            funpos = np.sum((sinogram - systemmat@attenuation)**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[tind] = connectionarray[neighbors[tind]][indn]
     #for cons in np.unique(connectionarray):
   
    conun = np.unique(connectionarray)
    con_dict = {}
    for c in range(len(conun)):
        con_dict[conun[c]] = [conun[c]]
    for c in range(len(triangles)):
        if connectionarray[c] != c:           
           con_dict[connectionarray[c]].extend([c])
    
    #del atlist

    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])
    projdif0ns = systemmat@attenuation - sinogram  
    maxgrad = np.max(gradoveredges) 
    gradorder = np.argsort(gradoveredges)
    compared = np.zeros((2, 2))
    funpos0 = projdif0 + scalerseg*lengrad
    for gind in range(len(gradoveredges)):
       mingrad = gradorder[gind]#np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       tmin = np.min(connectionarray[np.array([triangle1, triangle2])])
       tmax = np.max(connectionarray[np.array([triangle1, triangle2])])
       con1 = con_dict[connectionarray[triangle1]]
       con2 = con_dict[connectionarray[triangle2]]
       if len(con1) > 10 and len(con2) > 10:
           if np.abs(attenuation[triangle1] - attenuation[triangle2]) > 0.1:
               continue
       
       wheremin = np.where(compared[:, 0] == tmin)[0]
       if len(wheremin) > 0:
          wheremax = np.where(compared[wheremin, 1] == tmax)[0]
          if len(wheremax) > 0:
             continue
       atprev = attenuation[np.hstack((con1, con2))].copy()
       attenuation[np.hstack((con1, con2))] = np.sum(attenuation[np.hstack((con1, con2))]* \
                                   areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
       lennext = 0
       for con in neighbors[con1].ravel():
         if con in con2:
           lennext += 1
       projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]- atprev)
       funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
       if funpos < funpos0:
           funpos0 = funpos
           lengrad -= lennext
           minval = np.min(connectionarray[np.array([triangle1, triangle2])])
           otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
           connectionarray[np.hstack((con1, con2))] = minval
           con_dict[minval].extend(con_dict[otherval]) 
           del con_dict[otherval]
           compared = np.zeros((2, 2))
       else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]\
            @(attenuation[np.hstack((con1, con2))]- atprev)
            attenuation[np.hstack((con1, con2))] = atprev
            compared = np.vstack((compared, np.array([tmin, tmax])))

    posvals = np.unique(connectionarray)
    systmatcrop = np.zeros((numangles*numpix, len(posvals)))
    for col in range(len(posvals)):
       rowsor = np.where(connectionarray == posvals[col])[0]
       systmatcrop[:, col] = np.array(systemmat[:, rowsor].sum(axis=1)).flatten()
     #for cons in np.unique(connectionarray):
    atlist = bbREC(systmatcrop, attenuation[posvals], 200, sinogram)
    for i in range(len(atlist)):
        attenuation[np.where(connectionarray == posvals[i])[0]] = atlist[i]


    return attenuation, connectionarray 


@cuda.jit()
def raytracerparallel(angle, numrays, vertices, edges, attenuation, projdif, normals):
   k, l = cuda.grid(2)
   c = len(edges)
   d = numrays
   if k < c and l < d:       
      ray = numrays - (l) - 1
      if angle == 0:
         startx = -1
         starty = (ray + 0.5)/numrays
      elif angle == np.pi/2:
         startx = 1-(ray + 0.5)/numrays
         starty = -1
      elif angle == np.pi:
         startx = 2
         starty = 1 - (ray + 0.5)/numrays
      elif angle == 3*np.pi/2:
         startx = (ray + 0.5)/numrays
         starty = 2
      else:
         startx = 0.5 - 1.5*np.cos(angle)  - (ray + 0.5 - numrays/2)*np.sin(angle)/numrays
         starty = 0.5 - 1.5*np.sin(angle)  +  (ray + 0.5 - numrays/2)*np.cos(angle)/numrays
      dirx = -np.sin(angle)
      diry = np.cos(angle)
      dir3 = -startx*dirx - starty*diry
      v1x = vertices[edges[k][0]][0]
      v1y = vertices[edges[k][0]][1]      
      v2x = vertices[edges[k][1]][0]
      v2y = vertices[edges[k][1]][1]
      intpar = -(dirx*v1x +\
                 diry*v1y + dir3)/ \
                 (dirx*(v2x - \
                  v1x)+diry \
                  *(v2y - \
                  v1y))
      normalsx = vertices[edges[k][0]][1] - vertices[edges[k][1]][1]
      normalsy = -vertices[edges[k][0]][0] + vertices[edges[k][1]][0]
      edgeang = np.arctan2(normalsy, normalsx)
      if normalsx*diry - normalsy*dirx > 0:
        signcross = 1
      else: 
         signcross = -1
      if edgeang < 0:
         edgeang = 2*np.pi + edgeang 
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = v1x + intpar*(v2x - v1x)
         inty = v1y + intpar*(v2y - v1y)
         xdist = ((intx - startx)**2 + (inty - starty)**2)**(1/2) 
         cuda.atomic.add(projdif, l, signcross*xdist*attenuation[k])
         #cuda.atomic.add(projdif, ray, xdist2*normalsx*attenuation[k])
         #cuda.atomic.add(projdif, ray, xdist2*normalsy*attenuation[k])
         #  cuda.atomic.add(vertgrad, 2*edges[k][0], -scaleredge*(vertices[edges[k][1]][0] - vertices[edges[k][0]][0]))
         #  cuda.atomic.add(vertgrad, 2*edges[k][0]+1, -scaleredge*(vertices[edges[k][1]][1] - vertices[edges[k][0]][1]))


@cuda.jit()
def raytracerparallel1(angle, numrays, scaleredge, vertices, edges, attenuation, vertgrad, projdif):
   k, l = cuda.grid(2)
   c = len(edges)
   d = numrays
   if k < c and l < d:       
      if l == 0: 
        cuda.atomic.add(vertgrad, 2*edges[k][1], scaleredge*2*(vertices[edges[k][1]][0] - vertices[edges[k][0]][0]))
        cuda.atomic.add(vertgrad, 2*edges[k][1]+1, scaleredge*2*(vertices[edges[k][1]][1] - vertices[edges[k][0]][1]))
      ray = numrays - (l) - 1
      if angle == 0:
         startx = -1
         starty = (ray + 0.5)/numrays
      elif angle == np.pi/2:
         startx = 1-(ray + 0.5)/numrays
         starty = -1
      elif angle == np.pi:
         startx = 2
         starty = 1 - (ray + 0.5)/numrays
      elif angle == 3*np.pi/2:
         startx = (ray + 0.5)/numrays
         starty = 2
      else:
         startx = 0.5 - 1.5*np.cos(angle)  - (ray + 0.5 - numrays/2)*np.sin(angle)/numrays
         starty = 0.5 - 1.5*np.sin(angle)  +  (ray + 0.5 - numrays/2)*np.cos(angle)/numrays
      dirx = -np.sin(angle)
      diry = np.cos(angle)
      dir3 = -startx*dirx - starty*diry
      v1x = vertices[edges[k][0]][0]
      v1y = vertices[edges[k][0]][1]      
      v2x = vertices[edges[k][1]][0]
      v2y = vertices[edges[k][1]][1]
      intpar = -(dirx*v1x +\
                 diry*v1y + dir3)/ \
                 (dirx*(v2x - \
                  v1x)+diry \
                  *(v2y - \
                  v1y))
      normalsx = v1y - v2y
      normalsy = -v1x + v2x
      norms = (normalsx**2+normalsy**2)**(1/2)
      normalsx = normalsx/norms
      normalsy = normalsy/norms
      edgeang = np.arctan2(normalsy, normalsx)
      maxlen = ((v2x- v1x)**2 + (v2y-v1y)**2)**(1/2)
      if edgeang < 0:
         edgeang = 2*np.pi + edgeang 
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = v1x + intpar*(v2x - v1x)
         inty = v1y + intpar*(v2y - v1y)
         if (np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2 < 10**(-10):
            xdist = 0
            xdist2 = 0
         else:   
            xdist = ((((intx- v1x)**2 + (inty-v1y)**2)/ \
           ((v2x- v1x)**2 + (v2y-v1y)**2))\
           /(np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2)**(1/2) 
            xdist2 = ((((intx- v2x)**2 + (inty-v2y)**2)/ \
           ((v2x- v1x)**2 + (v2y-v1y)**2)) \
           /(np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2)**(1/2)
            
      
         cuda.atomic.add(vertgrad, 2*edges[k][1], xdist*normalsx*projdif[l]*attenuation[k])
         cuda.atomic.add(vertgrad, 2*edges[k][1]+1, xdist*normalsy*projdif[l]*attenuation[k])
         #  cuda.atomic.add(vertgrad, 2*edges[k][0], xdist2*normals[k][0]*projdif[l]*attenuation[k])
         #  cuda.atomic.add(vertgrad, 2*edges[k][0]+1, xdist2*normals[k][1]*projdif[l]*attenuation[k])
         #  cuda.atomic.add(vertgrad, 2*edges[k][0], -scaleredge*(vertices[edges[k][1]][0] - vertices[edges[k][0]][0]))
         #  cuda.atomic.add(vertgrad, 2*edges[k][0]+1, -scaleredge*(vertices[edges[k][1]][1] - vertices[edges[k][0]][1]))

@cuda.jit()
def raytracerparallel2(angle, numrays, scaleredge, vertices, edges, attenuation, vertgrad, projdif):
   k, l = cuda.grid(2)
   c = len(edges)
   d = numrays
   if k < c and l < d:        
      if l == 0:
         cuda.atomic.add(vertgrad, 2*edges[k][0], -scaleredge*2*(vertices[edges[k][1]][0] - vertices[edges[k][0]][0]))
         cuda.atomic.add(vertgrad, 2*edges[k][0]+1, -scaleredge*2*(vertices[edges[k][1]][1] - vertices[edges[k][0]][1]))
      ray = numrays - (l ) - 1
      if angle == 0:
         startx = -1
         starty = (ray + 0.5)/numrays
      elif angle == np.pi/2:
         startx = 1 - (ray + 0.5)/numrays
         starty = -1
      elif angle == np.pi:
         startx = 2
         starty = 1 - (ray + 0.5)/numrays
      elif angle == 3*np.pi/2:
         startx = (ray + 0.5)/numrays
         starty = 2
      else:
         startx = 0.5 - 1.5*np.cos(angle)  - (ray + 0.5 - numrays/2)*np.sin(angle)/numrays
         starty = 0.5 - 1.5*np.sin(angle)  +  (ray + 0.5 - numrays/2)*np.cos(angle)/numrays
      dirx = -np.sin(angle)
      diry = np.cos(angle)
      dir3 = -startx*dirx - starty*diry
      v1x = vertices[edges[k][0]][0]
      v1y = vertices[edges[k][0]][1]      
      v2x = vertices[edges[k][1]][0]
      v2y = vertices[edges[k][1]][1]
      intpar = -(dirx*v1x +\
                 diry*v1y + dir3)/ \
                 (dirx*(v2x - \
                  v1x)+diry \
                  *(v2y - \
                  v1y))

      normalsx = vertices[edges[k][0]][1] - vertices[edges[k][1]][1]
      normalsy = -vertices[edges[k][0]][0] + vertices[edges[k][1]][0]
      norms = (normalsx**2+normalsy**2)**(1/2)
      normalsx = normalsx/norms
      normalsy = normalsy/norms
      edgeang = np.arctan2(normalsy, normalsx)
      maxlen = ((v2x - v1x)**2 + (v2y-v1y)**2)**(1/2)
      if edgeang < 0:
         edgeang = 2*np.pi + edgeang 
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = v1x + intpar*(v2x - v1x)
         inty = v1y + intpar*(v2y - v1y)
         if (np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2 < 10**(-10): 
            xdist = 0
            xdist2 = 0
         else:   
            xdist = ((((intx- v1x)**2 + (inty-v1y)**2)/ \
           ((v2x- v1x)**2 + (v2y-v1y)**2))\
           /(np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2)**(1/2) 
            xdist2 = ((((intx- v2x)**2 + (inty-v2y)**2)/ \
           ((v2x- v1x)**2 + (v2y-v1y)**2)) \
           /(np.cos(angle)*normalsx + np.sin(angle)*normalsy)**2)**(1/2)
    
         #cuda.atomic.add(vertgrad, 2*edges[k][1], xdist*normals[k][0]*projdif[l]*attenuation[k])
         #cuda.atomic.add(vertgrad, 2*edges[k][1]+1, xdist*normals[k][1]*projdif[l]*attenuation[k])
         cuda.atomic.add(vertgrad, 2*edges[k][0], xdist2*normalsx*projdif[l]*attenuation[k])
         cuda.atomic.add(vertgrad, 2*edges[k][0]+1, xdist2*normalsy*projdif[l]*attenuation[k])
          # cuda.atomic.add(vertgrad, 2*edges[k][1], scaleredge*(vertices[edges[k][1]][0] - vertices[edges[k][0]][0]))
          # cuda.atomic.add(vertgrad, 2*edges[k][1]+1, scaleredge*(vertices[edges[k][1]][1] - vertices[edges[k][0]][1]))


@cuda.jit()
def lenscore(scaleredge, vertices, triangles, otherverts, vertgrad): 
   k = cuda.grid(1)
   c = len(triangles)
   if k < c:        
      if otherverts[triangles[k][0]] == 1:
         cuda.atomic.add(vertgrad, 2*triangles[k][0], scaleredge*(vertices[triangles[k][0]][0] - vertices[triangles[k][1]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][0] + 1, scaleredge*(vertices[triangles[k][0]][1] - vertices[triangles[k][1]][1]))
         cuda.atomic.add(vertgrad, 2*triangles[k][0], scaleredge*(vertices[triangles[k][0]][0] - vertices[triangles[k][2]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][0] + 1, scaleredge*(vertices[triangles[k][0]][1] - vertices[triangles[k][2]][1]))
      if otherverts[triangles[k][1]] == 1:
         cuda.atomic.add(vertgrad, 2*triangles[k][1], scaleredge*(vertices[triangles[k][1]][0] - vertices[triangles[k][0]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][1] + 1, scaleredge*(vertices[triangles[k][1]][1] - vertices[triangles[k][0]][1]))
         cuda.atomic.add(vertgrad, 2*triangles[k][1], scaleredge*(vertices[triangles[k][1]][0] - vertices[triangles[k][2]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][1] + 1, scaleredge*(vertices[triangles[k][1]][1] - vertices[triangles[k][2]][1]))
      if otherverts[triangles[k][2]] == 1:
         cuda.atomic.add(vertgrad, 2*triangles[k][2], scaleredge*(vertices[triangles[k][2]][0] - vertices[triangles[k][1]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][2] + 1, scaleredge*(vertices[triangles[k][2]][1] - vertices[triangles[k][1]][1]))
         cuda.atomic.add(vertgrad, 2*triangles[k][2], scaleredge*(vertices[triangles[k][2]][0] - vertices[triangles[k][0]][0]))
         cuda.atomic.add(vertgrad, 2*triangles[k][2] + 1, scaleredge*(vertices[triangles[k][2]][1] - vertices[triangles[k][0]][1]))



@cuda.jit()
def systmatrixpar(angle, numrays, vertices, triangles, passedtriangles):
   k, l = cuda.grid(2)
   c = len(triangles)
   d = numrays
   if k < c and l < d:   
      ray = numrays - l - 1
      if angle == 0:
         startx = -1
         starty = (ray + 0.5)/numrays
      elif angle == np.pi/2:
         startx = 1 - (ray + 0.5)/numrays
         starty = -1
      elif angle == np.pi:
         startx = 2
         starty = 1 - (ray + 0.5)/numrays
      elif angle == 3*np.pi/2:
         startx = (ray + 0.5)/numrays
         starty = 2
      else:
         startx = 0.5 - 1.5*np.cos(angle)  - (ray + 0.5 - numrays/2)*np.sin(angle)/numrays
         starty = 0.5 - 1.5*np.sin(angle)  +  (ray + 0.5 - numrays/2)*np.cos(angle)/numrays
      dist1 = 0
      dist2 = 0
      dist3 = 0
      dirx = -np.sin(angle)
      diry = np.cos(angle)
      dir3 = -startx*dirx - starty*diry
      intpar = -(dirx*vertices[triangles[k][0]][0] +\
                 diry*vertices[triangles[k][0]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][1]][0] - \
                  vertices[triangles[k][0]][0])+diry \
                  *(vertices[triangles[k][1]][1] - \
                  vertices[triangles[k][0]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][0]][0] + intpar*(vertices[triangles[k][1]][0] - vertices[triangles[k][0]][0])
         inty = vertices[triangles[k][0]][1] + intpar*(vertices[triangles[k][1]][1] - vertices[triangles[k][0]][1])
         dist1 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      intpar = -(dirx*vertices[triangles[k][1]][0] +\
                 diry*vertices[triangles[k][1]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][2]][0] - \
                  vertices[triangles[k][1]][0])+diry \
                  *(vertices[triangles[k][2]][1] - \
                  vertices[triangles[k][1]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][1]][0] + intpar*(vertices[triangles[k][2]][0] - vertices[triangles[k][1]][0])
         inty = vertices[triangles[k][1]][1] + intpar*(vertices[triangles[k][2]][1] - vertices[triangles[k][1]][1])
         dist2 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      intpar = -(dirx*vertices[triangles[k][2]][0] +\
                 diry*vertices[triangles[k][2]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][0]][0] - \
                  vertices[triangles[k][2]][0])+diry \
                  *(vertices[triangles[k][0]][1] - \
                  vertices[triangles[k][2]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][2]][0] + intpar*(vertices[triangles[k][0]][0] - vertices[triangles[k][2]][0])
         inty = vertices[triangles[k][2]][1] + intpar*(vertices[triangles[k][0]][1] - vertices[triangles[k][2]][1])
         dist3 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      dif1 = ((dist1 - dist2)**2)**(1/2)
      dif2 = ((dist2 - dist3)**2)**(1/2)
      dif3 = ((dist3 - dist1)**2)**(1/2)
      if dist1 == 0 or dist2 == 0:
         dif1 = 0
      if dist2 == 0 or dist3 == 0:
         dif2 = 0
      if dist1 == 0 or dist3 == 0:
         dif3 = 0
      difmax = dif1
      if dif2 > difmax:
         difmax = dif2
      if dif3 > difmax:
         difmax = dif3
      cuda.atomic.add(passedtriangles, l*c + k, difmax)


def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) 
                + x3 * (y1 - y2)) / 2.0)
 
 
# A function to check whether point P(x, y)
# lies inside the triangle formed by 
# A(x1, y1), B(x2, y2) and C(x3, y3) 
def isInside(x1, y1, x2, y2, x3, y3, x, y):
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC 
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC 
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB 
    A3 = area (x1, y1, x2, y2, x, y)
     
    # Check if sum of A1, A2 and A3 
    # is same as A
    if np.abs(A -( A1 + A2 + A3)) < 10**(-13):
        return True
    else:
        return False
def refinetrianglefasterns(triang, triangles, vertices, attenuation, splitnu,  rads, cent, angles, numpix): 
    neighbours = triang.neighbors[splitnu]
    addcent =  any((isInside(*vertices[triangles[splitnu]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[0]]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[1]]].ravel(), *cent[splitnu]), \
                isInside(*vertices[triangles[neighbours[2]]].ravel(), *cent[splitnu])))
    if addcent:
      #aroundsplit = np.unique(np.hstack((np.where(triangles == triangles[splitnu][0])[0], \
      #               np.where(triangles == triangles[splitnu][1])[0], \
      #               np.where(triangles == triangles[splitnu][2])[0])))
      aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
      aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      for aroundind in range(3):
        aroundsplit = triang.neighbors[aroundsplit]
        aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      aroundsplit = np.unique(aroundsplit[np.hstack((np.where(triangles[aroundsplit] == triangles[splitnu][0])[0], \
                              np.where(triangles[aroundsplit] == triangles[splitnu][1])[0], \
                                 np.where(triangles[aroundsplit] == triangles[splitnu][2])[0]))]) 
      
      oldtriangs = triangles[aroundsplit]
      areaor = [Polygon(vertices[t]).area for t in oldtriangs]
      vertices = np.vstack((vertices, cent[splitnu]))
      verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
      vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
      constraints = np.array([0, 0])
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1 ))
           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areanew = [Polygon(vertices[t]).area for t in newtriangles]
      if np.abs(np.sum(areanew) - np.sum(areaor)) > 10**(-11):
         searchround = 3
         while np.abs(np.sum(areanew) - np.sum(areaor)) > 10**(-11) and searchround < 10:
           searchround+=1  
           neiroundvert = searchround - 4       
           aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           for aroundind in range(searchround):
             aroundsplit = triang.neighbors[aroundsplit]
             aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           aroundsplit0 = aroundsplit.copy()
           aroundsplit2 = np.hstack((triang.neighbors[splitnu], splitnu))
           aroundsplit2 = np.unique(aroundsplit2[aroundsplit > -0.5])
           for aind2 in range(neiroundvert):
             aroundsplit2 = triang.neighbors[aroundsplit2]
             aroundsplit2 = np.unique(aroundsplit2[aroundsplit2 > -0.5])
           vertcheck =  np.setdiff1d(np.unique(triangles[aroundsplit2]), triangles[splitnu])
           aroundsplit = np.unique(aroundsplit[np.hstack((np.where(triangles[aroundsplit] == triangles[splitnu][0])[0], \
                              np.where(triangles[aroundsplit] == triangles[splitnu][1])[0], \
                                 np.where(triangles[aroundsplit] == triangles[splitnu][2])[0]))]) 
           for aind in range(len(vertcheck)):
              aroundsplit = np.hstack((aroundsplit, np.where(triangles[aroundsplit0] == vertcheck[aind])[0]))
           aroundsplit = np.unique(aroundsplit)
      
           oldtriangs = triangles[aroundsplit]
           areaor = [Polygon(vertices[t]).area for t in oldtriangs]
           verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
           vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
           constraints = np.array([0, 0])
           for trold in oldtriangs:
              for otind in range(3):
                 v1 = trold[otind]
                 v2 = trold[np.mod(otind + 1, 3)]
                 if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
                    constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
           triangdata = dict(vertices=verts, segments=constraints)
           newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
           newtriangles = vertinds[newtriangles]
           areanew = [Polygon(vertices[t]).area for t in newtriangles]


      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      triangles = np.delete(triangles, aroundsplit, axis=0)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #newtriangles = newtriangles[keepareas]
      #areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      del trmat
      del oldtriangs
      del verts
    else:
      for nei in neighbours:
          if edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][1]], vertices[triangles[nei][2]]):
            if len(np.intersect1d(triangles[nei][np.array([1, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][1]], vertices[triangles[nei][2]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][1]] + vertices[triangles[nei][2]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][1]] - vertices[triangles[nei][2]], 2)/2
              break   
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][2]], vertices[triangles[nei][0]]):
            if  len(np.intersect1d(triangles[nei][np.array([0, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][2]], vertices[triangles[nei][0]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][2]] + vertices[triangles[nei][0]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][2]] - vertices[triangles[nei][0]], 2)/2
              break
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][0]], vertices[triangles[nei][1]]):
            if  len(np.intersect1d(triangles[nei][np.array([1, 0])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][0]], vertices[triangles[nei][1]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][0]] + vertices[triangles[nei][1]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][0]] - vertices[triangles[nei][1]], 2)/2
              break
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused-10**(-11))[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
     
      aroundsplit = np.hstack((np.array(np.where(triangles == triangles[nei][0])[0]),\
                               np.array(np.where(triangles == triangles[nei][1])[0]) )) 
      for mind in range(len(vertrem)): 
          aroundsplit = np.hstack((aroundsplit,np.where(triangles == vertrem[mind])[0] ))
      aroundsplit = np.unique(aroundsplit) 

  
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, addvert))
      verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
      constraints = np.array([0, 0])
      vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #areas = np.array(areas)[keepareas]
      #newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts



    return triangles, vertices, attenuation, \
         aroundsplit, np.arange(len(triangles)-len(newtriangles), len(triangles))




def refinetrianglefaster(triang, triangles, vertices, attenuation, systemmat, splitnu,  rads, cent, angles, numpix): 
    neighbours = triang.neighbors[splitnu]
    addcent =  any((isInside(*vertices[triangles[splitnu]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[0]]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[1]]].ravel(), *cent[splitnu]), \
                isInside(*vertices[triangles[neighbours[2]]].ravel(), *cent[splitnu])))
    if addcent:
      #aroundsplit = np.unique(np.hstack((np.where(triangles == triangles[splitnu][0])[0], \
      #               np.where(triangles == triangles[splitnu][1])[0], \
      #               np.where(triangles == triangles[splitnu][2])[0])))
      aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
      aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      for aroundind in range(3):
        aroundsplit = triang.neighbors[aroundsplit]
        aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      aroundsplit = np.unique(aroundsplit[np.hstack((np.where(triangles[aroundsplit] == triangles[splitnu][0])[0], \
                              np.where(triangles[aroundsplit] == triangles[splitnu][1])[0], \
                                 np.where(triangles[aroundsplit] == triangles[splitnu][2])[0]))]) 
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, cent[splitnu]))
      verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
      vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
      constraints = np.array([0, 0])
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1 ))

           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      triangles = np.delete(triangles, aroundsplit, axis=0)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #newtriangles = newtriangles[keepareas]
      #areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      systemmat = systemmat[:, mask]
      del trmat
      del oldtriangs
      del verts
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      
      systex = csc_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))
      #data = np.concatenate([systemmat.data, systex.data])
      #indices = np.concatenate([systemmat.indices, systex.indices ])
      #indptr = np.concatenate([systemmat.indptr,  systex.indptr[1:] + len(systemmat.data)])
      #systemmat = csc_matrix((data, indices, indptr), shape=(systemmat.shape[0], systemmat.shape[1] + systex.shape[1]))
      systemmat = hstack((systemmat,systex), format='csc')
      #systemmat.data = np.hstack((systemmat.data,systex.data))
      #systemmat.indices = np.hstack((systemmat.indices,systex.indices))
      #systemmat.indptr = np.hstack((systemmat.indptr,(systex.indptr + systemmat.nnz)[1:]))
      #systemmat._shape = (systemmat.shape[0]+systex.shape[0],systex.shape[1])
    else:
      for nei in neighbours:
          if edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][1]], vertices[triangles[nei][2]]):
            if len(np.intersect1d(triangles[nei][np.array([1, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][1]], vertices[triangles[nei][2]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][1]] + vertices[triangles[nei][2]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][1]] - vertices[triangles[nei][2]], 2)/2
              break   
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][2]], vertices[triangles[nei][0]]):
            if  len(np.intersect1d(triangles[nei][np.array([0, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][2]], vertices[triangles[nei][0]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][2]] + vertices[triangles[nei][0]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][2]] - vertices[triangles[nei][0]], 2)/2
              break
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][0]], vertices[triangles[nei][1]]):
            if  len(np.intersect1d(triangles[nei][np.array([1, 0])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][0]], vertices[triangles[nei][1]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][0]] + vertices[triangles[nei][1]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][0]] - vertices[triangles[nei][1]], 2)/2
              break
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused - 10**(-11))[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
     
      aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
      aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      for aroundind in range(6):
        aroundsplit = triang.neighbors[aroundsplit]
        aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      posind =   np.where(triangles[aroundsplit] == vertrem[0])[0]
      for remind in range(1, len(vertrem)):
         posind = np.hstack((posind, np.where(triangles[aroundsplit] == vertrem[remind])[0])) 
      aroundsplit = aroundsplit[np.unique(posind)] 
  
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, addvert))
      verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
      constraints = np.array([0, 0])
      vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #areas = np.array(areas)[keepareas]
      #newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      systemmat = systemmat[:, mask]
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      systex = csc_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))
      #data = np.concatenate([systemmat.data, systex.data])
      #indices = np.concatenate([systemmat.indices, systex.indices ])
      #indptr = np.concatenate([systemmat.indptr,  systex.indptr[1:] + len(systemmat.data)])
      #systemmat = csc_matrix((data, indices, indptr), shape=(systemmat.shape[0], systemmat.shape[1] + systex.shape[1]))
      systemmat = hstack((systemmat,systex), format='csc')

    return triangles, vertices, attenuation, \
         systemmat,aroundsplit, np.arange(len(triangles)-len(newtriangles), len(triangles))


def refinetrianglefaster2(triang, triangles, vertices, attenuation, systemmat, splitnu,  rads, cent, angles, numpix): 
    neighbours = triang.neighbors[splitnu]
    addcent =  any((isInside(*vertices[triangles[splitnu]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[0]]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[1]]].ravel(), *cent[splitnu]), \
                isInside(*vertices[triangles[neighbours[2]]].ravel(), *cent[splitnu])))
    if addcent:
      #aroundsplit = np.unique(np.hstack((np.where(triangles == triangles[splitnu][0])[0], \
      #               np.where(triangles == triangles[splitnu][1])[0], \
      #               np.where(triangles == triangles[splitnu][2])[0])))
      start_timearound = time.time()
      aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
      aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      for aroundind in range(3):
        aroundsplit = triang.neighbors[aroundsplit]
        aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      aroundsplit = np.unique(aroundsplit[np.hstack((np.where(triangles[aroundsplit] == triangles[splitnu][0])[0], \
                              np.where(triangles[aroundsplit] == triangles[splitnu][1])[0], \
                                 np.where(triangles[aroundsplit] == triangles[splitnu][2])[0]))]) 
      print("time to calculate aroundsplit--- %s seconds ---" % (time.time() - start_timearound)) 
      start_timecons = time.time()
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, cent[splitnu]))
      verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
      vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
      constraints = np.array([0, 0])
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      print("time to calculate constraints --- %s seconds ---" % (time.time() - start_timecons))
      start_timeintersection = time.time()
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      print("time to to calculate intersection area--- %s seconds ---" % (time.time() - start_timeintersection)) 
      start_timeintear= time.time()
      triangles = np.delete(triangles, aroundsplit, axis=0)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #newtriangles = newtriangles[keepareas]
      #areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      print("time to manipulations1--- %s seconds ---" % (time.time() - start_timeintear)) 
      start_timeintear= time.time()
      systemmat = systemmat[:, mask]
      print("time to manipulations--- %s seconds ---" % (time.time() - start_timeintear)) 
      del trmat
      del oldtriangs
      del verts
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      start_timerunsysy = time.time()
      systex = csc_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))
      start_timesyst = time.time()
      #data = np.concatenate([systemmat.data, systex.data])
      #indices = np.concatenate([systemmat.indices, systex.indices ])
      #indptr = np.concatenate([systemmat.indptr,  systex.indptr[1:] + len(systemmat.data)])
      #systemmat = csc_matrix((data, indices, indptr), shape=(systemmat.shape[0], systemmat.shape[1] + systex.shape[1]))
      systemmat = hstack((systemmat,systex), format='csc')
      #systemmat.data = np.hstack((systemmat.data,systex.data))
      #systemmat.indices = np.hstack((systemmat.indices,systex.indices))
      #systemmat.indptr = np.hstack((systemmat.indptr,(systex.indptr + systemmat.nnz)[1:]))
      #systemmat._shape = (systemmat.shape[0]+systex.shape[0],systex.shape[1])
      print("time to stack systemmat --- %s seconds ---" % (time.time() - start_timesyst)) 
    else:
      for nei in neighbours:
          if edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][1]], vertices[triangles[nei][2]]):
            if len(np.intersect1d(triangles[nei][np.array([1, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][1]], vertices[triangles[nei][2]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][1]] + vertices[triangles[nei][2]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][1]] - vertices[triangles[nei][2]], 2)/2
              break   
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][2]], vertices[triangles[nei][0]]):
            if  len(np.intersect1d(triangles[nei][np.array([0, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][2]], vertices[triangles[nei][0]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][2]] + vertices[triangles[nei][0]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][2]] - vertices[triangles[nei][0]], 2)/2
              break
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][0]], vertices[triangles[nei][1]]):
            if  len(np.intersect1d(triangles[nei][np.array([1, 0])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][0]], vertices[triangles[nei][1]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][0]] + vertices[triangles[nei][1]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][0]] - vertices[triangles[nei][1]], 2)/2
              break
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused-10**(-11))[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
     
      aroundsplit = np.hstack((triang.neighbors[splitnu], splitnu))
      aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      for aroundind in range(5):
        aroundsplit = triang.neighbors[aroundsplit]
        aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
      posind =   np.where(triangles[aroundsplit] == vertrem[0])[0]
      for remind in range(1, len(vertrem)):
         posind = np.hstack((posind, np.where(triangles[aroundsplit] == vertrem[remind])[0])) 
      aroundsplit = aroundsplit[np.unique(posind)] 
  
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, addvert))
      verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
      constraints = np.array([0, 0])
      vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #areas = np.array(areas)[keepareas]
      #newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      systemmat = systemmat[:, mask]
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      systex = csc_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))
      start_timesyst = time.time()
      #data = np.concatenate([systemmat.data, systex.data])
      #indices = np.concatenate([systemmat.indices, systex.indices ])
      #indptr = np.concatenate([systemmat.indptr,  systex.indptr[1:] + len(systemmat.data)])
      #systemmat = csc_matrix((data, indices, indptr), shape=(systemmat.shape[0], systemmat.shape[1] + systex.shape[1]))
      systemmat = hstack((systemmat,systex), format='csc')

    return triangles, vertices, attenuation, \
         systemmat,aroundsplit, np.arange(len(triangles)-len(newtriangles), len(triangles))

def edgecollapse(vertices, triangles, attenuation, minvar, qmax, maxnum, scalcol, aspexp):


  triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

  corners = np.hstack((np.argmin(np.linalg.norm(vertices, 2, 1)), np.argmax(np.linalg.norm(vertices, 2, 1)), \
                    np.intersect1d(np.where(np.min(vertices, axis=1) == 0)[0], np.where(np.max(vertices, axis=1) == 1)[0])) )
  randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))
  
  
  if scalcol == 0: 
     edges, scorepervert, qualperedge, scalercollapse = createedges(triangles, corners, randverts, attenuation, 0)
  else:
     edges, scorepervert, qualperedge, scalercollapse = createedges(triangles, corners, randverts, attenuation,-1)
     scalercollapse = scalercollapse*scalcol 
  
  minmerg = np.min(qualperedge)
  edcol = np.argmin(qualperedge)
  edges = np.array(edges)

  minstart = minvar  - scalercollapse*aspexp
  counter = 0
  while minmerg < minstart  and counter < maxnum :
        vert1 = edges[edcol][0]
        vert2 = edges[edcol][1] 
        trianglecol = np.intersect1d(np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])
        trianglechange = np.setdiff1d(np.hstack((np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])), trianglecol)
        vertices = np.vstack((vertices, (vertices[vert1] + vertices[vert2])/2 ))
        triangnow = triangles[trianglechange].copy()
        triangnow[np.where(triangnow == vert1)] = len(vertices) -1
        triangnow[np.where(triangnow == vert2)] = len(vertices) -1
        radss, _ = circumcircle(vertices[triangnow])
        areas = [Polygon(vertices[t]).area for t in triangnow]
        triangold = np.hstack((trianglecol, trianglechange))
        areas0 = [Polygon(vertices[triangles[t]]).area for t in triangold]
        asnew = np.max(asppertr(vertices[triangnow], radss))
        vertices = np.delete(vertices, len(vertices)-1, axis=0)
        if  asnew < qmax and np.abs(np.sum(areas) - np.sum(areas0)) < 10**(-11):
             vertices, triangles, attenuation, edges, qualperedge, scorepervert, corners = \
                   collapseedge(vertices, triangles, attenuation, edges, qualperedge, scorepervert, edcol, corners, scalercollapse)         
        else:
            qualperedge[edcol] = np.max(qualperedge)
        minmerg = np.min(qualperedge)
        edcol = np.argmin(qualperedge)
        counter+=1


  

  return vertices, triangles, attenuation 

def refinetriangleret(triang, triangles, vertices, attenuation, splitnu, systemmat, rads, cent, angles, numpix): 
    neighbours = triang.neighbors[splitnu]
    addcent =  any((isInside(*vertices[triangles[splitnu]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[0]]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[1]]].ravel(), *cent[splitnu]), \
                isInside(*vertices[triangles[neighbours[2]]].ravel(), *cent[splitnu])))

    if addcent:
      aroundsplit = np.unique(np.hstack((np.where(triangles == triangles[splitnu][0])[0], \
                     np.where(triangles == triangles[splitnu][1])[0], \
                     np.where(triangles == triangles[splitnu][2])[0])))
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, cent[splitnu]))
      verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
      vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
      constraints = np.array([0, 0])
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      triangles = np.delete(triangles, aroundsplit, axis=0)
      numrep =len(aroundsplit)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #newtriangles = newtriangles[keepareas]
      #areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      systemmat = systemmat[:, mask]
      del trmat
      del oldtriangs
      del verts
      del aroundsplit
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      systemmat = hstack([systemmat, csr_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))])

      del newtriangles
    else:
      for nei in neighbours:
          if edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][1]], vertices[triangles[nei][2]]):
            if len(np.intersect1d(triangles[nei][np.array([1, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][1]], vertices[triangles[nei][2]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][1]] + vertices[triangles[nei][2]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][1]] - vertices[triangles[nei][2]], 2)/2
              break   
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][2]], vertices[triangles[nei][0]]):
            if  len(np.intersect1d(triangles[nei][np.array([0, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][2]], vertices[triangles[nei][0]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][2]] + vertices[triangles[nei][0]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][2]] - vertices[triangles[nei][0]], 2)/2
              break
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][0]], vertices[triangles[nei][1]]):
            if  len(np.intersect1d(triangles[nei][np.array([1, 0])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][0]], vertices[triangles[nei][1]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][0]] + vertices[triangles[nei][1]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][0]] - vertices[triangles[nei][1]], 2)/2
              break
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused-10**(-11))[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
     
      aroundsplit = np.hstack((np.array(np.where(triangles == triangles[nei][0])[0]),\
                               np.array(np.where(triangles == triangles[nei][1])[0]) )) 
      for mind in range(len(vertrem)): 
          aroundsplit = np.hstack((aroundsplit,np.where(triangles == vertrem[mind])[0] ))
      aroundsplit = np.unique(aroundsplit) 

  
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, addvert))
      verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
      constraints = np.array([0, 0])
      vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #areas = np.array(areas)[keepareas]
      #newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts




    return triangles, vertices, attenuation, systemmat, numrep


def refinetriangle(triang, triangles, vertices, attenuation, splitnu, systemmat, rads, cent, angles, numpix): 
    neighbours = triang.neighbors[splitnu]
    addcent =  any((isInside(*vertices[triangles[splitnu]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[0]]].ravel(), *cent[splitnu]), \
            isInside(*vertices[triangles[neighbours[1]]].ravel(), *cent[splitnu]), \
                isInside(*vertices[triangles[neighbours[2]]].ravel(), *cent[splitnu])))

    if addcent:
      aroundsplit = np.unique(np.hstack((np.where(triangles == triangles[splitnu][0])[0], \
                     np.where(triangles == triangles[splitnu][1])[0], \
                     np.where(triangles == triangles[splitnu][2])[0])))
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, cent[splitnu]))
      verts = np.vstack((vertices[np.unique(triangles[aroundsplit])], cent[splitnu]))
      vertinds = np.hstack((np.unique(triangles[aroundsplit]), len(vertices)-1))
      constraints = np.array([0, 0])
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) < 2:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      triangles = np.delete(triangles, aroundsplit, axis=0)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #newtriangles = newtriangles[keepareas]
      #areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      mask = np.ones(systemmat.shape[1], dtype=bool)
      mask[aroundsplit] = False
      systemmat = systemmat[:, mask]
      del trmat
      del oldtriangs
      del verts
      del aroundsplit
      del splitnu
      del mask
      block_x = 32
      block_y = 32
      block = (block_x, block_y)
      grid = ((len(newtriangles))//block_x+1, (numpix*len(angles))//block_x+1) 
      verticesc = cuda.to_device(vertices)
      trianglesc = cuda.to_device(newtriangles)
      systemmatc = cuda.to_device(np.zeros(numpix*len(newtriangles)*len(angles)))
      systmatrixparallangs[grid, block](angles, numpix, verticesc,\
        trianglesc, systemmatc)
      systemmat = hstack([systemmat, csr_matrix(systemmatc.copy_to_host().reshape((numpix*len(angles), len(newtriangles))))])

      del newtriangles
    else:
      for nei in neighbours:
          if edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][1]], vertices[triangles[nei][2]]):
            if len(np.intersect1d(triangles[nei][np.array([1, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][1]], vertices[triangles[nei][2]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][1]] + vertices[triangles[nei][2]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][1]] - vertices[triangles[nei][2]], 2)/2
              break   
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][2]], vertices[triangles[nei][0]]):
            if  len(np.intersect1d(triangles[nei][np.array([0, 2])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][2]], vertices[triangles[nei][0]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][2]] + vertices[triangles[nei][0]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][2]] - vertices[triangles[nei][0]], 2)/2
              break
          if  edge_intersection(cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])], vertices[triangles[nei][0]], vertices[triangles[nei][1]]):
            if  len(np.intersect1d(triangles[nei][np.array([1, 0])], triangles[splitnu])) < 2 and edge_intersection(vertices[triangles[nei][0]], vertices[triangles[nei][1]], cent[splitnu], vertices[np.setdiff1d(triangles[splitnu], triangles[nei])]):
              addvert = (vertices[triangles[nei][0]] + vertices[triangles[nei][1]])/2
              radiused = np.linalg.norm(vertices[triangles[nei][0]] - vertices[triangles[nei][1]], 2)/2
              break
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused-10**(-11))[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
     
      aroundsplit = np.hstack((np.array(np.where(triangles == triangles[nei][0])[0]),\
                               np.array(np.where(triangles == triangles[nei][1])[0]) )) 
      for mind in range(len(vertrem)): 
          aroundsplit = np.hstack((aroundsplit,np.where(triangles == vertrem[mind])[0] ))
      aroundsplit = np.unique(aroundsplit) 

  
      oldtriangs = triangles[aroundsplit]
      vertices = np.vstack((vertices, addvert))
      verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
      constraints = np.array([0, 0])
      vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
      for trold in oldtriangs:
         for otind in range(3):
            v1 = trold[otind]
            v2 = trold[np.mod(otind + 1, 3)]
            if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
               constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
      constraints = np.delete(constraints, 0, axis = 0)
      constraints = constraints.astype(int)
      if np.max(np.bincount(constraints.ravel())) > 2.5: 
         while np.max(np.bincount(constraints.ravel())) > 2.5: 
           aroundsplit = triang.neighbors[aroundsplit]
           aroundsplit = np.unique(aroundsplit[aroundsplit > -0.5])
           oldtriangs = triangles[aroundsplit]
           verts = np.vstack((vertices[np.unique(np.setdiff1d(triangles[aroundsplit], vertdel))], addvert))
           constraints = np.array([0, 0])
           vertinds = np.hstack((np.unique(np.setdiff1d(triangles[aroundsplit], vertdel)), len(vertices)-1 ))
           for trold in oldtriangs:
             for otind in range(3):
               v1 = trold[otind]
               v2 = trold[np.mod(otind + 1, 3)]
               if len(np.intersect1d(np.where(oldtriangs == v1)[0], np.where(oldtriangs == v2)[0])) == 1:
                 constraints = np.vstack((constraints, np.array([np.argmax(vertinds == v1), np.argmax(vertinds == v2)])))
           constraints = np.delete(constraints, 0, axis = 0)
           constraints = constraints.astype(int)
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      #keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      #areas = np.array(areas)[keepareas]
      #newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      #trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts

    return triangles, vertices, attenuation, systemmat


def edge_intersection(p1, p2, q1, q2):


    p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))

    r = p2 - p1
    s = q2 - q1

    rxs = np.cross(r, s)
    qp = q1 - p1
    qpxr = np.cross(qp, r)

    if rxs == 0:
        return False  

    t = np.cross(qp, s) / rxs
    u = qpxr / rxs
    #print(u)
    if -10**(-13) <= u <= 1+10**(-13):
        #intersection = p1 + t * r
        return True 

    return False  

@cuda.jit()
def lenpervertcalc(vertices, edges, lentermpervert):
   k = cuda.grid(1)
   maxk = len(edges)
   if k < maxk: 
     if vertices[edges[k][0]][0] > 0 and vertices[edges[k][0]][1] > 0 and vertices[edges[k][0]][0] < 1 and vertices[edges[k][0]][1] < 1:
       cuda.atomic.add(lentermpervert, 2*edges[k][0], vertices[edges[k][0]][0] - vertices[edges[k][1]][0])
       cuda.atomic.add(lentermpervert, 2*edges[k][0]+1, vertices[edges[k][0]][1] - vertices[edges[k][1]][1])
     if vertices[edges[k][1]][0] > 0 and vertices[edges[k][1]][1] > 0 and vertices[edges[k][1]][0] < 1 and vertices[edges[k][1]][1] < 1:
       cuda.atomic.add(lentermpervert, 2*edges[k][1], vertices[edges[k][1]][0] - vertices[edges[k][0]][0])
       cuda.atomic.add(lentermpervert, 2*edges[k][1]+1, vertices[edges[k][1]][1] - vertices[edges[k][0]][1])


@cuda.jit()
def systmatrixparallangs(angles, numrays, vertices, triangles, passedtriangles):
   k, l = cuda.grid(2)
   c = len(triangles)
   d = numrays*len(angles)
   if k < c and l < d:   
      angle = angles[l//numrays]
      ray = numrays - (l - (l//numrays)*numrays) - 1
      if angle == 0:
         startx = -1
         starty = (ray + 0.5)/numrays
      elif angle == np.pi/2:
         startx = 1 - (ray + 0.5)/numrays
         starty = -1
      elif angle == np.pi:
         startx = 2
         starty = 1 - (ray + 0.5)/numrays
      elif angle == 3*np.pi/2:
         startx = (ray + 0.5)/numrays
         starty = 2
      else:
         startx = 0.5 - 1.5*np.cos(angle)  - (ray + 0.5 - numrays/2)*np.sin(angle)/numrays
         starty = 0.5 - 1.5*np.sin(angle)  +  (ray + 0.5 - numrays/2)*np.cos(angle)/numrays
      dist1 = 0
      dist2 = 0
      dist3 = 0
      dirx = -np.sin(angle)
      diry = np.cos(angle)
      dir3 = -startx*dirx - starty*diry
      intpar = -(dirx*vertices[triangles[k][0]][0] +\
                 diry*vertices[triangles[k][0]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][1]][0] - \
                  vertices[triangles[k][0]][0])+diry \
                  *(vertices[triangles[k][1]][1] - \
                  vertices[triangles[k][0]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][0]][0] + intpar*(vertices[triangles[k][1]][0] - vertices[triangles[k][0]][0])
         inty = vertices[triangles[k][0]][1] + intpar*(vertices[triangles[k][1]][1] - vertices[triangles[k][0]][1])
         dist1 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      intpar = -(dirx*vertices[triangles[k][1]][0] +\
                 diry*vertices[triangles[k][1]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][2]][0] - \
                  vertices[triangles[k][1]][0])+diry \
                  *(vertices[triangles[k][2]][1] - \
                  vertices[triangles[k][1]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][1]][0] + intpar*(vertices[triangles[k][2]][0] - vertices[triangles[k][1]][0])
         inty = vertices[triangles[k][1]][1] + intpar*(vertices[triangles[k][2]][1] - vertices[triangles[k][1]][1])
         dist2 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      intpar = -(dirx*vertices[triangles[k][2]][0] +\
                 diry*vertices[triangles[k][2]][1] + dir3)/ \
                 (dirx*(vertices[triangles[k][0]][0] - \
                  vertices[triangles[k][2]][0])+diry \
                  *(vertices[triangles[k][0]][1] - \
                  vertices[triangles[k][2]][1]))
      if intpar < 1+10**(-9) and intpar > -10**(-9):
         intx = vertices[triangles[k][2]][0] + intpar*(vertices[triangles[k][0]][0] - vertices[triangles[k][2]][0])
         inty = vertices[triangles[k][2]][1] + intpar*(vertices[triangles[k][0]][1] - vertices[triangles[k][2]][1])
         dist3 = ((intx - startx)**2 + (inty - starty)**2)**(1/2)
      dif1 = ((dist1 - dist2)**2)**(1/2)
      dif2 = ((dist2 - dist3)**2)**(1/2)
      dif3 = ((dist3 - dist1)**2)**(1/2)
      if dist1 == 0 or dist2 == 0:
         dif1 = 0
      if dist2 == 0 or dist3 == 0:
         dif2 = 0
      if dist1 == 0 or dist3 == 0:
         dif3 = 0
      difmax = dif1
      if dif2 > difmax:
         difmax = dif2
      if dif3 > difmax:
         difmax = dif3
      cuda.atomic.add(passedtriangles, l*c + k, difmax)

def Computesytemmatrix(vertices, triangles, numpix, angles):

  block_x = 32
  block_y = 32
  block = (block_x, block_y)
  grid = ((len(triangles))//block_x+1, (numpix*10)//block_x+1) 
  verticesc = cuda.to_device(vertices)
  trianglesc = cuda.to_device(triangles)
  systemmat = []
  for k in range(0, len(angles), 10):
    systemmatc = cuda.to_device(np.zeros(numpix*len(triangles)*10))
    systmatrixparallangs[grid, block](angles[np.arange(k, k+10)], numpix, verticesc,\
         trianglesc, systemmatc)
    systemmat.append(csr_matrix(systemmatc.copy_to_host().reshape((numpix*10, len(triangles)))))
  systemmat = vstack(systemmat)
  return systemmat


def Refinement(vertices, triangles, attenuation, noise, res, systemmat, maxvert, stop1, updateat):

  triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    
  rads, cent = circumcircle(vertices[triangles])
  
  randdriehoeken = np.unique(np.where(triang.neighbors < 0)[0])
  
  splittriangles = np.setdiff1d(np.arange(len(triangles)), \
                np.hstack((randdriehoeken, np.where(rads < res)[0], \
                np.where(np.min(cent,axis=1) < 0)[0], \
                    np.where(np.min(cent,axis=1) > 1)[0])))
  atpertriangle = np.zeros((len(triangles), 3))
  atpertriangle[splittriangles, 0] = np.abs(attenuation[splittriangles] - attenuation[triang.neighbors[splittriangles]][:,0])
  atpertriangle[splittriangles, 1] =  np.abs(attenuation[splittriangles] - attenuation[triang.neighbors[splittriangles]][:,1])
  atpertriangle[splittriangles, 2] = np.abs(attenuation[splittriangles] - attenuation[triang.neighbors[splittriangles]][:,2])
  atpertriangle = np.max(atpertriangle, axis=1)
  conditionnumber = np.sqrt(np.max((eigs((systemmat.transpose())@systemmat)[0]).real)/np.min((eigs((systemmat.transpose())@systemmat)[0]).real))
  stopx = 2*((np.sqrt(len(sinogram))*noise*conditionnumber*np.linalg.norm(attenuation, 2)/np.linalg.norm(sinogram, 2))/np.sqrt(len(triangles)))
  verh = stopx/qmax
  maxatt = np.max(attenuation)
  stopcond = stopx + verh*np.sqrt(2)/2
  areas = [Polygon(vertices[t]).area for t in triangles]
  extrascore = np.zeros(len(triangles))
  extrascore[splittriangles] = asppertr(vertices[triangles[splittriangles]], rads[splittriangles]) 


  counter = 0
  if updateat: 
    Cmat =  np.array(1/np.sum(systemmat, axis = 1)).ravel()
    Rmat =  np.array((1/np.sum(systemmat, axis = 0))).ravel()
    Cmat[np.where(np.sum(systemmat, axis = 1) == 0)[0]] = 0
    Rmat[np.where(np.sum(systemmat, axis = 0) == 0)[0]] = 0
  stopref = 0
  stop2 = 0
  counter = 0
  systemmat = systemmat.tocsc()

  triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
  while counter < maxvert and np.max(atpertriangle + verh*extrascore) > stopcond:
      
       splitnu = np.argmax(atpertriangle + verh*extrascore)
       if updateat:
         triangles, vertices, attenuation, systemmat,  deletedtr, extratr = refinetrianglefaster(triang, triangles, vertices, attenuation, systemmat, splitnu, rads, cent, angles, numpix)
         Rmat = np.delete(Rmat, deletedtr)
         Rmatex = np.array((1/np.sum(systemmat[:, extratr], axis = 0)))[0].ravel()
         Rmatex[np.where(np.sum(systemmat[:, extratr], axis = 0) == 0)[0]] = 0
         Rmat = np.hstack((Rmat, Rmatex))
         attenuation, _, _ = SirtrecCR(systemmat, attenuation, 50, sinogram, stop1, stop2, Cmat, Rmat)
       else: 
         triangles, vertices, attenuation, deletedtr, extratr = refinetrianglefasterns(triang, triangles, vertices, attenuation, splitnu, rads, cent, angles, numpix)
       triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles) 
       counter +=1
       rads = np.delete(rads,deletedtr)
       cent = np.delete(cent, deletedtr, axis=0)
       radsex, centex = circumcircle(vertices[triangles[extratr]])
       rads = np.hstack((rads, radsex))
       cent = np.vstack((cent, centex))   
       atpertriangle = np.delete(atpertriangle, deletedtr)
       atex = np.zeros((len(extratr), 3))   
       atex[:, 0] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,0])
       atex[:, 1] =  np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,1])
       atex[:, 2] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,2])
       atex[np.where(radsex < res)] = 0
       atex[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       atpertriangle = np.hstack((atpertriangle, np.max(atex, axis=1)))
       extrascore = np.delete(extrascore, deletedtr)
       extraextrascore = asppertr(vertices[triangles[extratr]], rads[extratr])
       extraextrascore[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       extrascore = np.hstack((extrascore, extraextrascore)) 
 


  return vertices, triangles, attenuation 

def Startingclusterfunction(attenuation, vertices, triangles, sinogram, systemmat, neighbors):
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)

    startvalue = 1

    scalerseg = startvalue*projdif0/(2*len(np.where((neighbors).ravel() > -0.5)[0]))
    simplificdegree = 10
    nummax = 100
    attenuationor = attenuation.copy()
    connectionarray = np.arange(len(triangles))
    while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
     scalerseg = scalerseg*2


     connectionarray = np.arange(len(triangles))
     projdif0 = np.sum((sinogram - systemmat@attenuation)**2)
     numangles = int(len(sinogram)/numpix)
     startvalue = 1
     con_dict = {}
     for c in range(len(triangles)):
         con_dict[c] = [c]

     areas = [Polygon(vertices[t]).area for t in triangles]
     areas = np.array(areas)
     trianglist = np.zeros((len(np.where((neighbors) > -0.5)[0]), 2))
     trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
     trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
     trianglist = trianglist.astype(int)
     attenuation = attenuationor.copy()
     gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])
     projdif0ns = systemmat@attenuation - sinogram  

     lengrad = len(gradoveredges)/2
     funpos0 = projdif0 + scalerseg*lengrad
     simplificdegree = 10
     nummax = 100
     maxgrad = np.max(gradoveredges) 
     gradorder = np.argsort(gradoveredges)
     compared = np.zeros((2, 2))
     for gind in range(len(gradoveredges)):
        mingrad = gradorder[gind]#np.argmin(gradoveredges)
        triangle1 = trianglist[mingrad][0]
        triangle2 = trianglist[mingrad][1]
        if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
        tmin = np.min(connectionarray[np.array([triangle1, triangle2])])
        tmax = np.max(connectionarray[np.array([triangle1, triangle2])])
        con1 = con_dict[connectionarray[triangle1]]
        con2 = con_dict[connectionarray[triangle2]]
        if len(con1) > 10 and len(con2) > 10:
           if np.abs(attenuation[triangle1] - attenuation[triangle2]) > 0.1:
               continue
       
        wheremin = np.where(compared[:, 0] == tmin)[0]
        if len(wheremin) > 0:
           wheremax = np.where(compared[wheremin, 1] == tmax)[0]
           if len(wheremax) > 0:
             continue
        atprev = attenuation[np.hstack((con1, con2))].copy()
        attenuation[np.hstack((con1, con2))] = np.sum(attenuation[np.hstack((con1, con2))]* \
                                   areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])

        lennext = 0
        for con in neighbors[con1].ravel():
          if con in con2:
              lennext += 1
            

        projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuation[np.hstack((con1, con2))]- atprev)
        funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
        if funpos < funpos0:
           funpos0 = funpos
           lengrad -= lennext
           minval = np.min(connectionarray[np.array([triangle1, triangle2])])
           otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
           connectionarray[np.hstack((con1, con2))] = minval
           con_dict[minval].extend(con_dict[otherval]) 
           del con_dict[otherval]
           compared = np.zeros((2, 2))
        else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]\
            @(attenuation[np.hstack((con1, con2))]- atprev)
            attenuation[np.hstack((con1, con2))] = atprev
            compared = np.vstack((compared, np.array([tmin, tmax])))
       #gradoveredges[mingrad] = maxgrad

    
    return attenuation, connectionarray, scalerseg 

def Startingclusterfunctionor(attenuation, vertices, triangles, sinogram, systemmat, neighbors):
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)

    startvalue = 1

    scalerseg = startvalue*projdif0/(2*len(np.where((neighbors).ravel() > -0.5)[0]))
    simplificdegree = 10
    nummax = 100
    connectionarray = np.arange(len(triangles))
    while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
     scalerseg = scalerseg*2
     attenuation, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)
     
    
    return attenuation, connectionarray, scalerseg 
def collapseedgenoscores(vertices, triangles, attenuation, vert1, vert2, edges, trianglescheck):
    trianglecol = np.intersect1d(np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])
    edcol = np.intersect1d(np.where(edges == vert1)[0], np.where(edges == vert2)[0])
    trianglechange = np.setdiff1d(np.hstack((np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])), trianglecol)
    vertices = np.vstack((vertices, (vertices[vert1] + vertices[vert2])/2 ))
    triangnow = triangles[trianglechange].copy()
    triangnow[np.where(triangnow == vert1)] = len(vertices) -1
    triangnow[np.where(triangnow == vert2)] = len(vertices) -1
    trianglescheck = np.delete(trianglescheck, np.where(trianglescheck == trianglecol[0])[0])
    trianglescheck = np.delete(trianglescheck, np.where(trianglescheck == trianglecol[1])[0])
    trianglescheck[np.where(trianglescheck == trianglecol[1])]
    trianglescheck[np.where(trianglescheck > trianglecol[1])] = trianglescheck[np.where(trianglescheck > trianglecol[1])] - 1 
    trianglescheck[np.where(trianglescheck > trianglecol[0])] = trianglescheck[np.where(trianglescheck > trianglecol[0])] - 1
    edges[np.where(edges == vert1)] = len(vertices) -1
    edges[np.where(edges == vert2)] = len(vertices) -1

    trmat = triangle_area_intersection(triangnow, triangles[np.hstack((trianglecol, trianglechange))], vertices)
    areas = [Polygon(vertices[t]).area for t in triangnow]
    attenuation[trianglechange] =  (trmat@attenuation[np.hstack((trianglecol, trianglechange))])/areas
    attenuation = np.delete(attenuation, trianglecol)
    triangles[trianglechange] = triangnow

    triangles = np.delete(triangles, trianglecol, axis=0)
    edges = np.delete(edges, edcol, axis=0)



    vertices = np.delete(vertices, np.array([vert1, vert2]), axis=0)
    triangles[np.where(triangles > vert2)] = triangles[np.where(triangles > vert2)] - 1
    triangles[np.where(triangles > vert1)] = triangles[np.where(triangles > vert1)] - 1 
    edges[np.where(edges > vert2)] = edges[np.where(edges > vert2)] - 1
    edges[np.where(edges > vert1)] = edges[np.where(edges > vert1)] - 1 

    return vertices, triangles, attenuation, edges, trianglescheck

def collapseedge(vertices, triangles, attenuation, edges, qualperedge, scorepervert, edcol, corners, scalercollapse):
    vert1 = edges[edcol][0]
    vert2 = edges[edcol][1]   
    if len(np.intersect1d(edges[edcol], corners)):
       print("corners in edge list..")
    #print(vert1)
    #print(vert2)
    trianglecol = np.intersect1d(np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])
       
    #print("new it")
    #print(trianglecol)
    trianglechange = np.setdiff1d(np.hstack((np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])), trianglecol)
    #asold = np.max(asppertr(vertices[triangles[np.hstack((trianglecol, trianglechange))]], rads[np.hstack((trianglecol, trianglechange))]))
    vertices = np.vstack((vertices, (vertices[vert1] + vertices[vert2])/2 ))
    triangnow = triangles[trianglechange].copy()
    #print("old triangles are ")
    #print(triangnow)
    triangnow[np.where(triangnow == vert1)] = len(vertices) -1
    triangnow[np.where(triangnow == vert2)] = len(vertices) -1
    radss, _ = circumcircle(vertices[triangnow])

    scorepervert = np.hstack((scorepervert, (scorepervert[vert1] + scorepervert[vert2])/2))
    trmat = triangle_area_intersection(triangnow, triangles[np.hstack((trianglecol, trianglechange))], vertices)
    areas = [Polygon(vertices[t]).area for t in triangnow]
    attenuation[trianglechange] =  (trmat@attenuation[np.hstack((trianglecol, trianglechange))])/areas
    attenuation = np.delete(attenuation, trianglecol)
    vertscon = np.setdiff1d(np.hstack((triangles[np.where(triangles == vert1)[0]].ravel(), triangles[np.where(triangles == vert2)[0]].ravel())), np.array([vert1, vert2]))
    triangles[trianglechange] = triangnow
    #if len(np.intersect1d(triangles[trianglecol], corners)) > 0 and len(np.hstack((np.where(np.min(vertices[vertscon], axis=1) == 0)[0], \
    #              np.where(np.max(vertices[vertscon], axis=1) == 1)[0]))) > 1:
    if len(np.intersect1d(triangles[trianglecol], corners)) > 0 and len(np.hstack((np.where(np.min(vertices[edges[edcol]], axis=1) == 0)[0], \
                  np.where(np.max(vertices[edges[edcol]], axis=1) == 1)[0]))) > 1 :
       corner = np.intersect1d(triangles[trianglecol], corners)[0]
       edgeverts = np.setdiff1d(vertscon[np.hstack((np.where(np.min(vertices[vertscon], axis=1) == 0)[0], \
                  np.where(np.max(vertices[vertscon], axis=1) == 1)[0]))], corners)
       triangles = np.vstack((triangles, np.array([[corner, edgeverts[0], len(vertices)-1], \
                                                   [corner, edgeverts[1], len(vertices)-1]]))) 

       attenuation = np.hstack((attenuation, np.array([0, 0])))


    triangles = np.delete(triangles, trianglecol, axis=0)
    #print("new triangles are ")
    #print(triangnow)
    edges = np.delete(edges, edcol, axis=0)
    qualperedge = np.delete(qualperedge, edcol)
    deledges = np.unique(np.hstack((np.where(edges == vert1)[0], np.where(edges == vert2)[0])))
    #print("Deleted edges are ")
    #print(edges[deledges].ravel())
    edges = np.delete(edges, deledges, axis=0)
    qualperedge = np.delete(qualperedge, deledges)
    #print("Connected vertices are ")
    #print(vertscon)
    #print("newedges are ")
    randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))
    for eind in range(len(vertscon)):
        if len(np.intersect1d(corners, vertscon[eind])) < 1 and len(np.intersect1d(randverts, np.array([vertscon[eind], len(vertices)-1]))) != 1:
          edges = np.vstack((edges, np.array([vertscon[eind], len(vertices)-1])))
          e = len(edges) - 1
          #print(edges[e])
          neigbourtriangles = np.array([np.min(np.intersect1d(np.where(triangles ==edges[e][0])[0], \
                        np.where(triangles == edges[e][1])[0])), np.max(np.intersect1d(np.where(triangles == edges[e][0])[0], \
                        np.where(triangles == edges[e][1])[0]))])
          radss, _ = circumcircle(vertices[triangles[neigbourtriangles]])
          qualperedge = np.hstack((qualperedge, (scorepervert[edges[e][0]]+ scorepervert[edges[e][1]])/2- \
          scalercollapse*np.sum(asppertr(vertices[triangles[neigbourtriangles]], radss))/2)) 
    scorepervert = np.delete(scorepervert, np.array([vert1, vert2]))
    vertices = np.delete(vertices, np.array([vert1, vert2]), axis=0)
    triangles[np.where(triangles > vert2)] = triangles[np.where(triangles > vert2)] - 1
    triangles[np.where(triangles > vert1)] = triangles[np.where(triangles > vert1)] - 1 
    edges[np.where(edges > vert2)] = edges[np.where(edges > vert2)] - 1
    edges[np.where(edges > vert1)] = edges[np.where(edges > vert1)] - 1 
    corners[np.where(corners > vert2)] = corners[np.where(corners > vert2)] - 1
    corners[np.where(corners > vert1)] = corners[np.where(corners > vert1)] - 1

    return vertices, triangles, attenuation, edges, qualperedge, scorepervert, corners

def flipandcolclose(vertices, triangles, res, trianglesint, edges, attenuation):

   radss, _ = circumcircle(vertices[triangles])
   asnew = asppertr(vertices[triangles], radss)
   counted = 0
   trianglescheck = np.where(trianglesint == 1)[0]
   tind = 0
   while tind < len(trianglescheck):
       trr = triangles[trianglescheck[tind]]
       for edgin in range(3):
         triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        
         ed = np.array([trr[np.mod(edgin, 3)], trr[np.mod(edgin+1, 3)]])
         trianglesround = np.intersect1d(np.where(triangles == ed[0])[0], np.where(triangles == ed[1])[0])
         if len(trianglesround) > 1 and len(np.intersect1d(ed, edges)) == 2:
            if np.linalg.norm(vertices[ed[0]] - vertices[ed[1]]) < res:             
               vert1 = ed[0]
               vert2 = ed[1] 
               trianglecol = np.intersect1d(np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])
               trianglechange = np.setdiff1d(np.hstack((np.where(triangles == vert1)[0], np.where(triangles == vert2)[0])), trianglecol)
               vertices = np.vstack((vertices, (vertices[vert1] + vertices[vert2])/2 ))
               triangnow = triangles[trianglechange].copy()
               triangnow[np.where(triangnow == vert1)] = len(vertices) -1
               triangnow[np.where(triangnow == vert2)] = len(vertices) -1
               areas = [Polygon(vertices[t]).area for t in triangnow]
               triangold = np.hstack((trianglecol, trianglechange))
               areas0 = [Polygon(vertices[triangles[t]]).area for t in triangold]
               vertices = np.delete(vertices, len(vertices)-1, axis=0)
               if np.abs(np.sum(areas) - np.sum(areas0)) < 10**(-11):
                 vertices, triangles, attenuation, edges, trianglescheck = collapseedgenoscores(vertices, triangles, attenuation, np.min(ed), np.max(ed), edges, trianglescheck)
                 break

            
            elif attenuation[trianglesround[0]] == attenuation[trianglesround[1]]:
              trianglespos = np.array([[ed[0], np.setdiff1d(triangles[trianglesround[1]], ed)[0], np.setdiff1d(triangles[trianglesround[0]], ed)[0]], \
                                [ed[1], np.setdiff1d(triangles[trianglesround[1]], ed)[0], np.setdiff1d(triangles[trianglesround[0]], ed)[0]]])
              radspos, _ = circumcircle(vertices[trianglespos])   
              areasnew = [Polygon(vertices[t]).area for t in trianglespos]
              areaor = [Polygon(vertices[t]).area for t in triangles[trianglesround]]
              atround = np.unique(attenuation[triang.neighbors[trianglesround[0]]])
              if np.max(asppertr(vertices[trianglespos], radspos)) < np.max(asnew[trianglesround]) and \
                 np.abs(np.sum(areaor) - np.sum(areasnew)) < 10**(-10):
                triangles[trianglesround] = trianglespos
                asnew[trianglesround] = asppertr(vertices[trianglespos], radspos)
       counted += 1 
       tind += 1

   return vertices, triangles, attenuation

def createedges(triangles, corners, randverts, attenuation, manscal):
   
    scorepervert = np.zeros(len(vertices))

    edgesss = set()  # Using a set to store unique edges
        
    for trii in triangles:
        edgesss.add(tuple(sorted([trii[0], trii[1]])))  # Edge v1-v2
        edgesss.add(tuple(sorted([trii[1], trii[2]])))  # Edge v2-v3
        edgesss.add(tuple(sorted([trii[2], trii[0]]))) 
    edgesss = sorted(edgesss)
    edges = set()
    edge_dict = {}
    for e in range(len(edgesss)):
      edge = tuple(sorted(edgesss[e]))
      if len(np.intersect1d(randverts, edge)) !=1 and  len(np.intersect1d(edge, corners)) == 0:
        edge_dict[edge] = []
        edges.add(edge)
    edges = sorted(edges)
    for i in range(len(triangles)):
        edgess = [(triangles[i][a], triangles[i][b]) for a, b in [(0,1), (1,2), (0,2)]]
        for e in edgess:
            edge = tuple(sorted(e))
            if edge in edge_dict:
                edge_dict[edge].append(i)         

    del edgesss
    rads, _ = circumcircle(vertices[triangles])
    qualscore = np.zeros(len(triangles))
    meanatvert = np.zeros(len(vertices))
    for n in range(len(scorepervert)):
      trrond = np.where(triangles == n)[0]
      meanatvert[n] = np.mean(attenuation[trrond])
      scorepervert[n] = np.mean((meanatvert[n] - attenuation[trrond])**2) 
    qualscore = asppertr(vertices[triangles], rads)

    qualperedge = np.zeros((len(edges), 2))
    qualperedge[:, 0] = np.sum(scorepervert[np.array(edges)] , axis=1)/2
    inded = 0
    for ed in edges:
         qualperedge[inded][1] = np.mean(qualscore[edge_dict[ed]])       
         inded+=1  
    if manscal < -0.5:
      scalercollapse = np.sum(qualperedge[:, 0])/np.sum(qualperedge[:, 1])
    else:
       scalercollapse = manscal
    del edge_dict
    del qualscore
    qualperedge = qualperedge[:, 0] - scalercollapse*qualperedge[:, 1]
    return edges, scorepervert, qualperedge, scalercollapse

def asppertr(t, radii):
   asp = radii/np.min(np.vstack((np.linalg.norm(t[:, 1, :] - t[:, 2, :], 2, 1), \
                           np.linalg.norm(t[:, 2, :] - t[:, 0, :], 2, 1), \
                           np.linalg.norm(t[:, 0, :] - t[:, 1, :], 2, 1))), axis=0)
   return asp

def SirtrecCR(systemmat, attenuation, its, sinogram, stop1, stop2, Cmat, Rmat):
    enprev = 0
    en = 0
    for it in range(its):
        projdif = sinogram - systemmat@attenuation
        cprojdif = Cmat*projdif
        atchange = ((systemmat.transpose())@(cprojdif))
        normch = np.linalg.norm(atchange)
        if normch < stop1: #np.linalg.norm(attenuationchange, 2)**2 < stop1:
           break
        if it == 0:
            en = (projdif.transpose())@(cprojdif)
            systemmat = systemmat.tocsr()
        #print(np.sum(projdif**2))
        enprev = en 
        en = (projdif.transpose())@(cprojdif)
        attenuationchange = Rmat*atchange
        attenuation = attenuation + attenuationchange
        
        #if np.linalg.norm((1/Rmat)*attenuationchange) < stop1: #np.linalg.norm(attenuationchange, 2)**2 < stop1:
        #   break
        if enprev - en < stop2:
           break 
    if it > 0:
      systemmat = systemmat.tocsc()
    attenuation[np.where(attenuation < 0)] = 0
    return attenuation, normch, enprev - en     

def Sirtrec(systemmat, attenuation, its, sinogram, stop1, stop2):
    Cmat =  np.array(1/np.sum(systemmat, axis = 1)).ravel()
    Rmat =  np.array((1/np.sum(systemmat, axis = 0)))[0].ravel()
    en = ((sinogram - systemmat@attenuation).transpose())@(Cmat*(sinogram - systemmat@attenuation))
    enprev = en
    for it in range(its):
        projdif = sinogram - systemmat@attenuation
        normch = np.linalg.norm(((systemmat.transpose())@(Cmat*projdif)))
        if normch < stop1: #np.linalg.norm(attenuationchange, 2)**2 < stop1:
           break
        #print(np.sum(projdif**2))
        enprev = en 
        en = (projdif.transpose())@(Cmat*projdif)
        attenuationchange = Rmat*((systemmat.transpose())@(Cmat*projdif))
        attenuation = attenuation + attenuationchange
        attenuation[np.where(attenuation < 0)] = 0
        if np.linalg.norm((1/Rmat)*attenuationchange) < stop1: #np.linalg.norm(attenuationchange, 2)**2 < stop1:
           break
    return attenuation, normch, en     

def bbREC(systemmat, attenuation, its, sinogram):
    #Cmat =  np.array(1/np.sum(systemmat, axis = 1)).ravel()
    #Rmat =  np.array((1/np.sum(systemmat, axis = 0)))[0].ravel()
    gradat = (systemmat@attenuation - sinogram).transpose()@systemmat
    en0 = np.sum((systemmat@attenuation - sinogram)**2) 
    gammaBB = 1 
    attenuation = attenuation - gammaBB*gradat
    attenuation[np.where(attenuation < 0)] = 0
    en = np.sum((systemmat@attenuation - sinogram)**2) 
    counter = 0
    attenuationprev = attenuation.copy()
    gradprev = gradat.copy()
    while en > en0 and counter < 25:
      gammaBB = gammaBB/2
      counter += 1
      attenuation = attenuation + gammaBB*gradat
      attenuation[np.where(attenuation < 0)] = 0
      en = np.sum((systemmat@attenuation - sinogram)**2)
    atbest = attenuation.copy()
    enbest = en.copy()
    for _ in range(its):
        gradat = (systemmat@attenuation - sinogram).transpose()@systemmat
        gammaBB = np.sum(np.abs((attenuation - attenuationprev)*(gradat - gradprev)))/np.linalg.norm(gradat - gradprev)**2
        attenuationprev = attenuation.copy()
        gradprev = gradat.copy()  
        #print(np.sum(projdif**2))
        attenuation = attenuation  -  gammaBB*gradat
        attenuation[np.where(attenuation < 0)] = 0
        en = np.sum((systemmat@attenuation - sinogram)**2)
        if en < enbest:
           enbest = en.copy()
           atbest = attenuation.copy()
    return atbest



def Rectotalvar(systemmat, attenuation, its, sinogram, triang, scalerTV):
    #Cmat =  np.array(1/np.sum(systemmat, axis = 1)).ravel()
    #Rmat =  np.array((1/np.sum(systemmat, axis = 0)))[0].ravel()
    threenei = np.where(np.min(triang.neighbors, axis=1) > -0.5)[0]
    gradat = (systemmat@attenuation - sinogram).transpose()@systemmat
    grad2 = np.zeros(len(attenuation))
    grad2[threenei] = (attenuation[threenei] - attenuation[triang.neighbors][threenei,0]) + \
                 (attenuation[threenei]  - attenuation[triang.neighbors][threenei,1]) + \
        (attenuation[threenei] - attenuation[triang.neighbors][threenei,2])
    gradat = gradat + scalerTV*grad2
    en0 = np.sum((systemmat@attenuation - sinogram)**2) + np.sum(scalerTV*((attenuation[threenei] - attenuation[triang.neighbors][threenei,0])**2 + \
          (attenuation[threenei] - attenuation[triang.neighbors][threenei,1])**2 + \
        (attenuation[threenei] - attenuation[triang.neighbors][threenei,2])**2))
    gammaBB = 1 
    attenuation = attenuation - gammaBB*gradat
    en = np.sum((systemmat@attenuation - sinogram)**2) + np.sum(scalerTV*((attenuation[threenei] - attenuation[triang.neighbors][threenei,0])**2 + \
         (attenuation[threenei] - attenuation[triang.neighbors][threenei,1])**2 + \
         (attenuation[threenei] - attenuation[triang.neighbors][threenei,2])**2))
    counter = 0
    attenuationprev = attenuation.copy()
    gradprev = gradat.copy()
    while en > en0 and counter < 25:
      gammaBB = gammaBB/2
      counter += 1
      attenuation = attenuation + gammaBB*gradat
      en = np.sum((systemmat@attenuation - sinogram)**2) + scalerTV*((attenuation[threenei] - attenuation[triang.neighbors][threenei,0])**2 + \
         (attenuation[threenei] - attenuation[triang.neighbors][threenei,1])**2 + \
         (attenuation[threenei] - attenuation[triang.neighbors][threenei,2])**2)

    for _ in range(its):
        gradat = (systemmat@attenuation - sinogram).transpose()@systemmat
        grad2 = np.zeros(len(attenuation))
        grad2[threenei] = (attenuation[threenei] - attenuation[triang.neighbors][threenei,0]) +\
           (attenuation[threenei] - attenuation[triang.neighbors][threenei,1]) + \
           (attenuation[threenei] - attenuation[triang.neighbors][threenei,2])
        gradat = gradat + scalerTV*grad2
        gammaBB = np.sum((attenuation - attenuationprev)*(gradat - gradprev)/np.linalg.norm(gradat - gradprev)**2)
        attenuationprev = attenuation.copy()
        gradprev = gradat.copy()  
        #print(np.sum(projdif**2))
        attenuation = attenuation  -  gammaBB*gradat
        attenuation[np.where(attenuation < 0)] = 0

    return attenuation

def getinterfaces(vertices, triangles, attenuation):

     edges = np.array([0, 0])  # Using a set to store unique edges
     normals = np.array([0, 0])
     atlist = np.array([0])
     tind = 0
     for trii in triangles:
      for eind in range(3):
         trr = np.intersect1d(np.where(triangles == trii[np.mod(eind, 3)])[0], np.where(triangles == trii[np.mod(eind+1, 3)])[0])
         if len(trr) > 1:
            #if connectionarray[tr[0]] != connectionarray[tr[1]]:           
            if attenuation[trr[0]] != attenuation[trr[1]]:           
               edges =  np.vstack((edges, np.array([trii[np.mod(eind, 3)], trii[np.mod(eind + 1, 3)]])))
               normalsbij = np.array([vertices[trii[np.mod(eind, 3)]][1] - vertices[trii[np.mod(eind+1, 3)]][1], -(vertices[trii[np.mod(eind, 3)]][0] - vertices[trii[np.mod(eind + 1, 3)]][0])])
               normalsbij = normalsbij/np.linalg.norm(normalsbij, 2)
               if np.sign(np.dot(normalsbij, \
                  (vertices[trii[np.mod(eind, 3)]] + vertices[trii[np.mod(eind+1, 3)]])/2 -\
                     vertices[trii[np.mod(eind+2, 3)]])) < 0:
                  normalsbij = normalsbij*np.sign(np.dot(normalsbij, \
                  (vertices[trii[np.mod(eind, 3)]] + vertices[trii[np.mod(eind+1, 3)]])/2 -\
                     vertices[trii[np.mod(eind+2, 3)]]))
                  edges[len(edges) - 1] =  np.array([trii[np.mod(eind + 1, 3)], trii[np.mod(eind, 3)]])
               normals = np.vstack((normals, normalsbij))
               atlist = np.vstack((atlist,attenuation[tind]))
         else:
               edges = np.vstack((edges, np.array([trii[np.mod(eind, 3)], trii[np.mod(eind + 1, 3)]])))
               normalsbij = np.array([vertices[trii[np.mod(eind, 3)]][1] - vertices[trii[np.mod(eind+1, 3)]][1], -(vertices[trii[np.mod(eind, 3)]][0] - vertices[trii[np.mod(eind + 1, 3)]][0])])
               normalsbij = normalsbij/np.linalg.norm(normalsbij, 2)
               if np.sign(np.dot(normalsbij, \
                  (vertices[trii[np.mod(eind, 3)]] + vertices[trii[np.mod(eind+1, 3)]])/2 -\
                     vertices[trii[np.mod(eind+2, 3)]])) < 0:
                  normalsbij = normalsbij*np.sign(np.dot(normalsbij, \
                  (vertices[trii[np.mod(eind, 3)]] + vertices[trii[np.mod(eind+1, 3)]])/2 -\
                     vertices[trii[np.mod(eind+2, 3)]]))
                  edges[len(edges) - 1] =  np.array([trii[np.mod(eind + 1, 3)], trii[np.mod(eind, 3)]])
               normals = np.vstack((normals, normalsbij))
               atlist = np.vstack((atlist,attenuation[tind]))

      tind += 1
     normals = np.delete(normals, 0, axis=0)
     it = 0

     edges = np.delete(edges, 0, axis=0)
     atlist = np.delete(atlist, 0)    

     trianglesint = np.zeros(len(triangles))
     for trind in range(len(triangles)):
        if len(np.intersect1d(triangles[trind], edges)) > 0.5:
           trianglesint[trind] = 1   

     trianglesint = trianglesint.astype(int) 
     return edges, trianglesint, atlist, normals


def computeprojdif(sinogram, vertices, edges, atlist, normals, angles, numpix):
     
  projdif = np.zeros(len(sinogram))
  block_x = 32
  block_y = 32
  block = (block_x, block_y)
  grid = ((len(edges))//block_x+1, (numpix)//block_x+1)  
  verticesc = cuda.to_device(vertices)
  edgesc = cuda.to_device(edges)
  attenuationc = cuda.to_device(atlist)
  normalsc = cuda.to_device(normals)
  for anind in range(len(angles)):
   projdifc = cuda.to_device(projdif[np.arange(anind*numpix, (anind+1)*numpix)])
   raytracerparallel[grid, block](angles[anind], numpix, verticesc, edgesc, attenuationc, projdifc, normalsc)
   projdif[np.arange(anind*numpix, (anind+1)*numpix)] = projdifc.copy_to_host()


  projdif2 = np.zeros(len(sinogram))
  block_x = 32
  block_y = 32
  block = (block_x, block_y)
  grid = ((len(edges))//block_x+1, (numpix)//block_x+1)  
  verticesc = cuda.to_device((vertices-0.5)*(1+10**(-6))+0.5)
  edgesc = cuda.to_device(edges)
  attenuationc = cuda.to_device(atlist)
  normalsc = cuda.to_device(normals)
  for anind in range(len(angles)):
   projdifc = cuda.to_device(projdif2[np.arange(anind*numpix, (anind+1)*numpix)])
   raytracerparallel[grid, block](angles[anind], numpix, verticesc, edgesc, attenuationc, projdifc, normalsc)
   projdif2[np.arange(anind*numpix, (anind+1)*numpix)] = projdifc.copy_to_host()

  projdif3 = np.zeros(len(sinogram))
  block_x = 32
  block_y = 32
  block = (block_x, block_y)
  grid = ((len(edges))//block_x+1, (numpix)//block_x+1)  
  verticesc = cuda.to_device((vertices-0.5)*(1-10**(-6))+0.5)
  edgesc = cuda.to_device(edges)
  attenuationc = cuda.to_device(atlist)
  normalsc = cuda.to_device(normals)
  for anind in range(len(angles)):
   projdifc = cuda.to_device(projdif3[np.arange(anind*numpix, (anind+1)*numpix)])
   raytracerparallel[grid, block](angles[anind], numpix, verticesc, edgesc, attenuationc, projdifc, normalsc)
   projdif3[np.arange(anind*numpix, (anind+1)*numpix)] = projdifc.copy_to_host()

  projdif2 = (projdif2 + projdif3)/2
  projdif = projdif - sinogram
  projdif2 = projdif2 - sinogram
  projdif = np.vstack((projdif, projdif2))
  projdif = projdif.ravel()[np.argmin(projdif**2, axis = 0)*len(sinogram)+np.arange(len(sinogram))]
  return projdif

def displacement(vertices, triangles, attenuation, kappaopt, sinogram, angles, numpix, edges, trianglesint, atlist, normals, res):

     projdif = computeprojdif(sinogram, vertices, edges, atlist, normals, angles, numpix)
     verticesc = cuda.to_device(vertices)
     block_x = 32
     edgesc = cuda.to_device(edges)
     grid_x = len(edges)//block_x+1
     lentermpervertc = cuda.to_device(np.zeros(2*len(vertices)))
     lenpervertcalc[grid_x, block_x](verticesc, edgesc, lentermpervertc)
     lenterm = 0.5*np.linalg.norm(lentermpervertc.copy_to_host())**2
     scaleredge1 = kappaopt*(0.5*np.sum(projdif**2)/lenterm)
     areas = [Polygon(vertices[t]).area for t in triangles]
     scaleredge2 = 0.5*np.sum(projdif**2)/(funaspecttr(triangles, vertices, np.where(trianglesint == 0)[0]))
     areaor = np.array(areas)
     areas = [Polygon(vertices[t]).area for t in triangles]   

     for it in range(12):
       inded = 0
       #for ed in edges:
       # normed = np.array([vertices[ed[0]][1] - vertices[ed[1]][1], -(vertices[ed[0]][0] - vertices[ed[1]][0])]) 
       # normed = normed/np.linalg.norm(normed)
       # normals[inded][0] = normed[0]
       # normals[inded][1] = normed[1]
       # inded += 1
       vertices = vertices.ravel()

       areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]

      
       if it == 0:
         func0 = 0.5*np.sum(projdif**2) + scaleredge1*lenterm + scaleredge2*funaspecttr(triangles, np.reshape(vertices, (int(len(vertices)/2), 2)), np.where(trianglesint == 0)[0])
         projdif0 = projdif.copy()
       else:
          func0 = func
          projdif0 = projdif
       
       inded = 0


       vertgrad = gradoptfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation, trianglesint, scaleredge1, scaleredge2, areaor, projdif)
   
       if it < 1:
          hessiaaninv = pylops.Identity(len(vertices))*(1/np.linalg.norm(vertgrad))
       else:    
          sk = vertices - verticesprev
          yk = vertgrad - vertgradprev
          sy = np.dot(sk,yk)
          s_op = pylops.MatrixMult(sk[np.newaxis])
          Hiy = hessiaaninv @ yk
          Hiy_op = pylops.MatrixMult(Hiy[np.newaxis])
          hessiaaninv = hessiaaninv + (s_op.T @ s_op) * \
          ((sy + np.dot(yk, Hiy))/sy**2) - (Hiy_op.T @ s_op + s_op.T @ Hiy_op)*(1/sy)
          del Hiy
          del Hiy_op
          del s_op
          del sk
          del yk

       vertgradprev = vertgrad.copy()

       verticesprev = vertices.copy()

       dx = hessiaaninv@vertgrad
       del vertgrad
       #dx = gammaBB*vertgrad
       verticesprev = vertices.copy()


       damp = (res/2)/np.max(dx)
       vertices = vertices - damp*dx

       areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]
   
       if np.abs(np.sum(areas) - 1) < 10**(-11):
         func, projdif = optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation, trianglesint, scaleredge1, scaleredge2, areaor)
       else:
          func = func0*2
       its = 0
       while (func > func0 or np.abs(np.sum(areas) - 1) > 10**(-11)) and its < 10: 
       #while func > func0 + m*damp*0.00001 and its < 25: 
         damp = damp/2
         vertices = vertices + damp*dx
         areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]
         areas = np.array(areas)
         its += 1
         if  np.abs(np.sum(areas) - 1) > 10**(-11):
            continue
         func, projdif = optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation,  trianglesint, scaleredge1, scaleredge2, areaor)    
       if np.abs(np.sum(areas) - np.sum(areaor)) > 10**(-11) or func > func0:
          vertices = vertices + damp*dx
          vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
          func = func0 
          projdif = projdif0
          break
       vertices = np.reshape(vertices, (int(len(vertices)/2), 2))

     
     #systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 

     #projdif = np.sum((sinogram - systemmat@attenuation)**2)

     
     return vertices, 0.5*np.sum(projdif**2), func

def optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation, trianglesint, scaleredge1, scaleredge2, areaor): 
  vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
  #systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 
  #projdif = sinogram - systemmat@attenuation
  projdif = computeprojdif(sinogram, vertices, edges, atlist, normals, angles, numpix)
  verticesc = cuda.to_device(vertices)
  block_x = 32
  edgesc = cuda.to_device(edges)
  grid_x = len(edges)//block_x+1
  lentermpervertc = cuda.to_device(np.zeros(2*len(vertices)))
  lenpervertcalc[grid_x, block_x](verticesc, edgesc, lentermpervertc)
  lenterm1 = 0.5*np.linalg.norm(lentermpervertc.copy_to_host())**2
  trianglesout = np.where(trianglesint == 0)[0]
  #lenterm2 = funaspecttr(triangles, vertices, trianglesout)
  lenterm2 = funaspecttr(triangles, vertices, trianglesout)
  func = 0.5*np.sum(projdif**2) + scaleredge1*lenterm1 + scaleredge2*lenterm2
  #func = 0.5*np.sum(projdif**2) + 0.5*scaleredge*np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]])**2
  #func = 0.5*np.sum(projdif**2) + 0.25*scaleredge*np.linalg.norm(np.vstack((vertices[triangles[:, 0]] - vertices[triangles[:, 1]], \
  #                             vertices[triangles[:, 1]] - vertices[triangles[:, 2]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])))**2  
  return func, projdif


def gradoptfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation,  trianglesint, scaleredge1, scaleredge2, areaor, projdif):
  
  vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
  randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))


  #systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 


  #projdif =  systemmat@attenuation - sinogram 
  block_x = 32
  block_y = 32
  block = (block_x, block_y)
  grid = ((len(edges))//block_x+1, (numpix)//block_x+1)  
  verticesc = cuda.to_device(vertices)
  edgesc = cuda.to_device(edges)
  attenuationc = cuda.to_device(atlist)
  vertgradc = cuda.to_device(np.zeros(2*len(vertices)))
  for anind in range(len(angles)):
   projdifc = cuda.to_device(projdif[np.arange(anind*numpix, (anind+1)*numpix)])
   raytracerparallel1[grid, block](angles[anind], numpix, scaleredge1/len(angles), verticesc, edgesc, attenuationc, vertgradc, projdifc)
  vertgrad = vertgradc.copy_to_host()

  vertgradc = cuda.to_device(vertgrad)

  for anind in range(len(angles)):
   projdifc = cuda.to_device(projdif[np.arange(anind*numpix, (anind+1)*numpix)])
   raytracerparallel2[grid, block](angles[anind], numpix, scaleredge1/len(angles), verticesc, edgesc, attenuationc,  vertgradc, projdifc)
  vertgrad = vertgradc.copy_to_host()
  trianglesout = np.where(trianglesint == 0)[0]
  #gradex = gradaspecttr(triangles, vertices,trianglesout )
  gradex = gradaspecttr(triangles, vertices,trianglesout)
  gradex[np.unique(edges)] = 0
  
  
  #vertgrad = vertgradc.copy_to_host()

  #vertgradc = cuda.to_device(vertgrad)

  #trianglesc = cuda.to_device(triangles)

  #otherverts = np.zeros(len(vertices))
  #otherverts[np.setdiff1d(np.arange(len(vertices)), edges)] = 1

  #othervertsc = cuda.to_device(otherverts)
  
  #grid_x = len(triangles)//block_x +1
  #lenscore[grid_x, block_x](scaleredge, verticesc, trianglesc, othervertsc, vertgradc)
  #vertgrad = vertgradc.copy_to_host()
  vertgrad = vertgrad + scaleredge2*gradex.ravel()
  vertgrad = np.reshape(vertgrad, (int(len(vertices)), 2))
  vertices = vertices.ravel() 
  vertgrad[randverts, :] = 0
  vertgrad = vertgrad.ravel()


  return  vertgrad

def funaspecttr(triangles, vertices, trianglesout):
  vert_ = torch.from_numpy(vertices).requires_grad_(True) 
  tri_ = torch.from_numpy(triangles[trianglesout])
  tet_ = vert_[tri_.long()]
  aspectmesh =  asptorchall(tet_)
  aspectmesh = aspectmesh.detach().numpy()
  return  aspectmesh

def funarea(triangles, vertices, trianglesout, areaor):
  vert_ = torch.from_numpy(vertices).requires_grad_(True) 
  tri_ = torch.from_numpy(triangles)
  tet_ = vert_[tri_.long()]
  x1 = tet_[:, 0, :]- tet_[:, 1, :]
  x2 = tet_[:, 1, :]- tet_[:, 2, :]
  areas = (x1[:, 0]*x2[:, 1]-x1[:, 1]*x2[:, 0])/2
  return np.sum((areaor-areas.detach().numpy())**2)

def gradarea(triangles, vertices, trianglesout, areaor):
  vert_ = torch.from_numpy(vertices).requires_grad_(True) 
  tri_ = torch.from_numpy(triangles)
  tet_ = vert_[tri_.long()]
  areaor_ = torch.from_numpy(areaor)
  func = voltorchall(tet_, areaor_) 
  grad = torch.autograd.grad(outputs=func, inputs=vert_)[0]
  grad = grad.detach().numpy()
  return grad

def gradaspecttr(triangles, vertices, trianglesout):
  vert_ = torch.from_numpy(vertices).requires_grad_(True) 
  tri_ = torch.from_numpy(triangles[trianglesout])
  tet_ = vert_[tri_.long()]
  func = asptorchall(tet_) 
  grad = torch.autograd.grad(outputs=func, inputs=vert_)[0]
  grad = grad.detach().numpy()
  return grad

def voltorchall(tet_, areaor_):
  x1 = tet_[:, 0, :] - tet_[:, 1, :]
  x2 = tet_[:, 1, :] - tet_[:, 2, :]
  areas = torch.sum(((x1[:, 0]*x2[:, 1]-x1[:, 1]*x2[:, 0])/2-areaor_)**2)
  return areas

def asptorchall(tet_):
  A = tet_[:, 0, :] 
  B = tet_[:, 1, :] 
  C = tet_[:, 2, :] 
  a = torch.sum((A - B) ** 2, dim=1)**(1/2)
  b = torch.sum((B - C) ** 2, dim=1)**(1/2)
  c = torch.sum((C - A) ** 2, dim=1)**(1/2)
  mined = torch.min(torch.stack([a, b, c], dim=1), axis = 1).values
  aspect = torch.sum(((a*b*c/((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))**(1/2))/mined))
  return aspect

def asptorch(tet_):
  A = tet_[:, 0, :] 
  B = tet_[:, 1, :] 
  C = tet_[:, 2, :] 
  a = torch.sum((A - B) ** 2, dim=1)**(1/2)
  b = torch.sum((B - C) ** 2, dim=1)**(1/2)
  c = torch.sum((C - A) ** 2, dim=1)**(1/2)
  mined = torch.min(torch.stack([a, b, c], dim=1), axis = 1).values
  aspect = (a*b*c/((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))**(1/2))/mined
  return torch.max(aspect)


#%%
#No noise
path_to_file = os.getcwd()
fant = scipy.io.loadmat(f"{path_to_file}/fantoom3_2000x2000.mat")
N = 31  #Dimension starting mesh, increasing it over sqrt(2)/(2res) +1 didsables first refinement 
numpix = 1000
qmax = 2 
fant1 = fant['fantoom3']
fant1 = fant1.astype(float)
res = 0.01 #resoltuion parameter
N_sirtits = 500
resgr = len(fant1)
bounds1 = np.array([11.496, 15.465])
bounds2 = np.array([1.29, 1.411])  
fant = fant1.copy()
savebest = True
numangles = 150
angles = np.linspace(0,2*np.pi,numangles,False) 
minvar = 0.001**2
kappaopt = 1
kappaseg = 1
fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1

fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1

vol_geom = astra.create_vol_geom(resgr, resgr)
proj_geom = astra.create_proj_geom('parallel', len(fant1)/numpix, numpix, angles - np.pi/2)
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(fant1, proj_id)
   
sinogram = sinogram/len(fant1)
sinogram = sinogram.ravel()
#sinogram = sinogram + np.random.normal(0, 0.01*np.max(sinogram), len(sinogram))
#sinogram[np.where(sinogram < 0)] = 0
del fant1
noise = 0.01*np.max(sinogram)
idx = np.indices((N, N))  
X = idx[1] / (N - 1) 
Y = idx[0] / (N - 1)  
X = X.ravel()
Y = Y.ravel()
vertices = np.column_stack((X,  Y))
triangles = []
for i in range(N - 1):
  for j in range(N - 1):
      p1 = i * N + j
      p2 = p1 + 1
      p3 = p1 + N
      p4 = p3 + 1
      triangles.append([p1, p2, p3])  # Lower triangle
      triangles.append([p4, p3, p2])  # Upper triangle

triangles = np.array(triangles)
  
systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 
  

attenuation, stop1, _ = Sirtrec(systemmat, np.zeros(len(triangles)), N_sirtits, sinogram, 0, 0)



vertices, triangles, attenuation = Refinement(vertices, triangles, attenuation, noise, res, systemmat, 50000, stop1, True)


vertices, triangles, attenuation = edgecollapse(vertices, triangles, attenuation, minvar, qmax, 50000, 1, np.sqrt(2)/2)
  
systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 

triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        


_, _, scalerseg = Startingclusterfunction(attenuation, vertices, triangles, sinogram, systemmat, triang.neighbors)

scalerseg = scalerseg*kappaseg

attenuation, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)

if savebest: 
  verticesopt = vertices.copy()
  trianglesopt = triangles.copy()
  attenuationopt = attenuation.copy()

for optsteps in range(10):

     edges, trianglesint, atlist, normals = getinterfaces(vertices, triangles, attenuation)


     vertices, projdif, objfun = displacement(vertices, triangles, attenuation, kappaopt, sinogram, angles, numpix, edges, trianglesint, atlist, normals, res)
   

     if savebest: 
        if optsteps == 0:
          verticesopt = vertices.copy()
          trianglesopt = triangles.copy()
          attenuationopt = attenuation.copy()
          projdifbest = projdif
        else: 
           if projdif < projdifbest: 
             verticesopt = vertices.copy()
             trianglesopt = triangles.copy()
             attenuationopt = attenuation.copy()
             projdifbest = projdif
     else :
             verticesopt = vertices.copy()
             trianglesopt = triangles.copy()
             attenuationopt = attenuation.copy()


     if optsteps == 9:
         break    
    
     vertices, triangles, attenuation = flipandcolclose(vertices, triangles, res, trianglesint, edges, attenuation)

     vertices, triangles, attenuation = edgecollapse(vertices, triangles, attenuation, minvar, qmax, 50000, 0, qmax)


     vertices, triangles, attenuation = Refinement(vertices, triangles, attenuation, noise, res, systemmat, 50000, stop1, False)

     systemmat = Computesytemmatrix(vertices, triangles, numpix, angles) 


     attenuation, stop1, projdif = Sirtrec(systemmat, attenuation, N_sirtits, sinogram, 0, 0)

     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

     attenuation, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)



fig, ax = plt.subplots(figsize=(9,9))
triang = tri.Triangulation(verticesopt[:, 0], verticesopt[:, 1], trianglesopt)
ax.tripcolor(triang, attenuationopt, cmap='grey')
plt.axis("off")
plt.show()


# %%
#With noise
path_to_file = os.getcwd()
fant = scipy.io.loadmat(f"{path_to_file}/fantoom3_2000x2000.mat")
N = 31  #Dimension starting mesh, increasing it over sqrt(2)/(2res) +1 didsables first refinement 
numpix = 1000
qmax = 2 
fant1 = fant['fantoom3']
fant1 = fant1.astype(float)
res = 0.01 #resoltuion parameter
N_sirtits = 500
resgr = len(fant1)
bounds1 = np.array([11.496, 15.465])
bounds2 = np.array([1.29, 1.411])  
fant = fant1.copy()
savebest = True
numangles = 150
angles = np.linspace(0,2*np.pi,numangles,False) 
minvar = 0.001**2
kappaopt = 1
kappaseg = 1
fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1

fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1

vol_geom = astra.create_vol_geom(resgr, resgr)
proj_geom = astra.create_proj_geom('parallel', len(fant1)/numpix, numpix, angles - np.pi/2)
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(fant1, proj_id)
   
sinogram = sinogram/len(fant1)
sinogram = sinogram.ravel()
sinogram = sinogram + np.random.normal(0, 0.01*np.max(sinogram), len(sinogram))
sinogram[np.where(sinogram < 0)] = 0
del fant1
noise = 0.01*np.max(sinogram)
idx = np.indices((N, N))  
X = idx[1] / (N - 1) 
Y = idx[0] / (N - 1)  
X = X.ravel()
Y = Y.ravel()
vertices = np.column_stack((X,  Y))
triangles = []
for i in range(N - 1):
  for j in range(N - 1):
      p1 = i * N + j
      p2 = p1 + 1
      p3 = p1 + N
      p4 = p3 + 1
      triangles.append([p1, p2, p3])  # Lower triangle
      triangles.append([p4, p3, p2])  # Upper triangle

triangles = np.array(triangles)
  
systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 
  

attenuation, stop1, _ = Sirtrec(systemmat, np.zeros(len(triangles)), N_sirtits, sinogram, 0, 0)



vertices, triangles, attenuation = Refinement(vertices, triangles, attenuation, noise, res, systemmat, 50000, stop1, True)


vertices, triangles, attenuation = edgecollapse(vertices, triangles, attenuation, minvar, qmax, 50000, 1, np.sqrt(2)/2)
  
systemmat =  Computesytemmatrix(vertices, triangles, numpix, angles) 

triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        


_, _, scalerseg = Startingclusterfunction(attenuation, vertices, triangles, sinogram, systemmat, triang.neighbors)

scalerseg = scalerseg*kappaseg

attenuation, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)

if savebest: 
  verticesopt = vertices.copy()
  trianglesopt = triangles.copy()
  attenuationopt = attenuation.copy()

for optsteps in range(10):

     edges, trianglesint, atlist, normals = getinterfaces(vertices, triangles, attenuation)


     vertices, projdif, objfun = displacement(vertices, triangles, attenuation, kappaopt, sinogram, angles, numpix, edges, trianglesint, atlist, normals, res)
   

     if savebest: 
        if optsteps == 0:
          verticesopt = vertices.copy()
          trianglesopt = triangles.copy()
          attenuationopt = attenuation.copy()
          projdifbest = projdif
        else: 
           if projdif < projdifbest: 
             verticesopt = vertices.copy()
             trianglesopt = triangles.copy()
             attenuationopt = attenuation.copy()
             projdifbest = projdif
     else :
             verticesopt = vertices.copy()
             trianglesopt = triangles.copy()
             attenuationopt = attenuation.copy()

     if optsteps == 9:
         break
    
     vertices, triangles, attenuation = flipandcolclose(vertices, triangles, res, trianglesint, edges, attenuation)

     vertices, triangles, attenuation = edgecollapse(vertices, triangles, attenuation, minvar, qmax, 50000, 0, qmax)


     vertices, triangles, attenuation = Refinement(vertices, triangles, attenuation, noise, res, systemmat, 50000, stop1, False)

     systemmat = Computesytemmatrix(vertices, triangles, numpix, angles) 


     attenuation, stop1, projdif = Sirtrec(systemmat, attenuation, N_sirtits, sinogram, 0, 0)

     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

     attenuation, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)



fig, ax = plt.subplots(figsize=(9,9))
triang = tri.Triangulation(verticesopt[:, 0], verticesopt[:, 1], trianglesopt)
ax.tripcolor(triang, attenuationopt, cmap='grey')
plt.axis("off")
plt.show()

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

def  clusterfunction(attenuation, vertices, triangles, numpix, sinogram, systemmat, neighbors, scalerseg, res):

    connectionarray = np.arange(len(triangles))
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)
    numangles = int(len(sinogram)/numpix)
    startvalue = 1

    #scalerseg = startvalue*projdif0/len(np.where((neighbors).ravel() > -0.5)[0])
    #scalerseg0 = scalerseg
    areas = [Polygon(vertices[t]).area for t in triangles]
    areas = np.array(areas)
    trianglist = np.zeros((len(np.where((neighbors) > -0.5)[0]), 2))
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    trianglist = trianglist.astype(int)
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])

    lengrad = len(gradoveredges)/2
    funpos0 = projdif0 + scalerseg*lengrad
    simplificdegree = 10
    nummax = 100
    attenuationseg = attenuation.copy()
    #while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
    maxgrad = np.max(gradoveredges) 
    for _ in range(len(gradoveredges)):
       mingrad = np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       con1 = np.where(connectionarray[np.arange(connectionarray[triangle1], len(connectionarray))] == connectionarray[triangle1])[0] + connectionarray[triangle1]
       con2 = np.where(connectionarray[np.arange(connectionarray[triangle2], len(connectionarray))] == connectionarray[triangle2])[0] + connectionarray[triangle2]
       atprev = attenuationseg[np.hstack((con1, con2))].copy()
       attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                   areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
       lennext = len(np.intersect1d(neighbors[con1], con2))
       funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
       if funpos < funpos0:
           funpos0 = funpos
           lengrad -= lennext
           connectionarray[np.hstack((con1, con2))] = np.min(connectionarray[np.hstack((con1, con2))])
       else:   
           attenuationseg[np.hstack((con1, con2))] = atprev
       gradoveredges[mingrad] = maxgrad
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])    
    for _ in range(len(gradoveredges)):
       mingrad = np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       radss, _ = circumcircle(vertices[triangles[np.array([triangle1, triangle2])]])
       if (radss[0] > res or len(np.where(connectionarray == connectionarray[triangle1])[0]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle1]]), connectionarray[triangle1])) < 2) \
         and (radss[1] > res or len(np.where(connectionarray == connectionarray[triangle2])[0]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle2]]), connectionarray[triangle2])) < 2):
         con1 = np.where(connectionarray[np.arange(connectionarray[triangle1], len(connectionarray))] == connectionarray[triangle1])[0] + connectionarray[triangle1]
         con2 = np.where(connectionarray[np.arange(connectionarray[triangle2], len(connectionarray))] == connectionarray[triangle2])[0] + connectionarray[triangle2]
         atprev = attenuationseg[np.hstack((con1, con2))].copy()
         attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                    areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
         lennext = len(np.intersect1d(neighbors[con1], con2))
         funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
         if funpos < funpos0:
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[np.hstack((con1, con2))] = np.min(connectionarray[np.hstack((con1, con2))])
         else:   
            attenuationseg[np.hstack((con1, con2))] = atprev
         gradoveredges[mingrad] = maxgrad
       else:
            con1 = np.where(connectionarray[np.arange(connectionarray[triangle1], len(connectionarray))] == connectionarray[triangle1])[0] + connectionarray[triangle1]
            con2 = np.where(connectionarray[np.arange(connectionarray[triangle2], len(connectionarray))] == connectionarray[triangle2])[0] + connectionarray[triangle2]
            atprev = attenuationseg[np.hstack((con1, con2))].copy()
            attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                       areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
            lennext = len(np.intersect1d(neighbors[con1], con2))
            funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[np.hstack((con1, con2))] = np.min(connectionarray[np.hstack((con1, con2))])
            
    for tind in range(len(triangles)):
       if np.min(neighbors[tind]) < 0:
          continue
       if len(np.where(connectionarray[neighbors[tind]] != connectionarray[tind])[0]) > \
           len(np.where(connectionarray[neighbors[tind]] == connectionarray[tind])[0]):
           atprev = attenuationseg[tind].copy()
           values, counts = np.unique(attenuationseg[neighbors[tind]], return_counts=True)
           attenuationseg[tind] = values[np.argmax(counts)]
           indn = np.where(attenuationseg[neighbors[tind]] == values[np.argmax(counts)])[0][0]
           lennext = lennext - len(np.where(connectionarray[neighbors[tind]] ==\
                          connectionarray[neighbors[tind]][indn])[0])
           funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
           if funpos < funpos0: 
              connectionarray[tind] = connectionarray[neighbors[tind][indn]]
              lengrad = lengrad - lennext
              funpos0 = funpos 
           else:   
              attenuationseg[tind] = atprev
       if len(np.intersect1d(connectionarray[neighbors[tind]], connectionarray[tind])) == 0: 
            values, counts = np.unique(attenuationseg[neighbors[tind]], return_counts=True)
            attenuationseg[tind] = values[np.argmin(np.abs(values - attenuationseg[tind]))]
            indn = np.where(attenuationseg[neighbors[tind]] == values[np.argmin(np.abs(values - attenuationseg[tind]))])[0][0]
            lennext = len(np.where(connectionarray[neighbors[tind]] ==\
                          connectionarray[neighbors[tind]][indn])[0])
            funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[tind] = connectionarray[neighbors[tind]][indn]

     #for cons in np.unique(connectionarray):
   #  atlist = bbREC(systmatcrop, attenuationseg[posvals], 200, sinogram)
   #  for i in range(len(atlist)):
   #       attenuationseg[np.where(connectionarray == posvals[i])[0]] = atlist[i]
    
    funpos0 = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*lengrad
    

    
    #del atlist

    return attenuationseg, connectionarray 

def  clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, neighbors, scalerseg, res):
    systemmat = systemmat.tocsc()
    connectionarray = np.arange(len(triangles))
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)
    numangles = int(len(sinogram)/numpix)
    startvalue = 1
    con_dict = {}
    for c in range(len(triangles)):
        con_dict[c] = [c]

    #scalerseg = startvalue*projdif0/len(np.where((neighbors).ravel() > -0.5)[0])
    #scalerseg0 = scalerseg
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
    attenuationseg = attenuation.copy()
    #while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
    maxgrad = np.max(gradoveredges) 
    gradorder = np.argsort(gradoveredges)
    for gind in range(len(gradoveredges)):
       mingrad = gradorder[gind]#np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       con1 = con_dict[connectionarray[triangle1]]
       con2 = con_dict[connectionarray[triangle2]]
       if len(con1) > 10 and len(con2) > 10:
           if np.abs(attenuationseg[triangle1] - attenuationseg[triangle2]) > 0.1:
               continue

       atprev = attenuationseg[np.hstack((con1, con2))].copy()
       attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                   areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
       lennext = len(np.intersect1d(neighbors[con1], con2))
       projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuationseg[np.hstack((con1, con2))]- atprev)
       funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
       if funpos < funpos0:
           funpos0 = funpos
           lengrad -= lennext
           minval = np.min(connectionarray[np.array([triangle1, triangle2])])
           otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
           connectionarray[np.hstack((con1, con2))] = minval
           con_dict[minval].extend(con_dict[otherval]) 
           del con_dict[otherval]
       else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]\
            @(attenuationseg[np.hstack((con1, con2))]- atprev)
            attenuationseg[np.hstack((con1, con2))] = atprev
       #gradoveredges[mingrad] = maxgrad
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])  
    gradorder = np.argsort(gradoveredges)  
    for gind in range(len(gradoveredges)):
       mingrad = gradorder[gind]#np.argmin(gradoveredges)
       triangle1 = trianglist[mingrad][0]
       triangle2 = trianglist[mingrad][1]
       if connectionarray[triangle1] == connectionarray[triangle2]:
           gradoveredges[mingrad] = maxgrad
           continue
       radss, _ = circumcircle(vertices[triangles[np.array([triangle1, triangle2])]])
       if (radss[0] > res or len(np.where(connectionarray == connectionarray[triangle1])[0]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle1]]), connectionarray[triangle1])) < 2) \
         and (radss[1] > res or len(np.where(connectionarray == connectionarray[triangle2])[0]) > 3 or len(np.setdiff1d(np.unique(connectionarray[neighbors[triangle2]]), connectionarray[triangle2])) < 2):
         con1 = con_dict[connectionarray[triangle1]]
         con2 = con_dict[connectionarray[triangle2]]
         if len(con1) > 10 and len(con2) > 10:
            if np.abs(attenuationseg[triangle1] - attenuationseg[triangle2]) > 0.1:
                continue
         atprev = attenuationseg[np.hstack((con1, con2))].copy()
         attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                    areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
         lennext = len(np.intersect1d(neighbors[con1], con2))
         projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuationseg[np.hstack((con1, con2))]-atprev)
         funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)

         if funpos < funpos0:
            funpos0 = funpos
            lengrad -= lennext
            minval = np.min(connectionarray[np.array([triangle1, triangle2])])
            otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
            connectionarray[np.hstack((con1, con2))] = minval
            con_dict[minval].extend(con_dict[otherval])   
            del con_dict[otherval]       
         else:   
            projdif0ns = projdif0ns - systemmat[:, np.hstack((con1, con2))]@(attenuationseg[np.hstack((con1, con2))]-atprev)
            attenuationseg[np.hstack((con1, con2))] = atprev

         gradoveredges[mingrad] = maxgrad
       else:
            con1 = con_dict[connectionarray[triangle1]]
            con2 = con_dict[connectionarray[triangle2]]
            atprev = attenuationseg[np.hstack((con1, con2))].copy()
            attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                       areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
            lennext = len(np.intersect1d(neighbors[con1], con2))
            projdif0ns = projdif0ns + systemmat[:, np.hstack((con1, con2))]@(attenuationseg[np.hstack((con1, con2))]-atprev)
            funpos = np.sum(projdif0ns**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            minval = np.min(connectionarray[np.array([triangle1, triangle2])])
            otherval = np.max(connectionarray[np.array([triangle1, triangle2])])
            connectionarray[np.hstack((con1, con2))] = minval
            con_dict[minval].extend(con_dict[otherval])   
            del con_dict[otherval]       
            

    for tind in range(len(triangles)):
       if np.min(neighbors[tind]) < 0:
          continue
       if len(np.where(connectionarray[neighbors[tind]] != connectionarray[tind])[0]) > \
           len(np.where(connectionarray[neighbors[tind]] == connectionarray[tind])[0]):
           atprev = attenuationseg[tind].copy()
           values, counts = np.unique(attenuationseg[neighbors[tind]], return_counts=True)
           attenuationseg[tind] = values[np.argmax(counts)]
           indn = np.where(attenuationseg[neighbors[tind]] == values[np.argmax(counts)])[0][0]
           lennext = lennext - len(np.where(connectionarray[neighbors[tind]] ==\
                          connectionarray[neighbors[tind]][indn])[0])
           funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
           if funpos < funpos0: 
              connectionarray[tind] = connectionarray[neighbors[tind][indn]]
              lengrad = lengrad - lennext
              funpos0 = funpos 
           else:   
              attenuationseg[tind] = atprev
       if len(np.intersect1d(connectionarray[neighbors[tind]], connectionarray[tind])) == 0: 
            values, counts = np.unique(attenuationseg[neighbors[tind]], return_counts=True)
            attenuationseg[tind] = values[np.argmin(np.abs(values - attenuationseg[tind]))]
            indn = np.where(attenuationseg[neighbors[tind]] == values[np.argmin(np.abs(values - attenuationseg[tind]))])[0][0]
            lennext = len(np.where(connectionarray[neighbors[tind]] ==\
                          connectionarray[neighbors[tind]][indn])[0])
            funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[tind] = connectionarray[neighbors[tind]][indn]

     #for cons in np.unique(connectionarray):
   #  atlist = bbREC(systmatcrop, attenuationseg[posvals], 200, sinogram)
   #  for i in range(len(atlist)):
   #       attenuationseg[np.where(connectionarray == posvals[i])[0]] = atlist[i]
    
    

    
    #del atlist

    return attenuationseg, connectionarray 




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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      newtriangles = newtriangles[keepareas]
      areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      del trmat
      del oldtriangs
      del verts
      del splitnu
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
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused)[0]
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
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      areas = np.array(areas)[keepareas]
      newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
      attenuation = np.hstack((attenuation, (trmat@attenuation[aroundsplit])/areas))
      attenuation = np.delete(attenuation, aroundsplit)
      for nind in range(len(vertdel)):
         triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] = triangles[np.where(triangles > vertdel[len(vertdel) -1 -nind])] -1         
         newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] = newtriangles[np.where(newtriangles > vertdel[len(vertdel) -1 -nind])] -1
      del trmat
      del oldtriangs
      del verts
      del splitnu


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
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      triangles = np.delete(triangles, aroundsplit, axis=0)
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      newtriangles = newtriangles[keepareas]
      areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused)[0]
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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      areas = np.array(areas)[keepareas]
      newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      newtriangles = newtriangles[keepareas]
      areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused)[0]
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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      areas = np.array(areas)[keepareas]
      newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      newtriangles = newtriangles[keepareas]
      areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused)[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
      aroundsplit = np.array(np.where(triangles == vertrem[0])[0]) 
      for mind in range(1, len(vertrem)): 
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
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      areas = np.array(areas)[keepareas]
      newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      numrep =len(aroundsplit)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      newtriangles = newtriangles[keepareas]
      areas = np.array(areas)[keepareas]
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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
      vertdel = np.where(np.linalg.norm(vertices - addvert, 2, axis=1) < radiused)[0]
      verton = np.hstack((np.where(np.min(vertices[vertdel], axis=1) == 0)[0], \
                          np.where(np.max(vertices[vertdel], axis=1) == 1)[0]))
      vertdel = np.delete(vertdel, verton, axis=0)
      vertrem = np.unique(np.hstack((np.unique(np.vstack((triangles[splitnu], triangles[neighbours]))), vertdel)))
      aroundsplit = np.array(np.where(triangles == vertrem[0])[0]) 
      for mind in range(1, len(vertrem)): 
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
      triangdata = dict(vertices=verts, segments=constraints)
      newtriangles = tr.triangulate(triangdata, 'pA')['triangles']
      newtriangles = vertinds[newtriangles]
      areas = [Polygon(vertices[t]).area for t in newtriangles]
      trmat = triangle_area_intersection(newtriangles, triangles[aroundsplit], vertices)
      keepareas = np.intersect1d(np.where(np.abs(np.sum(trmat, axis=1) - areas) < 10**(-13))[0], np.where(np.array(areas)>10**(-7))[0])
      areas = np.array(areas)[keepareas]
      newtriangles = newtriangles[keepareas]
      triangles = np.delete(triangles, aroundsplit, axis=0)
      vertices = np.delete(vertices, vertdel, axis=0)
      triangles = np.vstack((triangles, newtriangles))
      trmat = trmat[keepareas]
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



def Startingclusterfunction(attenuation, vertices, triangles, sinogram, systemmat, neighbors):
    connectionarray = np.arange(len(triangles))
    projdif0 = np.sum((sinogram - systemmat@attenuation)**2)

    startvalue = 1

    scalerseg = startvalue*projdif0/len(np.where((neighbors).ravel() > -0.5)[0])
    areas = [Polygon(vertices[t]).area for t in triangles]
    areas = np.array(areas)
    trianglist = np.zeros((len(np.where((neighbors) > -0.5)[0]), 2))
    trianglist[:, 0] = np.where((neighbors) > -0.5)[0]
    trianglist[:, 1] = neighbors[np.where((neighbors) > -0.5)]
    trianglist = trianglist.astype(int)
    gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])
    
    lengrad = len(gradoveredges)/2
    funpos0 = projdif0 + scalerseg*lengrad
    simplificdegree = 10
    nummax = 100
    attenuationseg = attenuation.copy()
    while len(np.unique(connectionarray)) > np.max(np.array([len(triangles)/simplificdegree, nummax])):
      for _ in range(len(gradoveredges)):
        mingrad = np.argmin(gradoveredges)
        triangle1 = trianglist[mingrad][0]
        triangle2 = trianglist[mingrad][1]
        if connectionarray[triangle1] == connectionarray[triangle2]:
            gradoveredges[mingrad] = np.max(gradoveredges) 
            continue
        con1 = np.where(connectionarray[np.arange(connectionarray[triangle1], len(connectionarray))] == connectionarray[triangle1])[0] + connectionarray[triangle1]
        con2 = np.where(connectionarray[np.arange(connectionarray[triangle2], len(connectionarray))] == connectionarray[triangle2])[0] + connectionarray[triangle2]
        atprev = attenuationseg[np.hstack((con1, con2))].copy()
        attenuationseg[np.hstack((con1, con2))] = np.sum(attenuationseg[np.hstack((con1, con2))]* \
                                    areas[np.hstack((con1, con2))])/np.sum(areas[np.hstack((con1, con2))])
        lennext = len(np.intersect1d(neighbors[con1], con2))
        funpos = np.sum((sinogram - systemmat@attenuationseg)**2) + scalerseg*(lengrad - lennext)
        if funpos < funpos0:
            funpos0 = funpos
            lengrad -= lennext
            connectionarray[np.hstack((con1, con2))] = np.min(connectionarray[np.hstack((con1, con2))])
        else:   
            attenuationseg[np.hstack((con1, con2))] = atprev
        gradoveredges[mingrad] = np.max(gradoveredges) 
      gradoveredges = np.abs(attenuation[trianglist[:,0]] - \
                    attenuation[trianglist[:, 1]])
      scalerseg = 2*scalerseg
      funpos0 = np.sum((sinogram - systemmat@attenuation)**2) + scalerseg*lengrad
      #del atlist
    scalerseg = scalerseg/2
    return attenuationseg, connectionarray, scalerseg 

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

def collapseedge(vertices, triangles, attenuation, edges, qualperedge, scorepervert, edcol, corners):
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
    if len(np.intersect1d(triangles[trianglecol], corners)) > 0 and len(np.hstack((np.where(np.min(vertices[vertscon], axis=1) == 0)[0], \
                  np.where(np.max(vertices[vertscon], axis=1) == 1)[0]))) > 1:
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
    inded = 0
    for ed in edges:
       if len(edge_dict[ed]) == 2:
         qualperedge[inded][0] =  (scorepervert[np.array(edges)[inded, 0]] + scorepervert[np.array(edges)[inded, 1]])/2 
         qualperedge[inded][1] = (qualscore[edge_dict[ed][0]] + qualscore[edge_dict[ed][1]])/2     
       else:
         qualperedge[inded][0] =  (scorepervert[np.array(edges)[inded, 0]] + scorepervert[np.array(edges)[inded, 1]])/2 
         qualperedge[inded][1] =   (qualscore[edge_dict[ed]] )     
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
        print(en)
        #print(np.sum(projdif**2))
        enprev = en 
        en = (projdif.transpose())@(Cmat*projdif)
        attenuationchange = Rmat*((systemmat.transpose())@(Cmat*projdif))
        attenuation = attenuation + attenuationchange
        attenuation[np.where(attenuation < 0)] = 0
        if np.linalg.norm((1/Rmat)*attenuationchange) < stop1: #np.linalg.norm(attenuationchange, 2)**2 < stop1:
           break
        if enprev - en < stop2:
           break 
    print(it)
    return attenuation, normch, enprev - en     

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


def optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation, trianglesint, scaleredge1, scaleredge2, areaor): 
  vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
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
  projdif = sinogram - systemmat@attenuation
  verticesc = cuda.to_device(vertices)
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
  return func


def gradoptfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuation,  trianglesint, scaleredge1, scaleredge2, areaor):
  
  vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
  randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))

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
  projdif =  systemmat@attenuation - sinogram 
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


numpix = 1000
    #for valindex in range(len(valueslist)):
fant = scipy.io.loadmat("/home/vlab/Documents/Trianglemesh2D/fantoom3_2000x2000.mat")
N = 32
fant1 = fant['fantoom3']
fant1 = fant1.astype(float)
resgr = len(fant1)
bounds1 = np.array([11.496, 15.465])
bounds2 = np.array([1.29, 1.411])  
fant = fant1.copy()
numangles = 150#anglelist[angleindex]
print("number of angles is ")
print(numangles)
angles = np.linspace(0,2*np.pi,numangles,False)

fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1   
fant1[np.where(fant1 == 128)] = 0.5 
fant1[np.where(fant1 == 64)] = 0.5
fant1[np.where(fant1 == 255)] = 1
# yvalss = np.where(np.where(fant1 == 1)[0] < 875)[0]
# xvalss = np.where(np.where(fant1 == 1)[1] < 550)[0]
# indskeep = np.intersect1d(xvalss, yvalss)
# xvalss = np.where(fant1 == 1)[1][indskeep]
# yvalss = np.where(fant1 == 1)[0][indskeep]
# fant1 = fant1.ravel()
# fant1[yvalss*resgr + xvalss] = 0.55
# fant1 = fant1.reshape((resgr, resgr))

vol_geom = astra.create_vol_geom(resgr, resgr)
proj_geom = astra.create_proj_geom('parallel', len(fant1)/numpix, numpix, np.linspace(0,2*np.pi,numangles,False) - np.pi/2)

proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, sinogram = astra.create_sino(fant1, proj_id)

sinogram = sinogram/len(fant1)
sinogram = sinogram.ravel()
sinogram = sinogram + np.random.normal(0, 0.01*np.max(sinogram), len(sinogram))
sinogram[np.where(sinogram<0)] = 0
del fant1

minres = 0.01
res = minres
noise = 0.01*np.max(sinogram)
#noise = 0.01
idx = np.indices((N, N))  
X = idx[1] / (N - 1) 
Y = idx[0] / (N - 1)  

# Flatten to 1D arrays
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

# Convert list to NumPy array
triangles = np.array(triangles)
# Plot the Delaunay triangulation




#Solve system matrix
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

attenuation, stop1, stop2 = Sirtrec(systemmat, np.zeros(len(triangles)), 300, sinogram, 0, 0)

triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
qmax = 2 

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
verh = (1)*stopx/qmax
maxatt = np.max(attenuation)
stopcond = 2*((np.sqrt(len(sinogram))*noise*conditionnumber*np.linalg.norm(attenuation, 2)/np.linalg.norm(sinogram, 2))/np.sqrt(len(triangles))) + verh*np.sqrt(2)/2
areas = [Polygon(vertices[t]).area for t in triangles]
extrascore = np.zeros(len(triangles))
extrascore[splittriangles] = asppertr(vertices[triangles[splittriangles]], rads[splittriangles]) 


stopref = 0
stop2 = 0
counter = 0
maxvert = 20000

Cmat =  np.array(1/np.sum(systemmat, axis = 1)).ravel()
Rmat =  np.array((1/np.sum(systemmat, axis = 0))).ravel()
Cmat[np.where(np.sum(systemmat, axis = 1) == 0)[0]] = 0
Rmat[np.where(np.sum(systemmat, axis = 0) == 0)[0]] = 0
stopref = 0
stop2 = 0
counter = 0
maxvert = 20000
systemmat = systemmat.tocsc()

triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
while stopref == 0 and counter < maxvert:
      
       if np.max(atpertriangle + verh*extrascore) < stopcond :
           break
       splitnu = np.argmax(atpertriangle + verh*extrascore)
       triangles, vertices, attenuation, systemmat,  deletedtr, extratr = refinetrianglefaster(triang, triangles, vertices, attenuation, systemmat, splitnu, rads, cent, angles, numpix)
       triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
       Rmat = np.delete(Rmat, deletedtr)
       Rmatex = np.array((1/np.sum(systemmat[:, extratr], axis = 0)))[0].ravel()
       Rmatex[np.where(np.sum(systemmat[:, extratr], axis = 0) == 0)[0]] = 0
       Rmat = np.hstack((Rmat, Rmatex))
       attenuation, stop1ev, _ = SirtrecCR(systemmat, attenuation, 50, sinogram, stop1, stop2, Cmat, Rmat)
       #if np.mod(counter, 200) == 0:
       #  attenuation, stop1ev, _ = Sirtrec(systemmat, attenuation, 300, sinogram, 0, stop2)
       
       counter +=1
       rads = np.delete(rads,deletedtr)
       cent = np.delete(cent, deletedtr, axis=0)
       radsex, centex = circumcircle(vertices[triangles[extratr]])
       rads = np.hstack((rads, radsex))
       cent = np.vstack((cent, centex))
       #rads, cent = circumcircle(vertices[triangles])
       #randdriehoeken = np.unique(np.where(triang.neighbors < 0)[0])
       #splittriangles = np.setdiff1d(np.arange(len(triangles)), \
       #             np.hstack((randdriehoeken, np.where(rads < res)[0], \
       #             np.where(np.min(cent,axis=1) < 0)[0], \
       #                np.where(np.max(cent,axis=1) > 1)[0])))
    
       atpertriangle = np.delete(atpertriangle, deletedtr)
       atex = np.zeros((len(extratr), 3))   
       atex[:, 0] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,0])
       atex[:, 1] =  np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,1])
       atex[:, 2] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,2])
       atex[np.where(radsex < res)] = 0
       atex[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       atpertriangle = np.hstack((atpertriangle, np.max(atex, axis=1)))
       #areas = [Polygon(vertices[t]).area for t in triangles]
       extrascore = np.delete(extrascore, deletedtr)
       extraextrascore = asppertr(vertices[triangles[extratr]], rads[extratr])
       extraextrascore[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       extrascore = np.hstack((extrascore, extraextrascore)) 
  
  
  
selectdriehoeken = np.setdiff1d(np.arange(len(triangles)), np.unique(np.where(triang.neighbors < 0)[0]))
corners = np.hstack((np.argmin(np.linalg.norm(vertices, 2, 1)), np.argmax(np.linalg.norm(vertices, 2, 1)), \
                    np.intersect1d(np.where(np.min(vertices, axis=1) == 0)[0], np.where(np.max(vertices, axis=1) == 1)[0])) )
randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))
edges, scorepervert, qualperedge, scalercollapse = createedges(triangles, corners, randverts, attenuation,-1)
maxas = 1.5
minmerg = np.min(qualperedge)
edcol = np.argmin(qualperedge)
edges = np.array(edges)

minstart = 0.005**2 - scalercollapse*np.sqrt(2)/2
counter = 0
lenv = 20000
while minmerg < minstart  and counter < lenv :
        vert1 = edges[edcol][0]
        vert2 = edges[edcol][1] 
        #plt.triplot(vertices[:, 0], vertices[:, 1], triangles, color="gray")
        #plt.plot(vertices[edges[edcol]][:, 0], vertices[edges[edcol]][:, 1], 'y-')
        #plt.ylabel("Y")
        #plt.title("Delaunay Triangulation")
        #plt.show()
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
        areas = [Polygon(vertices[t]).area for t in triangnow]
        asnew = np.max(asppertr(vertices[triangnow], radss))
        vertices = np.delete(vertices, len(vertices)-1, axis=0)
        if any(np.isnan(areas)) :
             break
        del radss
        if  asnew < maxas: #or asnew < asold
             vertices, triangles, attenuation, edges, qualperedge, scorepervert, corners = \
                   collapseedge(vertices, triangles, attenuation, edges, qualperedge, scorepervert, edcol, corners)         
        else:
            #vertices = np.delete(vertices, len(vertices) - 1, axis=0)  
            qualperedge[edcol] = np.max(qualperedge)
        minmerg = np.min(qualperedge)
        edcol = np.argmin(qualperedge)
        counter+=1

triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

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

attenuationseg, connectionarray, scalerseg = Startingclusterfunction(attenuation, vertices, triangles, sinogram, systemmat, triang.neighbors)

attenuationseg, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)
#%%

for optsteps in range( 10):

     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/trianglesafterseg{optsteps}.npy", triangles) 
     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/attenuationafterseg{optsteps}.npy", attenuationseg)
     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/verticesafterafterseg{optsteps}.npy", vertices)

     edges = np.array([0, 0])  # Using a set to store unique edges
     normals = np.array([0, 0])
     atlist = np.array([0])
     tind = 0
     for trii in triangles:
      for eind in range(3):
         trr = np.intersect1d(np.where(triangles == trii[np.mod(eind, 3)])[0], np.where(triangles == trii[np.mod(eind+1, 3)])[0])
         if len(trr) > 1:
            #if connectionarray[tr[0]] != connectionarray[tr[1]]:           
            if attenuationseg[trr[0]] != attenuationseg[trr[1]]:           
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
               atlist = np.vstack((atlist,attenuationseg[tind]))
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
               atlist = np.vstack((atlist,0))

      tind += 1
     normals = np.delete(normals, 0, axis=0)
     it = 0

     edges = np.delete(edges, 0, axis=0)
     atlist = np.delete(atlist, 0)    


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
     projdif = sinogram - systemmat@attenuationseg
     verticesc = cuda.to_device(vertices)
     edgesc = cuda.to_device(edges)
     grid_x = len(edges)//block_x+1
     lentermpervertc = cuda.to_device(np.zeros(2*len(vertices)))
     lenpervertcalc[grid_x, block_x](verticesc, edgesc, lentermpervertc)
     lenterm = 0.5*np.linalg.norm(lentermpervertc.copy_to_host())**2
     scaleredge1 = (0.5*np.sum(projdif**2)/lenterm)
     randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))
     #scaleredge = 1*0.5*np.sum(projdif**2)/np.linalg.norm(np.hstack((vertices[triangles[:, 0]] - vertices[triangles[:, 1]], \
     #                    vertices[triangles[:, 1]] - vertices[triangles[:, 2]], \
     #s                     vertices[triangles[:, 2]] - vertices[triangles[:, 0]])))**2 
     areas = [Polygon(vertices[t]).area for t in triangles]

     trianglesint = np.zeros(len(triangles))
     for trind in range(len(triangles)):
        if len(np.intersect1d(triangles[trind], edges)) > 1.5:
           trianglesint[trind] = 1   

     trianglesint = trianglesint.astype(int)
     scaleredge2 = 0.5*np.sum(projdif**2)/(funaspecttr(triangles, vertices, np.where(trianglesint == 0)[0]))

     areaor = np.array(areas)
     for it in range(8):
       inded = 0
       for ed in edges:
        normed = np.array([vertices[ed[0]][1] - vertices[ed[1]][1], -(vertices[ed[0]][0] - vertices[ed[1]][0])]) 
        normed = normed/np.linalg.norm(normed)
        normals[inded][0] = normed[0]
        normals[inded][1] = normed[1]
        inded += 1
       vertices = vertices.ravel()

       areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]

      
       func0 = optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuationseg, trianglesint, scaleredge1, scaleredge2, areaor)
       inded = 0


       vertgrad = gradoptfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuationseg, trianglesint, scaleredge1, scaleredge2, areaor)
   
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
       print("demping")
       print(damp)
       func = optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuationseg, trianglesint, scaleredge1, scaleredge2, areaor)


       areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]
   

       its = 0
       print("starting value")
       print(func0)
       print("other values")
       print(func)
       while (func > func0 or np.abs(np.sum(areas) - 1) > 10**(-11)) and its < 10: 
       #while func > func0 + m*damp*0.00001 and its < 25: 
         damp = damp/2
         vertices = vertices + damp*dx
         areas = [Polygon(np.reshape(vertices, (int(len(vertices)/2), 2))[t]).area for t in triangles]
         print(np.sum(areas))
         areas = np.array(areas)
         its += 1
         if  np.abs(np.sum(areas) - 1) > 10**(-11):
            continue
         func = optfun(vertices, triangles, edges, angles, numpix, sinogram, atlist, attenuationseg,  trianglesint, scaleredge1, scaleredge2, areaor)    
         print(func)
       if np.abs(np.sum(areas) - np.sum(areaor)) > 10**(-11):
          vertices = vertices + damp*dx
          vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
          break
       vertices = np.reshape(vertices, (int(len(vertices)/2), 2))
       triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        
   
       radss, _ = circumcircle(vertices[triangles])
       #asnew = asppertr(vertices[triangles], radss)
       #fliptriangles = np.where(asnew > 3)
   

   
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
                 vertices, triangles, attenuationseg, edges, trianglescheck = collapseedgenoscores(vertices, triangles, attenuationseg, np.min(ed), np.max(ed), edges, trianglescheck)
                 break

            
            elif attenuationseg[trianglesround[0]] == attenuationseg[trianglesround[1]]:
              trianglespos = np.array([[ed[0], np.setdiff1d(triangles[trianglesround[1]], ed)[0], np.setdiff1d(triangles[trianglesround[0]], ed)[0]], \
                                [ed[1], np.setdiff1d(triangles[trianglesround[1]], ed)[0], np.setdiff1d(triangles[trianglesround[0]], ed)[0]]])
              radspos, _ = circumcircle(vertices[trianglespos])   
              areasnew = [Polygon(vertices[t]).area for t in trianglespos]
              areaor = [Polygon(vertices[t]).area for t in triangles[trianglesround]]
              atround = np.unique(attenuationseg[triang.neighbors[trianglesround[0]]])
              if np.max(asppertr(vertices[trianglespos], radspos)) < np.max(asnew[trianglesround]) and \
                 np.abs(np.sum(areaor) - np.sum(areasnew)) < 10**(-10):
                triangles[trianglesround] = trianglespos
                asnew[trianglesround] = asppertr(vertices[trianglespos], radspos)
       counted += 1 
       tind += 1


     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

     radss, _ = circumcircle(vertices[triangles])
     #asnew = asppertr(vertices[triangles], radss)
     #fliptriangles = np.where(asnew > 3)

     attenuation = attenuationseg
     attenuation = attenuationseg
     corners = np.hstack((np.argmin(np.linalg.norm(vertices, 2, 1)), np.argmax(np.linalg.norm(vertices, 2, 1)), \
                     np.intersect1d(np.where(np.min(vertices, axis=1) == 0)[0], np.where(np.max(vertices, axis=1) == 1)[0])) )
     randverts = np.unique(np.hstack((np.where(np.min(vertices, axis=1) == 0), np.where(np.max(vertices, axis=1) == 1))))
     edges, scorepervert, qualperedge, scalercollapse = createedges(triangles, corners, randverts, attenuationseg, 0)
     maxas = 1.5
     #maxas = 0
     minmerg = np.min(qualperedge)
     edcol = np.argmin(qualperedge)
     edges = np.array(edges)
     minstart = 0.00005**2 -scalercollapse*maxas
     counter = 0
     lenv = 20000
     while minmerg < minstart  and counter < lenv :
       vert1 = edges[edcol][0]
       vert2 = edges[edcol][1] 
       #plt.triplot(vertices[:, 0], vertices[:, 1], triangles, color="gray")
       #plt.plot(vertices[edges[edcol]][:, 0], vertices[edges[edcol]][:, 1], 'y-')
       #plt.xlabel("X")
       #plt.ylabel("Y")
       #plt.title("Delaunay Triangulation")
       #plt.show()
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
       areas = [Polygon(vertices[t]).area for t in triangnow]
       triangold = np.hstack((trianglecol, trianglechange))
       areas0 = [Polygon(vertices[triangles[t]]).area for t in triangold]
       asnew = np.max(asppertr(vertices[triangnow], radss))
       vertices = np.delete(vertices, len(vertices)-1, axis=0)
       if any(np.isnan(areas)) :
           break
       del radss
       if  asnew < maxas and np.abs(np.sum(areas) - np.sum(areas0)) < 10**(-11): #or asnew < asold
           vertices, triangles, attenuation, edges, qualperedge, scorepervert, corners = \
                 collapseedge(vertices, triangles, attenuation, edges, qualperedge, scorepervert, edcol, corners)         
       else:
          #vertices = np.delete(vertices, len(vertices) - 1, axis=0)  
          qualperedge[edcol] = np.max(qualperedge)
       minmerg = np.min(qualperedge)
       edcol = np.argmin(qualperedge)
       counter+=1
     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

     radss, _ = circumcircle(vertices[triangles])


     attenuationseg = attenuation

     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/1pcnoise/{namelist[valindex]}/volcor/trianglesafterexmerg{optsteps}_scaleraspop100.npy", triangles) 
     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/1pcnoise/{namelist[valindex]}/volcor/attenuationafterexmerg{optsteps}_scaleraspop100.npy", attenuationseg)
     #np.save(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/1pcnoise/{namelist[valindex]}/volcor/verticesafterexmerg{optsteps}_scaleraspop100.npy", vertices)



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
     verh = (1)*stopx/qmax
     maxatt = np.max(attenuation)
     stopcond = 2*((np.sqrt(len(sinogram))*noise*conditionnumber*np.linalg.norm(attenuation, 2)/np.linalg.norm(sinogram, 2))/np.sqrt(len(triangles))) + verh*np.sqrt(2)/2
     areas = [Polygon(vertices[t]).area for t in triangles]
     #stopcond = (0.02*len(sinogram)/np.linalg.norm(sinogram, 2))*conditionnumber*np.sqrt(np.sum(attenuation**2*np.array(areas)**2)) + verh*np.sqrt(2)/2
     extrascore = np.zeros(len(triangles))
     extrascore[splittriangles] = asppertr(vertices[triangles[splittriangles]], rads[splittriangles]) 


     stopref = 0
     stop2 = 0
     counter = 0
     maxvert = 20000

     counter = 0
     maxvert = 20000
     systemmat = systemmat.tocsc()

     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
     while stopref == 0 and counter < maxvert:
      
       if np.max(atpertriangle + verh*extrascore) < stopcond :
           break
       splitnu = np.argmax(atpertriangle + verh*extrascore)
       triangles, vertices, attenuation, deletedtr, extratr = refinetrianglefasterns(triang, triangles, vertices, attenuation, splitnu, rads, cent, angles, numpix)
       triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
       #if np.mod(counter, 200) == 0:
       #  attenuation, stop1ev, _ = Sirtrec(systemmat, attenuation, 300, sinogram, 0, stop2)
       
       counter +=1
       rads = np.delete(rads,deletedtr)
       cent = np.delete(cent, deletedtr, axis=0)
       radsex, centex = circumcircle(vertices[triangles[extratr]])
       rads = np.hstack((rads, radsex))
       cent = np.vstack((cent, centex))
       #rads, cent = circumcircle(vertices[triangles])
       #randdriehoeken = np.unique(np.where(triang.neighbors < 0)[0])
       #splittriangles = np.setdiff1d(np.arange(len(triangles)), \
       #             np.hstack((randdriehoeken, np.where(rads < res)[0], \
       #             np.where(np.min(cent,axis=1) < 0)[0], \
       #                np.where(np.max(cent,axis=1) > 1)[0])))
    
       atpertriangle = np.delete(atpertriangle, deletedtr)
       atex = np.zeros((len(extratr), 3))   
       atex[:, 0] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,0])
       atex[:, 1] =  np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,1])
       atex[:, 2] = np.abs(attenuation[extratr] - attenuation[triang.neighbors[extratr]][:,2])
       atex[np.where(radsex < res)] = 0
       atex[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       atpertriangle = np.hstack((atpertriangle, np.max(atex, axis=1)))
       #areas = [Polygon(vertices[t]).area for t in triangles]
       extrascore = np.delete(extrascore, deletedtr)
       extraextrascore = asppertr(vertices[triangles[extratr]], rads[extratr])
       extraextrascore[np.where(np.min(triang.neighbors[extratr], axis=1) < -0.5)] = 0
       extrascore = np.hstack((extrascore, extraextrascore)) 
  
  
     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)



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


     attenuation, stop1, _ = Sirtrec(systemmat, attenuation, 500, sinogram, 0, 0)

     triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)        

     radss, _ = circumcircle(vertices[triangles])


     attenuationseg, connectionarray = clusterfunctionf(attenuation, vertices, triangles, numpix, sinogram, systemmat, triang.neighbors, scalerseg, res)
     



# %%
#plot colorbar
# fig, ax = plt.subplots(figsize=(9,9))
# im = ax.imshow(fant1, cmap="grey", aspect=1)
# plt.axis("off")
# divider = make_axes_locatable(ax)
# #cax = divider.new_vertical(size="5%", pad=0.25,  pack_start=True)
# #cax = divider.new_vertical(pad=0.25, shrink=0.9, pack_start=True)
# cax = ax.inset_axes([0, -0.05, 1, 0.04], transform=ax.transAxes)
# fig.add_axes(cax)
# cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
# cbar.ax.set_xlabel("Attenuation",size=21,  labelpad=5)
# cbar.ax.xaxis.set_ticks_position('bottom')
# cbar.ax.tick_params(labelsize=21) 
# cbar.ax.xaxis.set_label_position('bottom')
# plt.show()
#%%
'''(
reslist = np.array([0.02, 0.015, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003])
namelist = ["res_0_02", "res_0_015", "res_0_01", "res_0_009", \
             "res_0_008", "res_0_007", "res_0_006", "res_0_005",\
                   "res_0_004", "res_0_003"]
MSEperres = np.zeros(len(reslist))
PSNRperres = np.zeros(len(reslist))
for resindex in range(len(namelist)):
   optsteps = 4
   contvalue = 100*(valueslist[valindex]-0.5)/0.5
   triangles = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/{namelist[resindex]}/trianglesafteropt{optsteps}_scaleraspop100.npy") 
   attenuation = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/{namelist[resindex]}/attenuationafteropt{optsteps}_scaleraspop100.npy")
   vertices = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/{namelist[resindex]}/verticesafteropt{optsteps}_scaleraspop100.npy")
   fig, ax = plt.subplots(figsize=(9,9))
   triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
   ax.tripcolor(triang, attenuation, cmap='grey')

   plt.axis("off")
   #divider = make_axes_locatable(ax)
   #plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/{namelist[resindex]}/res_{namelist[resindex]}_DAMMER.png",bbox_inches='tight', pad_inches=0.0)
   plt.show()

   fant = scipy.io.loadmat("/home/vlab/Documents/Trianglemesh2D/fantoom3_2000x2000.mat")
   valuenow = valueslist[valindex]
   fant1 = fant['fantoom3']
   fant1 = fant1.astype(float)
   resgr = len(fant1)
   fant = fant1.copy()
   fant1[np.where(fant1 == 128)] = 0.5 
   fant1[np.where(fant1 == 64)] = 0.5
   fant1[np.where(fant1 == 255)] = 1
   # yvalss = np.where(np.where(fant1 == 1)[0] < 875)[0]
   # xvalss = np.where(np.where(fant1 == 1)[1] < 550)[0]
   # indskeep = np.intersect1d(xvalss, yvalss)
   # xvalss = np.where(fant1 == 1)[1][indskeep]
   # yvalss = np.where(fant1 == 1)[0][indskeep]
   # fant1 = fant1.ravel()
   # fant1[yvalss*resgr + xvalss] = valuenow
   fant1 = fant1.reshape((resgr, resgr))
   fig, ax = plt.subplots(figsize=(9,9))
   ax.imshow(fant1, cmap="grey", aspect=1)
   plt.axis("off")
   #divider = make_axes_locatable(ax)
   #plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/1pcnoise/{namelist[valindex]}/contrast_{contvalue}pc_groundtruthzoom.png")
   plt.show()
   mse, psnr = MSEPSNRnonumba(triangles, vertices, attenuation, fant1)
   MSEperres[resindex] = mse
   PSNRperres[resindex] = psnr

np.save("/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/MSEperres.npy", MSEperres)
np.save("/home/vlab/Documents/Trianglemesh2D/phantom2/nonoise/150projs/PSNRperres.npy", PSNRperres)
#%%
valueslist = np.array([0.8, 0.7, 0.6, 0.55, 0.525, 0.51])
namelist = ["attenuation0_8", "attenuation0_7", "attenuation0_6", \
              "attenuation0_55", "attenuation0_525", "attenuation0_51"]
for valindex in range(2,6):
   #  optsteps = 4
    contvalue = int(100*(valueslist[valindex]-0.5)/0.5)
    triangles = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/trianglesafteropt{optsteps}_scaleraspop100.npy") 
    attenuation = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/attenuationafteropt{optsteps}_scaleraspop100.npy")
    vertices = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/verticesafteropt{optsteps}_scaleraspop100.npy")
    fig, ax = plt.subplots(figsize=(9,9))
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.tripcolor(triang, attenuation, cmap='grey')
    ax.plot([0.145, 0.245], [0.59, 0.59], color='red')
    ax.plot([0.145, 0.245], [0.72, 0.72], color='red')
    ax.plot([0.145, 0.145], [0.59, 0.72], color='red')
    ax.plot([0.245, 0.245], [0.59, 0.72], color='red')
    plt.axis("off")
   #  #divider = make_axes_locatable(ax)
    plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/contrast_{contvalue}pc_DAMMER_square.png")
    plt.show()
   
    optsteps = 4
    triangles = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise//{namelist[valindex]}/scalerman/scaleredge1_0_01/trianglesafteropt{optsteps}_scaleraspop100.npy") 
    attenuation = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/attenuationafteropt{optsteps}_scaleraspop100.npy")
    vertices = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/verticesafteropt{optsteps}_scaleraspop100.npy")
    fig, ax = plt.subplots(figsize=(10,13))
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.tripcolor(triang, attenuation, cmap='grey')
    ax.plot([0.145, 0.245], [0.59, 0.59], color='red', linewidth=8)
    ax.plot([0.145, 0.245], [0.72, 0.72], color='red', linewidth=8)
    ax.plot([0.145, 0.145], [0.59, 0.72], color='red', linewidth=8)
    ax.plot([0.245, 0.245], [0.59, 0.72], color='red', linewidth=8)
    plt.xlim(0.145, 0.245)
    plt.ylim(0.59, 0.72)
    plt.axis("off")
    #divider = make_axes_locatable(ax)
    plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/contrast_{contvalue}pc_DAMMER_squarezoom.png")
    plt.show()

    #Shepp Logan phantom:
    fig, ax = plt.subplots(figsize=(20,10))
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.tripcolor(triang, attenuation, cmap='grey')
    ax.plot([0.4, 0.6], [0.15, 0.15], color='red', linewidth=8)
    ax.plot([0.4, 0.6], [0.25, 0.25], color='red', linewidth=8)
    ax.plot([0.4, 0.4], [0.15, 0.25], color='red', linewidth=8)
    ax.plot([0.6, 0.6], [0.15, 0.25], color='red', linewidth=8)
    plt.xlim(0.4, 0.6)
    plt.ylim(0.15, 0.25)
    plt.axis("off")
    #divider = make_axes_locatable(ax)
    plt.show() 
   fig, ax = plt.subplots(figsize=(9,9))
   ax.imshow(fant1, cmap="grey", aspect=1)
   plt.axis("off")
   ax.set_xticks([])
   ax.set_yticks([])
   dimvox = 2000
   ax.plot([0.43*dimvox, 0.56*dimvox], [(1-0.25)*dimvox, (1-0.25)*dimvox], color='red', linewidth=8)
   ax.plot([0.43*dimvox, 0.43*dimvox], [(1-0.15)*dimvox, (1-0.25)*dimvox], color='red', linewidth=8)
   ax.plot([0.43*dimvox, 0.56*dimvox], [(1-0.15)*dimvox, (1-0.15)*dimvox], color='red', linewidth=8)
   ax.plot([0.56*dimvox, 0.56*dimvox], [(1-0.25)*dimvox, (1-0.15)*dimvox], color='red', linewidth=8)
   plt.xlim(0.43*dimvox, 0.56*dimvox)
   plt.ylim((1-0.15)*dimvox, (1-0.25)*dimvox)
   plt.axis("off")

   plt.show()
           fant = scipy.io.loadmat("/home/vlab/Documents/Trianglemesh2D/fantoom3_2000x2000.mat")
    valuenow = valueslist[valindex]
    fant1 = fant['fantoom3']
    fant1 = fant1.astype(float)
    resgr = len(fant1)
    fant = fant1.copy()
    fant1[np.where(fant1 == 128)] = 0.5 
    fant1[np.where(fant1 == 64)] = 0.5
    fant1[np.where(fant1 == 255)] = 1
    yvalss = np.where(np.where(fant1 == 1)[0] < 875)[0]
    xvalss = np.where(np.where(fant1 == 1)[1] < 550)[0]
    indskeep = np.intersect1d(xvalss, yvalss)
    xvalss = np.where(fant1 == 1)[1][indskeep]
    yvalss = np.where(fant1 == 1)[0][indskeep]
    fant1 = fant1.ravel()
    fant1[yvalss*resgr + xvalss] = valuenow
    fant1 = fant1.reshape((resgr, resgr))
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(fant1, cmap="grey")
    ax.set_xticks([])
    ax.set_yticks([])
    dimvox = 2000
    ax.plot([0.145*dimvox, 0.245*dimvox], [(1-0.59)*dimvox, (1-0.59)*dimvox], color='red', linewidth=8)
    ax.plot([0.145*dimvox, 0.245*dimvox], [(1-0.72)*dimvox, (1-0.72)*dimvox], color='red', linewidth=8)
    ax.plot([0.145*dimvox, 0.145*dimvox], [(1-0.72)*dimvox, (1-0.59)*dimvox], color='red', linewidth=8)
    ax.plot([0.245*dimvox, 0.245*dimvox], [(1-0.72)*dimvox, (1-0.59)*dimvox], color='red', linewidth=8)
    plt.xlim(0.145*dimvox, 0.245*dimvox)
    plt.ylim((1-0.59)*dimvox, (1-0.72)*dimvox)
    plt.axis("off")
    #divider = make_axes_locatable(ax)
    plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/nonoise/{namelist[valindex]}/scalerman/scaleredge1_0_01/contrast_{contvalue}pc_groundtruthzoom.png")
    plt.show()


# %%
#Experiment beam hardening /home/vlab/Documents/Trianglemesh2D/phantom2/beamhard


#  optsteps = 4
contvalue = int(100*(valueslist[valindex]-0.5)/0.5)
triangles = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/trianglesafteropt{optsteps}_scaleraspop100.npy") 
attenuation = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/attenuationafteropt{optsteps}_scaleraspop100.npy")
vertices = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/verticesafteropt{optsteps}_scaleraspop100.npy")
fig, ax = plt.subplots(figsize=(9,9))
triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
ax.tripcolor(triang, attenuation, cmap='grey')
ax.plot([0.145, 0.245], [0.59, 0.59], color='red')
ax.plot([0.145, 0.245], [0.72, 0.72], color='red')
ax.plot([0.145, 0.145], [0.59, 0.72], color='red')
ax.plot([0.245, 0.245], [0.59, 0.72], color='red')
plt.axis("off")
   #  #divider = make_axes_locatable(ax)
plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/contrast_{contvalue}pc_DAMMER_square.png")
plt.show()
   
optsteps = 4
triangles = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/lowercontrast/1pcnoise//{namelist[valindex]}/trianglesafteropt{optsteps}_scaleraspop100.npy") 
attenuation = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/attenuationafteropt{optsteps}_scaleraspop100.npy")
vertices = np.load(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/verticesafteropt{optsteps}_scaleraspop100.npy")
fig, ax = plt.subplots(figsize=(10,13))
triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
ax.tripcolor(triang, attenuation, cmap='grey')
ax.plot([0.145, 0.245], [0.59, 0.59], color='red', linewidth=8)
ax.plot([0.145, 0.245], [0.72, 0.72], color='red', linewidth=8)
ax.plot([0.145, 0.145], [0.59, 0.72], color='red', linewidth=8)
ax.plot([0.245, 0.245], [0.59, 0.72], color='red', linewidth=8)
plt.xlim(0.145, 0.245)
plt.ylim(0.59, 0.72)
plt.axis("off")
#divider = make_axes_locatable(ax)
plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/contrast_{contvalue}pc_DAMMER_squarezoom.png")
plt.show()

     vertices = np.load("/home/vlab/Documents/Trianglemesh2D/phantomSL/lowerangles/nonoise/150angles/attempt3/verticesafteropt12_scaleraspop100.npy")
     attenuationseg = np.load("/home/vlab/Documents/Trianglemesh2D/phantomSL/lowerangles/nonoise/150angles/attempt3/attenuationafteropt12_scaleraspop100.npy")
     attenuation = attenuationseg
     triangles = np.load("/home/vlab/Documents/Trianglemesh2D/phantomSL/lowerangles/nonoise/150angles/attempt3/trianglesafteropt12_scaleraspop100.npy")    

fant = scipy.io.loadmat("/home/vlab/Documents/Trianglemesh2D/fantoom3_2000x2000.mat")
valuenow = valueslist[valindex]
fant1 = fant['fantoom3']
fant1 = fant1.astype(float)
resgr = len(fant1)
fant = fant1.copy()
fant1[np.where(fant1 == 128)] = 1.35
fant1[np.where(fant1 == 64)] = 1.35
fant1[np.where(fant1 == 255)] = 13.5
yvalss = np.where(np.where(fant1 == 1)[0] < 875)[0]
xvalss = np.where(np.where(fant1 == 1)[1] < 550)[0]
indskeep = np.intersect1d(xvalss, yvalss)
xvalss = np.where(fant1 == 1)[1][indskeep]
yvalss = np.where(fant1 == 1)[0][indskeep]
fant1 = fant1.ravel()
fant1[yvalss*resgr + xvalss] = valuenow
fant1 = fant1.reshape((resgr, resgr))
fig, ax = plt.subplots(figsize=(9,9))
ax.imshow(fant1, cmap="grey")
ax.set_xticks([])
ax.set_yticks([])
dimim = 2000
ax.plot([0.145*dimim, 0.245*dimim], [(1-0.59)*dimim, (1-0.59)*dimim], color='red', linewidth=8)
ax.plot([0.145*dimim, 0.245*dimim], [(1-0.72)*dimim, (1-0.72)*dimim], color='red', linewidth=8)
ax.plot([0.145*dimim, 0.145*dimim], [(1-0.72)*dimim, (1-0.59)*dimim], color='red', linewidth=8)
ax.plot([0.245*dimim, 0.245*dimim], [(1-0.72)*dimim, (1-0.59)*dimim], color='red', linewidth=8)
plt.xlim(0.145*dimim, 0.245*dimim)
plt.ylim((1-0.59)*dimim, (1-0.72)*dimim)
plt.axis("off")
#divider = make_axes_locatable(ax)
plt.savefig(f"/home/vlab/Documents/Trianglemesh2D/phantom2/beamhard/contrast_{contvalue}pc_groundtruthzoom.png")
plt.show()


#phantom 1:
# 
#  

fig, ax = plt.subplots(figsize=(9,9))
ax.imshow(reconstructionbest, cmap="grey")
ax.set_xticks([])
ax.set_yticks([])
dimim = len(reconstructionbest)
plt.axis("off")
ax.plot([0.5*dimim, 0.6*dimim], [(1-0.65)*dimim, (1-0.65)*dimim], color='red', linewidth=8)
ax.plot([0.5*dimim, 0.6*dimim], [(1-0.55)*dimim, (1-0.55)*dimim], color='red', linewidth=8)
ax.plot([0.5*dimim, 0.5*dimim], [(1-0.65)*dimim, (1-0.55)*dimim], color='red', linewidth=8)
ax.plot([0.6*dimim, 0.6*dimim], [(1-0.65)*dimim, (1-0.55)*dimim], color='red', linewidth=8)
plt.xlim(0.5*dimim, 0.6*dimim)
plt.ylim((1-0.65)*dimim, (1-0.55)*dimim)
#divider = make_axes_locatable(ax)
plt.show()

)'''
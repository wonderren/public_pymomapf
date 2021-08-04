"""
Author: Zhongqiang (Richard) Ren
Version@2021
Remark: Entry point to planners.
"""
import context
import time

import numpy as np
import matplotlib.pyplot as plt

import common 
import mocbs
import moastar # NAMOA*
import momstar


def RunToyExample():
  """
  """

  # a tiny world with very simple cost vectors.

  grids = np.zeros((3,3))
  grids[0,:]=1
  grids[1,0]=1
  grids[1,2]=1

  sy = np.array([2,2]) # start y = rows in grid image, the kth component corresponds to the kth robot.
  sx = np.array([0,2]) # start x = column in grid image
  gy = np.array([2,2]) # goal y
  gx = np.array([2,0]) # goal x

  cvecs = [np.array([2,2]), np.array([0,5])] # the kth component corresponds to the kth robot.
  cgrids = [np.ones((3,3)), np.ones((3,3))] # the mth component corresponds to the mth objective.
  # cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.

  cdim = len(cvecs[0])

  ##################################################################
  #### choose one of the planner to run by uncommenting the code ###
  ##################################################################

  #### Invoke MO-CBS planner ###
  # res = mocbs.RunMocbsMAPF(grids, sx, sy, gx, gy, cvecs, cgrids, cdim, 1.0, 0.0, np.inf, 10, 2)

  #### Invoke NAMOA* planner ###
  # res = moastar.RunMoAstarMAPF(grids, sx, sy, gx, gy, cvecs, cgrids, cdim, 1.0, 0.0, np.inf, 10)

  #### Invoke MOM* planner ###
  res = momstar.RunMoMstarMAPF(grids, sx, sy, gx, gy, cvecs, cgrids, cdim, 1.0, 0.0, np.inf, 10)

  print(res)
  
  return


def main():
  """
  """
  RunToyExample()
  return


if __name__ == '__main__':
  print("begin of main")
  main()
  print("end of main")


  
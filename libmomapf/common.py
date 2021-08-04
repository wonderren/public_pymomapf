"""
Author: Zhongqiang (Richard) Ren
Version@2021
Remark: some of the code is redundant and needs a clean up.
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq as hpq
import matplotlib.cm as cm
import json
import copy

NUM_EPSILON = 1e-6

class PrioritySet(object):
  """
  priority queue, min-heap
  """
  def __init__(self):
    """
    no duplication allowed
    """
    self.heap_ = []
    self.set_ = set()
  def add(self, pri, d):
    """
    will check for duplication and avoid.
    """
    if not d in self.set_:
        hpq.heappush(self.heap_, (pri, d))
        self.set_.add(d)
  def pop(self):
    """
    impl detail: return the first(min) item that is in self.set_
    """
    pri, d = hpq.heappop(self.heap_)
    while d not in self.set_:
      pri, d = hpq.heappop(self.heap_)
    self.set_.remove(d)
    return pri, d
  def size(self):
    return len(self.set_)
  def print(self):
    print(self.heap_)
    print(self.set_)
    return
  def has(self, d):
    if d in self.set_:
      return True
    return False
  def remove(self, d):
    """
    implementation: only remove from self.set_, not remove from self.heap_ list.
    """
    if not d in self.set_:
      return False
    self.set_.remove(d)
    return True

def IsDictSubset(dict1, dict2):
  """return if dict1 is a subset of dict2"""
  for k in dict1:
    if k not in dict2:
      return False
  return True

def ItvOverlap(ita,itb,jta,jtb):
  """
  check if two time interval are overlapped or not.
  """
  if ita >= jtb or jta >= itb: # non-overlap
    return False, -1.0, -1.0
  # must overlap now
  tlb = jta # find the larger value among ita and jta, serve as lower bound
  if ita >= jta:
    tlb = ita
  tub = jtb # find the smaller value among itb and jtb, serve as upper bound
  if itb <= jtb:
    tub = itb
  return True, tlb, tub

def Equal(v1,v2):
  for idx in range(len(v1)):
    if abs(v1[idx] - v2[idx]) > NUM_EPSILON:
      return False
  return True

def DominantLess(v1,v2):
  """
  given two vector v1,v2 (list or tuple), return if v1 dominates v2.
  """
  # if len(v1) == 1:
  #   return v1 < v2
  exist_strict_less = False
  for idx in range(len(v1)):
    if v1[idx] > v2[idx] + NUM_EPSILON:
      return False # v1 does not dominate v2
    else:
      if v1[idx] < v2[idx] - NUM_EPSILON:
        exist_strict_less = True
  if exist_strict_less:
    return True
  else:
    return False

def StrongDominantLess(v1,v2):
  """
  given two vector v1,v2 (list or tuple), return if v1 strictly dominates v2.
  """
  for idx in range(len(v1)):
    if v1[idx] >= v2[idx]:
      return False # v1 does not dominate v2
  return True

def DomOrEqual(v1,v2):
  """
  """
  if DominantLess(v1,v2) or Equal(v1,v2):
    return True
  return False

def WeakDom(v1,v2, eps=0.0):
  """
  every element in v1 is <= the corresponding lement in v2.
  If eps > 0, then this is epsilon-dominance.
  """
  if np.sum( np.array(v1) <= (1.0+eps) * np.array(v2) ) == len(v1):
    return True
  return False

def UFFind(dic, i):
  """
  dic = Union-find data structure, find operation. Find root of i.
  if i is not in dic, ERROR!
  """
  if i not in dic:
    print("[ERROR] UFFind, i not in dic!!!")
    return "WTF??"
  while(dic[i] != i):
    dic[i] = dic[dic[i]] # path compression
    i = dic[i]
  return i

def UFUnion(dic, i, j):
  """
  dic = Union-find data structure, union operation, 
   union the set of i and the set of j.
  """
  rooti = UFFind(dic,i)
  rootj = UFFind(dic,j)
  if rooti == rootj:
    return
  else:
    maxij = max(rooti,rootj)
    if rooti == maxij:
      dic[rootj] = rooti
    else:
      dic[rooti] = rootj

def UFDict2ListSet(dic):
  """
  Convert a dict that represents a union find data structure to a list of (disjoint) sets.
  """
  out = list()
  aux = dict() # aux is a dic that maps set id to set index in the list.
  for k in dic:
    # print("k:",k)
    root = UFFind(dic,k)
    # print("root:",root)
    if root in aux:
      out[aux[root]].add(k)
    else:
      # print("root:",root)
      out.append(set())
      aux[root] = len(out)-1
      # print("aux[root]:",aux[root])
      out[aux[root]].add(k)
  return out,aux

def L1Norm(vec1,vec2):
  return np.sum(np.abs(vec1-vec2))

def L2Norm(vec1,vec2):
  return np.linalg.norm(vec1-vec2)

def HausdorffDistMinPart(vec, vecs):
  """
  find the nearest dist between vec and some vec in vecs
  """
  mind = np.inf
  for vec2 in vecs:
    d = L2Norm(vec,vec2)
    if d < mind:
      mind = d
  return mind

def HausdorffDist(vecs1,vecs2):
  """
  given two list of vectors, measure the Hausdorff distance.
  The metric between two vec is choose to be the max()
  """
  dmax1 = 0
  for vec1 in vecs1:
    d = HausdorffDistMinPart(vec1, vecs2)
    # print(" min d = ", d)
    if d > dmax1:
      dmax1 = d
  # print(" max d1 = ", dmax1)
  dmax2 = 0
  for vec2 in vecs2:
    d = HausdorffDistMinPart(vec2, vecs1)
    # print(" min d = ", d)
    if d > dmax2:
      dmax2 = d
  # print(" max d2 = ", dmax2)
  return max(dmax1, dmax2)

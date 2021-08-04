"""
Author: Zhongqiang (Richard) Ren
Version@2021
Remark: some of the code is redundant and needs a clean up.
"""
import numpy as np
import heapq as hpq
import common as cm
import itertools as itt
import time
import copy
import moastar
import sys

MAX_NGH_SIZE=1e7

class MoMstar(moastar.MoAstarMAPF):
  """MoMstar is derived from MoAstarMAPF"""
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol):
    super(MoMstar, self).__init__(grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol)
    self.collision_dict = dict() 
    self.backprop_dict = dict() 
    self.max_col_set = 0

  def AddToBackSet(self, sid, nsid):
    """
    """
    if nsid not in self.backprop_dict:
      self.backprop_dict[nsid] = set()
    self.backprop_dict[nsid].add(sid)
    return

  def ReopenState(self, sid):
    """
    """
    self.open_list.add(np.sum(self.f_value[sid]), sid )
    return

  def UnionColSet(self, s, cset):
    """
    """
    for k in cset: # union
      self.collision_dict[s.id][k] = 1
    if len(self.collision_dict[s.id]) > self.max_col_set:
      self.max_col_set = len(self.collision_dict[s.id])
    return 

  def BackPropagation(self, s, cset):
    """
    collision set back propagation
    """
    if len(cset) > self.max_col_set:
      self.max_col_set = len(cset)
    if cm.IsDictSubset(cset, self.collision_dict[s.id]):
      return
    self.UnionColSet(s, cset)
    self.ReopenState(s.id)
    if s.id not in self.backprop_dict:
      return # reach init state
    for ps_id in self.backprop_dict[s.id]:
      self.BackPropagation(self.all_visited_s[ps_id], cset)

  def GetNeighbors(self, s, tstart):
    """
    """
    nid_dict = dict()
    ngh_size = 1
    for idx in range(self.num_robots): # loop over all robot
      tnow = time.perf_counter() # check if timeout.
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      nid_dict[idx] = list()
      cy = int(np.floor(s.loc[idx]/self.nxt)) # current x
      cx = int(s.loc[idx]%self.nxt) # current y
      if idx in self.collision_dict[s.id]:
        ngh_size = ngh_size * len(self.action_set_x)
        for action_idx in range(len(self.action_set_x)):
          nx = cx+self.action_set_x[action_idx] # next x
          ny = cy+self.action_set_y[action_idx] # next y 
          if not (ny < 0 or ny >= self.nyt or nx < 0 or nx >= self.nxt):
            # do not exceed border
            if self.grids[ny,nx] == 0: # not obstacle
              nid_dict[idx].append( ny*self.nxt+nx )
      else:
        ngh_size = ngh_size * len(self.optm_policis[idx][cy][cx])
        for next_xy in self.optm_policis[idx][cy][cx]: 
          nx = next_xy[0] # follow the convention in GridPolicy in common.py
          ny = next_xy[1]
          nid_dict[idx].append( ny*self.nxt+nx )
    if ngh_size > MAX_NGH_SIZE: # too many ngh, doom to fail
      print(" !!! ngh_size too large:", ngh_size, " > ", MAX_NGH_SIZE, " !!!")
      return list(), False
    s_ngh = list()
    all_loc = list( itt.product(*(nid_dict[ky] for ky in sorted(nid_dict))) )
    for ida in range(len(all_loc)):
      tnow = time.perf_counter()
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      ns = moastar.MoAstarState(self.state_gen_id, tuple(all_loc[ida]), s.cost_vec+self.GetCost(s.loc,all_loc[ida]) )
      self.state_gen_id = self.state_gen_id + 1 
      s_ngh.append(ns)
    return s_ngh, True

  def FilterState(self,s,f_array):
    """
    """
    if self.s_f.loc not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[self.s_f.loc]: # notice: only states that reaches goal !!
      if sid == s.id:
        continue
      if cm.DomOrEqual( self.f_value[sid], f_array ):
        return True # filtered
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def RefineFrontier(self, s):
    # A detail. MOM*, no refinement, otherwise need to properly maintain back_set.
    return

  def RefineGoalFrontier(self, s): 
    """
    Use s to remove dominated states in frontier set at goal.
    """
    if s.loc != self.s_f.loc:
      return
    if self.s_f.loc not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[self.s_f.loc])
    for sid in temp_frontier:
      if cm.DominantLess(s.cost_vec, self.all_visited_s[sid].cost_vec):
        self.frontier_map[self.s_f.loc].remove(sid)
    return

  def DominanceBackprop(self, s):
    """
    s is dominated and pruned. backprop s relavant info.
    """
    if s.loc not in self.frontier_map:
      return
    if s.id not in self.backtrack_dict:
      return
    parent_sid = self.backtrack_dict[s.id]
    for sid in self.frontier_map[s.loc]:
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        self.AddToBackSet(parent_sid, sid)
        if len(self.collision_dict[sid]) > 0:
          self.BackPropagation(self.all_visited_s[parent_sid], self.collision_dict[sid])
    return
  
  def Search(self, search_limit=np.inf, time_limit=10):
    print("--------- MOM* Search begin , nr = ", self.num_robots, "---------")
    if self.heu_failed:
      print(" xxxxx MOM* direct terminates because heuristics computation failed...")
      output_res = ( 0, [], 0, -1, self.GetRemainTime(), self.max_col_set )
      return dict(), output_res
    self.time_limit = time_limit
    tstart = time.perf_counter()
    self.all_visited_s[self.s_o.id] = self.s_o
    self.f_value[self.s_o.id] = self.s_o.cost_vec + self.weight*self.GetHeuristic(self.s_o)
    self.open_list.add(np.sum(self.f_value[self.s_o.id]), self.s_o.id)
    self.collision_dict[self.s_o.id] = dict()
    self.AddToFrontier(self.s_o)
    search_success = True
    rd = 0
    while(True):
      tnow = time.perf_counter()
      rd = rd + 1
      if (rd > search_limit) or (tnow - tstart > self.GetRemainTime()):
        print(" Fail! timeout! ")
        search_success = False
        break
      if (self.open_list.size()) == 0:
        print(" Done! openlist is empty! ")
        search_success = True
        break
      pop_node = self.open_list.pop()
      curr_s = self.all_visited_s[pop_node[1]]
      self.RefineGoalFrontier(curr_s)
      # filter state
      if self.FilterState(curr_s, self.f_value[curr_s.id]):
        if curr_s.id in self.frontier_map[self.s_f.loc]:
          self.frontier_map[self.s_f.loc].remove(curr_s.id)
        continue
      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors(curr_s, tnow) # neighboring states
      if not ngh_success:
        search_success = False
        break
      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx] 
        self.backtrack_dict[ngh_s.id] = curr_s.id
        h_array = self.GetHeuristic(ngh_s)
        f_array = ngh_s.cost_vec + self.weight*h_array
        if self.FilterState(ngh_s, f_array):
          if ngh_s.id in self.frontier_map[self.s_f.loc]:
            self.frontier_map[self.s_f.loc].remove(ngh_s.id)
          continue
        self.AddToBackSet(curr_s.id,ngh_s.id)
        # compute collision set
        ngh_collision_set = self.CollisionCheck(curr_s,ngh_s)
        if ngh_s.id in self.collision_dict:
          for k in ngh_collision_set: # union with prev collision set.
            self.collision_dict[ngh_s.id][k] = 1
        else:
          self.collision_dict[ngh_s.id] = ngh_collision_set # first time, create a dict
        if len(ngh_collision_set) > 0: 
          self.BackPropagation(curr_s, ngh_collision_set)
          continue # this ngh state is in collision
        if (not self.Pruning(ngh_s, f_array)):
          self.AddToFrontier(ngh_s)
          self.f_value[ngh_s.id] = ngh_s.cost_vec + self.weight*h_array
          self.open_list.add(np.sum(self.f_value[ngh_s.id]), ngh_s.id)
        else: # dominated
          self.DominanceBackprop(ngh_s)
    if True:
      # output jpath is in reverse order, from goal to start
      all_jpath = self.ReconstructPathAll()
      all_cost_vec = dict()
      for k in all_jpath:
        all_cost_vec[k] = self.all_visited_s[k].cost_vec
      output_res = ( int(rd), all_cost_vec, int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()), self.max_col_set )
      print(" MOM* search terminates with ", len(all_jpath), " solutions.")
      return all_jpath, output_res
    else:
      output_res = ( int(rd), dict(), int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()), self.max_col_set )
      return dict(), output_res

def RunMoMstarMAPF(grids, sx, sy, gx, gy, cvecs, cost_grids, cdim, w, eps, search_limit, time_limit):
  """
  sx,sy = starting x,y coordinates of agents.
  gx,gy = goal x,y coordinates of agents.
  cdim = M, cost dimension.
  cvecs = cost vectors of agents, the kth component is a cost vector of length M for agent k.
  cost_grids = a tuple of M matrices, the mth matrix is a scaling matrix for the mth cost dimension.
  cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.
  w is the heuristic inflation rate. E.g. w=1.0, no inflation, w>1.0, use inflation. 
  eps is useless. set eps=0.
  search_limit is the maximum rounds of expansion allowed. set it to np.inf if you don't want to use.
  time_limit is the maximum amount of time allowed for search (in seconds), typically numbers are 60, 300, etc.
  """
  print("...RunMoMstarMAPF... ")
  truncated_cvecs = list()
  truncated_cgrids = list()

  # ensure cost dimension.
  for idx in range(len(cvecs)):
    truncated_cvecs.append(cvecs[idx][0:cdim])
  for idx in range(cdim):
    truncated_cgrids.append(cost_grids[idx])

  mom = MoMstar(grids, sx, sy, gx, gy, truncated_cvecs, truncated_cgrids, w, eps, time_limit, True)
  t_remain = mom.GetRemainTime() # in constructor defined in MoAstarMAPF, compute policy takes time
  return mom.Search(search_limit, t_remain)

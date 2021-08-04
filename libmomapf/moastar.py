"""
Author: Zhongqiang (Richard) Ren
Version@2021
Remark: some of the code is redundant and needs a clean up.
"""
import numpy as np
import heapq as hpq
import itertools as itt
import time
import copy
import sys

import common as cm

"""
This file implements NAMOA* algorithm on four-connected grids.
"""

class MoAstarState:
  def __init__(self, sid=-1, loc=(), cost_vec=np.array(0)):
    self.id = sid
    self.loc = loc # location id
    self.cost_vec = cost_vec # M-dimensional objective space.
  
  def __str__(self):
    return "{"+str(self.id)+","+str(self.loc)+","+str(self.cost_vec)+"}" 

  def ConflictSet(self):
    """
    Should only be called for MAPF.
    """
    out_dict = dict()
    for ix in range(len(self.loc)):
      for iy in range(ix+1, len(self.loc)):
        if self.loc[ix] == self.loc[iy]: # occupy same node
          if ix not in out_dict:
            if iy not in out_dict: # both not in dic
              out_dict[ix] = iy
              out_dict[iy] = iy
            else: # iy in dic, i.e. in some col set
              out_dict[ix] = cm.UFFind(out_dict, iy)
          else:
            if iy not in out_dict:
              out_dict[iy] = cm.UFFind(out_dict, ix)
            else: # both in dict
              cm.UFUnion(out_dict, ix, iy)
    return out_dict

class MoAstarMAPFBase:
  """
  MoAstarMAPF, no heuristic.
  This class is a base class.
  """
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit):
    """
    NAMOA* algorithm.
    cvecs e.g. = [np.array(1,2),np.array(3,4),np.array(1,5)] means 
      robot 1 has cost (1,2) over every edge, 
      robot 2 has cost (3,4) over every edge and 
      robot 3 has cost (1,5) over every edge.
    """
    self.grids = grids
    (self.nyt, self.nxt) = self.grids.shape
    self.cost_vecs = copy.deepcopy(cvecs)
    self.cdim = len(cvecs[0])
    self.num_robots = len(sx)
    self.state_gen_id = 3 # 1 is start, 2 is goal
    self.cost_grids = copy.deepcopy(cost_grids)

    # start state
    self.sx = copy.deepcopy(sx)
    self.sy = copy.deepcopy(sy)
    tmp_sloc = list()
    for idx in range(len(self.sx)):
      tmp_sloc.append( self.sy[idx]*self.nxt + self.sx[idx] )
    self.s_o = MoAstarState(1,tuple(tmp_sloc),np.zeros(self.cdim))

    # goal state
    self.gx = copy.deepcopy(gx)
    self.gy = copy.deepcopy(gy)
    tmp_gloc = list()
    for idx in range(len(self.gx)):
      tmp_gloc.append( self.gy[idx]*self.nxt + self.gx[idx] )
    self.s_f = MoAstarState(2,tuple(tmp_gloc),np.array([np.inf for i in range(self.cdim)]))

    # search params and data structures
    self.weight = w
    self.eps = eps
    self.action_set_x = [0,-1,0,1,0]
    self.action_set_y = [0,0,-1,0,1]
    self.all_visited_s = dict()
    self.frontier_map = dict()
    self.open_list = cm.PrioritySet()
    self.f_value = dict()
    self.close_set = set()
    self.backtrack_dict = dict() # track parents

    self.time_limit = time_limit
    self.remain_time = time_limit # record remaining time for search
    self.heu_failed = False # by default

    return

  def GetRemainTime(self):
    return self.remain_time

  def GetHeuristic(self, s):
    """
    Get estimated M-dimensional cost-to-goal.
    In this base class, there is no heuristic, just zero h-vector...
    """
    if self.num_robots == 1: # if only 1 robot, no heuristic.
      return np.zeros(self.cdim)
    else:
      return np.zeros(self.cdim)

  def GetCost(self, loc, nloc):
    """
    Get M-dimenisonal cost vector of moving from loc to nloc.
    """
    out_cost = np.zeros(self.cdim)
    for idx in range(self.num_robots): # loop over all robots
      if nloc[idx] != loc[idx]: # there is a move, not wait in place.
        if len(self.cost_grids) > 0 and len(self.cost_grids) >= self.cdim :
          cy = int(np.floor(nloc[idx]/self.nxt)) # ref x
          cx = int(nloc[idx]%self.nxt) # ref y, 
          # ! CAVEAT, this loc is ralavant to cost_grid, must be consistent with Policy search.
          for ic in range(self.cdim):
            out_cost[ic] = out_cost[ic] + self.cost_vecs[idx][ic]*self.cost_grids[ic][cy,cx]
        else:
          out_cost = out_cost + self.cost_vecs[idx]
      else: # robot stay in place.
        if len(self.s_f.loc) == 0: # policy mode, no goal specified.
          out_cost = out_cost + self.cost_vecs[idx]*1 # stay-in-place fixed cost
        elif loc[idx] != self.s_f.loc[idx]: # nloc[idx] = loc[idx] != s_f.loc[idx], robot not reach goal.
          out_cost = out_cost + self.cost_vecs[idx]*1 # np.ones((self.cdim)) # stay in place, fixed energy cost for every robot
    return out_cost

  def AddToFrontier(self, s):
    """Add a state into frontier"""
    self.all_visited_s[s.id] = s
    if s.loc not in self.frontier_map:
      self.frontier_map[s.loc] = set()
      self.frontier_map[s.loc].add(s.id)
    else:
      self.RefineFrontier(s)
      self.frontier_map[s.loc].add(s.id)
    return

  def RefineFrontier(self, s):
    """Use s to remove dominated states in frontier set"""
    if s.loc not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[s.loc])
    for sid in temp_frontier:
      if sid == s.id: # do not compare with itself !
        continue
      if cm.DomOrEqual(s.cost_vec, self.all_visited_s[sid].cost_vec):
        self.frontier_map[s.loc].remove(sid)
        self.open_list.remove(sid)
    return

  def CollisionCheck(self,s,ns):
    out_dict = ns.ConflictSet() # conflict in ns
    for idx in range(len(s.loc)):
      for idy in range(idx+1, len(s.loc)):
        if idx == idy:
          continue
        if ns.loc[idy] == s.loc[idx] and s.loc[idy] == ns.loc[idx]:
          # two robots swap location
          if idx not in out_dict:
            if idy not in out_dict: # both not in dic
              out_dict[idx] = idy
              out_dict[idy] = idy
            else: # iy in dic, i.e. in some col set
              out_dict[idx] = cm.UFFind(out_dict, idy)
          else:
            if idy not in out_dict:
              out_dict[idy] = cm.UFFind(out_dict, idx)
            else: # both in dict
              cm.UFUnion(out_dict, idx, idy)
    return out_dict

  def FilterState(self,s,f_array):
    if self.FrontierFilterState(s,f_array):
      return True
    if self.GoalFilterState(s,f_array):
      return True
    return False

  def FrontierFilterState(self,s,f_array):
    """
    filter state s, if s is dominated by any states in frontier other than s.
    """
    if s.loc not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[s.loc]:
      if sid == s.id:
        continue # do not compare with itself...
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def GoalFilterState(self,s,f_array):
    """
    filter state s, if s is dominated by any states that reached goal. (non-negative cost vec).
    """
    if self.s_f.loc not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[self.s_f.loc]:
      if cm.DomOrEqual(self.f_value[sid], f_array):
        return True
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def GetNeighbors(self, s, tstart):
    """
    input a tjs state s, compute its neighboring states.
    output a list of states.
    """
    nid_dict = dict() # key is robot index, value is ngh(location id) for robot idx
    for idx in range(self.num_robots): # loop over all robot

      tnow = time.perf_counter() # check if timeout.
      if (int(tnow - tstart) > self.GetRemainTime()):
        print(" FAIL! timeout in get ngh! " )
        return [], False

      nid_dict[idx] = list() # neighbors for each robot idx

      # its nid_dict[idx] is empty list
      cy = int(np.floor(s.loc[idx]/self.nxt)) # current x
      cx = int(s.loc[idx]%self.nxt) # current y

      # explore all neighbors
      for action_idx in range(len(self.action_set_x)):
        nx = cx+self.action_set_x[action_idx] # next x
        ny = cy+self.action_set_y[action_idx] # next y 
        if not (ny < 0 or ny >= self.nyt or nx < 0 or nx >= self.nxt):
          # do not exceed border
          if self.grids[ny,nx] == 0: # not obstacle
            nid_dict[idx].append( ny*self.nxt+nx )
 
    # generate all joint neighbors s_ngh from nid_dict
    s_ngh = list()
    all_loc = list( itt.product(*(nid_dict[ky] for ky in sorted(nid_dict))) )

    for ida in range(len(all_loc)): # loop over all neighboring joint edges
      tnow = time.perf_counter()
      if (int(tnow - tstart) > self.GetRemainTime()):
        print(" FAIL! timeout in get ngh! " )
        return [], False

      ns = MoAstarState(self.state_gen_id, tuple(all_loc[ida]), s.cost_vec+self.GetCost(s.loc,all_loc[ida]) )
      self.state_gen_id = self.state_gen_id + 1 
      s_ngh.append(ns)
    return s_ngh, True

  def Pruning(self, s, f_array):
    """
    """
    if s.loc not in self.frontier_map:
      return False # this je is never visited before, should not prune
    for fid in self.frontier_map[s.loc]: # loop over all states in frontier set.
      if fid == s.id:
        continue
      if cm.DomOrEqual(self.all_visited_s[fid].cost_vec, s.cost_vec):
        return True # should be pruned
    # end of for
    return False # should not be pruned

  def ReconstructPath(self, sid):
    """
    input state is the one that reached, 
    return a list of joint vertices in right order.
    """
    jpath = []
    while sid in self.backtrack_dict:
      jpath.append(self.all_visited_s[sid].loc)
      sid = self.backtrack_dict[sid] 
    jpath.append(self.all_visited_s[sid].loc)
    # reverse output path here.
    out = []
    for idx in range(len(jpath)):
      out.append(jpath[len(jpath)-1-idx])
    return out

  def ReconstructPathAll(self):
    jpath_all = dict()
    if self.s_f.loc not in self.frontier_map:
      return jpath_all # no solution found
    for gid in self.frontier_map[self.s_f.loc]:
      jpath_all[int(gid)] = self.ReconstructPath(gid)
    return jpath_all

  def Search(self, search_limit=np.inf):
    """
    """
    # print(" NAMOA* Search begin ")
    if self.heu_failed :
      print("[CAVEAT] MOA* direct terminates because heuristics computation failed...")
      output_res = ( 0, [], 0, -1, self.GetRemainTime() )
      return dict(), output_res
    # self.time_limit = time_limit
    tstart = time.perf_counter()
    self.all_visited_s[self.s_o.id] = self.s_o
    self.f_value[self.s_o.id] = self.s_o.cost_vec + self.GetHeuristic(self.s_o)
    self.open_list.add(np.sum(self.f_value[self.s_o.id]), self.s_o.id)
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
      pop_node = self.open_list.pop() # ( sum(f), sid )
      curr_s = self.all_visited_s[pop_node[1]]
      # filter state
      if self.FilterState(curr_s, self.f_value[curr_s.id]):
        continue
      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors(curr_s, tnow) # neighboring states
      if not ngh_success:
        search_success = False
        break

      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx]
        if len(self.CollisionCheck(curr_s,ngh_s)):
          continue # discard state that are in conflict.
        # if reach here, joint state is collision free
        h_array = self.GetHeuristic(ngh_s)
        f_array = ngh_s.cost_vec + self.weight*h_array
        if self.FilterState(ngh_s, f_array):
          continue
        if (not self.Pruning(ngh_s, f_array)):
          self.AddToFrontier(ngh_s)
          self.backtrack_dict[ngh_s.id] = curr_s.id
          self.f_value[ngh_s.id] = ngh_s.cost_vec + self.weight*h_array
          self.open_list.add(np.sum(self.f_value[ngh_s.id]), ngh_s.id)

    if True:
      # output jpath is in reverse order, from goal to start
      all_jpath = self.ReconstructPathAll()
      all_cost_vec = dict()
      for k in all_jpath:
        all_cost_vec[k] = self.all_visited_s[k].cost_vec
      output_res = ( int(rd), all_cost_vec, int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()) )
      return all_jpath, output_res
    else:
      output_res = ( int(rd), dict(), int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()) )
      return dict(), output_res

class MoMapfPolicy(MoAstarMAPFBase):
  """
  MomapfPolicy is derived from MoAstarMAPFBase and computes 
   individual policies for a SINGLE robot.
  """
  def __init__(self, grids, sx, sy, cvecs, cost_grids, time_limit):
    super(MoMapfPolicy, self).__init__(grids, sx, sy, [], [], cvecs, cost_grids, 1.0, 0.0, time_limit)
    
  def FilterState(self,s):
    """
    Only compare states with frontier set at its joint vertex.
    """
    if s.loc not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[s.loc]:
      if sid == s.id:
        continue
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def GetCost(self, loc, nloc):
    """
    Get M-dimenisonal cost vector of moving from nid to nid2.
    """
    out_cost = np.zeros(self.cdim)
    for idx in range(self.num_robots): # loop over all robots
      if nloc[idx] != loc[idx]: # there is a move, not wait in place.
        if len(self.cost_grids) > 0 and len(self.cost_grids) >= self.cdim :
          cy = int(np.floor(loc[idx]/self.nxt)) # ref x
          cx = int(loc[idx]%self.nxt) # ref y, 
          for ic in range(self.cdim):
            out_cost[ic] = out_cost[ic] + self.cost_vecs[idx][ic]*self.cost_grids[ic][cy,cx]
        else:
          out_cost = out_cost + self.cost_vecs[idx]
      else: # robot stay in place.
        if len(self.s_f.loc) == 0:
          out_cost = out_cost + self.cost_vecs[idx]*1 # stay-in-place fixed cost
        elif loc[idx] != self.s_f.loc[idx]:
          out_cost = out_cost + self.cost_vecs[idx]*1 

    return out_cost

  def Pruning(self, s):
    """
    before sync
    """
    if s.loc not in self.frontier_map:
      return False # this je is never visited before, should not prune
    for fid in self.frontier_map[s.loc]: # loop over all states in frontier set.
      if fid == s.id:
        continue
      if cm.DomOrEqual(self.all_visited_s[fid].cost_vec, s.cost_vec):
        return True # should be pruned
    # end of for
    return False # should not be pruned

  def ReconstructPathAll(self):
    return dict()

  def Search(self, search_limit, time_limit):
    """
    """
    self.time_limit = time_limit
    tstart = time.perf_counter()

    self.all_visited_s[self.s_o.id] = self.s_o
    self.f_value[self.s_o.id] = self.s_o.cost_vec + self.GetHeuristic(self.s_o)
    self.open_list.add(np.sum(self.f_value[self.s_o.id]), self.s_o.id)
    self.AddToFrontier(self.s_o)
  
    search_success = True
    rd = 0
    while(True):
      tnow = time.perf_counter()
      rd = rd + 1
      if (rd > search_limit) or (tnow - tstart > self.time_limit):
        print(" Fail! timeout! ")
        search_success = False
        break
      if (self.open_list.size()) == 0:
        print(" Done! openlist is empty! ")
        search_success = True
        break
      pop_node = self.open_list.pop() # ( sum(f), sid )
      curr_s = self.all_visited_s[pop_node[1]]

      # filter state
      if self.FilterState(curr_s): # guarantee that, the state being expanded next has a non-dominated cost vector.
        continue

      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors(curr_s, tnow) # neighboring states

      if not ngh_success:
        search_success = False
        break

      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx] 
        if self.FilterState(ngh_s):
          continue
        if (not self.Pruning(ngh_s)):
          self.AddToFrontier(ngh_s)
          self.backtrack_dict[ngh_s.id] = curr_s.id
          self.f_value[ngh_s.id] = ngh_s.cost_vec
          # print("f_vec:",self.f_value[ngh_s.id], ", sum:", np.sum(self.f_value[ngh_s.id]))
          self.open_list.add(np.sum(self.f_value[ngh_s.id]), ngh_s.id)

    if True:
      # output jpath is in reverse order, from goal to start
      all_jpath = self.ReconstructPathAll()
      all_cost_vec = dict()
      for k in all_jpath:
        all_cost_vec[k] = self.all_visited_s[k].cost_vec
      output_res = ( int(rd), all_cost_vec, int(search_success), float(time.perf_counter()-tstart) )
      return all_jpath, output_res
    else:
      output_res = ( int(rd), dict(), int(search_success), float(time.perf_counter()-tstart) )
      return dict(), output_res

  def BuildPolicy(self):
    # policy: map a vertex to a set of next states
    policy = [[ list() for x in range(self.nxt)] for y in range(self.nyt)] 

    # distmat: map a vertex to a set of cost vectors, which are the pareto distance to goal.
    distmat = [[ list() for x in range(self.nxt)] for y in range(self.nyt)] 
    for iy in range(self.nyt):
      for ix in range(self.nxt):
        nid = iy*self.nxt+ix
        help_dict = dict() # help avoid duplicated actions in policy, re-init for every ix,iy loop.
        if tuple([nid]) not in self.frontier_map:
          continue
        for sid in self.frontier_map[tuple([nid])]: # loop over frontier set at location nid.
          if sid not in self.backtrack_dict:
            continue
          pid = self.backtrack_dict[sid]
          nid = self.all_visited_s[pid].loc[0]
          cy = int(np.floor(nid/self.nxt)) # next x 
          cx = int(nid%self.nxt) # next y
          if (cx,cy) not in help_dict:
            policy[iy][ix].append((cx,cy)) # follow the convention in GridPolicy() in common.py 
            distmat[iy][ix].append(self.all_visited_s[sid].cost_vec)
            help_dict[(cx,cy)] = len(policy[iy][ix]) - 1 # last action in dic
          else:
            help_idx = help_dict[(cx,cy)] # locate that action index in policy[iy][ix] (which is a list).
            for ic in range(self.cdim): # take element-wise minimum.
              if self.all_visited_s[sid].cost_vec[ic] < distmat[iy][ix][help_idx][ic]:
                distmat[iy][ix][help_idx][ic] = self.all_visited_s[sid].cost_vec[ic]
          # end of else
        # end for sid
      # end for ix
    # end for iy
    policy[self.sy[0]][self.sx[0]].append( (self.sx[0],self.sy[0]) ) # at goal location
    return policy, distmat

def GetMoMapfPolicy(grids, gx, gy, cvec, cost_grids, time_limit):
  """
  This function wrap up class MoMapfPolicy into a function to make it easier to call.
  """
  momapf = MoMapfPolicy(grids, [gx], [gy], [cvec], cost_grids, time_limit)
  jpath, search_res = momapf.Search(999999999, time_limit)
  pol,dm = momapf.BuildPolicy()
  return pol, dm, search_res

class MoAstarMAPF(MoAstarMAPFBase):
  """
  MoAstarMAPF, derived from MoAstarMAPFBase, heuristic used.
  This is the NAMOA* algorithm used for comparison in MOMAPF problem.
  """
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol):
    super(MoAstarMAPF, self).__init__(grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit)
    self.time_limit = time_limit
    self.remain_time = time_limit # record remaining time for search
    # optimal policy
    self.optm_policis = dict()
    self.optm_distmats = dict()
    self.heu_failed = False
    if compute_pol:
      for ix in range(len(self.sx)):
        self.optm_policis[ix], self.optm_distmats[ix], search_res = \
          GetMoMapfPolicy(self.grids, gx[ix], gy[ix], cvecs[ix], cost_grids, self.remain_time)
        self.remain_time = self.remain_time - search_res[-1]
        if search_res[2] == 0:
          # policy build failed....
          self.heu_failed = True
          break
        if self.remain_time <= 0:
          self.heu_failed = True
          break
  
  def GetHeuristic(self, s):
    """
    Get estimated M-dimensional cost-to-goal, 
    override the func in MoAstarMAPFBase class.
    """
    if len(self.sx)==1:
      return np.zeros(1)
    h_vec = np.zeros(self.cdim)
    for ri in range(self.num_robots):
      nid = s.loc[ri]
      cy = int(np.floor(nid/self.nxt)) # current x
      cx = int(nid%self.nxt) # current y
      dist_vec_list = self.optm_distmats[ri][cy][cx]
      if len(dist_vec_list) > 0:
        h_vec = h_vec + np.min(np.stack(dist_vec_list),axis=0)
    return h_vec

def RunMoAstarMAPF(grids, sx, sy, gx, gy, cvecs, cost_grids, cdim, w, eps, search_limit, time_limit):
  """
  sx,sy = starting x,y coordinates of agents.
  gx,gy = goal x,y coordinates of agents.
  cdim = M, cost dimension.
  cvecs = cost vectors of agents, the kth component is a cost vector of length M for agent k.
  cost_grids = a tuple of M matrices, the mth matrix is a scaling matrix for the mth cost dimension.
  cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.
  w, eps are not in use! Make w=1, eps=0.
  search_limit is the maximum rounds of expansion allowed. set it to np.inf if you don't want to use.
  time_limit is the maximum amount of time allowed for search (in seconds), typically numbers are 60, 300, etc.
  """
  print("...Run NAMOA* for MOMAPF... ")
  truncated_cvecs = list()
  truncated_cgrids = list()
  for idx in range(len(cvecs)):
    # print(cvecs[idx][0:cdim])
    truncated_cvecs.append(cvecs[idx][0:cdim])
  for idx in range(cdim):
    truncated_cgrids.append(cost_grids[idx])
  moa = MoAstarMAPF(grids, sx, sy, gx, gy, truncated_cvecs, truncated_cgrids, w, eps, time_limit, True)
  return moa.Search(search_limit)

def RunMoAstarSingleAgent(grids, sx, sy, gx, gy, cvecs, cost_grids, cdim, w, eps, search_limit, time_limit):
  """
  Wrap up MoAstarMAPFBase class as a function to compute for single agent multi-objective graph search.
  This is essentially NAMOA* algorithm. See RunMoAstarMAPF for info about input args.
  """
  truncated_cvecs = list()
  truncated_cgrids = list()
  truncated_cvecs.append(cvecs[0:cdim])
  for idx in range(cdim):
    truncated_cgrids.append(cost_grids[idx])
  moa = MoAstarMAPFBase(grids, [sx], [sy], [gx], [gy], truncated_cvecs, truncated_cgrids, w, eps, time_limit)
  return moa.Search(search_limit)


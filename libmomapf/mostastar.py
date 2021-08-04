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
import sys

"""
This file is about running MOA* in a time-augmented graph. (let's call it MOSTA* for now.)
"""

DEBUG_MOSTASTAR=False

TIME_MAX=65536

class MoSTAstarState:
  """
  Search state for MOSTA*
  """
  def __init__(self, sid, loc, cost_vec, t):
    """
    State for single agent MOA* search support ST constraints.
    """
    self.id = sid
    self.loc = loc # location id
    self.cost_vec = cost_vec # M-dimensional objective space.
    self.t = t #
  
  def __str__(self):
    return "{"+str(self.id)+","+str(self.loc)+",cvec"+str(self.cost_vec)+",t("+str(self.t)+")}" 

  def Equal(self, other):
    return cm.Equal(self.cost_vec, other.cost_vec)

class MoSTAstar:
  """
  NAMOA* search in time-augmented graph.
  """
  def __init__(self, grids, sx, sy, gx, gy, cvec, cost_grids, w=1.0, eps=0.0, action_set_x = [-1,0,1,0,0], action_set_y = [0,-1,0,1,0]):
    """
    Multi-Objective Astar algorithm.
    cvecs e.g. = [np.array(1,2),np.array(3,4),np.array(1,5)] means 
      robot 1 has cost (1,2) over every edge, 
      robot 2 has cost (3,4) over every edge and 
      robot 3 has cost (1,5) over every edge.
    """
    self.grids = grids
    (self.nyt, self.nxt) = self.grids.shape
    self.cost_vec = cvec
    self.cdim = len(cvec)
    self.state_gen_id = 3 # 1 is start, 2 is goal
    self.cost_grids = cost_grids
    # start state
    self.sx = sx
    self.sy = sy
    self.s_o = MoSTAstarState(1,self.sy*self.nxt+self.sx,np.zeros(self.cdim),0)
    # goal state
    self.gx = gx
    self.gy = gy
    self.s_f = MoSTAstarState(2,self.gy*self.nxt+self.gx,np.array([TIME_MAX for i in range(self.cdim)]),-1)
    # search params and data structures
    self.weight = w
    self.eps = eps
    self.action_set_x = action_set_x
    self.action_set_y = action_set_y
    self.all_visited_s = dict() 
    self.frontier_map = dict() 
    self.open_list = cm.PrioritySet()
    self.f_value = dict() # map a state id to its f-value vector (np.array)
    self.close_set = set()
    self.backtrack_dict = dict() # track parents
    self.reached_goal_states = set()
    self.time_limit = 30 # seconds, default
    self.node_constr = dict()
      # a dict that maps a vertex id to a list of forbidden timestamps
    self.swap_constr = dict()
      # a dict that maps a vertex id to a dict with (forbidden time, set(forbidden next-vertex)) as key-value pair.

    return

  def AddNodeConstr(self, nid, t):
    """
    robot is forbidden from entering nid at time t.
    This is a naive implementation using list.
    """
    # a new node id
    t = int(t) # make sure int!
    if nid not in self.node_constr:
      self.node_constr[nid] = list()
      self.node_constr[nid].append(t)
      return
    # locate the index for t
    idx = 0
    while idx < len(self.node_constr[nid]):
      if t <= self.node_constr[nid][idx]:
        break
      idx = idx + 1
    if idx == len(self.node_constr[nid]):
      self.node_constr[nid].append(t)
      return
    # avoid duplication
    if t == self.node_constr[nid][idx]: 
      return
    # update list
    tlist = list()
    for idy in range(len(self.node_constr[nid])):
      if idy == idx:
        tlist.append(t)
      tlist.append(self.node_constr[nid][idy])
    self.node_constr[nid] = tlist
    return

  def AddSwapConstr(self, nid1, nid2, t):
    """
    robot is forbidden from transfering from (nid1,t) to (nid2,t+1).
    """
    if nid1 not in self.swap_constr:
      self.swap_constr[nid1] = dict()
    if t not in self.swap_constr[nid1]:
      self.swap_constr[nid1][t] = set()
    self.swap_constr[nid1][t].add(nid2)
    return

  def GetHeuristic(self, s):
    """
    Get estimated M-dimensional cost-to-goal.
    Manhattan heuristic vector or just no heuristic.
    """
    ### Manhattan heuristic vector
    ### whether this is an underestimate depends on the cost vector of edges.
    # cy = int(np.floor(s.loc/self.nxt)) 
    # cx = int(s.loc%self.nxt) 
    # return ( abs(cy-self.gy) + abs(cx - self.gx) ) * np.ones(self.cdim)
    
    # no heuristic
    return np.zeros(self.cdim)

  def GetCost(self, loc, nloc, dt=1):
    """
    Get M-dimenisonal cost vector of moving from loc to nloc.
    """
    out_cost = np.zeros(self.cdim)
    if nloc != loc: # there is a move, not wait in place.
      if len(self.cost_grids) > 0 and len(self.cost_grids) >= self.cdim : # there is cost_grid, use it.
        cy = int(np.floor(nloc/self.nxt)) # ref y
        cx = int(nloc%self.nxt) # ref x, 
        for ic in range(self.cdim):
          out_cost[ic] = out_cost[ic] + self.cost_vec[ic]*self.cost_grids[ic][cy,cx]
      else: # there is no cost_grid, no use
        out_cost = out_cost + self.cost_vec
    else: # robot stay in place.
      if loc != self.s_f.loc:
        out_cost = out_cost + self.cost_vec
    out_cost = out_cost + (dt-1)*self.cost_vec
    return out_cost

  def GetStateIdentifier(self, s):
    """
    return an identifier of state, such as a tuple of (nid, t), which characterize frontiers.
    """
    return (s.loc, s.t)

  def CheckReachGoal(self, s):
    """
    verify if s is a state that reaches goal and robot can stay there forever !!
    """
    if (s.loc != self.s_f.loc):
      return False
    if s.loc not in self.node_constr:
      return True
    if s.t > self.node_constr[s.loc][-1]:
      return True
    return False

  def AddToFrontier(self, s):
    """Add a state into frontier"""
    self.all_visited_s[s.id] = s
    cfg_t = self.GetStateIdentifier(s)
    if (cfg_t == self.GetStateIdentifier(self.s_f)): 
      cfg_t = self.GetStateIdentifier(self.s_f) 
    if cfg_t not in self.frontier_map:
      self.frontier_map[cfg_t] = set()
      self.frontier_map[cfg_t].add(s.id)
    else:
      # not first time visit of a jc
      self.RefineFrontier(s)
      self.frontier_map[cfg_t].add(s.id)
    return

  def RefineFrontier(self, s):
    """Use s to remove dominated states in frontier set"""
    cfg_t = self.GetStateIdentifier(s)
    if cfg_t not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[cfg_t])
    for sid in temp_frontier:
      if sid == s.id :
        continue
      if cm.DomOrEqual(s.cost_vec, self.all_visited_s[sid].cost_vec):
        self.frontier_map[cfg_t].remove(sid)
        self.open_list.remove(sid)
    return

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
    cfg_t = self.GetStateIdentifier(s)
    if cfg_t not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[cfg_t]:
      if sid == s.id:
        continue # do not compare with itself...
      if cm.DomOrEqual(self.f_value[sid], f_array):
        return True
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def GoalFilterState(self,s,f_array):
    """
    filter state s, if s is dominated by any states that reached goal. (non-negative cost vec).
    """
    for sid in self.reached_goal_states:
      if cm.DomOrEqual(self.f_value[sid], f_array):
        return True
      if cm.DomOrEqual(self.all_visited_s[sid].cost_vec, s.cost_vec):
        return True # filtered
    return False # not filtered

  def RefineGoalReached(self,s):
    """
    """
    temp_set = copy.deepcopy(self.reached_goal_states)
    for sid in self.reached_goal_states:
      if sid == s.id:
        continue
      if cm.DomOrEqual(s.cost_vec, self.all_visited_s[sid].cost_vec):
        temp_set.remove(sid)
    self.reached_goal_states = temp_set
    return 

  def IfNodeOccupied(self, loc, t):
    if loc not in self.node_constr:
      return False
    for tt in self.node_constr[loc]:
      if tt == t:
        return True
      else:
        if tt > t:
          return False
    return False

  def GetNeighbors(self, s, tstart):
    """
    input a tjs state s, compute its neighboring states.
    output a list of states.
    """
    s_ngh = list()
    cy = int(np.floor(s.loc/self.nxt)) # current x
    cx = int(s.loc%self.nxt) # current y
    # loop over all four possible actions
    for action_idx in range(len(self.action_set_x)):
      nx = cx+self.action_set_x[action_idx] # next x
      ny = cy+self.action_set_y[action_idx] # next y 
      nnid = ny*self.nxt+nx
      if (nx >= self.nxt) or (nx < 0) or (ny >= self.nyt) or (ny < 0): # out of border of grid
        continue
      if (self.grids[ny,nx] > 0): # static obstacle
        continue
      # check swap conflict
      if (s.loc in self.swap_constr) and (s.t in self.swap_constr[s.loc]) \
          and (nnid in self.swap_constr[s.loc][s.t]):
        continue # robot is not allowed to transite to nnid at this time 
                 # due to swap constraints!
      if self.IfNodeOccupied(s.loc, s.t): # node constraint violated !! continue !!
        continue
      ns = MoSTAstarState(self.state_gen_id, nnid, s.cost_vec+self.GetCost(s.loc,nnid,1), s.t+1)
      self.state_gen_id = self.state_gen_id + 1 
      s_ngh.append(ns)

    return s_ngh, True

  def Pruning(self, s, f_array):
    """
    """
    cfg_t = self.GetStateIdentifier(s)
    if cfg_t not in self.frontier_map:
      return False 
    for fid in self.frontier_map[cfg_t]:
      if cm.DomOrEqual(self.all_visited_s[fid].cost_vec, s.cost_vec):
        return True # should be pruned
    # end of for
    return False # should not be pruned

  def ReconstructPath(self, sid):
    """
    input state is the one that reached, 
    return a list of joint vertices in right order.
    """
    jpath = [] # in reverse order
    tt = [] # in reverse order
    while sid in self.backtrack_dict:
      jpath.append(self.all_visited_s[sid].loc)
      tt.append(self.all_visited_s[sid].t)
      sid = self.backtrack_dict[sid] 
    jpath.append(self.all_visited_s[sid].loc)
    tt.append(self.all_visited_s[sid].t)
    # reverse output path here.
    nodes = []
    times = []
    for idx in range(len(jpath)):
      nodes.append(jpath[len(jpath)-1-idx])
      times.append(tt[len(jpath)-1-idx])
    return nodes, times

  def ReconstructPathAll(self):
    traj_all = dict()
    for sid in self.reached_goal_states:
      # print("))))) reconstruct:", sid)
      traj_all[int(sid)] = self.ReconstructPath(sid)
    return traj_all
  
  def InitSearch(self):
    """
    move part of the code to this func, to make it easy to be derived from.
    """
    self.s_o.t = 0
    self.all_visited_s[self.s_o.id] = self.s_o
    self.f_value[self.s_o.id] = self.s_o.cost_vec + self.GetHeuristic(self.s_o)
    self.open_list.add(np.sum(self.f_value[self.s_o.id]), self.s_o.id)
    self.AddToFrontier(self.s_o)
    return
    
  def Search(self, search_limit=100000, time_limit=10):
    if DEBUG_MOSTASTAR:
      print(" MOSTA* Search begin ")
    self.time_limit = time_limit
    tstart = time.perf_counter()
    self.InitSearch()
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
        search_success = True
        break
      pop_node = self.open_list.pop() # ( sum(f), sid )
      curr_s = self.all_visited_s[pop_node[1]]
      # filter state
      if self.FilterState(curr_s, self.f_value[curr_s.id]):
        continue
      if self.CheckReachGoal(curr_s): 
        self.reached_goal_states.add(curr_s.id)
        self.RefineGoalReached(curr_s)
      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors(curr_s, tnow)
      if not ngh_success:
        search_success = False
        break
      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx]
        h_array = self.GetHeuristic(ngh_s)
        f_array = ngh_s.cost_vec + self.weight*h_array
        if self.FilterState(ngh_s, f_array):
          continue
        if (not self.Pruning(ngh_s, f_array)):
          self.AddToFrontier(ngh_s)
          self.backtrack_dict[ngh_s.id] = curr_s.id
          self.f_value[ngh_s.id] = ngh_s.cost_vec + self.weight*h_array
          self.open_list.add(np.sum(self.f_value[ngh_s.id]), ngh_s.id)
        else:
          if DEBUG_MOSTASTAR:
            print(" XX dom pruned XX ")
    if search_success:
      # output jpath is in reverse order, from goal to start
      all_path = self.ReconstructPathAll()
      all_cost_vec = dict()
      for k in all_path:
        all_cost_vec[k] = self.all_visited_s[k].cost_vec
      if len(all_path) == 0: # @2021-04-30, this can leads to error in the derived class like MO-SIPP, MO-SIPP-landmark
        search_success = 0
      output_res = ( int(rd), all_cost_vec, int(search_success), float(time.perf_counter()-tstart) )
      return all_path, output_res
    else:
      output_res = ( int(rd), dict(), int(search_success), float(time.perf_counter()-tstart) )
      return dict(), output_res

def RunMoSTAstar(grids, sx, sy, gx, gy, cvec, cost_grids, cdim, w, eps, search_limit, time_limit, use_same_cost_grid=False, node_cstrs=[], swap_cstrs=[]):
  if DEBUG_MOSTASTAR:
    print("...RunMoSTAstar... ")
    print("sx:",sx," sy:",sy, " gx:",gx, " gy:",gy)
    print("node_cstrs:", node_cstrs, " swap_cstrs:", swap_cstrs)
  truncated_cvec = cvec[0:cdim]
  truncated_cgrids = list()
  for idx in range(cdim):
    if use_same_cost_grid:
      truncated_cgrids.append(cost_grids[0])
    else:
      truncated_cgrids.append(cost_grids[idx])
  mosta = MoSTAstar(grids, sx,sy,gx,gy, truncated_cvec, truncated_cgrids, w, eps)
  for node_cstr in node_cstrs:
    mosta.AddNodeConstr(node_cstr[0], node_cstr[1])
  for swap_cstr in swap_cstrs:
    mosta.AddSwapConstr(swap_cstr[0], swap_cstr[1], swap_cstr[2])
  return mosta.Search(search_limit, time_limit)

"""
Author: Zhongqiang (Richard) Ren
Version@2021
Remark: some of the code is redundant and needs a clean up.
"""
import numpy as np
import copy
import time
import sys

import common as cm
import moastar
import mostastar as mosta
import itertools as itt

######
MOCBS_INIT_SIZE_LIMIT = 800*1000
OPEN_ADD_MODE = 2
######

def EnforceUnitTimePath(lv,lt):
  """
  Given a path (without the final node with infinite timestamp), 
   insert missing (v,t) to ensure every pair of adjacent (v,t) 
   has time difference of one.
  """
  dt = 1
  nlv = list()
  nlt = list()
  for ix in range(len(lt)-1):
    nlv.append(lv[ix])
    nlt.append(lt[ix])
    if lt[ix+1]-lt[ix] > 1.001:
      ct = lt[ix]
      while lt[ix+1] - ct > 1.001:
        nlv.append(lv[ix])
        nlt.append(ct+1)
        ct = ct + 1
  # end for
  nlv.append(lv[-1])
  nlt.append(lt[-1])
  return nlv, nlt
  
class MocbsConstraint:
  """
  MocbsConstraint
  """
  def __init__(self, i, va,vb, ta,tb, j=-1, flag=-1):
    """
    create a constraint, if a single point, then va=vb
    """
    self.i = i # i<0, iff not valid
    self.va = va
    self.vb = vb
    self.ta = ta
    self.tb = tb
    self.j = j # undefined by default, this is used for MA-CBS
    self.flag = flag # flag = 1, vertex conflict, flag = 2 swap conflict

  def __str__(self):
    return "{i:"+str(self.i)+",va:"+str(self.va)+",vb:"+str(self.vb)+\
      ",ta:"+str(self.ta)+",tb:"+str(self.tb)+",j:"+str(self.j)+",flag:"+str(self.flag)+"}"

class MocbsSol:
  """
  The solution in CBS high level node. A dict of paths for all robots.
  """
  def __init__(self):
    self.paths = dict()
    return

  def __str__(self):
    return str(self.paths)

  def AddPath(self, i, lv, lt):
    """
    lv is a list of loc id
    lt is a list of time (adjacent element increases with a step size of one)
    """
    # add a final infinity interval
    nlv,nlt = EnforceUnitTimePath(lv,lt)
    nlv.append(nlv[-1])
    nlt.append(np.inf)
    self.paths[i] = [nlv,nlt]
    return 

  def DelPath(self, i):
    self.paths.pop(i)
    return

  def GetPath(self, i):
    return self.paths[i]

  def CheckConflict(self, i,j):
    """
    """
    ix = 0
    while ix < len(self.paths[i][1])-1:
      for jx in range(len(self.paths[j][1])-1):
        jtb = self.paths[j][1][jx+1]
        jta = self.paths[j][1][jx]
        itb = self.paths[i][1][ix+1]
        ita = self.paths[i][1][ix]
        iva = self.paths[i][0][ix] 
        ivb = self.paths[i][0][ix+1]
        jva = self.paths[j][0][jx]
        jvb = self.paths[j][0][jx+1]
        overlaps, t_lb, t_ub = cm.ItvOverlap(ita,itb,jta,jtb)
        if not overlaps:
          continue
        if ivb == jvb: # vertex conflict at ivb (=jvb)
          return [MocbsConstraint(i, ivb, ivb, t_lb+1, t_lb+1, j, 1), MocbsConstraint(j, jvb, jvb, t_lb+1, t_lb+1, i, 1)] # t_ub might be inf?
          # use min(itb,jtb) to avoid infinity
        if (ivb == jva) and (iva == jvb): # swap location
          return [MocbsConstraint(i, iva, ivb, t_lb, t_lb+1, j, 2), MocbsConstraint(j, jva, jvb, t_lb, t_lb+1, i, 2)]
      ix = ix + 1
    return []

class MocbsNode:
  """
  High level search tree node
  """
  def __init__(self, id0, cvec, sol=MocbsSol(), cstr=MocbsConstraint(-1,-1,-1,-1,-1,-1), parent=-1):
    """
    id = id of this high level CT node
    sol = an object of type CCbsSol.
    cstr = a list of constraints, either empty or of length 2.
      newly added constraint in this node, to get all constraints, 
      need to backtrack from this node down to the root node.
    parent = id of the parent node of this node.
    """
    self.id = id0
    self.sol = sol
    self.cstr = cstr
    self.cvec = cvec
    self.parent = -1 
    self.root = -1 # to which tree it belongs
    return

  def __str__(self):
    str1 = "{id:"+str(self.id)+",cvec:"+str(self.cvec)+",par:"+str(self.parent)
    return str1+",cstr:"+str(self.cstr)+",sol:"+str(self.sol)+"}"

  def CheckConflict(self):
    """
    check for conflicts along paths of all pairs of robots.
    record the first one conflict.
    Notice that one conflict should be splitted to 2 constraints.
    """
    done_set = set()
    for k1 in self.sol.paths:
      for k2 in self.sol.paths:
        if k2 in done_set or k2 == k1:
          continue
        # check for collision
        res = self.sol.CheckConflict(k1,k2)
        if len(res) > 0:
          return res
      # end for k2
      done_set.add(k1) # auxiliary
    return [] # no conflict

class MocbsSearch:
  """
  """
  def __init__(self, grids, sx_list, sy_list, gx_list, gy_list, cvecs, cgrids, expansion_mode, time_limit):
    """
    arg grids is a 2d static grid.
    """
    self.grids = copy.deepcopy(grids)
    (self.yd, self.xd) = self.grids.shape
    self.sx_list = copy.deepcopy(sx_list)
    self.sy_list = copy.deepcopy(sy_list)
    self.gx_list = copy.deepcopy(gx_list)
    self.gy_list = copy.deepcopy(gy_list)
    self.num_robots = len(sx_list)
    self.cdim = len(cvecs[0])
    self.cvecs = copy.deepcopy(cvecs) # for multi-dimensional cost.
    self.cgrids = copy.deepcopy(cgrids) # for multi-dimensional cost.
    self.expansion_mode = expansion_mode
    self.time_limit = time_limit
    self.nodes = dict() # high level nodes
    self.open_list = cm.PrioritySet()
    self.open_by_tree = dict()
    self.closed_set = set()
    self.num_closed_low_level_states = 0
    self.num_low_level_calls = 0
    self.total_low_level_time = 0
    self.node_id_gen = 1
    self.num_roots = 0
    self.curr_root = -1
    self.root_generated = 0
    self.nondom_goal_nodes = set() # a set of HL nodes that reaches goal with non-dom cost vec.
    return

  def BacktrackCstrs(self, nid):
    """
    given a node, trace back to the root, find all constraints relavant.
    """
    node_cs = list()
    swap_cs = list()
    cid = nid
    ri = self.nodes[nid].cstr.i
    while cid != -1:
      if self.nodes[cid].cstr.i == ri: # not a valid constraint
        # init call of mocbs will not enter this.
        cstr = self.nodes[cid].cstr
        if self.nodes[cid].cstr.flag == 1: # vertex constraint
          node_cs.append( (cstr.vb, cstr.tb) )
        elif self.nodes[cid].cstr.flag == 2: # edge constraint
          swap_cs.append( (cstr.va, cstr.vb, cstr.ta) )
          node_cs.append( (cstr.va, cstr.tb) ) # since another robot is coming to v=va at t=tb
      cid = self.nodes[cid].parent
    return node_cs, swap_cs

  def LsearchPlanner(self, ri, node_cs, swap_cs):
    """
    """
    path_dict, lsearch_stats = mosta.RunMoSTAstar(self.grids, self.sx_list[ri], self.sy_list[ri], self.gx_list[ri], self.gy_list[ri], \
      self.cvecs[ri], self.cgrids, self.cdim, 1.0, 0.0, np.inf, self.time_limit-(time.perf_counter()-self.tstart), False, node_cs, swap_cs)
    return path_dict, lsearch_stats

  def Lsearch(self, nid):
    """
    low level search
    """
    nd = self.nodes[nid]
    ri = nd.cstr.i
    node_cs, swap_cs = self.BacktrackCstrs(nid)

    # call constrained NAMOA*
    path_dict, lsearch_stats = self.LsearchPlanner(ri, node_cs, swap_cs) # replace the following two lines of code.
    
    ct = 0 # count of node generated
    path_list = self.PathDictLexSort(path_dict, lsearch_stats[1]) # enforce order
    for k in range(len(path_list)): # loop over all individual Pareto paths 
      new_nd = copy.deepcopy(self.nodes[nid]) # generate new node
      new_nd.sol.DelPath(ri)
      new_nd.sol.AddPath(ri, path_list[k][0], path_list[k][1]) # inf end is add here.
      new_nd.cvec = self.ComputeNodeCostObject(new_nd)

      if self.GoalFilterNodeObject(new_nd):
        continue # skip this dominated node

      # a non-dom node, add to OPEN
      new_id = new_nd.id # the first node, self.nodes[nid] is ok
      if ct > 0: # generate a new node, assign a new id
        new_id = self.node_id_gen
        self.node_id_gen = self.node_id_gen + 1
        new_nd.id = new_id
      self.nodes[new_id] = new_nd # add to self.nodes

      ### ADD OPEN BEGIN
      if OPEN_ADD_MODE == 1:
        self.open_list.add(np.sum(new_nd.cvec), new_nd.id) # add to OPEN
        self.open_by_tree[new_nd.root].add(np.sum(new_nd.cvec), new_nd.id) # add to OPEN in the search tree it belongs to
      elif OPEN_ADD_MODE == 2:
        self.open_list.add(tuple(new_nd.cvec), new_nd.id) # add to OPEN
        self.open_by_tree[new_nd.root].add(tuple(new_nd.cvec), new_nd.id) # add to OPEN in the search tree it belongs to
      ### ADD OPEN END

      ct = ct + 1 # count increase
    return lsearch_stats

  def ComputeNodeCostObject(self, nd):
    """
    Given a high level search node, compute the cost of paths in that node.
    """
    out_cost = np.zeros(self.cdim) # init M-dim cost vector
    for k in nd.sol.paths: # loop over each individual paths in the solution.
      last_idx = -2
      for idx in range(len(nd.sol.paths[k][0])): # find last loc_id that reach goal (remember, last element in lt is timestamp inf !)
        # self.paths[k][0] is a list of loc_id
        i1 = len(nd.sol.paths[k][0]) - 1 - idx # kth loc id
        i2 = i1-1 # (k-1)th loc id
        if i2 < 0:
          break
        if nd.sol.paths[k][0][i2] == nd.sol.paths[k][0][i1]:
          last_idx = i2
        else:
          break
      # find last_idx
      for idx in range(last_idx):
        nidx = idx + 1 # next index
        nloc = nd.sol.paths[k][0][nidx]
        loc = nd.sol.paths[k][0][idx]
        ntt = nd.sol.paths[k][1][nidx]
        tt = nd.sol.paths[k][1][idx]
        cy = int(np.floor(nloc/self.xd)) # ref x
        cx = int(nloc%self.xd) # ref y, 
        if nloc != loc: # there is a move, not wait in place.
          for ic in range(self.cdim):
            out_cost[ic] = out_cost[ic] + self.cvecs[k][ic]*self.cgrids[ic][cy,cx]
        else: # robot stay in place. We know it has not reach goal yet. (until reach last_idx-1)
          out_cost = out_cost + self.cvecs[k]*int(ntt-tt) # np.ones((self.cdim)) # stay in place, fixed energy cost for every robot
    nd.cvec = out_cost # update cost in that node
    return out_cost

  def GoalFilterNode(self,nid):
    """
    filter HL node nid, if self.nodes[nid] is dominated by any HL nodes that reached goal.
    """
    for fid in self.nondom_goal_nodes:
      if cm.DomOrEqual( self.nodes[fid].cvec, self.nodes[nid].cvec ):
        return True
    return False # not filtered

  def GoalFilterNodeObject(self,hnode):
    """
    filter HL node hnode, if self.nodes hnode is dominated by any HL nodes that reached goal.
    """
    for fid in self.nondom_goal_nodes:
      if cm.DomOrEqual( self.nodes[fid].cvec, hnode.cvec ):
        return True
    return False # not filtered

  def RefineNondomGoals(self,nid): # should never be called ???
    """
    nid is a new HL node that reaches goal. Use it to filter self.nondom_goal_nodes
    """
    temp_set = copy.deepcopy(self.nondom_goal_nodes)
    for fid in self.nondom_goal_nodes:
      if cm.DomOrEqual( self.nodes[nid].cvec, self.nodes[fid].cvec):
        temp_set.remove(fid)
    self.nondom_goal_nodes = temp_set
    return

  def InitSearch(self):
    """
    called at the beginning of the search. 
    generate first High level node.
    compute individual optimal path for each robot.
    """
    self.pareto_idvl_path_dict = dict()
    for ri in range(self.num_robots):
      tnow = time.perf_counter()
      time_left = self.time_limit - (tnow-self.tstart)
      
      single_pareto_path, others = moastar.RunMoAstarSingleAgent(self.grids, self.sx_list[ri], self.sy_list[ri], \
        self.gx_list[ri], self.gy_list[ri], self.cvecs[ri], self.cgrids, self.cdim, \
        1.0, 0.0, 1e10, time_left )
      
      self.pareto_idvl_path_dict[ri] = list()
      for ref_key in single_pareto_path:
        lv = list()
        lt = list()
        curr_time = 0
        for pt in single_pareto_path[ref_key]:
          lv.append(pt[0]) # pt is a tuple of 1 element, e.g. (34,)
          lt.append(curr_time)
          curr_time = curr_time + 1
        self.pareto_idvl_path_dict[ri].append( (lv,lt) )

      tnow = time.perf_counter()
      if (tnow - self.tstart > self.time_limit):
        print(" FAIL! timeout! ")
        return 0
    # end for

    # for too many root nodes, just terminates...
    init_size = 1
    for k in self.pareto_idvl_path_dict:
      init_size = init_size*len(self.pareto_idvl_path_dict[k])
    if (init_size > MOCBS_INIT_SIZE_LIMIT):
      print("[CAVEAT] Too many roots to be generated for MO-CBS. Terminate. (why not use MO-CBS-t?)")
      self.num_roots = init_size
      return 0
    self.num_roots = init_size

    all_combi = list( itt.product(*(self.pareto_idvl_path_dict[ky] for ky in sorted(self.pareto_idvl_path_dict))) )

    for jpath in all_combi:
      nid = self.node_id_gen
      self.nodes[nid] = copy.deepcopy(MocbsNode(nid, np.zeros(self.cdim)))
      self.nodes[nid].root = nid
      self.node_id_gen = self.node_id_gen + 1
      for ri in range(len(jpath)):
        self.nodes[nid].sol.AddPath(ri,jpath[ri][0],jpath[ri][1])
      cvec = self.ComputeNodeCostObject(self.nodes[nid]) # update node cost vec and return cost vec

      self.open_by_tree[nid] = cm.PrioritySet()

      if OPEN_ADD_MODE == 1:
        self.open_list.add(np.sum(cvec),nid)
        self.open_by_tree[nid].add(np.sum(cvec), nid)
      elif OPEN_ADD_MODE == 2:
        self.open_list.add(tuple(cvec),nid)
        self.open_by_tree[nid].add(tuple(cvec), nid)
  
    return 1

  def PathDictLexSort(self, path_dict, cvec_dict):
    """
    sort path_dict by lex order and return a list.
    """
    out_path_list = list()
    pq = cm.PrioritySet()
    for k in cvec_dict:
      pq.add( tuple( cvec_dict[k] ), k)
    while pq.size() > 0:
      cvec, k = pq.pop()
      out_path_list.append(path_dict[k])
    return out_path_list

  def InitSearch_OnDemand(self):
    """
    called at the beginning of the search. 
    generate first High level node.
    compute individual optimal path for each robot.
    """
    self.pareto_idvl_path_dict = dict()
    for ri in range(self.num_robots):
      tnow = time.perf_counter()

      time_left = self.time_limit - (tnow-self.tstart)

      # call constrained NAMOA*
      single_pareto_path, others = moastar.RunMoAstarSingleAgent(self.grids, self.sx_list[ri], self.sy_list[ri], \
        self.gx_list[ri], self.gy_list[ri], self.cvecs[ri], self.cgrids, self.cdim, \
        1.0, 0.0, 1e10, time_left )

      self.pareto_idvl_path_dict[ri] = list()

      for ref_key in single_pareto_path:
        lv = list()
        lt = list()
        curr_time = 0
        for pt in single_pareto_path[ref_key]:
          lv.append(pt[0]) # pt is a tuple of 1 element, e.g. (34,)
          lt.append(curr_time)
          curr_time = curr_time + 1
        self.pareto_idvl_path_dict[ri].append( (lv,lt) )
        
      tnow = time.perf_counter()
      if (tnow - self.tstart > self.time_limit):
        print(" FAIL! timeout! ")
        return 0
    # end for

    # update number of roots.
    init_size = 1
    for k in self.pareto_idvl_path_dict:
      init_size = init_size*len(self.pareto_idvl_path_dict[k])
    self.num_roots = init_size

    # init indices 
    self.init_tree_index_dict = dict()
    for ri in range(self.num_robots):
      self.init_tree_index_dict[ri] = 0

    # generate a root
    if not self.GenRoot_OnDemand():
      return 2 # this does not matter, if fail to generate, then OPEN depletes and the program terminates.

    self.UpdateIndices_OnDemand()
    return 1

  def GenRoot_OnDemand(self):

    if (self.init_tree_index_dict[self.num_robots-1] >= len(self.pareto_idvl_path_dict[self.num_robots-1]) ):
      return False, -1

    nid = self.node_id_gen
    self.nodes[nid] = copy.deepcopy(MocbsNode(nid, np.zeros(self.cdim)))
    self.nodes[nid].root = nid
    self.node_id_gen = self.node_id_gen + 1
    self.root_generated = self.root_generated + 1 
    
    for ri in range(self.num_robots):
      ri_path = self.pareto_idvl_path_dict[ri][self.init_tree_index_dict[ri]]
      self.nodes[nid].sol.AddPath(ri, ri_path[0], ri_path[1])
    cvec = self.ComputeNodeCostObject(self.nodes[nid]) # update node cost vec and return cost vec

    self.open_by_tree[nid] = cm.PrioritySet()
    if OPEN_ADD_MODE == 1:
      self.open_list.add(np.sum(cvec),nid)
      self.open_by_tree[nid].add(np.sum(cvec), nid)
    elif OPEN_ADD_MODE == 2:
      self.open_list.add(tuple(cvec),nid)
      self.open_by_tree[nid].add(tuple(cvec), nid)

    return True, nid

  def UpdateIndices_OnDemand(self):
    self.init_tree_index_dict[0] = self.init_tree_index_dict[0] + 1
    idx = 0
    while (idx < (self.num_robots-1)) and \
      (self.init_tree_index_dict[idx] >= len(self.pareto_idvl_path_dict[idx])):
      # add one to the "next digit"
      self.init_tree_index_dict[idx] = 0
      self.init_tree_index_dict[idx+1] = self.init_tree_index_dict[idx+1] + 1
      idx = idx + 1
    return 

  def UpdateStats(self, stats):
    """
    """
    self.num_closed_low_level_states = self.num_closed_low_level_states + stats[0]
    self.total_low_level_time = self.total_low_level_time + stats[3]
    self.num_low_level_calls = self.num_low_level_calls + 1
    return

  def ReconstructPath(self, nid):
    """
    """
    path_set = dict()
    for i in range(self.num_robots):
      lx = list()
      ly = list()
      lv = self.nodes[nid].sol.GetPath(i)[0]
      for v in lv:
        y = int(np.floor(v / self.xd))
        x = int(v % self.xd)
        ly.append(y)
        lx.append(x)
      lt = self.nodes[nid].sol.GetPath(i)[1]
      path_set[i] = [lx,ly,lt]
    return path_set

  def FirstConflict(self, nd):
    return nd.CheckConflict()

  def SelectNode(self):
    """
    Pop a node from OPEN. 
    consider self.expansion_mode, either MO-CBS, or MO-CBS-t.
    """
  
    if self.expansion_mode == 0 and self.open_list.size() == 0:
      return False, []
  
    if self.open_by_tree[self.curr_root].size() == 0:
      self.open_by_tree.pop(self.curr_root) # delete current tree 
      for k in self.open_by_tree: # move to next tree
        self.curr_root = k
        break
    if len(self.open_by_tree) == 0: # if all trees are depleted
      return False, []

    popped = (-1,-1) 
    if (self.expansion_mode == 1 or self.expansion_mode == 2):
      popped = self.open_by_tree[self.curr_root].pop()
      self.open_list.remove(popped[1])
    else:
      popped = self.open_list.pop() # pop_node = (f-value, high-level-node-id)
    
    return True, popped

  def SelectNode_OnDemand(self):
    """
    Pop a node from OPEN. 
    consider self.expansion_mode=2, for MO-CBS-t on demand generation of roots.
    """
    if self.open_by_tree[self.curr_root].size() == 0:
      print(">>> MO-CBS-t, Move to next tree or terminate...")

      # generate a root
      genroot_success, root_id = self.GenRoot_OnDemand()
    
      if not genroot_success:
        return False, [] # fail to generate new node
      self.UpdateIndices_OnDemand()

      # delete old tree and move to newly generated tree
      self.open_by_tree.pop(self.curr_root) # delete curr tree 
      for k in self.open_by_tree: # move to next tree
        self.curr_root = k
        break

    # pop a node from current tree
    popped = (-1,-1) 
    popped = self.open_by_tree[self.curr_root].pop()
    self.open_list.remove(popped[1])

    return True, popped

  def Search(self, search_limit):
    """
    high level search
    """
    ########################################################
    #### init search ####
    ########################################################
    self.tstart = time.perf_counter()
    if (self.expansion_mode == 0 or self.expansion_mode == 1):
      init_success = self.InitSearch()
    else:
      init_success = self.InitSearch_OnDemand()

    find_first_feasible_sol = False
    first_sol = []
    first_sol_gvec = []

    if init_success == 0:
      output_res = ( int(len(self.closed_set)), dict(), 0, \
        float(time.perf_counter()-self.tstart), 0, int(self.num_roots), \
        int(self.open_list.size()), find_first_feasible_sol, first_sol_gvec,
        float(self.total_low_level_time), int(self.num_low_level_calls) )
      return dict(), output_res
    
    search_success = False
    best_g_value = -1
    reached_goal_id = -1

    # the following few lines is useful when self.expansion_mode == 1, init self.curr_root
    for k in self.open_by_tree:
      self.curr_root = k
      break

    ########################################################
    #### init search, END ####
    ########################################################
    while True :
      tnow = time.perf_counter()
      rd = len(self.closed_set)
      if (rd > search_limit) or (tnow - self.tstart > self.time_limit):
        search_success = False
        break
      ########################################################
      #### pop a non-dominated from OPEN ####
      ########################################################
      pop_succeed = False
      popped = []
      if (self.expansion_mode == 0 or self.expansion_mode == 1):
        pop_succeed, popped = self.SelectNode()
      else:
        pop_succeed, popped = self.SelectNode_OnDemand()

      if not pop_succeed:
        break
      if self.GoalFilterNode(popped[1]):
        continue
      ########################################################
      #### pop a non-dominated from OPEN, END ####
      ########################################################

      ########################################################
      #### Expand a node ####
      ########################################################
      self.closed_set.add(popped[1]) # only used to count numbers
      curr_node = self.nodes[popped[1]]
      cstrs = self.FirstConflict(curr_node)

      if len(cstrs) == 0: # no conflict, find a sol !!
        if not find_first_feasible_sol:
          find_first_feasible_sol = True
          first_sol = copy.deepcopy(curr_node)
          first_sol_gvec = curr_node.cvec
        self.RefineNondomGoals(curr_node.id)
        self.nondom_goal_nodes.add(curr_node.id)
        continue # keep going

      for cstr in cstrs:
        new_id = self.node_id_gen
        self.node_id_gen = self.node_id_gen + 1
        self.nodes[new_id] = copy.deepcopy(curr_node)
        self.nodes[new_id].id = new_id
        self.nodes[new_id].parent = curr_node.id
        self.nodes[new_id].cstr = cstr
        ri = cstr.i
        sstats = self.Lsearch(new_id) # this can generate multiple nodes
        self.UpdateStats(sstats)
        if sstats[2] == 0:
          # this branch fails, robot ri cannot find a consistent path.
          continue
        # node insertion into OPEN is done in Lsearch()
      ########################################################
      #### Expand a node, END ####
      ########################################################
      # end of for
    # end of while

    all_path_set = dict()
    all_cost_vec = dict()
    for nid in self.nondom_goal_nodes:
      hnode = self.nodes[nid]
      all_path_set[hnode.id] = hnode.sol.paths
      all_cost_vec[hnode.id] = hnode.cvec
    if (self.open_list.size() == 0) and (len(self.nondom_goal_nodes) > 0) and (self.root_generated == self.num_roots):
      search_success = True

    output_res = ( int(len(self.closed_set)), all_cost_vec, int(search_success), \
      float(time.perf_counter()-self.tstart), int(self.num_closed_low_level_states), \
      int(self.num_roots), int(self.open_list.size()), find_first_feasible_sol, first_sol_gvec,
      float(self.total_low_level_time), int(self.num_low_level_calls) )

    return all_path_set, output_res

def RunMocbsMAPF(grids, sx, sy, gx, gy, cvecs, cost_grids, cdim, w, eps, search_limit, time_limit, expansion_mode=2):
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
  expansion_mode = 0, MO-CBS
  expansion_mode = 2, MO-CBS-t (treewise expansion), default.
  """
  if expansion_mode == 2:
    print("... Run MO-CBS-t ... ")
  elif expansion_mode == 0:
    print("... Run MO-CBS ... ")
  else:
    sys.exit("[ERROR] Are you kidding? Unknown expansion mode for MO-CBS!!!")

  truncated_cvecs = list()
  truncated_cgrids = list()

  for idx in range(len(cvecs)):
    truncated_cvecs.append(cvecs[idx][0:cdim])
  for idx in range(cdim):
    truncated_cgrids.append(cost_grids[idx])

  mocbs = MocbsSearch(grids, sx, sy, gx, gy, truncated_cvecs, truncated_cgrids, expansion_mode, time_limit)

  return mocbs.Search(search_limit)
  
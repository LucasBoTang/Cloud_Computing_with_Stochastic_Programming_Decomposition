import time
from gurobipy import *
import numpy as np


def extensiveForm(d):
    """
    solve with extensive form
    """
    tick = time.time()
    # ceate a new model
    m = Model('EF')

    # first-stage variables:
    # VMs reservation
    xr = m.addVars(d.users, d.VMs, d.providers, vtype=GRB.INTEGER, name='xr')
    # routers reservation
    yr = m.addVars(d.users, d.routers, vtype=GRB.CONTINUOUS, name='yr')

    # second-stage variables:
    # VMs utilization
    xu = m.addVars(d.scenarios, d.users, d.VMs, d.providers, vtype=GRB.CONTINUOUS, name='xu')
    # routers utilization
    yu = m.addVars(d.scenarios, d.users, d.routers, vtype=GRB.CONTINUOUS, name='yu')
    # VMs on-demand
    xo = m.addVars(d.scenarios, d.users, d.VMs, d.providers, vtype=GRB.CONTINUOUS, name='xo')
    # routers on-demand
    yo = m.addVars(d.scenarios, d.users, d.routers, vtype=GRB.CONTINUOUS, name='yo')
    # network flow
    f = m.addVars(d.scenarios, d.users, d.arcs, vtype=GRB.CONTINUOUS, name='f')

    # objective function
    # first stage
    obj = quicksum(d.VM_rcost[v] * xr[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
        + quicksum(d.R_rcost[r] * yr[u,r] for r in d.routers for u in d.users)
    # scecond stage
    for s in d.scenarios:
        # utilization
        obj += d.prob[s] * (quicksum(d.VM_ucost[s,v] * xu[s,u,v,p] for p in d.providers for v in d.VMs for u in d.users) + \
                            quicksum(d.R_ucost[s,r] * yu[s,u,r] for r in d.routers for u in d.users))
        # on-demand
        obj += d.prob[s] * (quicksum(d.VM_ocost[s,v] * xo[s,u,v,p] for p in d.providers for v in d.VMs for u in d.users) + \
                            quicksum(d.R_ocost[s,r] * yo[s,u,r] for r in d.routers for u in d.users))
    m.setObjective(obj)
    # sense
    m.modelSense = GRB.MINIMIZE

    # constraints:
    # utilization bound
    m.addConstrs((xu[s,u,v,p] <= xr[u,v,p] for p in d.providers for v in d.VMs for u in d.users for s in d.scenarios),
                 name='VM utilization bound')
    m.addConstrs((yu[s,u,r] <= yr[u,r] for r in d.routers for u in d.users for s in d.scenarios),
                 name='R utilization bound')
    # capacity
    m.addConstrs((quicksum(d.VM_CDemand[v] * (xu[s,u,v,p] + xo[s,u,v,p]) for u in d.users for v in d.VMs) <= d.P_CCapacity[p]
                  for p in d.providers for s in d.scenarios),
                 name='CPU capacity')
    m.addConstrs((quicksum(d.VM_SDemand[v] * (xu[s,u,v,p] + xo[s,u,v,p]) for u in d.users for v in d.VMs) <= d.P_SCapacity[p]
                  for p in d.providers for s in d.scenarios),
                 name='Storage capacity')
    m.addConstrs((quicksum(d.VM_MDemand[v] * (xu[s,u,v,p] + xo[s,u,v,p]) for u in d.users for v in d.VMs) <= d.P_MCapacity[p]
                  for p in d.providers for s in d.scenarios),
                 name='Memory capacity')
    m.addConstrs((quicksum(yu[s,u,r] + yo[s,u,r] for u in d.users) <= d.R_BCapacity[r]
                  for r in d.routers for s in d.scenarios),
                 name='Bandwidth capacity')
    # VM demand
    m.addConstrs((quicksum(xu[s,u,v,p] + xo[s,u,v,p] for p in d.providers) >= d.VM_demands[s,u,v]
                  for v in d.VMs for u in d.users for s in d.scenarios),
                 name='VM demand')
    # network flow
    m.addConstrs((quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_in == r) == \
                  quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_out == r)
                  for u in d.users for r in d.routers for s in d.scenarios),
                 name='R balance')
    m.addConstrs((quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_out == r) == yu[s,u,r] + yo[s,u,r]
                  for u in d.users for r in d.routers for s in d.scenarios),
                 name='R usage')
    m.addConstrs((quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_out == p) >= \
                  quicksum(d.VM_BDemand[v] * (xu[s,u,v,p] + xo[s,u,v,p]) for v in d.VMs)
                  for p in d.providers for u in d.users for s in d.scenarios),
                 name='R demand')
    m.addConstrs((quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_in == u) == \
                  quicksum(f[s,u,e_out,e_in] for e_out, e_in in d.arcs if e_out[0] == 'P')
                  for u in d.users for s in d.scenarios),
                 name='U balance')

    # optimze
    m.update()
    m.optimize()
    tock = time.time()

    # output
    print()
    obj = m.objVal
    print('Optimal obj: {:.4f}'.format(obj))
    print('VM:')
    for u in d.users:
        for v in d.VMs:
            for p in d.providers:
                print('  xr[{}, {}, {}] = {}'.format(u,v,p,int(xr[u,v,p].x)))
    print('Routers:')
    for u in d.users:
        for r in d.routers:
            print('  yr[{}, {}] = {:.2f}'.format(u,r,int(yr[u,r].x)))
    elapsed = tock - tick
    print('Elapse Time: {:.4f}'.format(elapsed))

    return obj, elapsed

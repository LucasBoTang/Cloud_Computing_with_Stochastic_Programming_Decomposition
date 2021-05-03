import time
from gurobipy import *
import numpy as np


def levelMethod(d, lambd=0.2929, epsilon=1e-4):
    """
    Run level method
    """
    # build the master problem
    MP, xr, yr, t = buildMP(d)
    # build subproblems
    SP, xu, xo, yu, yo, f = buildSP(d)
    # build levelproblems
    LP, xr_l, yr_l = buildLP(d)

    # init
    bestUB = GRB.INFINITY
    noIters = 0
    noCuts = 0
    tick = time.time()

    # constraints for LP
    betas = []
    m_constrs = []

    # init x1
    xrsol_t, yrsol_t, MPobj = solveMP(MP, xr, yr, d)

    optimal = False
    print()
    while True:
        noIters += 1
        print('Iteration {}:'.format(noIters))

        #============================== Solve subproblems ==========================
        fx, beta, alpha_x, alpha_y = computeSubgradient(xrsol_t, yrsol_t, SP, xr, yr, xu, xo, yu, yo, xr_l, yr_l, d)
        # update MP
        noCuts += 1
        MP.addConstr(t >= beta + \
                          quicksum(alpha_x[u,v,p] * xr[u,v,p] for p in d.providers for v in d.VMs for u in d.users) + \
                          quicksum(alpha_y[u,r] * yr[u,r] for r in d.routers for u in d.users))
        betas.append(beta)

        #============================== Minimize model =============================
        # compute best lower bound
        _, _, MPobj = solveMP(MP, xr, yr, d)
        print('Best LB: {:.2f}'.format(MPobj))
        # update best upper bound
        bestUB = min(fx, bestUB)
        print('Current UB f(x_t): {:.2f}'.format(fx))
        print('Best UB: {:.2f}'.format(bestUB))
        # convergence check
        if (bestUB - MPobj) / (1 + abs(bestUB)) <= epsilon:
            print('The algorithm converges.')
            optimal = True
            break

        #============================== Project ====================================
        lt = MPobj + lambd * (bestUB - MPobj)
        # update mt(x) expression
        m_constrs.append(LP.addConstr(beta + \
                                      quicksum((alpha_x[u,v,p] + d.VM_rcost[v]) * xr_l[u,v,p]
                                               for p in d.providers for v in d.VMs for u in d.users) + \
                                      quicksum((alpha_y[u,r] + d.R_rcost[r]) * yr_l[u,r]
                                               for r in d.routers for u in d.users) <= 0))
        xrsol_t, yrsol_t = modifyAndSolveLP(LP, xr_l, yr_l, m_constrs, lt, betas, xrsol_t, yrsol_t, d)
        print()

    tock = time.time()
    print('Original problem is optimal.')
    print('Optimal obj: {:.4f}'.format(fx))
    print('VM:')
    for u in d.users:
        for v in d.VMs:
            for p in d.providers:
                print('  xr[{}, {}, {}] = {}'.format(u,v,p,int(xr[u,v,p].x)))
    print('Routers:')
    for u in d.users:
        for r in d.routers:
            print('  yr[{}, {}] = {:.2f}'.format(u,r,int(yr[u,r].x)))
    print('NoIters: {}'.format(noIters))
    print('NoCuts: {}'.format(noCuts))
    elapsed = tock - tick
    print('Elapse Time: {:.4f}'.format(elapsed))

    return fx, elapsed, noIters, noCuts


def buildMP(d):
    """
    Build the master problem
    """
    # create a new master
    MP = Model('MP')

    # turn off output
    MP.Params.outputFlag = 0
    # dual simplex
    MP.Params.method = 1

    # first-stage variables:
    # VMs reservation
    xr = MP.addVars(d.users, d.VMs, d.providers, vtype=GRB.INTEGER, name='xr')
    # routers reservation
    yr = MP.addVars(d.users, d.routers, vtype=GRB.CONTINUOUS, name='yr')

    # second stage expectation
    t = MP.addVar(vtype=GRB.CONTINUOUS, name='theta')

    # objective function
    obj = quicksum(d.VM_rcost[v] * xr[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
        + quicksum(d.R_rcost[r] * yr[u,r] for r in d.routers for u in d.users) \
        + t
    MP.setObjective(obj)
    # model sense
    MP.modelSense = GRB.MINIMIZE

    return MP, xr, yr, t


def buildSP(d):
    """
    Build the subproblems
    """
    # create a new primal subproblem
    SP = Model('SP')

    # turn off output
    SP.Params.outputFlag = 0
    # dual simplex
    SP.Params.method = 1

    # second-stage varibles:
    # VMs utilization
    xu = SP.addVars(d.users, d.VMs, d.providers, vtype=GRB.CONTINUOUS, name='xu')
    # routers utilization
    yu = SP.addVars(d.users, d.routers, vtype=GRB.CONTINUOUS, name='yu')
    # VMs on-demand
    xo = SP.addVars(d.users, d.VMs, d.providers, vtype=GRB.CONTINUOUS, name='xo')
    # routers on-demand
    yo = SP.addVars(d.users, d.routers, vtype=GRB.CONTINUOUS, name='yo')
    # network flow
    f = SP.addVars(d.users, d.arcs, vtype=GRB.CONTINUOUS, name='f')

    # model sense
    SP.modelSense = GRB.MINIMIZE

    # constraints:
    # utilization bound (init rhs = 0)
    SP.addConstrs((xu[u,v,p] <= 0 for p in d.providers for v in d.VMs for u in d.users),
                  name='VM utilization bound')
    SP.addConstrs((yu[u,r] <= 0 for r in d.routers for u in d.users),
                  name='R utilization bound')
    # capacity
    SP.addConstrs((quicksum(d.VM_CDemand[v] * (xu[u,v,p] + xo[u,v,p]) for u in d.users for v in d.VMs) <= d.P_CCapacity[p]
                   for p in d.providers), name='CPU capacity')
    SP.addConstrs((quicksum(d.VM_SDemand[v] * (xu[u,v,p] + xo[u,v,p]) for u in d.users for v in d.VMs) <= d.P_SCapacity[p]
                   for p in d.providers), name='Storage capacity')
    SP.addConstrs((quicksum(d.VM_MDemand[v] * (xu[u,v,p] + xo[u,v,p]) for u in d.users for v in d.VMs) <= d.P_MCapacity[p]
                   for p in d.providers), name='Memory capacity')
    SP.addConstrs((quicksum(yu[u,r] + yo[u,r] for u in d.users) <= d.R_BCapacity[r]
                   for r in d.routers), name='Bandwidth capacity')
    # VM demand (init rhs = 0)
    SP.addConstrs((quicksum(xu[u,v,p] + xo[u,v,p] for p in d.providers) >= 0
                   for v in d.VMs for u in d.users), name='VM demand')
    # network flow
    SP.addConstrs((quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_in == r) == \
                   quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_out == r)
                   for u in d.users for r in d.routers), name='R balance')
    SP.addConstrs((quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_out == r) == yu[u,r] + yo[u,r]
                   for u in d.users for r in d.routers), name='R usage')
    SP.addConstrs((quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_out == p) >= \
                   quicksum(d.VM_BDemand[v] * (xu[u,v,p] + xo[u,v,p]) for v in d.VMs)
                   for p in d.providers for u in d.users), name='R demand')
    SP.addConstrs((quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_in == u) == \
                   quicksum(f[u,e_out,e_in] for e_out, e_in in d.arcs if e_out[0] == 'P')
                   for u in d.users), name='U balance')

    SP.update()
    return SP, xu, xo, yu, yo, f


def buildLP(d):
    """
    Build the level problem to find new xt
    """
    # create a new model
    LP = Model('LP')

    # turn off output
    LP.Params.outputFlag = 0

    # variables
    # VMs reservation
    xr_l = LP.addVars(d.users, d.VMs, d.providers, vtype=GRB.INTEGER, name='xr')
    # routers reservation
    yr_l = LP.addVars(d.users, d.routers, vtype=GRB.CONTINUOUS, name='yr')

    # model sense
    LP.modelSense = GRB.MINIMIZE

    return LP, xr_l, yr_l


def solveMP(MP, xr, yr, d):
    """
    Get optimal solution and objective value for current master problem
    """
    # optimize
    MP.update()
    MP.optimize()
    MPobj = MP.objVal

    # get first-stage varibles
    xrsol_t = {}
    for u in d.users:
        for v in d.VMs:
            for p in d.providers:
                xrsol_t[u,v,p] = xr[u,v,p].x
    yrsol_t = {}
    for u in d.users:
        for r in d.routers:
            yrsol_t[u,r] = yr[u,r].x

    return xrsol_t, yrsol_t, MPobj


def computeSubgradient(xrsol_t, yrsol_t, SP, xr, yr, xu, xo, yu, yo, xr_l, yr_l, d):
    """
    Evaluate f(yt) and compute subgradient by solving subproblems
    """
    # current upper bound (first part)
    fx = sum(d.VM_rcost[v] * xrsol_t[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
       + sum(d.R_rcost[r] * yrsol_t[u,r] for r in d.routers for u in d.users)

    # init cut constraint coefficients
    beta = 0
    alpha_x = {(u,v,p): 0 for p in d.providers for v in d.VMs for u in d.users}
    alpha_y = {(u,r):0 for r in d.routers for u in d.users}

    for s in d.scenarios:
        # solve subproblem and get dual solution
        qvalue, vu_pisol, ru_pisol, c_pisol, s_pisol, m_pisol, b_pisol, d_pisol = \
        modifyAndSolveSP(s, xrsol_t, yrsol_t, SP, xu, xo, yu, yo, d)
        # check dual solution
        dualSPobj = sum(xrsol_t[u,v,p] * vu_pisol[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
                  + sum(yrsol_t[u,r] * ru_pisol[u,r] for r in d.routers for u in d.users) \
                  + sum(c_pisol[p] * d.P_CCapacity[p] if d.P_CCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(s_pisol[p] * d.P_SCapacity[p] if d.P_SCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(m_pisol[p] * d.P_MCapacity[p] if d.P_MCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(b_pisol[r] * d.R_BCapacity[r] for r in d.routers) \
                  + sum(d_pisol[u,v] * d.VM_demands[s,u,v] for v in d.VMs for u in d.users)
        assert abs(dualSPobj - qvalue) < 1e-4, 'Strong duality'

        # current upper bound (second part)
        fx += d.prob[s] * qvalue

        # accumulate subgradient constraint
        beta += (sum(c_pisol[p] * d.P_CCapacity[p] if d.P_CCapacity[p] != float('inf') else 0 for p in d.providers) + \
                 sum(s_pisol[p] * d.P_SCapacity[p] if d.P_SCapacity[p] != float('inf') else 0 for p in d.providers) + \
                 sum(m_pisol[p] * d.P_MCapacity[p] if d.P_MCapacity[p] != float('inf') else 0 for p in d.providers) + \
                 sum(b_pisol[r] * d.R_BCapacity[r] for r in d.routers) + \
                 sum(d_pisol[u,v] * d.VM_demands[s,u,v] for v in d.VMs for u in d.users)) * d.prob[s]
        for u in d.users:
            for v in d.VMs:
                for p in d.providers:
                    alpha_x[u,v,p] += d.prob[s] * vu_pisol[u,v,p]
        for u in d.users:
            for r in d.routers:
                alpha_y[u,r] += d.prob[s] * ru_pisol[u,r]

    return fx, beta, alpha_x, alpha_y

def modifyAndSolveSP(s, xrsol, yrsol, SP, xu, xo, yu, yo, d):
    """
    modify constraints rhs with scenario and fixed first-stage varibles
    solve and return dual solution
    """
    # modify rhs
    for constr in SP.getConstrs():
        name, index = constr.constrName[:-1].split('[')
        index = index.split(',')
        if name == 'VM utilization bound':
            p, v, u = index
            constr.rhs = xrsol[u,v,p]
        elif name == 'R utilization bound':
            r, u = index
            constr.rhs = yrsol[u,r]
        elif name == 'VM demand':
            v, u = index
            constr.rhs = d.VM_demands[s,u,v]

    # modify obj
    obj = quicksum(d.VM_ucost[s,v] * xu[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
        + quicksum(d.R_ucost[s,r] * yu[u,r] for r in d.routers for u in d.users) \
        + quicksum(d.VM_ocost[s,v] * xo[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
        + quicksum(d.R_ocost[s,r] * yo[u,r] for r in d.routers for u in d.users)
    SP.setObjective(obj)

    # solve
    SP.optimize()
    SPobj = SP.objVal
    # print('Subproblem {}'.format(s))
    # print('SPobj: {:.2f}'.format(SPobj))

    # dual solution
    vu_pisol = {} # VM utilization
    ru_pisol = {} # router utilization
    c_pisol = {} # CPU capacity
    s_pisol = {} # storage capacity
    m_pisol = {} # memory capacity
    b_pisol = {} # bandwith capacity
    d_pisol = {} # VM demand
    for constr in SP.getConstrs():
        name, index = constr.constrName[:-1].split('[')
        if name == 'VM utilization bound':
            p, v, u = index.split(',')
            vu_pisol[u,v,p] = constr.pi
        elif name == 'R utilization bound':
            r, u = index.split(',')
            ru_pisol[u,r] = constr.pi
        elif name == 'CPU capacity':
            c_pisol[index] = constr.pi
        elif name == 'Storage capacity':
            s_pisol[index] = constr.pi
        elif name == 'Memory capacity':
            m_pisol[index] = constr.pi
        elif name == 'Bandwidth capacity':
            b_pisol[index] = constr.pi
        elif name == 'VM demand':
            v, u = index.split(',')
            d_pisol[u,v] = constr.pi

    return SPobj, vu_pisol, ru_pisol, c_pisol, s_pisol, m_pisol, b_pisol, d_pisol


def modifyAndSolveLP(LP, xr_l, yr_l, m_constrs, lt, betas, xrsol_t, yrsol_t, d):
    """
    modify constraints rhs with different lt to get new xt
    """
    # update obj with xt
    obj = quicksum((xr_l[u,v,p] - xrsol_t[u,v,p]) * (xr_l[u,v,p] - xrsol_t[u,v,p])
                   for p in d.providers for v in d.VMs for u in d.users) + \
          quicksum((yr_l[u,r] - yrsol_t[u,r]) * (yr_l[u,r] - yrsol_t[u,r])
                   for r in d.routers for u in d.users)
    LP.setObjective(obj)

    # update constraints with lt
    for i, constr in enumerate(m_constrs):
        constr.rhs = lt - betas[i]

    # optimize
    LP.update()
    LP.optimize()

    # new xt
    # get first-stage varibles
    xrsol_t = {}
    for u in d.users:
        for v in d.VMs:
            for p in d.providers:
                xrsol_t[u,v,p] = xr_l[u,v,p].x
    yrsol_t = {}
    for u in d.users:
        for r in d.routers:
            yrsol_t[u,r] = yr_l[u,r].x

    return xrsol_t, yrsol_t

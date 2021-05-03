import time
from gurobipy import *
import numpy as np


def regularizedDecomposition(d, rho=1, gamma=0.1, epsilon=1e-4):
    """
    Regularized decomposition
    """
    # build the master problem
    MP, xr, yr, t = buildMP(d)
    # build subproblems
    SP, xu, xo, yu, yo, f = buildSP(d)

    # init
    bestUB = GRB.INFINITY
    noIters = 0
    noCuts = 0
    tick = time.time()

    # t = 1
    noIters += 1
    print('\nIteration {}:'.format(noIters))
    # compute x1, y1
    xrsol_yt, yrsol_yt, _ = solveMP(MP, xr, yr, d)
    xrsol_xt, yrsol_xt = xrsol_yt, yrsol_yt
    # compute subgradient
    fy, expr = computeSubgradient(xrsol_yt, yrsol_yt, SP, xr, yr, xu, xo, yu, yo, d)
    fx = fy
    # update MP
    noCuts += 1
    MP.addConstr(t >= expr)
    # update best upper bound
    bestUB = fy
    print('Current UB f(y_t): {:.2f}'.format(fy))
    print('Best UB: {:.2f}'.format(bestUB))
    print()

    optimal = False
    while True:
        noIters += 1
        print('Iteration {}:'.format(noIters))

        # ============================== Solve master ==============================
        xrsol_yt, yrsol_yt, m_yt = modifyAndSolveMP(MP, xr, yr, t, xrsol_xt, yrsol_xt, d, rho)
        print('Current m(y_t): {:.2f}'.format(m_yt))

        # ============================== Convergence check =========================
        vt = m_yt - fx
        # check if terminate
        if vt / (1 + abs(fx)) >= - epsilon:
            print('The algorithm converges.')
            optimal = True
            break

        #============================== Solve subproblems ==========================
        fy, expr = computeSubgradient(xrsol_yt, yrsol_yt, SP, xr, yr, xu, xo, yu, yo, d)
        # update MP
        noCuts += 1
        MP.addConstr(t >= expr)

        # update best upper bound
        bestUB = min(fy, bestUB)
        print('Current UB f(y_t): {:.2f}'.format(fy))
        print('Best UB: {:.2f}'.format(bestUB))
        print()

        #============================== Update rho =================================
        mu = 2 * rho * (1 - (fy - fx) / vt)
        rho = min(max(mu, rho/10, 1e-4), 10*rho)

        #============================== Serious step ===============================
        if fy < fx + gamma * vt:
            xrsol_xt, yrsol_xt = xrsol_yt, yrsol_yt
            fx = fy

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


def computeSubgradient(xrsol_yt, yrsol_yt, SP, xr, yr, xu, xo, yu, yo, d):
    """
    Evaluate f(yt) and compute subgradient by solving subproblems
    """
    # current upper bound (first part)
    fy = sum(d.VM_rcost[v] * xrsol_yt[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
       + sum(d.R_rcost[r] * yrsol_yt[u,r] for r in d.routers for u in d.users)

    # init cut constraint
    expr = 0

    for s in d.scenarios:
        # solve subproblem and get dual solution
        qvalue, vu_pisol, ru_pisol, c_pisol, s_pisol, m_pisol, b_pisol, d_pisol = \
        modifyAndSolveSP(s, xrsol_yt, yrsol_yt, SP, xu, xo, yu, yo, d)
        # check dual solution
        dualSPobj = sum(xrsol_yt[u,v,p] * vu_pisol[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
                  + sum(yrsol_yt[u,r] * ru_pisol[u,r] for r in d.routers for u in d.users) \
                  + sum(c_pisol[p] * d.P_CCapacity[p] if d.P_CCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(s_pisol[p] * d.P_SCapacity[p] if d.P_SCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(m_pisol[p] * d.P_MCapacity[p] if d.P_MCapacity[p] != float('inf') else 0 for p in d.providers) \
                  + sum(b_pisol[r] * d.R_BCapacity[r] for r in d.routers) \
                  + sum(d_pisol[u,v] * d.VM_demands[s,u,v] for v in d.VMs for u in d.users)
        assert abs(dualSPobj - qvalue) < 1e-4, 'Strong duality'

        # current upper bound (second part)
        fy += d.prob[s] * qvalue

        # accumulate subgradient constraint
        expr_s = sum(xr[u,v,p] * vu_pisol[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
               + sum(yr[u,r] * ru_pisol[u,r] for r in d.routers for u in d.users) \
               + sum(c_pisol[p] * d.P_CCapacity[p] if d.P_CCapacity[p] != float('inf') else 0 for p in d.providers) \
               + sum(s_pisol[p] * d.P_SCapacity[p] if d.P_SCapacity[p] != float('inf') else 0 for p in d.providers) \
               + sum(m_pisol[p] * d.P_MCapacity[p] if d.P_MCapacity[p] != float('inf') else 0 for p in d.providers) \
               + sum(b_pisol[r] * d.R_BCapacity[r] for r in d.routers) \
               + sum(d_pisol[u,v] * d.VM_demands[s,u,v] for v in d.VMs for u in d.users)
        expr += d.prob[s] * expr_s

    return fy, expr


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


def modifyAndSolveMP(MP, xr, yr, t, xrsol_xt, yrsol_xt, d, rho):
    """
    modify and solve master probelm (QP)
    """
    # change objective function
    obj = quicksum(d.VM_rcost[v] * xr[u,v,p] for p in d.providers for v in d.VMs for u in d.users) \
        + quicksum(d.R_rcost[r] * yr[u,r] for r in d.routers for u in d.users) \
        + t \
        + rho / 2 * (quicksum((xr[u,v,p] - xrsol_xt[u,v,p]) * (xr[u,v,p] - xrsol_xt[u,v,p])
                              for p in d.providers for v in d.VMs for u in d.users) + \
                     quicksum((yr[u,r] - yrsol_xt[u,r]) * (yr[u,r] - yrsol_xt[u,r])
                              for r in d.routers for u in d.users))
    MP.setObjective(obj)

    # optimize
    xrsol_yt, yrsol_yt, MPobj = solveMP(MP, xr, yr, d)
    m_yt = sum(d.VM_rcost[v] * xr[u,v,p].x for p in d.providers for v in d.VMs for u in d.users) \
         + sum(d.R_rcost[r] * yr[u,r].x for r in d.routers for u in d.users) \
         + t.x

    return xrsol_yt, yrsol_yt, m_yt

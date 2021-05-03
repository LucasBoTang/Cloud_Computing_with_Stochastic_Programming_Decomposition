import numpy as np

class simulatedDataset:
    """
    A simulated dataset
    """

    def __init__(self, num_scenarios):
        # random seed
        np.random.seed(139)
        # set
        self.users = ['U0', 'U1']
        self.VMs = ['V0', 'V1']
        self.providers = ['P0', 'P1', 'P2', 'P3']
        self.routers = ['R{}'.format(i) for i in range(9)]
        self.scenarios = ['S{}'.format(i) for i in range(num_scenarios)]
        # probability
        self.prob = {s:1/num_scenarios for s in self.scenarios}
        # VMs demand
        self.VM_CDemand = {'V0':24, 'V1':96} # CPU
        self.VM_SDemand = {'V0':12, 'V1':16} # storage
        self.VM_MDemand = {'V0':4, 'V1':16} # memory
        self.VM_BDemand = {'V0':0.8, 'V1':1.2} # Bandwith
        # VMs cost
        self.VM_rbasecost = {'V0':0.011, 'V1':0.023} # reservation cost
        self.VM_ubasecost = {'V0':0.02, 'V1':0.0488} # utilization cost
        self.VM_obasecost = {'V0':0.1, 'V1':0.1952} # on-demand cost
        self.VM_rcost = {}
        self.VM_ucost = {}
        self.VM_ocost = {}
        self.genVMCost()
        # provider capacity
        self.P_CCapacity = {p:float('inf') for p in self.providers} # CPU
        self.P_SCapacity = {p:float('inf') for p in self.providers} # storage
        self.P_MCapacity = {p:float('inf') for p in self.providers} # memory
        # routers capacity
        self.R_BCapacity = {**{'R{}'.format(i):100 for i in range(6)},
                            **{'R{}'.format(i):50 for i in range(6,9)}}
        # routers cost
        self.R_rbasecost = {**{'R{}'.format(i):0.651 for i in range(6)},
                            **{'R{}'.format(i):0.268 for i in range(6,9)}} # reservation cost
        self.R_ubasecost = {**{'R{}'.format(i):0.038 for i in range(6)},
                            **{'R{}'.format(i):0.013 for i in range(6,9)}} # utilization cost
        self.R_obasecost = {**{'R{}'.format(i):1.301 for i in range(6)},
                            **{'R{}'.format(i):0.587 for i in range(6,9)}} # on-demand cost
        self.R_rcost = {}
        self.R_ucost = {}
        self.R_ocost = {}
        self.genRCost()
        # VMs demands for users
        self.VM_demands = {}
        self.genVMDemands()
        # edges
        self.arcs = [('P0','R1'), ('P0','R6'), ('P1','R2'), ('P1','R7'), ('P2','R5'), ('P2','R8'), ('P3','R4'), ('P3','R8'),
                     ('R1','R2'), ('R2','R3'), ('R3','R4'), ('R5','R0'),
                     ('R0','R5'), ('R5','R4'), ('R4','R3'), ('R2','R1'), ('R1','R0'),
                     ('R0','R6'), ('R6','R1'), ('R3','R7'), ('R7','R2'), ('R4','R8'), ('R8','R5'),
                     ('R0','U0'), ('R6','U0'), ('R3','U1'), ('R7','U1')]

    def genVMCost(self):
        """
        generate VM cost with different senarios
        """
        # fix reservation cost
        self.VM_rcost = self.VM_rbasecost
        # utilization and on-demand cost with scenarios
        for s in self.scenarios:
            for v in self.VMs:
                self.VM_ucost[s,v] = np.round(self.VM_ubasecost[v] + np.random.uniform(-0.1,0.1) * self.VM_ubasecost[v], 4)
                self.VM_ocost[s,v] = np.round(self.VM_obasecost[v] + np.random.uniform(-0.1,0.1) * self.VM_obasecost[v], 4)

    def genRCost(self):
        """
        generate router cost with different senarios
        """
        # fix reservation cost
        self.R_rcost = self.R_rbasecost
        # utilization and on-demand cost with scenarios
        for s in self.scenarios:
            for r in self.routers:
                self.R_ucost[s,r] = np.round(self.R_ubasecost[r] + np.random.uniform(-0.2,0.2) * self.R_ubasecost[r], 4)
                self.R_ocost[s,r] = np.round(self.R_obasecost[r] + np.random.uniform(-0.2,0.2) * self.R_obasecost[r], 4)

    def genVMDemands(self):
        """
        generate VM demands per user with different senarios
        """
        for s in self.scenarios:
            self.VM_demands[s,'U0','V0'] = max(int(np.random.normal(25, 6)),0)
            self.VM_demands[s,'U0','V1'] = max(int(np.random.normal(36, 6)),0)
            self.VM_demands[s,'U1','V0'] = max(int(np.random.normal(30, 8)),0)
            self.VM_demands[s,'U1','V1'] = max(int(np.random.normal(50, 10)),0)

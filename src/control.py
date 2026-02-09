import numpy as np
from src.utils.utils import find_global_stress_mat
from tqdm import tqdm

def tracking_error(config, target_config):

    max_itr = config.shape[2]
    target_config = np.tile(target_config[:, :, np.newaxis], (1, 1, max_itr))

    return np.linalg.norm(config - target_config, axis=(0,1))

def affine_formation_control_global(target_config, stress_mat, leader_id, gain = 1, max_itr = 10000, dt = 0.01):

    # leader_id: list of leader node indices

    D, N = target_config.shape

    if leader_id is None:
        leader_id = list(range(D + 1))

    assert len(leader_id) >= D + 1, "leader nodes must be at least D + 1"

    config = np.random.rand(D, N)
    config_track = np.zeros((D, N, max_itr))
    # tracking_error = np.zeros(max_itr)
    
    for i in range(max_itr):
        control_input = stress_mat @ config.T
        config -= gain * dt * control_input.T
        config[:, leader_id] = target_config[:, leader_id]
        # log current configuration
        config_track[:, :, i] = config
        # tracking_error[i] = np.linalg.norm(config - target_config)
    
    return config_track


# def multicluster_afc_global(target_config: np.ndarray, 
#                         subgraph_glb_idx: list, 
#                         leaders_glb_idx: list, 
#                         stress_mat_list: list,
#                         max_itr: int = 10000, 
#                         dt: float = 0.01, 
#                         gain: float = 10.0):
    
#     D, N = target_config.shape

#     stress_mat_global = find_global_stress_mat(stress_mat_list, subgraph_glb_idx, target_config.shape[1])

#     init_config = np.random.rand(D, N)
#     # global control
#     config_track_glb = np.zeros((D, N, max_itr+1))
#     config_track_glb[:, :, 0] = init_config  # initial configuration
#     for itr in range(max_itr):
#         control_input_glb = - gain * config_track_glb[:, :, itr] @ stress_mat_global.T  # shape (D,N)
#         config_track_glb[:, :, itr+1] = config_track_glb[:, :, itr] + dt * control_input_glb
#         # leaders fixed at target positions
#         for leader_idx in leaders_glb_idx:
#             config_track_glb[:, leader_idx, itr+1] = target_config[:, leader_idx]

#     return config_track_glb

def multicluster_afc_global(target_config: np.ndarray, 
                        subgraph_glb_idx: list, 
                        leaders_glb_idx: list, 
                        stress_mat_list: list,
                        init_config: np.ndarray = None,  # <--- NEW ARGUMENT
                        max_itr: int = 10000, 
                        dt: float = 0.01, 
                        gain: float = 10.0):
    
    D, N = target_config.shape

    # 1. Compute Global Matrix
    stress_mat_global = find_global_stress_mat(stress_mat_list, subgraph_glb_idx, N)

    # 2. Initialize Trajectory Array
    config_track_glb = np.zeros((D, N, max_itr+1))

    # 3. Set Initial State
    if init_config is None:
        # Default: Start from random noise if no history provided
        config_track_glb[:, :, 0] = np.random.rand(D, N)
    else:
        # Chaining: Start exactly where the previous phase ended
        config_track_glb[:, :, 0] = init_config

    # 4. Control Loop
    for itr in range(max_itr):
        # Control Law: u = - gain * P * Omega^T
        control_input_glb = - gain * config_track_glb[:, :, itr] @ stress_mat_global.T
        
        # Integration
        config_track_glb[:, :, itr+1] = config_track_glb[:, :, itr] + dt * control_input_glb
        
        # Hard Constraint: Fix Leaders to Target Positions
        for leader_idx in leaders_glb_idx:
            config_track_glb[:, leader_idx, itr+1] = target_config[:, leader_idx]

    return config_track_glb


class Agent:
    def __init__(self, idx_glb: int,
                 profiles: list,
                 target_pos: np.ndarray, 
                 position: np.ndarray,   
                 is_leader: bool, 
                 max_itr: int, 
                 dt: float,):

        self.dt = dt
        self.max_itr = max_itr
        self.idx_glb = idx_glb
        self.is_leader = is_leader
        self.dim = position.shape[0] 
        self.initial_position = position.copy() # Store the initial position
        self.position = position  # current position, shape (D,)
        self.target_pos = target_pos  # target position, shape (D,)
        
        self.profiles = profiles
        # print(f"Agent {self.idx_glb} has {len(self.profiles)} profile(s)")
        # change the local neighbor id for all profiles
        for profile in self.profiles:
            profile['local_neighbors_idx'] = self.get_neighbors(profile['local_idx'], profile['stress_mat'])
            # print(f"  Profile subgraph {profile['subgraph_idx']} has neighbors {profile['local_neighbors_idx']}")

    def reset(self):
        """Resets the agent's position to its initial state."""
        self.position = self.initial_position.copy()

    def get_neighbors(self, local_idx, stress_mat):
        # based on the one stress matrices, find local IDs of neighbors
        neighbor_ID = list(np.nonzero(stress_mat[local_idx, :])[0])
        neighbor_ID.remove(local_idx)  # remove self
        return neighbor_ID

    def step(self, Zij, gain = 10.0):
        # Zij: for node i, all zijs, [D,N]
        
        # followers control
        if self.is_leader == 0:

            control_input = np.zeros(self.dim)
            # randomly select one profile to work with
            profile = self.profiles[np.random.randint(len(self.profiles))]
            
            # Get the mapping from local subgraph indices to global agent indices
            global_idx_map = profile['global_idx_map'] # <-- GET THE MAP

            for j_local in profile['local_neighbors_idx']: # <-- This is the LOCAL index
                
                # Find the corresponding global index
                j_global = global_idx_map[j_local] # <-- THIS IS THE FIX
                
                stress_ij = profile['stress_mat'][profile['local_idx'], j_local]
                
                # Use the correct GLOBAL index to access Zij

                control_input +=  gain * stress_ij * Zij[:, j_global] 
            
            #################################################################
            # implement gain compensation
            control_input  = control_input * len(self.profiles)
            #################################################################
            
            self.position = self.position + self.dt * control_input
        
        # leader fixed at target position
        else:
            self.position = self.target_pos

        return self.position
    
def multicluster_afc_distributed(target_config: np.ndarray, 
                                subgraph_glb_idx: list, 
                                leaders_glb_idx: list, 
                                stress_mat_list: list,
                                max_itr: int = 10000, 
                                dt: float = 0.01, 
                                gain: float = 10.0):
    """
    Distributed AFC for multicluster formations.
    Each agent only has access to local information from its subgraphs.

    """
    D, N = target_config.shape

    init_config = np.random.rand(D, N) * 10  # random position in 10x10 area

    # initialize agents
    agents = []
    for idx_glb in range(N):
        is_leader = 1 if idx_glb in leaders_glb_idx else 0

        # find which subgraphs this agent belongs to
        in_subgraphs = []
        idx_local = []
        for subgraph_idx, subgraph in enumerate(subgraph_glb_idx):
            if idx_glb in subgraph:
                in_subgraphs.append(subgraph_idx)
                idx_local.append(subgraph.index(idx_glb))
        
        # collect profiles for this agent
        profiles = []

        for subgraph_idx, local_idx in zip(in_subgraphs, idx_local):
            stress_mat = stress_mat_list[subgraph_idx]
            profiles.append({
                'subgraph_idx': subgraph_idx,
                'local_idx': local_idx,
                'local_neighbors_idx': None,  # to be filled in Agent class
                'stress_mat': stress_mat,  # calculate stress matrix for this subgraph
                'global_idx_map': subgraph_glb_idx[subgraph_idx]
            })

            # Fill in local_neighbors_idx for each profile

        agent = Agent(idx_glb=idx_glb,
                      profiles=profiles,
                      target_pos=target_config[:,idx_glb],
                      position=init_config[:, idx_glb],
                      is_leader=is_leader,
                      max_itr=max_itr,
                      dt=dt)
        agents.append(agent)

    print(f"All agents initialized.")

    config_track = np.zeros((D, N, max_itr+1))
    config_track[:, :, 0] = init_config  # initial configuration
    for itr in tqdm(range(max_itr)):
        for i in range(N):
            Zij = config_track[:, [i], itr] - config_track[:, :, itr]  # shape (D,N) 
            config_track[:, i, itr+1] = agents[i].step(Zij, gain=gain)

    return config_track
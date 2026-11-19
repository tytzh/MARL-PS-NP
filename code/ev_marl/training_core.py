

import math
import copy
import time
import os
import hashlib
from scipy.spatial.distance import cdist
from ddpg_agents import Agent_b,Agent_c
import scipy.io as scio
import random
import pickle
from constants import scale_nonEV,adr_Load_nonEV,path_case,step_interval,rate_CS_b,rate_CS_c,scores_average_window
from training_utils import refresh_price_nonEVload,save_show_scores, update_dur_action_soc3,\
    get_evs_group,choose_actions_mask3,get_real_actions2,get_next_statev11, save_transitions_mask,update_score_ls
from environment_entities import EV
import torch
import numpy as np

EPISODES = globals().get("EPISODES", 10)


def time_to_hours(time_str):
    
    hours, minutes, seconds = map(int, time_str.split(':'))
    
    return hours + minutes / 60 + seconds / 3600
def initialize_data(adr_case_ev,adr_datapt):
    # 1.
    list_hour = []
    n_idx_hr = 1*60*60/step_interval  
    temp = [i for i in range(int(n_idx_hr))]
    for i in range(24):
        temp_i =    [i*int(n_idx_hr) + j for j in temp]
        list_hour.append(temp_i)

    d = scio.loadmat(adr_case_ev)
    with open(adr_datapt, "rb") as file:
        d_pt2 = pickle.load(file)
    n_evs24 = d_pt2['n_evs24']
    p_n_24 = [item/sum(n_evs24) for item in n_evs24]
    p_dur_24 = d_pt2['p_dur_24']
    ev_iniSoC = d['ev_iniSoC'][0]
    ev_duration = d['ev_duration'][0]
    ev_starttime = d['ev_starttime']
    ev_starttime = [round(time_to_hours(item),1) for item in ev_starttime]
    # 2.
    price = [87.40, 78.79, 75.39, 70.21, 69.15, 75.01, 80.75, 105.53, 111.14, 99.72, 93.04, 85.57, 83.27, 81.53, 85.77,
             83.32, 91.00, 112.94, 150.32, 178.01, 169.93, 138.05, 115.50, 98.10]
    price = [i* 1.25 * 0.001 for i in price]
    n_price = len(price)
    price_exp = sum(price)/len(price)
    Load_NonEV = scio.loadmat(adr_Load_nonEV)['Load_NonEV_pu'][0] * scale_nonEV

    
    
    price_pu = get_pu(price)
    Load_NonEV_pu = get_pu(Load_NonEV)

    return  price,price_pu,price_exp,n_price,Load_NonEV,Load_NonEV_pu,list_hour,ev_iniSoC,ev_duration,ev_starttime,p_n_24,p_dur_24
def initialize_model(n_knn,prune_percent,n_group,train_seed):
    
    base_dim = 19 if os.environ.get("VX_HETEROGENEOUS_STRESS") == "1" else 18
    state_dim = (base_dim - 1 if os.environ.get("VX_NO_MASK_SPECIALIZATION") == "1"
                 else base_dim + n_group)
    n_total = [state_dim * n_knn, state_dim * n_knn]
    n_mask = n_group
    episode_scores = []
    episode_scores_b = []
    episode_scores_c = []

    window_scores = []
    window_scores_b = []
    window_scores_c = []


    state_sizes = [state_dim, state_dim]
    action_size = [1, 1]  

    agent_b = Agent_b(id_agent=0, state_size=state_sizes[0], state_total_size=n_total[0],
                    action_size=action_size[0], num_agents=1, random_seed=train_seed)
    agent_c = Agent_c(id_agent=1, state_size=state_sizes[1], state_total_size=n_total[1],
                    action_size=action_size[1], num_agents=1, random_seed=train_seed + 1)


    

    list_actor_masked = torch.tensor([])
    list_critic_masked = torch.tensor([])
    agent_b.make_prune_masks2(n_mask,prune_percent,list_actor_masked,list_critic_masked)
    list_actor_masked = torch.tensor([])
    list_critic_masked = torch.tensor([])
    agent_c.make_prune_masks2(n_mask, prune_percent, list_actor_masked, list_critic_masked)


    return agent_b, agent_c, episode_scores, episode_scores_b, episode_scores_c, window_scores, window_scores_b, window_scores_c,n_total
def initialize_episode():
    
    agent_scores_b = np.zeros(2)  # num_agents = 2
    agent_scores_c = np.zeros(2)
    
    ev_finished = []
    all_energy_ch = []
    all_energy_dch = []
    score_ls = np.array([0. for _ in range(2)])  
    n_score_ls = np.array([0. for _ in range(2)])  
    energy_ch = [0, 0]  
    energy_dch = [0, 0]


    return agent_scores_b, agent_scores_c, ev_finished, all_energy_ch, all_energy_dch \
        ,  score_ls, n_score_ls, energy_ch,energy_dch
def mask_list_EVs(feas_group,list_EVs,n_clusters):
    # =============================== C fuzzy K means ===================================
    
    m = 2  
    max_iter = 100
    tolerance = 1e-5

    
    feas_group = np.array(feas_group)
    data = feas_group
    U = np.random.dirichlet(np.ones(n_clusters), size=data.shape[0])

    def update_centers(U, data, m):
        um = U ** m
        centers = um.T @ data / np.sum(um, axis=0)[:, None]
        return centers

    def update_membership(U, centers, data, m):
        power = 2 / (m - 1)
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            new_U = 1.0 / np.sum((distances[:, :, None] / distances[:, None, :]) ** power, axis=2)
        new_U = np.nan_to_num(new_U)
        zero = distances <= np.finfo(float).eps
        zero_rows = zero.any(axis=1)
        if np.any(zero_rows):
            matches = zero[zero_rows]
            new_U[zero_rows] = matches / matches.sum(axis=1, keepdims=True)
        return new_U

    
    for iteration in range(max_iter):
        centers = update_centers(U, data, m)
        new_U = update_membership(U, centers, data, m)

        
        if np.linalg.norm(new_U - U) < tolerance:
            break

        U = new_U
    cluster_membership = np.argmax(U, axis=1)

    
    n_evs = len(list_EVs)
    for i in range(n_evs):
        U_i = U[i, :]
        cluster_i = cluster_membership[i]
        list_EVs[i].id_mask  = cluster_i
        list_EVs[i].weight_mask =  U_i

        distances = cdist([feas_group[i]], feas_group, metric='euclidean')[0]
        sorted_indices = np.argsort(distances)
        temp_i = [item for item in sorted_indices if item != i]
        list_EVs[i].idx_ev_near = temp_i


    return list_EVs,centers


    # for i in range(0, len(list_EVs0_start)):
    #     list_EVs0_start[i].id_mask = cluster_membership[i]
    #     list_EVs0_start[i].weight_mask = U[i, :]
def get_exist_t(list_inCS,list_inPT,idx_t,list_EVs):
    # list_inCS:
    EVs_in = [[],[],[]]  

    for i in range(len(list_inCS)):
        list_idxt = list_inCS[i]
        if len(list_idxt):
            list_h = list_idxt[0]
            if idx_t in list_h:
                idx = list_h.index(idx_t)
                infrai = 0
                duri = list_idxt[1][idx]
                EVs_in[0].extend([i])
                EVs_in[1].extend([duri])
                EVs_in[2].extend([infrai])

    for i in range(len(list_inPT)):
        if i in EVs_in[0]:
            continue
        list_idxt = list_inPT[i]
        if len(list_idxt):
            list_h = list_idxt[0]
            if idx_t in list_h:
                idx = list_h.index(idx_t)
                infrai = 1
                duri = list_idxt[1][idx]
                EVs_in[0].extend([i])
                EVs_in[1].extend([duri])
                EVs_in[2].extend([infrai])

    # [0] EV idx
    # [1] hr:duration
    # [2] time index from
    # [3] time index to
    # [4] index infrastructure
    return EVs_in
def get_EVs_exist24(list_inCS, list_inPT, list_EVs):
    
    EVs_exist24= []

    for idx_h in range(24):
        EVs_exist = get_exist_t(list_inCS, list_inPT, idx_h, list_EVs)
        EVs_exist24.append(EVs_exist)
    
    return EVs_exist24
def get_pu(data):
    mean = sum(data) / len(data)

    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5

    
    z_score_normalized = [(x - mean) / std_dev for x in data]

    
    z_min = min(z_score_normalized)
    z_max = max(z_score_normalized)
    min_max_normalized = [(z - z_min) / (z_max - z_min) for z in z_score_normalized]

    return min_max_normalized
def get_data_episode(n_evs,list_EVs,list_inCS, list_inPT, selection_seed=None):
    
    dual_selected = set()
    dual_pt_records = {}
    if os.environ.get("VX_SUMO_DUAL_SHARE") is not None:
        cs_pool = [i for i in range(len(list_EVs)) if list_inCS[i]]
        pt_pool = [i for i in range(len(list_EVs)) if list_inPT[i] and not list_inCS[i]]
        inactive_pool = [i for i in range(len(list_EVs))
                         if not list_inCS[i] and not list_inPT[i]]
        n_dual = round(n_evs * float(os.environ["VX_SUMO_DUAL_SHARE"]))
        n_cs_only = round(n_evs * float(os.environ.get("VX_SUMO_CS_ONLY_SHARE", "0.3")))
        n_pt_only = round(n_evs * float(os.environ.get("VX_SUMO_TER_ONLY_SHARE", "0.3")))
        n_inactive = n_evs - n_dual - n_cs_only - n_pt_only
        tight_share = float(os.environ.get("VX_SUMO_TIGHT_CS_SHARE", "0"))
        if tight_share > 0:
            tight_margin = float(os.environ.get("VX_SUMO_TIGHT_CS_MARGIN", "0.1"))
            tight_pool = []
            for idx in cs_pool:
                ev = list_EVs[idx]
                contact_h = sum(list_inCS[idx][1])
                reachable_soc = min(
                    float(ev.soc_t_b) + float(ev.p_max_b) * contact_h / float(ev.cap_b), 1.0)
                margin = reachable_soc - float(ev.soc_exp_b)
                if ev.soc_t_b < ev.soc_exp_b and 0 <= margin <= tight_margin:
                    tight_pool.append(idx)
            n_cs_total = n_dual + n_cs_only
            n_tight = round(n_cs_total * tight_share)
            if len(tight_pool) < n_tight:
                raise ValueError(
                    f"SUMO pool has {len(tight_pool)} tight-reachable CS profiles; "
                    f"{n_tight} required")
            tight_selected = random.sample(tight_pool, n_tight)
            regular_pool = [idx for idx in cs_pool if idx not in tight_selected]
            cs_selected = tight_selected + random.sample(regular_pool, n_cs_total - n_tight)
            random.shuffle(cs_selected)
        else:
            cs_selected = random.sample(cs_pool, n_dual + n_cs_only)
        pt_selected = random.sample(pt_pool, n_dual + n_pt_only)
        inactive_selected = random.sample(inactive_pool, n_inactive)
        dual_selected = set(cs_selected[:n_dual])
        dual_pt_records = dict(zip(cs_selected[:n_dual], pt_selected[:n_dual]))
        idx_ev_select = cs_selected + pt_selected[n_dual:] + inactive_selected
        random.shuffle(idx_ev_select)
    elif os.environ.get("VX_SUMO_STRATIFIED_SAMPLE") == "1":
        cs_pool = [i for i in range(len(list_EVs)) if list_inCS[i]]
        pt_pool = [i for i in range(len(list_EVs)) if list_inPT[i] and not list_inCS[i]]
        inactive_pool = [i for i in range(len(list_EVs))
                         if not list_inCS[i] and not list_inPT[i]]
        n_cs = round(n_evs * float(os.environ.get("VX_SUMO_CS_SHARE", "0.5")))
        n_pt = round(n_evs * float(os.environ.get("VX_SUMO_TER_SHARE", "0.4")))
        n_inactive = n_evs - n_cs - n_pt
        if min(n_cs, n_pt, n_inactive) < 0:
            raise ValueError("SUMO sampling shares exceed the requested EV population")
        if n_cs > len(cs_pool) or n_pt > len(pt_pool) or n_inactive > len(inactive_pool):
            raise ValueError("SUMO contact pools cannot satisfy stratified sampling")
        idx_ev_select = (random.sample(cs_pool, n_cs) +
                         random.sample(pt_pool, n_pt) +
                         random.sample(inactive_pool, n_inactive))
        random.shuffle(idx_ev_select)
    else:
        if selection_seed is not None:
            idx_ev_select = random.Random(selection_seed).sample(
                range(len(list_EVs)), n_evs)
        else:
            idx_ev_select = random.sample([i for i in range(len(list_EVs))], n_evs)
    list_EVs2 = []
    list_inCS2 = []  #
    list_inPT2 = []
    for i in range(n_evs):
        ii = idx_ev_select[i]
        # Episodes must not mutate the initial vehicle population.
        list_EVs2.append(copy.deepcopy(list_EVs[ii]))
        list_EVs2[-1]._vx_profile_id = int(ii)
        pt_record = copy.deepcopy(
            list_inPT[dual_pt_records[ii]] if ii in dual_selected else list_inPT[ii])
        if (pt_record and pt_record[0]
                and os.environ.get("VX_SUMO_TER_WINDOW_MINUTES") is not None):
            duration = float(os.environ["VX_SUMO_TER_WINDOW_MINUTES"]) / 60.0
            contact_hour = int(pt_record[0][0])
            cs_hours = set(list_inCS[ii][0]) if list_inCS[ii] else set()
            if ii in dual_selected and contact_hour in cs_hours:
                contact_hour = next(h for h in range(24) if h not in cs_hours)
            pt_record = [[contact_hour], [duration]]
        list_inPT2.append(pt_record)
        list_inCS2.append(list_inCS[ii])

    # Nested synthetic TER deployment coverage. This only adds TER availability
    # to the already sampled EV profiles; it never changes profile attributes or
    # the random streams used to construct/sample those profiles.
    coverage = os.environ.get("VX_TER_COVERAGE_LEVEL", "default").lower()
    if coverage not in ("default", "w1", "w2", "w3"):
        raise ValueError(f"Unknown TER coverage level: {coverage}")
    if coverage in ("w2", "w3"):
        target_count = 12 if coverage == "w2" else n_evs
        ter_present = [i for i, record in enumerate(list_inPT2)
                       if record and record[0]]
        # Nesting takes precedence if a sampled contact set already exceeds 60%.
        target_count = max(target_count, len(ter_present))
        candidates = [i for i in range(n_evs) if i not in set(ter_present)]
        coverage_seed = os.environ.get("VX_TER_COVERAGE_SEED", "73021")

        def stable_rank(text):
            payload = f"{coverage_seed}:{text}".encode("utf-8")
            return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

        candidates.sort(key=lambda i: stable_rank(list_EVs2[i]._vx_profile_id))
        additions = candidates[:max(0, target_count - len(ter_present))]
        donor_indices = [i for i, record in enumerate(list_inPT)
                         if record and record[0]]
        if additions and not donor_indices:
            raise ValueError("Cannot expand TER coverage: no TER time templates")
        donor_indices.sort(key=lambda i: stable_rank(("template", i)))
        for rank, i in enumerate(additions):
            donor = list_inPT[donor_indices[rank % len(donor_indices)]]
            contact_hour = int(donor[0][0])
            duration = (float(os.environ["VX_SUMO_TER_WINDOW_MINUTES"]) / 60.0
                        if os.environ.get("VX_SUMO_TER_WINDOW_MINUTES") is not None
                        else float(donor[1][0]))
            list_inPT2[i] = [[contact_hour], [duration]]

    for ev, cs_record, pt_record in zip(list_EVs2, list_inCS2, list_inPT2):
        ev._vx_remaining_contact_h = sum(cs_record[1] if cs_record else []) + sum(
            pt_record[1] if pt_record else [])

    EVs_exist_24_2 = get_EVs_exist24(list_inCS2, list_inPT2, list_EVs2)

    return list_EVs2,list_inCS2,list_inPT2,EVs_exist_24_2


def apply_infrastructure_scenario(list_inCS, list_inPT, scenario):
    """Disable unavailable service events without regenerating EV demand."""
    if scenario == "hybrid":
        return list_inCS, list_inPT
    if scenario == "cs_only":
        return list_inCS, [[] for _ in list_inPT]
    if scenario == "ter_only":
        return [[] for _ in list_inCS], list_inPT
    raise ValueError(f"Unknown infrastructure scenario: {scenario}")


def resolve_controlled_contact_conflicts(list_inCS, list_inPT):
    """Give wired CS fixed priority when both contacts occupy the same hour."""
    resolved_pt = copy.deepcopy(list_inPT)
    for i, (cs_record, ter_record) in enumerate(zip(list_inCS, resolved_pt)):
        if not cs_record or not ter_record:
            continue
        cs_hours = set(cs_record[0])
        keep = [j for j, hour in enumerate(ter_record[0]) if hour not in cs_hours]
        ter_record[0] = [ter_record[0][j] for j in keep]
        if len(ter_record) > 1:
            ter_record[1] = [ter_record[1][j] for j in keep]
        if not ter_record[0]:
            resolved_pt[i] = []
    return list_inCS, resolved_pt


def synchronize_infrastructure_state(list_EVs, list_inCS, list_inPT):
    """Make duration features reflect only currently enabled contacts."""
    for ev, cs_record, pt_record in zip(list_EVs, list_inCS, list_inPT):
        records = [record for record in (cs_record, pt_record)
                   if record and record[0]]
        durations = [duration for record in records for duration in record[1]]
        hours = [hour for record in records for hour in record[0]]
        remaining = float(sum(durations))
        ev.dur_h = remaining
        ev._vx_remaining_contact_h = remaining
        if hours:
            ev.t_start = min(hours)
    return list_EVs
def calculate_hourly_duration(start_time, duration):
    
    start_hour = math.floor(start_time)
    start_minute_fraction = start_time - start_hour

    
    hourly_stamps = []
    hourly_durations = []

    
    remaining_duration = duration
    current_time = start_time

    while remaining_duration > 1e-10:
        
        current_hour = math.floor(current_time)
        time_in_current_hour = min(1 - (current_time - current_hour), remaining_duration)

        
        rounded_duration = round(time_in_current_hour, 2)
        if rounded_duration > 0:
            hourly_stamps.append(current_hour)
            hourly_durations.append(rounded_duration)

        
        remaining_duration -= time_in_current_hour
        current_time = current_hour + 1

    return [hourly_stamps, hourly_durations]


def initialization(n_group):
    # constant
    n_evs = 5000
         
    i_case = 1
    adr_case_ev = path_case + str(i_case) + '/data_ev.mat'
    adr_datapt = path_case + str(i_case) + '/data_pt.pkl'
    price, price_pu, price_exp, n_price, Load_NonEV, Load_NonEV_pu, list_hour, ev_iniSoC, ev_duration, \
    list_ev_starttime,p_n_24,p_dur_24 = initialize_data(adr_case_ev,adr_datapt)

    price_min = min(price)
    price_max = max(price)
    price_exp_max = (price_exp - price_min)
    price_exp_min = (price_exp - price_max)
    price_degrad = 100 * 0.000125 / 0.2  
    price_deg_max = max(price) - price_exp - price_degrad
    price_deg_min = min(price) - price_exp - price_degrad
    price_trade_min = min(price_deg_min, price_exp_min)
    price_trade_max = max( price_deg_max, price_exp_max)

    stress_scenario = os.environ.get("VX_HETEROGENEOUS_STRESS") == "1"
    # =========================
    
    # =========================

    # 0.2 0.3 0.4
    EV_TYPE_CONFIGS = {
        0: {  
            "soc_exp": 0.5,
            "w": [0.3, 0.7],
            "price_threshold_pu": 0.1
        },
        1: {  
            "soc_exp": 0.90,
            "w": [0.7, 0.3],
            "price_threshold_pu": 0.3
        },
        2: {  
            "soc_exp": 0.75,
            "w": [0.5, 0.5],
            "price_threshold_pu": 0.2
        }
    }

    
    EV_TYPE_PROBS = [0.3, 0.3, 0.4]
    EV_TYPE_IDS = list(EV_TYPE_CONFIGS.keys())

    # =========================
    
    # =========================
    list_EVs = []
    feas_group = []
    list_inCS = [[] for _ in range(n_evs)]
    list_inPT = [[] for _ in range(n_evs)]

    for i in range(n_evs):
        
        soc_ini_b = round(random.random(), 1)
        soc_ini_c = round(random.random(), 1)

        cap_b = 60

        
        ev_type = np.random.choice(EV_TYPE_IDS, p=EV_TYPE_PROBS)
        config = EV_TYPE_CONFIGS[ev_type]
        if stress_scenario:
            soc_ini_b = (0.2, 0.5, 0.8)[ev_type]
            soc_ini_c = (0.15, 0.5, 0.85)[ev_type]
            if ev_type == 0 and os.environ.get("VX_TRADER_INITIAL_SOC") is not None:
                soc_ini_b = float(os.environ["VX_TRADER_INITIAL_SOC"])

        
        fixed_target = os.environ.get("VX_FIXED_TARGET_SOC")
        target_scale = float(os.environ.get("VX_TARGET_SOC_SCALE", "1"))
        soc_exp_b = (float(fixed_target) if fixed_target is not None else
                     float(np.clip(config["soc_exp"] * target_scale, 0.05, 1.0)))

        fixed_threshold = os.environ.get("VX_FIXED_PRICE_THRESHOLD")
        threshold_scale = float(os.environ.get("VX_PRICE_THRESHOLD_SCALE", "1"))
        price_threshold_pu = (float(fixed_threshold) if fixed_threshold is not None else
                              float(np.clip(config["price_threshold_pu"] * threshold_scale, 0.0, 1.0)))
        price_threshold = price_min + price_threshold_pu * (price_max - price_min)

        fixed_w_soc = os.environ.get("EV_FIXED_W_SOC")
        if fixed_w_soc is not None:
            w = [float(fixed_w_soc), 1.0 - float(fixed_w_soc)]
        elif os.environ.get("VX_FIXED_PRICE_WEIGHT") is not None:
            fixed_price = float(os.environ["VX_FIXED_PRICE_WEIGHT"])
            w = [1.0 - fixed_price, fixed_price]
        else:
            price_weight_scale = float(os.environ.get("VX_PRICE_WEIGHT_SCALE", "1"))
            weighted_price = config["w"][1] * price_weight_scale
            total_weight = config["w"][0] + weighted_price
            w = [config["w"][0] / total_weight, weighted_price / total_weight]
        w_soc = w[0]
        w_price = w[1]

        
        
        preference_feature = [
            soc_exp_b,
            price_threshold_pu,
            w_soc,
            w_price
            
        ]

        list_EVs.append(
            EV(
                soc_ini_b,
                soc_exp_b,
                cap_b,
                rate_CS_b,
                rate_CS_c,
                soc_ini_c,
                price_threshold,
                price_threshold_pu,
                w
            )
        )
        list_EVs[-1].vx_type = int(ev_type)
        if stress_scenario:
            list_EVs[-1].cap_c = (2.0, 4.0, 6.0)[ev_type]
            preference_feature.append(list_EVs[-1].cap_c / 6.0)

        feas_group.append(preference_feature)


    infrastructure_assignment = globals().get("INFRASTRUCTURE_ASSIGNMENT", "exclusive")
    p_cs = 0.7 if stress_scenario else 0.3
    if infrastructure_assignment not in ("exclusive", "dual_access", "controlled_overlap"):
        raise ValueError(f"Unknown infrastructure assignment: {infrastructure_assignment}")
    n_cs = n_evs if infrastructure_assignment == "dual_access" else int(n_evs * p_cs)
    if infrastructure_assignment == "dual_access":
        n_pt = n_evs
    elif infrastructure_assignment == "controlled_overlap":
        n_pt = int(n_evs * 0.3)
    else:
        n_pt = n_evs - n_cs
    idx_evs = [i for i in range(n_evs)]
    idx_incs = list(idx_evs) if infrastructure_assignment == "dual_access" else random.sample(idx_evs, n_cs)
    if infrastructure_assignment == "dual_access":
        idx_inpt = list(idx_evs)
    elif infrastructure_assignment == "controlled_overlap":
        idx_inpt = random.sample(idx_evs, n_pt)
    else:
        idx_inpt = list(set(idx_evs) - set(idx_incs))

    idx_incs_guoye = random.sample(idx_incs,int(n_cs*0.1))
    idx_incs_baitian = random.sample(list(set(idx_incs) - set(idx_incs_guoye)), int(n_cs * 0.1))

    
    for incsi in idx_incs:
        duri =  round(ev_duration[incsi],1)
        t_starti = list_ev_starttime[incsi]

        if stress_scenario:
            duri = min((2.0, 5.0, 9.0)[list_EVs[incsi].vx_type], 24 - t_starti)

        if incsi in idx_incs_guoye:
            duri = 24-t_starti
        if incsi in idx_incs_baitian:
            t_starti = 0

        list_EVs[incsi].dur_h = duri
        list_EVs[incsi].t_start = t_starti
        tempt = calculate_hourly_duration(t_starti, duri)
        list_inCS[incsi] = tempt

    
    hours_inpt = np.random.choice(range(24), size=n_pt, p=p_n_24)
    xiaoshu_inpt = np.random.uniform(0, 1, n_pt)
    time_inpt = [hri + round(mini,1) for (hri,mini) in zip(hours_inpt,xiaoshu_inpt)]

    dur_inpt = []
    for i in range(len(hours_inpt)):
        hi = hours_inpt[i]
        durations, probs = zip(*p_dur_24[hi].items())
        duration = random.choices(durations, weights=probs, k=1)[0]
        dur_inpt.append(duration)

    for i in range(len(idx_inpt)):
        inpti = idx_inpt[i]
        duri = dur_inpt[i]/3600
        t_starti = time_inpt[i]
        if stress_scenario:
            duri = min((1.0, 3.0, 6.0)[list_EVs[inpti].vx_type], 24 - t_starti)
        list_EVs[inpti].dur_h = duri
        list_EVs[inpti].t_start = t_starti
        tempt = calculate_hourly_duration(t_starti, duri)
        list_inPT[inpti] = tempt
    if infrastructure_assignment in ("dual_access", "controlled_overlap"):
        # Preserve both independently sampled opportunity sets. Execution applies
        # fixed wired priority to same-hour overlaps after scenario availability.
        for i in idx_evs:
            active_hours = ((list_inCS[i][0] if list_inCS[i] else [])
                            + (list_inPT[i][0] if list_inPT[i] else []))
            active_durations = ((list_inCS[i][1] if list_inCS[i] else [])
                                + (list_inPT[i][1] if list_inPT[i] else []))
            if active_hours:
                list_EVs[i].t_start = min(active_hours)
                list_EVs[i].dur_h = sum(active_durations)

    EVs_exist_24 = get_EVs_exist24(list_inCS, list_inPT, list_EVs)
    list_EVs, centers = mask_list_EVs(feas_group, list_EVs, n_group)

    return price, price_pu, Load_NonEV, Load_NonEV_pu,price_exp_max,price_exp_min,price_max,price_min,price_degrad,\
           price_trade_max,price_trade_min,list_EVs,list_inCS,list_inPT,centers,EVs_exist_24
def add_reward(reward_history,reward):
    reward_history.append(reward)
    if len(reward_history) > 100:  
        reward_history.pop(0)
    return reward_history
def is_outlier(reward_history,reward):
    if len(reward_history) < 100:  
        return False
    mean = np.mean(reward_history)
    threshold = 0.8*abs(mean)
    diff = reward - mean  
    if diff <0:
        diff_abs = abs(diff)
        if diff_abs > threshold:
            return True
        else:
            return False
    else:
        return False


def save_show_scores(score_ls,n_score_ls,episode_scores,episode_scores_b,episode_scores_c,i_episode,
                     scores_average_window,window_scores,window_scores_b,window_scores_c,scores_mask,episode_scores_mask):
    score_ls_mean = np.divide(score_ls, n_score_ls,
                              out=np.full(2, np.nan), where=n_score_ls > 0)
    episode_scores.append(np.max(score_ls_mean))
    episode_scores_b.append(score_ls_mean[0])
    episode_scores_c.append(score_ls_mean[1])
    average_score = np.mean(episode_scores[-scores_average_window:])
    average_score_b = np.mean(episode_scores_b[-scores_average_window:])
    average_score_u = np.mean(episode_scores_c[-scores_average_window:])

    window_scores.append(average_score)
    window_scores_b.append(average_score_b)
    window_scores_c.append(average_score_u)

    print(f"Episode {i_episode}/{EPISODES} | Battery reward: {score_ls_mean[0]:.4f}"
          f" | Capacitor reward: {score_ls_mean[1]:.4f}", flush=True)


    list_averages = [np.mean(sublist) if len(sublist) else np.nan for sublist in scores_mask]
    episode_scores_mask.append(list_averages)

    return episode_scores,episode_scores_b,episode_scores_c,window_scores,window_scores_b,window_scores_c,episode_scores_mask


if __name__ == "__main__":
    # ============================================================
    
    # ============================================================

    train_seed = globals().get("TRAIN_SEED", 3)
    infrastructure_scenario = globals().get("INFRASTRUCTURE_SCENARIO", "hybrid")
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    n_evs = globals().get("TRAIN_EVS", 100)
    n_group = globals().get("N_GROUP", 3)
    n_knn = globals().get("N_KNN", 5)
    prune_percent = globals().get("PRUNE_PERCENT", 10)

    reward_history_b = []  
    reward_history_c = []
    reward_history_bm = [[] for _ in range(n_group)]

    k = 2  

    (
        price,
        price_pu,
        Load_NonEV,
        Load_NonEV_pu,
        price_exp_max,
        price_exp_min,
        price_max,
        price_min,
        price_degrad,
        price_trade_max,
        price_trade_min,
        list_EVs0,
        list_inCS0,
        list_inPT0,
        centers,
        EVs_exist_24,
    ) = initialization(n_group)

    (
        agent_b,
        agent_c,
        episode_scores,
        episode_scores_b,
        episode_scores_c,
        window_scores,
        window_scores_b,
        window_scores_c,
        n_total,
    ) = initialize_model(n_knn, prune_percent, n_group, train_seed)

    episode_scores_mask = []
    episode_filtered_samples = []
    episode_total_samples = []
    episode_returns_b = []
    episode_returns_c = []
    reward_filter_enabled = globals().get("REWARD_FILTER", False)
    start_time = time.perf_counter()
    tt0 = time.time()

    # ============================================================
    
    # ============================================================

    for i_episode in range(1, EPISODES + 1):
        agent_b.vx_current_episode = i_episode
        agent_c.vx_current_episode = i_episode
        episode_filtered_samples.append(0)
        episode_total_samples.append(0)
        # --------------------------------------------------------
        
        # --------------------------------------------------------


        list_EVs, list_inCS, list_inPT, EVs_exist24 = get_data_episode(
            n_evs,
            list_EVs0,
            list_inCS0,
            list_inPT0,
            selection_seed=(train_seed * 100000 + i_episode
                            if globals().get("INFRASTRUCTURE_ASSIGNMENT", "exclusive")
                            == "controlled_overlap" else None),
        )
        list_inCS, list_inPT = apply_infrastructure_scenario(
            list_inCS, list_inPT, infrastructure_scenario)
        if globals().get("INFRASTRUCTURE_ASSIGNMENT", "exclusive") == "controlled_overlap":
            list_inCS, list_inPT = resolve_controlled_contact_conflicts(list_inCS, list_inPT)
        list_EVs = synchronize_infrastructure_state(
            list_EVs, list_inCS, list_inPT)
        EVs_exist24 = get_EVs_exist24(list_inCS, list_inPT, list_EVs)

        
        agent_b.reset()
        agent_c.reset()

        agent_b.initial_state_dict()  
        agent_c.initial_state_dict()  

        (
            agent_scores_b,
            agent_scores_c,
            ev_finished,
            all_energy_ch,
            all_energy_dch,
            score_ls,
            n_score_ls,
            energy_ch,
            energy_dch,
        ) = initialize_episode()

        scores_mask = [[] for _ in range(n_group)]

        # --------------------------------------------------------
        
        # --------------------------------------------------------

        for idx_h in range(24):
            EVs_exist = EVs_exist24[idx_h]
            idx_h_next = idx_h + 1

            if idx_h != 23:
                EV_exist_next = EVs_exist24[idx_h_next]
            else:
                EV_exist_next = (get_exist_t(list_inCS, list_inPT, 24, list_EVs)
                                 if globals().get("CORRECTED_VX", False) else EVs_exist24[0])

            n_exist = len(EVs_exist[0])

            if n_exist == 0:
                continue

            
            EVs_done = list(set(EVs_exist[0]) - set(EV_exist_next[0]))
            if globals().get("INFRASTRUCTURE_ASSIGNMENT", "exclusive") in ("dual_access", "controlled_overlap"):
                EVs_done = [i for i in EVs_done if not any(
                    h > idx_h for record in (list_inCS[i], list_inPT[i]) if record
                    for h in record[0])]
            done_ls = [i in EVs_done for i in EVs_exist[0]]
            if globals().get("INFRASTRUCTURE_ASSIGNMENT", "exclusive") == "controlled_overlap":
                EVs_soc_done = [
                    ev_id for ev_id in EVs_exist[0]
                    if not any(hour > idx_h for record in (list_inCS[ev_id], list_inPT[ev_id])
                               if record for hour in record[0])
                ]
            else:
                EVs_soc_done = [
                    ev_id for ev_id, infra in zip(EVs_exist[0], EVs_exist[2])
                    if infra == 0 and not any(
                        hour > idx_h for hour in
                        (list_inCS[ev_id][0] if list_inCS[ev_id] else []))
                ]

            # ----------------------------------------------------
            
            # ----------------------------------------------------

            (
                price_seq,
                price_pu_seq,
                Load_NonEV_seq,
                Load_NonEV_pu_seq,
            ) = refresh_price_nonEVload(
                idx_h,
                len(price),
                price,
                price_pu,
                Load_NonEV,
                Load_NonEV_pu,
            )

            # ----------------------------------------------------
            
            # ----------------------------------------------------

            list_EVs, feas_c, feas_b = get_evs_group(
                EVs_exist,
                list_EVs,
                price_pu_seq,
                Load_NonEV_pu_seq,
                n_knn,
            )

            # ----------------------------------------------------
            
            
            
            # ----------------------------------------------------

            (
                actions_b,
                actions_c,
                actions_b_pu,
                actions_c_pu,
                mark_ls,
                weight_ls,
            ) = choose_actions_mask3(
                EVs_exist,
                list_EVs,
                feas_c,
                feas_b,
                agent_b,
                agent_c,
                rate_CS_b,
                rate_CS_c,
                i_episode,
                n_group,
            )

            actions_b_real, actions_c_real, list_EVs = get_real_actions2(
                EVs_exist,
                list_EVs,
                actions_b,
                actions_c,
            )

            list_EVs = update_dur_action_soc3(
                n_exist,
                EVs_exist,
                actions_b_real,
                actions_c_real,
                list_EVs,
                price_seq[0],
                actions_b_pu,
                actions_c_pu,
            )

            # ----------------------------------------------------
            
            # ----------------------------------------------------

            (
                obs_next_ls_b,
                obs_next_ls_c,
                reward_ls_b,
                reward_ls_c,
                energy_ch,
                energy_dch,
                reward_ls_bm,
            ) = get_next_statev11(
                idx_h_next,
                len(price),
                price,
                price_pu,
                Load_NonEV,
                Load_NonEV_pu,
                list_EVs,
                EVs_done,
                energy_ch,
                energy_dch,
                price_seq,
                EVs_exist,
                EV_exist_next,
                price_min,
                price_max,
                price_exp_min,
                price_exp_max,
                price_trade_min,
                price_trade_max,
                price_degrad,
                n_group,
                EVs_soc_done,
            )

            # ----------------------------------------------------
            
            
            # ----------------------------------------------------

            if not (np.isfinite(reward_ls_b).all() and np.isfinite(reward_ls_c).all()):
                raise FloatingPointError(f"Non-finite reward: episode={i_episode}, hour={idx_h}")
            # Record actual rewards before filtering replay-buffer samples.
            episode_total_samples[-1] += n_exist
            score_ls, n_score_ls = update_score_ls(
                score_ls, n_score_ls, reward_ls_b, reward_ls_c, n_exist)
            for kk, ev_id in enumerate(EVs_exist[0]):
                scores_mask[list_EVs[ev_id].id_mask].append(reward_ls_b[kk])
            all_energy_ch.append(energy_ch)
            all_energy_dch.append(energy_dch)
            mean_reward_b = np.mean(reward_ls_b)
            mean_reward_c = np.mean(reward_ls_c)
            mean_reward_bm = [np.mean(item) if len(item) else np.nan for item in reward_ls_bm]

            if not reward_filter_enabled or i_episode < 10:
                reward_history_b = add_reward(reward_history_b, mean_reward_b)
                reward_history_c = add_reward(reward_history_c, mean_reward_c)

                for jj in range(n_group):
                    if not np.isnan(mean_reward_bm[jj]):
                        reward_history_bm[jj] = add_reward(
                            reward_history_bm[jj],
                            mean_reward_bm[jj],
                        )

            else:
                out_b = is_outlier(reward_history_b, mean_reward_b)
                out_c = is_outlier(reward_history_c, mean_reward_c)

                flag_out_m = []

                for jj in range(n_group):
                    if not np.isnan(mean_reward_bm[jj]):
                        out_jj = is_outlier(
                            reward_history_bm[jj],
                            mean_reward_bm[jj],
                        )
                        flag_out_m.append(out_jj)

                if out_b or out_c or any(flag_out_m):
                    episode_filtered_samples[-1] += n_exist
                    continue

                reward_history_b = add_reward(reward_history_b, mean_reward_b)
                reward_history_c = add_reward(reward_history_c, mean_reward_c)

                for jj in range(n_group):
                    if not np.isnan(mean_reward_bm[jj]):
                        reward_history_bm[jj] = add_reward(
                            reward_history_bm[jj],
                            mean_reward_bm[jj],
                        )

            # ----------------------------------------------------
            
            # ----------------------------------------------------


            # ----------------------------------------------------
            
            # ----------------------------------------------------

            actions_b_before = []
            actions_c_before = []

            for i in EVs_exist[0]:
                evi = list_EVs[i]
                actionbi = evi.actions_b_pu
                actionci = evi.actions_c_pu

                n_actions = len(actionbi)

                if n_actions == 1:
                    actions_b_before.append(0)
                    actions_c_before.append(0)
                else:
                    actions_c_before.append(actionbi[-2])
                    actions_b_before.append(actionbi[-2])

            # ----------------------------------------------------
            
            # ----------------------------------------------------

            agent_c, agent_b = save_transitions_mask(
                agent_b,
                agent_c,
                feas_b,
                feas_c,
                obs_next_ls_b,
                obs_next_ls_c,
                actions_b_pu,
                actions_c_pu,
                n_exist,
                reward_ls_b,
                reward_ls_c,
                n_knn,
                mark_ls,
                EVs_exist,
                EV_exist_next,
                list_EVs,
                done_ls,
                weight_ls,
                n_total,
            )

        from actor_critic_consistency import finish_episode_constraint_metrics
        finish_episode_constraint_metrics()

        # --------------------------------------------------------
        
        # --------------------------------------------------------

        (
            episode_scores,
            episode_scores_b,
            episode_scores_c,
            window_scores,
            window_scores_b,
            window_scores_c,
            episode_scores_mask,
        ) = save_show_scores(
            score_ls,
            n_score_ls,
            episode_scores,
            episode_scores_b,
            episode_scores_c,
            i_episode,
            scores_average_window,
            window_scores,
            window_scores_b,
            window_scores_c,
            scores_mask,
            episode_scores_mask,
        )
        episode_returns_b.append(float(score_ls[0]))
        episode_returns_c.append(float(score_ls[1]))


    # ============================================================
    
    # ============================================================

    end_time = time.perf_counter()
    total_time = end_time - start_time


    


import copy
import os
import random

import numpy as np
import torch
from constants import step_interval

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_EVs_exist24(list_inCS_start, list_inPT_start, list_EVs0_start, list_hour):
    
    EVs_exist24_start = []

    for idx_h in range(24):
        EVs_exist_start = get_exist_t(list_inCS_start, list_inPT_start, idx_h, list_EVs0_start, list_hour)
        EVs_exist24_start.append(EVs_exist_start)
    
    return EVs_exist24_start


def get_exist_t(list_inCS,list_inPT,idx_t,list_EVs,list_hour):
    # list_inCS:
    EVs_in = [[],[],[],[],[],[],[]]  

    for i in range(len(list_inCS[0])):
        list_idxt = list_inCS[1][i]
        idxt_in = list(set(list_hour[idx_t])&set(list_idxt ))
        if len(idxt_in):
            EV_idx_i = list_inCS[0][i]
            dur_hr_i = len(idxt_in)*step_interval/3600
            fromi = list_idxt[0]
            toi = list_idxt[-1]
            infrai = 0
            EVs_in[0].extend([EV_idx_i])
            EVs_in[1].extend([dur_hr_i])
            EVs_in[2].extend([fromi])
            EVs_in[3].extend([toi])
            EVs_in[4].extend([infrai])
            EVs_in[5].extend([0])

    
    for i in range(len(list_inPT)):
        list_idxt = list_inPT[i][0]
        idxt_in = list(set(list_hour[idx_t]) & set(list_idxt))
        if len(idxt_in):
            EV_idx_i = i
            dur_hr_i = len(list_idxt)*step_interval/3600
            fromi = list_idxt[0]
            toi = list_idxt[-1]
            distance_i = list_inPT[i][1][-1]

            infrai = 1

            EVs_in[0].extend([EV_idx_i])
            EVs_in[1].extend([dur_hr_i])
            EVs_in[2].extend([fromi])
            EVs_in[3].extend([toi])
            EVs_in[4].extend([infrai])
            EVs_in[5].extend([distance_i])

    # [0] EV idx
    # [1] hr:duration
    # [2] time index from
    # [3] time index to
    # [4] index infrastructure
    return EVs_in
def refresh_price_nonEVload(t_step,n_price,price,price_pu,Load_NonEV,Load_NonEV_pu):
    
    
    
    # 3) Non-EVload_y
    # 4) non-EVload_squ

    
    if t_step == 0:
        idx_seq = range(0,n_price)
    elif t_step == n_price-1:
        idx_seq = [n_price-1]
        idx_seq.extend(range(0,n_price-1))
    else:
        idx_seq = []
        idx_seq.extend(range(t_step,n_price))
        idx_seq.extend(range(0,t_step))

    # ---> price
    price_seq = [price[i] for i in idx_seq]
    price_pu_seq = [price_pu[i] for i in idx_seq]

    # --> non-EV load
    Load_NonEV_seq = [Load_NonEV[i] for i in idx_seq]
    Load_NonEV_pu_seq =  [ Load_NonEV_pu[i] for i in idx_seq]

    return price_seq,price_pu_seq, Load_NonEV_seq,Load_NonEV_pu_seq


def refresh_EVs_features(EVs_exist,list_EVs,price_seq,Load_NonEV_pu_seq):
    
    n_exist = len(EVs_exist[0])
    n_evs = len(list_EVs)
    feas_grouping = []
    feas_capacitor = []
    feas_battery = []
    p_exist = n_exist/n_evs
    if n_exist == 0:
        p_0 = 0
        p_1 = 0
    else:
        p_0 = EVs_exist[2].count(1)/n_exist
        p_1 = EVs_exist[2].count(0)/n_exist

    for i in range(n_exist):
        idx_i= EVs_exist[0][i]
        hr_i = EVs_exist[1][i]
        infra_i = EVs_exist[2][i]  
        dur_i = list_EVs[idx_i].dur_h

        list_EVs[idx_i].refresh_soc()  
        # 1） features for battery
        list_EVs[idx_i].get_fea_b_and_c_iddpg(price_seq,Load_NonEV_pu_seq, infra_i,hr_i,dur_i)
        # list_EVs[idx_i].get_fea_b_and_c2(price_seq, Load_NonEV_pu_seq, infra_i, hr_i, dur_i, p_exist, p_0, p_1)
        list_EVs[idx_i].get_fea_grouping()
        
        feas_grouping.append(list_EVs[idx_i].fea_grouping)
        feas_capacitor.append(list_EVs[idx_i].fea_capacitor)
        feas_battery.append(list_EVs[idx_i].fea_battery)
    return list_EVs, feas_grouping,feas_capacitor,feas_battery
def get_evs_group(EVs_exist,list_EVs,price_seq,Load_NonEV_pu_seq,n_knn):
    
    list_EVs, feas_group,feas_capacitor,feas_battery \
        = refresh_EVs_features(EVs_exist,list_EVs,price_seq,Load_NonEV_pu_seq)
    return  list_EVs, feas_capacitor,feas_battery


def get_mask_critic(agent,idx_mark):
    initial_state_dict= agent.dict_critic
    mask_temp = agent.critic_masks[idx_mark]
    model = agent.critic_local

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    return agent  
def get_mask_actor(agent,idx_mark):
    initial_state_dict= agent.dict_actor
    mask_temp = agent.actor_masks[idx_mark]
    model = agent.actor_local

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    return agent  
def choose_actions_mask3(EVs_exist,list_EVs,feas_capacitor,feas_battery,agent_b,agent_c,rate_CS_b,rate_CS_c,i_episode,n_group):

    # -------------------------------------------------------------------
    
    actions_b = []
    actions_c = []
    actions_b_pu = []
    actions_c_pu = []
    mark_ls = []
    weight_ls = []
    idx_weight_ls = []
    n_exist = len(EVs_exist[0])
    for i in range(n_exist):  

        
        fea_capacitor = np.array(feas_capacitor[i])
        fea_capacitor = fea_capacitor.reshape(1,len(fea_capacitor))
        fea_battery = np.array(feas_battery[i])
        fea_battery = fea_battery .reshape(1,len(fea_battery))

        
        
        idx_ev = EVs_exist[0][i]
        idx_mark = list_EVs[idx_ev].id_mask  
        mark_ls.append(idx_mark)
        weight_u = list_EVs[idx_ev].weight_mask  
        weight_ls.append(weight_u)

        
        
        a_p_b = 0
        a_p_c = 0
        for j in range(n_group):
            action_p_battery = agent_b.act_mask2(fea_battery,j,add_noise=True)
            action_p_capacitor = agent_c.act_mask2(fea_capacitor,j,add_noise=True)
            a_p_b = a_p_b + weight_u[j]* action_p_battery
            a_p_c = a_p_c + weight_u[j] * action_p_capacitor

        epsilon = max([0.001,0.5*(0.95**(i_episode))])
        if np.random.rand() < epsilon:
            a_p_b = random.uniform(-1, 1)
            a_p_c = random.uniform(-1, 1)
        else:
            a_p_b = a_p_b[0][0]
            a_p_c = a_p_c [0][0]
        actions_b_pu.append(a_p_b)
        actions_c_pu.append(a_p_c)
        action_p_battery = a_p_b * rate_CS_b
        action_p_capacitor = a_p_c * rate_CS_c

        # for battery
        actions_b.append(action_p_battery)
        actions_c.append(action_p_capacitor)

    return actions_b,actions_c, actions_b_pu, actions_c_pu, mark_ls , weight_ls
def get_real_actions2(EVs_exist,list_EVs,actions_b,actions_c):
    n_exist = len(EVs_exist[0])
    actions_b_real = []
    actions_c_real = []

    for i in range(n_exist):
        idx_ev = EVs_exist[0][i]
        hr_ev = EVs_exist[1][i]  
        cap_b = list_EVs[idx_ev].cap_b
        cap_c = list_EVs[idx_ev].cap_c

        
        actions_b0 = list_EVs[idx_ev].actions_b  
        actions_c0 = list_EVs[idx_ev].actions_c
        t_actions_b0 = list_EVs[idx_ev].hr_b
        t_actions_c0 = list_EVs[idx_ev].hr_c

        
        socb0 = list_EVs[idx_ev].soc_ini_b
        socc0 = list_EVs[idx_ev].soc_ini_c
        list_soc_b = [socb0]
        list_soc_c = [socc0]
        l_energy_action_b = [x * y for x, y in zip(t_actions_b0, actions_b0)]
        l_energy_action_c = [x * y for x, y in zip(t_actions_c0, actions_c0)]
        

        for jj in range(len(t_actions_b0)):   #
            
            e_bi = l_energy_action_b[jj]
            e_ci = l_energy_action_c[jj]
            soc_bi = e_bi/cap_b
            soc_ci = e_ci/cap_c

            
            soc_bi_new = min([1,list_soc_b[-1] + soc_bi])
            soc_ci_new = min([1,list_soc_c[-1] + soc_ci])

            
            list_soc_b.append(soc_bi_new)
            list_soc_c.append(soc_ci_new)
        list_EVs[idx_ev].list_soc_b = list_soc_b
        list_EVs[idx_ev].list_soc_c = list_soc_c

        
        soc_remain = 0.01
        e_remain_b = soc_remain * cap_b
        e_remain_c = soc_remain * cap_c
        a_ch_b_max = round((cap_b - list_soc_b[-1]*cap_b - e_remain_b)/hr_ev,1)
        a_dch_b_max = -round(list_soc_b[-1] * cap_b/hr_ev - e_remain_b, 1)
        a_ch_c_max = round((cap_c - list_soc_c[-1]*cap_c - e_remain_c )/hr_ev,1)
        a_dch_c_max = -round(list_soc_c[-1] * cap_c/hr_ev - e_remain_c , 1)

        
        action_bi = actions_b[i]
        action_ci = actions_c[i]

        # ---> b
        if action_bi > 0:
            action_bi = min([action_bi,a_ch_b_max])
        else:
            action_bi = max([action_bi,a_dch_b_max])
        if action_ci > 0:
            action_ci = min([action_ci,a_ch_c_max])
        else:
            action_ci = max([action_ci,a_dch_c_max])

        if action_bi*actions_b[i]<0:
            action_bi = 0
        if action_ci*actions_c[i]<0:
            action_ci = 0


        actions_b_real.append(round(action_bi,1))
        actions_c_real.append(round(action_ci,1))

    return actions_b_real,actions_c_real,list_EVs


def get_next_statev11(idx_h_next,n_price,price,price_pu,Load_NonEV,Load_NonEV_pu,list_EVs,EVs_done,energy_ch,
                     energy_dch,price_seq,EVs_exist,EVs_exist_next,price_min,price_max,price_exp_min,price_exp_max,price_trade_min,price_trade_max,price_degrad,n_group,
                     EVs_soc_done=None):
    n_exist = len(EVs_exist[0])
    n_exist_next = len(EVs_exist_next[0])

    if n_exist_next == 0:
        p_exist_next = 0
        p_0_next = 0
        p_1_next = 0
    else:
        p_exist_next =  n_exist_next/len(list_EVs)
        p_0_next = EVs_exist_next[2].count(0)/n_exist_next
        p_1_next = EVs_exist_next[2].count(1)/n_exist_next
    p_next = [p_exist_next,p_0_next,p_1_next]
    obs_next_ls_b =[]
    reward_ls_b = []
    obs_next_ls_c =[]
    reward_ls_c = []
    idx_ev_trans = [[],[]]
    id_ev_trans = [[],[]]
    result_b = []
    result_c = []
    reward_ls_bm = [[] for _ in range(n_group)]

    price_seq_next, price_pu_seq_next, Load_NonEV_seq_next, Load_NonEV_pu_seq_next = refresh_price_nonEVload(
        idx_h_next,n_price,price,price_pu,Load_NonEV,Load_NonEV_pu)

    for i in range(n_exist):
        # ================================= states/observations ================================================
        idx_i = EVs_exist[0][i]
        hr_i = EVs_exist[1][i]
        infra_i = EVs_exist[2][i]
        dur_hi = list_EVs[idx_i].dur_h

        trans_i = 0
        flag_done = idx_i in EVs_done
        flag_soc_done = (idx_i in EVs_soc_done
                         if EVs_soc_done is not None else flag_done)
        price_t = price_seq[0]

        idx_ev_trans[trans_i].append(i)  
        id_ev_trans[trans_i].append(idx_i)  

        
        fea_b,fea_c = list_EVs[idx_i].get_fea_b_and_c_iddpg(price_pu_seq_next, Load_NonEV_pu_seq_next, infra_i, hr_i,dur_hi)
        obs_next_ls_b.append(fea_b)
        obs_next_ls_c.append(fea_c)


        # 3. # ================================= rewards ================================================
        reward_b1, action_b = list_EVs[idx_i].get_reward_b(price_t,flag_soc_done, price_min, price_max, price_degrad,price_trade_max, price_trade_min,infra_i)
        reward_c1, action_c = list_EVs[idx_i].get_reward_c2(flag_done,price_t,price_max, price_min)
        
        reward_ls_b.append(reward_b1)
        reward_ls_c.append(reward_c1)
        reward_ls_bm[list_EVs[idx_i].id_mask].append(reward_b1)

        
        # i_ev,idx_trans,energy
        temp_b = [i,trans_i,action_b]
        temp_c = [i,trans_i,action_c]
        result_b.append(temp_b)
        result_c.append(temp_c)

    load_allow0 = n_exist*0.6 
    punish = 2
    eps = 1e-8

    eps = 1e-8

    
    idx_cs = [i for i in range(n_exist) if result_b[i][1] == 0]

    
    contrib_b = {
        i: result_b[i][2]
        for i in idx_cs
    }

    contrib_c = {
        i: 0.1 * result_c[i][2]
        for i in idx_cs
    }

    
    load_transformer = sum(contrib_b.values()) + sum(contrib_c.values())

    
    violation = abs(load_transformer) - load_allow0

    if violation > 0:
        
        violation_ratio = violation / (load_allow0 + eps)

        if load_transformer > load_allow0:
            
            pos_b = {
                i: max(contrib_b[i], 0.0)
                for i in idx_cs
            }

            pos_c = {
                i: max(contrib_c[i], 0.0)
                for i in idx_cs
            }

            total_pos = sum(pos_b.values()) + sum(pos_c.values())

            if total_pos > eps:
                for i in idx_cs:
                    w_b = pos_b[i] / total_pos
                    w_c = pos_c[i] / total_pos

                    reward_ls_b[i] -= w_b * punish * violation_ratio
                    reward_ls_c[i] -= w_c * punish * violation_ratio

        elif load_transformer < -load_allow0:
            
            neg_b = {
                i: max(-contrib_b[i], 0.0)
                for i in idx_cs
            }

            neg_c = {
                i: max(-contrib_c[i], 0.0)
                for i in idx_cs
            }

            total_neg = sum(neg_b.values()) + sum(neg_c.values())

            if total_neg > eps:
                for i in idx_cs:
                    w_b = neg_b[i] / total_neg
                    w_c = neg_c[i] / total_neg

                    reward_ls_b[i] -= w_b * punish * violation_ratio
                    reward_ls_c[i] -= w_c * punish * violation_ratio


    return obs_next_ls_b,obs_next_ls_c,reward_ls_b,reward_ls_c,energy_ch,energy_dch,reward_ls_bm


def update_score_ls(score_ls,n_score_ls,reward_ls_b,reward_ls_c,n_ev):
    
    for i in range(n_ev):
        score_ls[0] = score_ls[0] + reward_ls_b[i]
        score_ls[1] = score_ls[1] + reward_ls_c[i]
        n_score_ls[0] = n_score_ls[0] + 1
        n_score_ls[1] = n_score_ls[1] + 1


    return score_ls,n_score_ls
def get_total_s_and_next(obs_ls,obs_next_ls,d_obs,n_exist,n_knn,EVs_exist, EV_exist_next,list_EVs,n_total):

    total_s = []
    total_s_next = []

    
    
    idx_exist = EVs_exist[0]

    
    for i in range(n_exist):
        totali = obs_ls[i].copy()
        total_nexti = obs_next_ls[i].copy()


        idx_ev = idx_exist[i]
        idx_exist_other = [item for item in idx_exist if item != idx_ev]

        
        idx_ev_nearest = list_EVs[idx_ev].idx_ev_near
        #idx_ev_nearest = list_EVs[idx_ev].idx_ev_near[::-1]
        idx_ev_select =  [item for item in idx_exist_other if item in idx_ev_nearest][0:n_knn-1]  


        idx_select = [idx_exist.index(item) if item in idx_exist else None for item in idx_ev_select]
        n_select = len(idx_ev_select)


        for j in range(n_select):
            tempj = obs_ls[idx_select[j]]
            totali.extend(tempj)

            tempj2 = obs_next_ls[idx_select[j]]
            total_nexti.extend(tempj2)

        totali = totali + [0] * (n_total - len(totali))
        total_nexti = total_nexti + [0] * (n_total - len(total_nexti))
        total_s.append(totali)
        total_s_next.append(total_nexti)
    return total_s, total_s_next
def save_transitions_mask(agent_b,agent_c,feas_b,feas_c,obs_next_ls_b,obs_next_ls_c,
                     action_b,action_c,n_exist,reward_ls_b,reward_ls_c,n_knn, mark_ls,EVs_exist,EV_exist_next,list_EVs,
                           done_ls,weight_ls,n_total):

    obs_ls_b = copy.deepcopy(feas_b)
    obs_ls_c = copy.deepcopy(feas_c)
    d_obs_b = len(obs_ls_b[0])
    d_obs_c = len(obs_ls_c[0])

    
    total_s_b, total_s_next_b = get_total_s_and_next(obs_ls_b,obs_next_ls_b,d_obs_b,n_exist, n_knn,EVs_exist,EV_exist_next,list_EVs,n_total[0])
    total_s_c, total_s_next_c = get_total_s_and_next(obs_ls_c, obs_next_ls_c, d_obs_c,n_exist, n_knn,EVs_exist,EV_exist_next,list_EVs,n_total[1])

    
    action_ls_b = [np.array([i]) for i in action_b]
    action_ls_c = [np.array([i]) for i in action_c]

    

    for i in range(n_exist):
        
        state_b = np.array(obs_ls_b[i])
        state_b = state_b.reshape(1,len(state_b))
        state_c = np.array(obs_ls_c[i])
        state_c = state_c.reshape(1,len(state_c))
        state_next_b = np.array(obs_next_ls_b[i])
        state_next_b = state_next_b.reshape(1,len(state_next_b))
        state_next_c= np.array(obs_next_ls_c[i])
        state_next_c = state_next_c.reshape(1,len(state_next_c))

        state_total_b = np.array(total_s_b[i])
        state_total_b = state_total_b.reshape(1,len(state_total_b))
        state_total_c = np.array(total_s_c[i])
        state_total_c = state_total_c.reshape(1, len(state_total_c))

        id_mask = mark_ls[i]
        weight_i = weight_ls[i]

        # done
        done_i = done_ls[i]

        

        # state, state_next, state_total, state_total_next, action, action_other, reward, done
        agent_b.step_mask3(state_b,state_next_b, state_total_b, total_s_next_b[i], action_ls_b[i],
                     reward_ls_b[i], done_i,id_mask,weight_i)
        agent_c.step_mask3(state_c,state_next_c, state_total_c, total_s_next_c[i], action_ls_c[i],
                     reward_ls_c[i], done_i,id_mask,weight_i)
    return agent_c,agent_b
def add_reward(reward_history,reward):
    reward_history.append(reward)
    if len(reward_history) > 100:  
        reward_history.pop(0)
    return reward_history
def is_outlier(reward_history,reward,k):
    if len(reward_history) < 10:  
        return False
    mean = np.mean(reward_history)
    std = np.std(reward_history)
    return abs(reward - mean) > k * std


def save_show_scores(score_ls,n_score_ls,episode_scores,episode_scores_b,episode_scores_c,i_episode,
                     scores_average_window,window_scores,window_scores_b,window_scores_c):
    if 0 in n_score_ls or 0 in score_ls:
        window_scores.append(window_scores[-1])
        window_scores_b.append(window_scores_b[-1])
        window_scores_c.append(window_scores_c[-1])
        episode_scores.append(episode_scores[-1])
        episode_scores_b.append(episode_scores_b[-1])
        episode_scores_c.append(episode_scores_c[-1])
        return episode_scores, episode_scores_b, episode_scores_c, window_scores, window_scores_b, window_scores_c

    score_ls_mean = score_ls / n_score_ls
    episode_scores.append(np.max(score_ls_mean))
    episode_scores_b.append(score_ls_mean[0])
    episode_scores_c.append(score_ls_mean[1])
    average_score = np.mean(episode_scores[i_episode - min(i_episode, scores_average_window):i_episode + 1])
    average_score_b = np.mean(episode_scores_b[i_episode - min(i_episode, scores_average_window):i_episode + 1])
    average_score_u = np.mean(episode_scores_c[i_episode - min(i_episode, scores_average_window):i_episode + 1])

    window_scores.append(average_score)
    window_scores_b.append(average_score_b)
    window_scores_c.append(average_score_u)

    print("Epoch:{}".format(i_episode))
    print(" score avergae:{:.3f}; score b average:{:.3f}; score u average:{:.3f} ".format(average_score, average_score_b, average_score_u))

    return episode_scores,episode_scores_b,episode_scores_c,window_scores,window_scores_b,window_scores_c
def update_dur_action_soc3(n_exist,EVs_exist,actions_b_real,actions_c_real,list_EVs,price_t,actions_b_pu, actions_c_pu):
    for i in range(n_exist):
        idx_i = EVs_exist[0][i]
        hr_i = EVs_exist[1][i]
        infra_i = EVs_exist[2][i]
        action_bi = actions_b_real[i]
        action_ci = actions_c_real[i]
        action_bi_pu = actions_b_pu[i]
        action_ci_pu = actions_c_pu[i]

        
        list_EVs[idx_i].refresh_action_with_time(action_bi, action_ci, hr_i)
        list_EVs[idx_i]. refresh_action_pu_with_time(action_bi_pu,action_ci_pu)

        
        list_EVs[idx_i].refresh_soc()
        list_EVs[idx_i].l_pricet.append(price_t)
        list_EVs[idx_i].refresh_dur_cs(hr_i)

    return list_EVs

def extract_by_indices_tensor(original_list, indices):
    extracted_elements = [ original_list[i].tolist() for i in indices if i < len(original_list) ]
    extracted_elements_tensor = torch.tensor(extracted_elements)
    return extracted_elements_tensor


# save the agents


# ============


def choose_actions_mask5test(EVs_exist,list_EVs,feas_capacitor,feas_battery,agent_b,agent_c,rate_CS_b,rate_CS_c,n_group):

    # -------------------------------------------------------------------
    
    actions_b = []
    actions_c = []
    actions_b_pu = []
    actions_c_pu = []
    mark_ls = []
    weight_ls = []
    idx_weight_ls = []
    n_exist = len(EVs_exist[0])
    for i in range(n_exist):  

        
        fea_capacitor = np.array(feas_capacitor[i])
        fea_capacitor = fea_capacitor.reshape(1,len(fea_capacitor))
        fea_battery = np.array(feas_battery[i])
        fea_battery = fea_battery .reshape(1,len(fea_battery))

        
        
        idx_ev = EVs_exist[0][i]
        idx_mark = list_EVs[idx_ev].id_mask  
        mark_ls.append(idx_mark)
        weight_u = list_EVs[idx_ev].weight_mask  
        weight_ls.append(weight_u)

        
        
        a_p_b = 0
        a_p_c = 0
        for j in range(n_group):
            action_p_battery = agent_b.act_mask2(fea_battery,j,add_noise=False)
            action_p_capacitor = agent_c.act_mask2(fea_capacitor,j,add_noise=False)
            a_p_b = a_p_b + weight_u[j]* action_p_battery
            a_p_c = a_p_c + weight_u[j] * action_p_capacitor

        a_p_b = a_p_b[0][0]
        a_p_c = a_p_c [0][0]
        actions_b_pu.append(a_p_b)
        actions_c_pu.append(a_p_c)
        action_p_battery = a_p_b * rate_CS_b
        action_p_capacitor = a_p_c * rate_CS_c

        # for battery
        actions_b.append(action_p_battery)
        actions_c.append(action_p_capacitor)

    return actions_b,actions_c, actions_b_pu, actions_c_pu



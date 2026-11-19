
import numpy as np
import os

def custom_mapping(x,x_max,x_min):
    
    if x >= 0:
        return np.interp(x, (0, x_max), (0, 1))
    else:
        return np.interp(abs(x), (0, abs(x_min)), (0, -1))

class CS(object):
    
    def __init__(self,id,lane,n_server, node_start,node_end,n_t_slots):
        self.id = id
        self.lane = lane
        self.edge = "_".join([i for i in lane.split('_')[0:2]])
        self.n_server = int(n_server)  
        self.node_start = node_start
        self.node_end = node_end
        self.n_server_ch = int(self.n_server*1)  
        self.n_server_wt = self.n_server - self.n_server_ch
        self.reserve_ch = np.zeros((self.n_server_ch,int(n_t_slots)),dtype='<U5')
        self.n_wt = 0


        
class EV(object):
    
    
    def __init__(self,soc_ini_b,soc_exp_b,cap_b,p_max_b,p_max_c,soc_ini_c,price_exp,price_exp_pu,w):

        
        self.soc_ini_b = soc_ini_b
        self.soc_ini_c = soc_ini_c
        self.soc_exp_b = soc_exp_b
        self.cap_b = cap_b
        self.p_max_b = p_max_b
        self.p_max_c = p_max_c
        self.price_exp = price_exp
        self.price_exp_pu = price_exp_pu
        self.w = w
        self.l_pricet = []


        self.cap_c = 1  
        self.energy_consume = 0.2e-3  # kwh/m

        self.soc_t_b = soc_ini_b
        self.soc_t_c = soc_ini_c
        self.actions_b = []
        self.actions_c = []
        self.actions_b_pu = []
        self.actions_c_pu = []
        self.hr_b = [] 
        self.hr_c = []


    


    def refresh_soc(self):
        #
        actions_b = self.actions_b  
        t_actions_b = self.hr_b
        actions_c = self.actions_c
        t_actions_c = self.hr_c

        #
        energy_action_b = sum([x * y for x, y in zip(t_actions_b, actions_b)])
        energy_action_c = sum([x * y for x, y in zip(t_actions_c, actions_c)])

        
        self.soc_t_b = max(min(1,self.soc_ini_b + energy_action_b/self.cap_b),0)
        self.soc_t_c = max(min(1,self.soc_ini_c + energy_action_c/self.cap_c),0)


    def refresh_dur_cs(self,hr_i):
        
        self.dur_h = max(0,self.dur_h - hr_i) # unit: s

    def refresh_action_with_time(self,action_b,action_c,hr):
        
        self.hr_b.append(hr)
        self.hr_c.append(hr)
        self.actions_c.append(action_c)
        self.actions_b.append(action_b)
    def refresh_action_pu_with_time(self,action_b,action_c):
        
        self.actions_c_pu.append(action_c)
        self.actions_b_pu.append(action_b)

    def get_fea_b_and_c_iddpg(self,price_pu_seq,Load_NonEV_pu_seq,infra_i,hr_i,dur_i):

        f_soc_exp = self.soc_exp_b - self.soc_t_b  
        f = []

        f.extend([infra_i, dur_i/24,hr_i, self.soc_t_b,self.soc_t_c, f_soc_exp, self.price_exp_pu])
        if os.environ.get("VX_NO_MASK_SPECIALIZATION") != "1":
            f.extend([self.id_mask])
            f.extend(list(self.weight_mask))
        if os.environ.get("VX_HETEROGENEOUS_STRESS") == "1":
            f.append(self.cap_c / 6.0)
        
        dur_i = int(dur_i)+1
        price_list = price_pu_seq[0:dur_i]
        load_list = Load_NonEV_pu_seq[0:dur_i]

        temp_price = [price_list[0], max(price_list),min(price_list),np.mean(price_list),np.std(price_list)]
        temp_load = [load_list[0], max(load_list),min(load_list),np.mean(load_list),np.std(load_list)]

        f.extend(temp_price)
        f.extend(temp_load )

        self.fea_battery = f
        self.fea_capacitor = f
        return f,f

    def get_fea_grouping(self):
        f = []
        f.extend(self.fea_battery)  
        if len(self.actions_b):
            f.extend([self.soc_t_c,self.actions_b[-1]/self.p_max_b,self.actions_c[-1]/self.p_max_c])  
        else:
            f.extend([self.soc_t_c, 0,0])

        self.fea_grouping = f
        return f

    # reward

    def get_reward_b(self,price_t, flag_done, price_min, price_max, price_degrad,price_trade_max, price_trade_min, infra_i):
        
        
        
        
        w = self.w  
        action_pu_t = self.actions_b_pu[-1]
        action_t = self.actions_b[-1]
        hr_t = self.hr_b[-1]
        energy_b = action_t * hr_t
        cap = self.cap_b
        price_exp = self.price_exp
        soc1 =  self.soc_t_b
        soc0 = soc1  - energy_b /cap
        soc_exp  = self.soc_exp_b

        
        price_t_pu = (price_t - price_min) / (price_max - price_min)  
        price_t_exp_pu = custom_mapping(price_exp - price_t, price_trade_max, price_trade_min)
        price_t_trade_pu = custom_mapping(price_t - price_exp - price_degrad, price_trade_max, price_trade_min)

        if soc0 < soc_exp:
            r_elec = (action_pu_t * (1 - price_t_pu) - 1) / 2
            r_soc = np.interp(action_pu_t, (-1, 1), (-1, 0))
            r_b = w[0] * r_soc + w[1] * r_elec - 1  

        else:  
            if action_pu_t > 0:
                r_elec = action_pu_t * price_t_exp_pu
            else:
                r_elec = abs(action_pu_t) * price_t_trade_pu
            r_soc = 0
            r_b = r_elec

        r_soc_done = 0
        if flag_done == True and infra_i == 0:
            if soc1 < soc_exp:
                r_soc_done = - 10 * (soc_exp - soc1) / soc_exp
                r_b = r_b + r_soc_done  

        return r_b, action_pu_t


    def get_reward_c2(self,flag_done,price_t,price_max, price_min):
        action_pu_t = self.actions_c_pu[-1]
        soc = self.soc_t_c
        soc_ini_c = self.soc_ini_c
        price_exp = self.price_exp
        price_exp_max = price_exp - price_min
        price_exp_min = price_exp -price_max


        price_t_exp = price_exp - price_t  
        price_t_exp_pu = custom_mapping(price_t_exp, price_exp_max, price_exp_min)
        r_c = action_pu_t * price_t_exp_pu


        return r_c,action_pu_t



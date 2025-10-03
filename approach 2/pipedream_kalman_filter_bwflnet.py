import io
import numpy as np
import pandas as pd
import wntr
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nutils import interpolate_sample, _kalman_semi_implicit
import time
import pickle
import pipedream_utility as pdu
from pipedream_utility import *
import pipedream_simulation as pd_sim
from pipedream_simulation import *


#Don't show future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def kalman_filter(model, Z, H, C, Qcov, Rcov, P_x_k_k,
                  dt, **kwargs):
    """
    Apply Kalman Filter to fuse observed data into model.

    Inputs:
    -------
    Z : np.ndarray (b x 1)
        Observed data
    H : np.ndarray (M x b)
        Observation matrix
    C : np.ndarray (a x M)
        Signal-input matrix
    Qcov : np.ndarray (M x M)
        Process noise covariance
    Rcov : np.ndarray (M x M)
        Measurement noise covariance
    P_x_k_k : np.ndarray (M x M)
        Posterior error covariance estimate at previous timestep
    dt : float
        Timestep (seconds)
    """
    A_1, A_2, b = model._semi_implicit_system(_dt=dt)
    b_hat, P_x_k_k = _kalman_semi_implicit(Z, P_x_k_k, A_1, A_2, b, H, C,
                                           Qcov, Rcov)
    model.b = b_hat
    model.iter_count -= 1
    model.t -= dt
    model._solve_step(dt=dt, **kwargs)
    return P_x_k_k

def apply_EKF(inp, sensors, t_run = 24, dt = 3600, Rcov = None, Rcov_case = 2, Qcov = None, banded = False, num_iter=10, use_tank_init_cond=False, sensor_std_dev=0.1, full_model = None, Qcov_full = None):
    
    wn = wntr.network.WaterNetworkModel(inp)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    if t_run == None:
        t_run=int(wn.options.time.duration/wn.options.time.hydraulic_timestep)+1

    H_df_model, Q_df_model, Q_pump_model, Q_prv_model, model, Q_in_all_df_model , _, _, _, _, _ = run_pipedream_simulation(inp, t_run, dt, banded = banded)
            
    superjunctions, superlinks, orifices, pumps, prvs, H_bc, Q_in, pats, mult_df, tank_min, tank_max, tank_dict, time_controls_compiled, events_controls_pairs, pumps_init = pdu.wntr_2_pd(wn, t_run, dt)
    
    if full_model != None:
        wn2 = wntr.network.WaterNetworkModel(full_model)
     
    sensor_inds=[superjunctions['name'].to_list().index(s) for s in sensors]
    sensor_rows=list(np.arange(0,len(sensors)))
    
    # Set up Kalman filtering parameters
    n = wn.num_nodes + wn.num_tanks # we are adding the "fake" nodes connected to tank orifices
    p = n
    m = len(sensors)
    
    process_std_dev = .1
    measurement_std_dev = .1
    neg_dem_nodes  = [x for x in range(len(Q_in)) if Q_in[x] > 0]
    neg_list = []
    for id in neg_dem_nodes:
        neg_list.append(list(model.superjunctions.loc[model.superjunctions['id']==id,'name'])[0])

    if Rcov is None:
        if Rcov_case == 1:
            Rcov = [0.00001]*np.eye(m)
        
        elif Rcov_case == 2:
            Rcov = [sensor_std_dev**2]*np.eye(m)
            
        elif Rcov_case == 3:
            Rcov = np.square(np.array(np.std(H_df_model[sensors])))*np.eye(m)
            
        elif Rcov_case == 4:
            Rcov = [0.5**2]*np.eye(m)
            for node in sensors:
                # print(node)
                if node in wn.tank_name_list:
                    # print('tank!!!!')
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                if node in neg_list:
                    # print('neg dem node!!!!')
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                    
        else:
            Rcov = np.square(np.array(np.std(H_df_model[sensors])))*np.eye(m)
            for node in sensors:
                if node in wn.tank_name_list:
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                if node in neg_list:
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
    
    if Qcov is None:
        Qcov = np.cov(results.node['head'], bias=True, rowvar = False)
        #Qcov_full = np.cov(results.node['head'], bias=True, rowvar=False)    

    #%% Implement Kalman Filter
                
    #Enter the sensor locations
    H = np.zeros((m, n))
    H[sensor_rows, sensor_inds] = 1.
    
    C = np.zeros((n, p))
    C[np.arange(n), np.arange(p)] = 1.
    
    P_x_k_k = C @ Qcov @ C.T
    
    # pull in data from Excel
    
    # first check which node corresponds to which logger
    
    # sensors=['node_2012', 'node_2535']
    # t_run = 6

    if full_model!=None:

        sensor_info = pd.read_csv('Networks/bwflnet-_-data/data/sensor_info.csv')
        logger_dict = {}
    
        for element in list(sensor_info['Node/Link']):
            logger_dict[element] = sensor_info['Logger_ID'].loc[sensor_info['Node/Link']==element].values[0]
            
        measurements = pd.DataFrame(columns = sensors, dtype=object) #, index = np.linspace(0,t_run*3600, t_run*4+1))
            
        for sensor_name in sensors:
            if sensor_name == 'node_2548':
                logger = logger_dict['node_2012']
                meas_data_all = pd.read_csv(f'Networks/bwflnet-_-data/data/15min/24hr_training_data/{logger}.csv')
                measurements[sensor_name] = meas_data_all['p_mean'].loc[:t_run*4] + wn.get_node('node_2012').elevation+0.05
            else:
                logger = logger_dict[sensor_name]
                meas_data_all = pd.read_csv(f'Networks/bwflnet-_-data/data/15min/24hr_training_data/{logger}.csv')
                measurements[sensor_name] = meas_data_all['p_mean'].loc[:t_run*4] + wn.get_node(sensor_name).elevation
            
        measurements.index = np.linspace(0,t_run*3600, t_run*4+1)
        
    else:
        
        sensor_info = pd.read_csv('Networks/bwflnet-_-data/data/sensor_info.csv')
        logger_dict = {}
    
        for element in list(sensor_info['Node/Link']):
            logger_dict[element] = sensor_info['Logger_ID'].loc[sensor_info['Node/Link']==element].values[0]
            
        measurements = pd.DataFrame(columns = sensors, dtype=object) #, index = np.linspace(0,t_run*3600, t_run*4+1))
            
        for sensor_name in sensors:
            logger = logger_dict[sensor_name]
            meas_data_all = pd.read_csv(f'Networks/bwflnet-_-data/data/15min/24hr_training_data/{logger}.csv')
            measurements[sensor_name] = meas_data_all['p_mean'].loc[:t_run*4] + wn.get_node(sensor_name).elevation
            
        measurements.index = np.linspace(0,t_run*3600, t_run*4+1)
    
    t_end = np.max(H_df_model.index)
    
    mult_df['-'] = 0.
    multipliers = mult_df
    
    #%% Run Model- Baseline
    
    # Specify number of internal links in each superlink and timestep size in seconds
    internal_links = 1
    num_tanks=wn.num_tanks
    u_o = np.zeros(num_tanks)
    u_p = np.ones(wn.num_pumps)
    
    model = SuperLink(superlinks, superjunctions,  
                      internal_links=internal_links, orifices=orifices, pumps=pumps, prvs=prvs, auto_permute=banded)
    
    is_tank = model.superjunctions['tank'].values
    tank_min = model.superjunctions['tank_min'].values
    tank_max = model.superjunctions['tank_max'].values 
    
    Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[0]).fillna(0.) * model.superjunctions['base_demand']).values
    model.spinup(n_steps=10, dt=60, Q_in=Q_in_t, H_bc=H_bc, u_o=u_o, u_p=u_p, banded=banded, num_iter=num_iter, head_tol=0.0001)
    
    Hj = []
    Q = []
    Q_pump = []
    Q_prv = []
    Q_in_all = []
    t=[]
        
    #Run model for 24 hours
    # While time is less than 24 hours
    
    while model.t < (t_run * 3600):    
        t1 = time.time()
        hour=model.t/3600
        j=int(np.floor(model.t//wn.options.time.pattern_timestep))
        Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[j]).fillna(0.) * model.superjunctions['base_demand']).values
        Q_in_all.append(Q_in_t)
        H_bc_t = H_bc.copy()
        
        # Set tank initial conditions
        if use_tank_init_cond:
            if model.t == 0:
                H_bc_t[is_tank] = model.superjunctions['h_0'].values[is_tank] + model.superjunctions['z_inv'].values[is_tank]
                model.bc[is_tank] = True
                H_bc_t_0=H_bc_t
            else:
                model.bc[is_tank] = False
        
        u_o = ((model.H_j[is_tank] > tank_min[is_tank]) & (model.H_j[is_tank] < tank_max[is_tank])).astype(np.float64)
#        # Check control rule status
#        # open link --> 1, close link --> 0
                
        # Event based controls -- only assuming pump - tank control rules for here. Modify for NWC
        # this is also for > upper limit
        
        for key in events_controls_pairs.keys():
            node = events_controls_pairs[key]['Node']
            link = events_controls_pairs[key]['Link']
            node_id = list(model.superjunctions.loc[model.superjunctions['name']==node,'id'])[0]
            if link in wn.pump_name_list:
                pump_id = model.pumps.loc[model.pumps['name']==link,'id'].values
                
            if model.H_j[node_id] > events_controls_pairs[key]['Upper lim']:
                u_p[pump_id] = events_controls_pairs[key]['Upper lim stat']
            if model.H_j[node_id] < events_controls_pairs[key]['Lower lim']:
                u_p[pump_id] = events_controls_pairs[key]['Lower lim stat']
                    
        #Run model
        model.step(dt=dt, H_bc = H_bc_t, Q_in=Q_in_t, u_o=u_o, u_p=u_p, 
                   banded=banded, num_iter=num_iter, head_tol=0.00001) # initial conditions
        
        next_measurement = measurements.loc[model.t].values
        #next_measurement = interpolate_sample(model.t,
        #                                      measurements.index.values,
        #                                      measurements.values)
        
        
        P_x_k_k = kalman_filter(model, next_measurement, H, C, Qcov, Rcov, P_x_k_k,
                                dt, u_o=u_o, u_p = u_p, banded=banded)
        #Extract results at each timestep
        Hj.append(model.H_j.copy())
        Q.append(model.Q_ik.copy())
        Q_pump.append(model.Q_p)
        Q_prv.append(model.Q_prv)
        t.append(model.t)
        t2 = time.time()
        
    Hj = np.vstack(Hj)
    Q = np.vstack(Q)
    Q_pump_ekf = np.vstack(Q_pump)
    Q_prv_ekf = np.vstack(Q_prv)
    t=np.vstack(t)
        
    #Sample down the Q matrix to only every column from a new link, not each sub-link
    #i.e. if there are 12 internal links in each superlink, then each of the first 
    #12 columns in q will basically be the same
    
    n_superlinks,x=superlinks.shape
    Q_superlinks=Q[:,0:n_superlinks*internal_links:internal_links]
    
    #put H and Q into a dataframe
    #Unscramble the head matrix
    perm_inv = np.argsort(model.permutations)
    Hj = Hj[:, perm_inv]
    
    #Do not use model.superjunctions because I want to use the head matrix in the same order
    #as the original (unpermuted) superjunctions DF because then the columns correspond
    #to the wntr results
    H_df_filtered = pd.DataFrame(columns=superjunctions['name'],index=np.arange(0,t_run*3600,dt),data=Hj)
    Q_df_filtered=pd.DataFrame(columns=model.superlinks['name'],index=np.arange(0,t_run*3600,dt),data=Q_superlinks)
    
    
    return H_df_filtered, H_df_model, model, Q_pump_model, Q_pump_ekf, Q_prv_model, Q_prv_ekf, measurements, Q_df_model, Q_df_filtered
       
#cython: boundscheck = False
#cython: language_level = 3

import numpy as np

cdef int[:,:] sites_random(int num_lanes, int num_places, int num_cars):
    '''
    input:
    num_lanes - number of lanes
    num_places - number of cells in the lane
    num_cars - number of cars in the system

    returns sites[num_lanes,num_places] - int matrix with exactly num_cars randomly distributed zeros.
    Other fields are equal to -1 (which represents absence of the car).
    '''
    cdef int dim = num_lanes*num_places
    random_places = np.random.choice(dim, num_cars, replace = False)
    a = np.full(dim, -1, dtype = int)
    a[random_places] = 0
    cdef int[:,:] sites = a.reshape(num_lanes, num_places)
    return sites


cdef int[:,:] cars_list_fill_from_sites(int[:,:] sites, int num_cars):
    '''
    fills the cars array with a values from sites matrix
    '''
    cdef int num_lanes = sites.shape[0]
    cdef int num_places = sites.shape[1]
    cdef int k = 0
    cdef int i
    cdef int j
    cdef int[:,:] cars_list = np.zeros((num_cars, 3), dtype = int)
    cdef int[:] random_index = np.asarray(np.random.default_rng().choice(num_cars, size=num_cars, replace=False), dtype = int)

    for i in range(num_lanes):
        for j in range(num_places):
            if sites[i,j] > -1:
                cars_list[random_index[k],0] = i
                cars_list[random_index[k],1] = j
                cars_list[random_index[k],2] = sites[i,j]
                k += 1
    return cars_list


cdef initialize_random_config(int num_lanes, int num_places, int num_cars):
    '''
    create initial random configuration with num_lanes lines and
    num_places in each line, and num_cars placed with velocity = 0
    '''
    cdef int[:,:] X = sites_random(num_lanes, num_places, num_cars)
    cdef int[:,:] C = cars_list_fill_from_sites(X, num_cars)
    return X, C


cdef int[:,:] empy_car_sites(int[:,:] cars_list, int[:,:] sites):
    '''
    updates sites: make empty those sites, where cars placed
    cars_list - a list of cars, each element is [number of lane, number of place, velocity]
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    cdef int num_cars = len(cars_list)
    cdef int i
    cdef int lane
    cdef int place
    
    for i in range(num_cars):
        lane = cars_list[i, 0]
        place = cars_list[i,1]
        sites[lane, place] = -1
    return sites
        
    
cdef int[:,:] update_car_sites(int[:,:] cars_list, int[:,:] sites):
    '''
    updates sites: place cars in cells according to cars_list array
    cars_list - a list of cars, each element is [number of lane, number of place, velocity]
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    cdef int num_cars = len(cars_list)
    cdef int i
    cdef int lane
    cdef int place
    
    for i in range(num_cars):
        lane = cars_list[i, 0]
        place = cars_list[i,1]
        sites[lane, place] = cars_list[i, 2]
    return sites


cdef g_v_next(int[:,:] cars_list, int[:,:] sites, int S):
    '''
    cars_list (ndarray) - a list of cars, each element is [number of lane, number of place, velocity]
    sites (ndarray) - a (number of lanes)x(number of places) array with values are the cars velocities
    S (int) - a number of cars that account in a perspective view (model parameter)
    
    Returns: g, v_next in present line (only forward)
    '''  
    cdef int num_cars = len(cars_list)
    cdef int num_cells = sites.shape[1]
    cdef int i
    cdef int dist
    cdef int present_lane
    cdef int current_position
    cdef int[:,:] g_next = np.zeros((num_cars, S), dtype = int)
    cdef int[:,:] v_next = np.zeros_like(g_next)
        
    for i in range(num_cars):
        present_lane = cars_list[i,0]
        current_position = cars_list[i,1]
        dist = 0
        for j in range(S):
            while sites[present_lane,(current_position + dist + 1)%num_cells] < 0:
                dist += 1
            g_next[i,j] = dist
            v_next[i, j] = sites[present_lane,(current_position + dist + 1)%num_cells]
            dist += 1
    return g_next, v_next


cdef double[:] prob_slow_down(int[:,:] cars_list, int[:,:] g, int[:,:] v_next, int G, double[:] P):
    '''
    Returns probability of random brake according to S-NFS model
    '''
    cdef int num_cars = len(cars_list)
    cdef double[:] p = np.zeros(num_cars)
    cdef int i
    cdef int v_current
    
    for i in range(num_cars):
        v_current = cars_list[i,2]
        if g[i,0] >= G:
            p[i] = P[0]
        else:
            if v_current < v_next[i, 0]:
                p[i] = P[1]
            elif v_current == v_next[i, 0]:
                p[i] = P[2]
            else:
                p[i] = P[3]
    return p


cdef int neighb_car_type(int [:,:] cars_list, int lane, int position, int distance, int max_distance, int n_hdv):
    cdef int i
    cdef int num_cars = len(cars_list)
    cdef type_identification = False
    
    if not type_identification or (distance >= max_distance):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return 0    #0 means there is no car within max_distance
    else:
        for i in range(num_cars):
            if (cars_list[i, 0] == lane and cars_list[i, 1] == position):
                if i < n_hdv:
                    return 1
                else:
                    return 2


cdef dist_and_vels(int [:,:] cars_list, int[:,:] sites, int n_hdv):
    # this works properly for 2 lanes only!
    cdef int num_lanes = sites.shape[0]
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int H = 1
    cdef int g_max = 15
    cdef int [:] dist = np.zeros(4*H, dtype = int)  
    cdef int [:,:,:,:] g = np.zeros((num_cars, num_lanes, 2, H), dtype = int)
    cdef int [:,:,:,:] v = np.zeros_like(g)
    cdef int [:,:,:,:] t = np.zeros_like(g)
    cdef int present_lane
    cdef int current_position
    cdef int i
    cdef int lane
    cdef int j
    cdef int k
    cdef int position
    
    for i in range(num_cars):
        present_lane = cars_list[i,0]
        current_position = cars_list[i,1]
        for k in range(4*H):
            dist[k] = 0
        k = 0
        
        for j in range(H):        
            for lane in range(num_lanes):
                if lane != present_lane:
                    dist[k:k+1] = -1
                    
                # forward cars positions (g_0), speed (v_0) and type (t_0) in the lane
                while sites[lane, (current_position + dist[k] + 1)%num_cells] < 0:
                    dist[k] += 1
                    if dist[k] >= g_max:
                        break
                g[i,lane,0,j] += dist[k]
                position = (current_position + dist[k] + 1)%num_cells
                v[i,lane,0,j] = sites[lane, position]
                t[i,lane,0,j] = neighb_car_type(cars_list, lane, position, dist[k], g_max, n_hdv)
                dist[k] += 1
                k += 1
                
                # backward cars positions (g_1) and speed (v_1) in the lane
                while sites[lane, (current_position - (dist[k] + 1))] < 0:
                    dist[k] += 1
                    if dist[k] >= g_max:
                        break
                g[i,lane,1,j] += dist[k]
                position = current_position - (dist[k] + 1)
                v[i,lane,1,j] = sites[lane, position]
                t[i,lane,1,j] = neighb_car_type(cars_list, lane, position, dist[k], g_max, n_hdv)
                dist[k] += 1
                k += 1
    return g, v, t


cdef bint CL_incentive_criterion(int vel, int g_f_PL, int v_f_PL, int g_f_AL, int v_f_AL, int t_f_PL, int t_f_AL, str model, str agent_type):
    cdef bint CAV_change
    cdef double P_CAV_CL = 0.15

    if (model == 'S-NFS') or (model == 'KKW'):
        if agent_type == 'HDV':
            return (vel > g_f_PL + v_f_PL) and (vel <= g_f_AL + v_f_AL)
        elif agent_type == 'CAV':
            return (vel > g_f_PL + v_f_PL) and (vel <= g_f_AL + v_f_AL)
        elif agent_type == 'HAV':
            return True

    elif model == 'W184':
        if agent_type == 'HDV':
            return g_f_PL <= 1 and g_f_AL >= 2
            #(g_f_PL <  g_f_AL) and (g_f_PL < 5) (vel > 0) and 
        elif agent_type == 'CAV':
            return g_f_PL <= 1 and g_f_AL >= 2
            #CAV_change = (t_f_AL == 2) 
            #(vel > 0) and (g_f_PL < 5) and (g_f_PL <  g_f_AL) #and (np.random.random() < P_CAV_CL)) or CAV_change
        elif agent_type == 'HAV':
            return True

        
cdef bint CL_safety_criterion(int vel, int g_b_AL, int v_b_AL, str model):
    cdef int W184_gap = 2 
    
    if (model == 'S-NFS') or (model == 'KKW'):
        return vel >= v_b_AL - g_b_AL
    elif model == 'W184':
        return vel >= v_b_AL - g_b_AL + W184_gap

    
cdef str agent_type(int ID, int n_hdv, int n_cav):
    if ID < n_hdv:
        return 'HDV'
    elif (ID >= n_hdv) and (ID < n_hdv + n_cav):
        return 'CAV'
    else:
        return 'HAV'

    
cdef change_lane(int[:,:] cars_list, int[:,:] sites, int n_hdv, double P_lc, int num_change_lane, str model):
    cdef int[:] vel = cars_list[:,2]
    cdef int num_cars = len(cars_list)
    cdef int num_lanes = sites.shape[0]
    cdef double[:] r = np.random.random(num_cars)    
    cdef int i
    cdef int pres_lane
    cdef int adj_lane
    cdef int [:,:,:,:] g
    cdef int [:,:,:,:] v
    cdef int [:,:,:,:] t
    
    g, v, t = dist_and_vels(cars_list, sites, n_hdv)
    sites = empy_car_sites(cars_list, sites)
    
    for i in range(num_cars):
        if (r[i] < P_lc or i >= n_hdv):
            pres_lane = cars_list[i,0]
            adj_lane = (pres_lane + 1)%num_lanes
            if CL_incentive_criterion(vel[i], g[i,pres_lane,0,0], v[i,pres_lane,0,0], g[i,adj_lane,0,0], v[i,adj_lane,0,0], t[i,pres_lane,0,0], t[i,adj_lane,0,0], model, agent_type(i, n_hdv, 0)):
                if CL_safety_criterion(vel[i], g[i,adj_lane,1,0], v[i,adj_lane,1,0], model):
                    cars_list[i,0] = adj_lane
                    num_change_lane += 1

    sites = update_car_sites(cars_list, sites)
    return sites, cars_list, num_change_lane


cdef int current_index(int[:] arranged_ind, int list_length, int i):
    cdef int j
    cdef int cur_ind = 0
    
    for j in range(list_length):
        if arranged_ind[j] == i:
            cur_ind = j
            break
    return cur_ind


cdef int[:] vel_step_SNFS(int[:,:] cars_list, int[:,:] g, int[:,:] v_next, double[:] p, int S, int G, double q, double r, int V_max, int N_c):
    '''
    S-NFS model velocity update in a single lane
    '''
    cdef int num_cars = len(cars_list)
    cdef int[:,:] v = np.zeros((num_cars, 6), dtype = int)
    cdef int[:] s = np.random.choice((1, S), num_cars, p = [1-r, r])
    cdef int i
    cdef int[:] v_return = np.zeros(num_cars, dtype = int)
    cdef int cur_ind
    cdef int ind
    cdef int v4_next
    cdef double[:] rand_array = np.random.random(num_cars)
    cdef int[:] arranged_ind = np.asarray(np.argsort(cars_list[:,1]), dtype = int)
       
    for i in range(num_cars):
        v[i,0] = cars_list[i,2]
        
    # Rule 1. Acceleration    
    for i in range(num_cars):
        if (g[i,0] >= G or v[i,0] <= v_next[i,0]):
            v[i,1] = min(V_max, v[i,0] + 1)
        else:
            v[i,1] = v[i,0]
            
    # Rule 2. Slow-to-start
    for i in range(num_cars):
        if (rand_array[i] < q)&(g[i,s[i]-1] + 1 - (v_next[i,s[i]-1] - v[i,0]) - s[i] >= 0):
            v[i,2] = min(v[i,1], g[i,s[i]-1] + 1 - (v_next[i,s[i]-1] - v[i,0]) - s[i]) #if there wasn't a line change in here
        else:
            v[i,2] = v[i,1]
            
    # Rule 3. Quick start
    for i in range(num_cars):
        v[i,3] = min(v[i,2], g[i,s[i]-1] + 1 - s[i])
        
    # Rule 4. Random brake
    cdef double[:] rand_array2 = np.random.random(num_cars)
    
    for i in range(num_cars):
        if (rand_array2[i] < 1-p[i] and v[i,3] > 0):
            v[i,4] = max(1, v[i,3] - 1)
        else:
            v[i,4] = v[i,3]
    
    if False:
    #if num_cars > N_c:
        for i in range(num_cars - N_c, num_cars):
            cur_ind = current_index(arranged_ind, num_cars, i)
            ind = (cur_ind + 1)%num_cars
            while cars_list[i,0] != cars_list[arranged_ind[ind],0]:
                ind = (ind + 1)%num_cars
            v4_next = v[arranged_ind[ind], 4]
            if (v[i,4] == v4_next) and (g[i,0] < G) and (v[i,4]>3):
                v[i, 4] -= 1

    # Rule 5. Avoid collision
    for i in range(num_cars):
        cur_ind = current_index(arranged_ind, num_cars, i)
        ind = (cur_ind + 1)%num_cars
        while cars_list[i,0] != cars_list[arranged_ind[ind],0]:
            ind = (ind + 1)%num_cars
        v4_next = v[arranged_ind[ind], 4]

        v[i,5] = min(v[i,4], g[i,0] + v4_next)
    
    for i in range(num_cars):
        v_return[i] = v[i,5]
    
    return v_return


cdef move_SNFS(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c, double P_lc, int num_change_lane):
    '''
    Single move action: update system from t to t+1 with revised S-NFS model
    '''
    cdef int S = 2
    cdef int G = 15
    cdef double q = 0.99
    cdef double r = 0.99
    cdef int V_max = 5
    cdef double[:] P_rbrake = np.array([0.999, 0.99, 0.98, 0.01])  #np.array([1.0, 1.0, 1.0, 1.0])
    
    cdef int[:] v_new
    cdef int num_lanes = sites.shape[0]
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int i

    if num_cars == num_lanes*num_cells:
        return sites, cars_list, num_change_lane    
    
    if (num_lanes > 1 and P_lc > 0):
        sites, cars_list, num_change_lane = change_lane(cars_list, sites, n_hdv, P_lc, num_change_lane, 'S-NFS') # try to change lane
    
    g, v_next = g_v_next(cars_list, sites, S)
    p = prob_slow_down(cars_list, g, v_next, G, P_rbrake)

    v_new = vel_step_SNFS(cars_list, g, v_next, p, S, G, q, r, V_max, N_c)
    
    #update cars positions and velocities
    sites = empy_car_sites(cars_list, sites)
    for i in range(num_cars):
        cars_list[i,2] = v_new[i]
        cars_list[i,1] = (cars_list[i,1] + v_new[i])%num_cells
    sites = update_car_sites(cars_list, sites)
    return sites, cars_list, num_change_lane


cdef move_W184(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c, double P_lc, int num_change_lane, bint red_light):
    '''
    Single move action: update system from t to t+1
    '''
    cdef int num_lanes = sites.shape[0]
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef double[:] p = np.array([0.1, 0.3, 0.95])
    cdef double[:] r_0 = np.random.random(num_cars)
    cdef double[:] r_1 = np.random.random(num_cars)
    cdef double[:] r_2 = np.random.random(num_cars)
    cdef int[:] v = np.zeros(num_cars, dtype = int)
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int[:] arranged_ind = np.asarray(np.argsort(cars_list[:,1]), dtype = int)
    cdef int cur_ind
    cdef int ind
    cdef int s
    cdef int i

    cdef int tr_light_pos = 50000

    if num_cars == num_lanes*num_cells:
        return sites, cars_list, num_change_lane

    if (num_lanes > 1): # and (P_lc > 0 or n_hdv < num_cars)):
        sites, cars_list, num_change_lane = change_lane(cars_list, sites, n_hdv, P_lc, num_change_lane, 'W184')
        
    g, v_next = g_v_next(cars_list, sites, 1)

    # Acceleration and random stop for human-driven vehicles
    for i in range(n_hdv):
        if (cars_list[i,1] == tr_light_pos and red_light == True):
            v[i] = 0
        elif ((g[i,0] > 4) or (g[i,0] >= 3 and r_2[i] < p[2]) or (g[i,0] == 2 and r_1[i] < p[1]) or (g[i,0] == 1 and r_0[i] < p[0])):
            v[i] = 1
        else:
            v[i] = 0

    # For automated vehicles
    for i in range(n_hdv, num_cars):
        if (cars_list[i,1] == tr_light_pos and red_light == True and cars_list[i,0] == 0):
            v[i] = 0
        elif (g[i,0] > 0):
            v[i] = 1
        else:
            v[i] = 0
            cur_ind = current_index(arranged_ind, num_cars, i)
            ind = (cur_ind + 1)%num_cars
            s = 0
            while s < max_platoon_size:
                while cars_list[i,0] != cars_list[arranged_ind[ind],0]:
                    ind = (ind + 1)%num_cars
                if arranged_ind[ind] >= n_hdv: #next vehicle - automated
                    if (g[arranged_ind[ind],0] > 0) and (cars_list[arranged_ind[ind], 1] != tr_light_pos):
                        v[i] = 1
                        break
                    else:
                        ind = (ind + 1)%num_cars
                        s += 1
                else: #next vehicle - human-driven
                    v[i] = 0
                    break
    #update cars positions and velocities
    sites = empy_car_sites(cars_list, sites)
    for i in range(num_cars):
        cars_list[i,2] = v[i]
        cars_list[i,1] = (cars_list[i,1] + v[i])%num_cells
    sites = update_car_sites(cars_list, sites)
    return sites, cars_list, num_change_lane


cdef move_KKW(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c, double P_lc, int num_change_lane):
    '''
    Single move action: update system from t to t+1 with Kerner-Klenov-Wolf model
    '''
    cdef int k = 3
    cdef double p_brake = 0.25
    cdef double p_accel = 0.25
    cdef int V_max = 5
    
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int[:] v = np.zeros(num_cars, dtype = int)
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int i
    cdef double[:] rand_arr = np.random.random(num_cars)
    cdef int G
    cdef int v_c
    cdef int v_w
    cdef int eta
    
    sites, cars_list, num_change_lane = change_lane(cars_list, sites, n_hdv, P_lc, num_change_lane, 'S-NFS') # try to change lane
    g, v_next = g_v_next(cars_list, sites, 1)

    for i in range(num_cars):
        v[i] = cars_list[i,2]
        G = k*v[i]
        if g[i, 0] > G:
            v_c = v[i] + 1
        else:
            if v_next[i,0] > v[i]:
                v_c = v[i] + 1
            elif v_next[i,0] == v[i]:
                v_c = v[i]
            else:
                v_c = v[i] - 1
        v_w = max(0, min(V_max, v_c, g[i,0]))
        if rand_arr[i] < p_brake:
            eta = -1
        elif (p_brake <= rand_arr[i]) and (rand_arr[i] < p_brake + p_accel):
            eta = 1
        else:
            eta = 0
        v[i] = max(0, min(V_max, v_w + eta, v[i] + 1, g[i,0]))
    
    #update cars positions and velocities
    sites = empy_car_sites(cars_list, sites)
    for i in range(num_cars):
        cars_list[i,2] = v[i]
        cars_list[i,1] = (cars_list[i,1] + v[i])%num_cells
    sites = update_car_sites(cars_list, sites)
    return sites, cars_list, num_change_lane


cdef double flux_calc(int [:,:] sites, str method, str model):
    '''
    Returns: density (float) and flux (float)
 
    Input: 
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    cdef int num_lanes = sites.shape[0] 
    cdef int num_places = sites.shape[1]
    #cdef int num_cars = 0
    cdef int V_max = 5
    cdef int i
    cdef int j
    cdef int veh_flow = 0
    cdef double flux
    cdef int average_vel = 0

    if method == 'Loop':
        if model == 'W184':
            V_max = 1
        for j in range(num_lanes):
            for i in range(V_max):
                if sites[j, i+1] >= V_max - i:
                    veh_flow =+ 1
        flux = veh_flow
        
    elif method == 'Average':
        for i in range(num_lanes):
            for j in range(num_places):
                if sites[i,j] > -1:
                    average_vel += sites[i,j]
                    #num_cars += 1
        flux = average_vel/(num_places*num_lanes)

    return flux  


cdef double average_velocity(int [:,:] cars_list, str model):
    '''
    Calculates average velocity of all vehicles except counteracting ones
    '''
    cdef int num_cars = len(cars_list)
    cdef double av_vel = 0
    cdef int i
    
    for i in range(num_cars):
        av_vel += cars_list[i,2]    
    
    if (num_cars > 0):
        av_vel = av_vel / (num_cars)   
    else:
        if model == 'W184':
            av_vel = 1
        else:
            av_vel = 5
    return av_vel


cdef fill_episode(int [:,:] sites, int time_iter, int[:,:,:] episodes):
    cdef int num_lanes = sites.shape[0] 
    cdef int num_places = sites.shape[1]
    cdef int lane
    cdef int place

    for lane in range(num_lanes):
        for place in range(num_places):
            if sites[lane,place] > -1:
                episodes[lane,time_iter,place] = 1
            else:
                episodes[lane,time_iter,place] = 0

                
cdef traffic_lights(int i, int T_last_switch, int T_g, int T_r, bint red_light):
    cdef bint previous_light = red_light
    
    if (i - T_last_switch >= T_g) and (red_light == False):
        red_light = True
        if (red_light != previous_light):
            T_last_switch = i
            #print('lights switched at ', i)

    if (i - T_last_switch >= T_r) and (red_light == True):
        red_light = False
        if (red_light != previous_light):
            T_last_switch = i
            #print('lights switched at ', i)
    return T_last_switch, red_light


cpdef run_simulation(int num_lanes, int num_places, double density, double P_lc, int time_steady, int num_iters, double R_hdv, int max_platoon_size, int N_c, str model, bint visualise):
    '''
    Single run of a traffic simulation: returns density and flux
    num_lanes - a number of lanes
    num_places - a number of car places (cells) in one lane
    num_iters - a number of time iterations
    N_c - a number of counter agents
    visualise - a logical marker: if True then draw an x-t plot
    '''
    cdef int num_cars = int(density*num_lanes*num_places)
    cdef double flux = 0
    cdef double v_av = 0
    cdef double change_lane_rate = 0
    cdef int num_change_lane = 0
    cdef int[:,:] X
    cdef int[:,:] C
    cdef int i
    cdef int [:,:,:] episodes = np.zeros((num_lanes, time_steady+num_iters+1, num_places), dtype = int)
    cdef int n_hdv = int(R_hdv*num_cars)

    #traffic lights
    cdef bint traffic_lights_on = False
    cdef int T_g = 60
    cdef int T_r = 20
    cdef int T_last_switch = 0
    cdef bint red_light = False

    X, C = initialize_random_config(num_lanes, num_places, num_cars)

    if visualise == True:
        fill_episode(X, 0, episodes)

    for i in range(time_steady+num_iters):
        if model == 'S-NFS':
            X, C, num_change_lane = move_SNFS(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        elif model == 'W184':
            if traffic_lights_on:
                T_last_switch, red_light = traffic_lights(i, T_last_switch, T_g, T_r, red_light)
            X, C, num_change_lane = move_W184(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane, red_light)
        elif model == 'KKW':
            X, C, num_change_lane =  move_KKW(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        else:
            print('ERROR. No model available called ', model)
            break

        if i > time_steady:
            flux += flux_calc(X, 'Average', model)
            v_av += average_velocity(C, model)
        
        if visualise == True:
            fill_episode(X, i+1, episodes)

    flux /= num_iters
    v_av /= num_iters
    #flux = v_av*density
    change_lane_rate = num_change_lane/(num_iters*num_places)

    if visualise == True:
        return flux, change_lane_rate, v_av, episodes
    else:
        return flux, change_lane_rate, v_av
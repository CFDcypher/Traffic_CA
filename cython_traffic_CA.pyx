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

    cdef int[:,:] g = np.zeros((num_cars, S), dtype = int)
    cdef int[:,:] v_next = np.zeros_like(g)
        
    for i in range(num_cars):
        present_lane = cars_list[i,0]
        current_position = cars_list[i,1]
        dist = 0
        for j in range(S):
            while sites[present_lane,(current_position + dist + 1)%num_cells] < 0:
                dist += 1
            g[i,j] = dist
            v_next[i, j] = sites[present_lane,(current_position + dist + 1)%num_cells]
            dist += 1
    return g, v_next

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

cdef dist_and_vels(int [:,:] cars_list, int[:,:] sites):
    # this works properly for 2 lanes only!
    cdef int num_lanes = sites.shape[0]
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int H = 1
    cdef int g_max = 20
    cdef int[:] dist = np.zeros(4*H, dtype = int)  
    cdef int [:,:,:,:] g = np.zeros((num_cars, num_lanes, 2, H), dtype = int)
    cdef int [:,:,:,:] v = np.zeros_like(g)
    cdef int present_lane
    cdef int current_position
    cdef int i
    cdef int lane
    cdef int j
    cdef int k
    
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
                # forward cars positions (g_0) and speed (v_0) in the lane
                while sites[lane, (current_position + dist[k] + 1)%num_cells] < 0:
                    dist[k] += 1
                    if dist[k] > g_max:
                        break
                g[i,lane,0,j] += dist[k]
                v[i,lane,0,j] = sites[lane, (current_position + dist[k] + 1)%num_cells]
                dist[k] += 1
                k += 1
                
                # backward cars positions (g_1) and speed (v_1) in the lane
                while sites[lane, (current_position - (dist[k] + 1))] < 0:
                    dist[k] += 1
                    if dist[k] > g_max:
                        break
                g[i,lane,1,j] += dist[k]
                v[i,lane,1,j] = sites[lane, (current_position - (dist[k] + 1))]
                dist[k] += 1
                k += 1
    return g, v


cdef bint CL_incentive_criterion(int vel, int g_next_CL, int v_next_CL, str model, str agent_type):
    if model == 'S-NFS':
        if agent_type == 'HDV':
            return vel >= g_next_CL + v_next_CL
        elif agent_type == 'CAV':
            return vel >= g_next_CL + v_next_CL
        elif agent_type == 'HAV':
            return True

    elif model == 'W184':
        if agent_type == 'HDV':
            return (g_next_CL == 1 or g_next_CL == 2)
        elif agent_type == 'CAV':
            return (g_next_CL == 1 or g_next_CL == 2)
        elif agent_type == 'HAV':
            return True

cdef bint CL_safety_criterion(int vel, int g_next_AL, int v_next_AL, int g_beh_AL, int v_beh_AL, str model, str agent_type):
    if model == 'S-NFS':
        if agent_type == 'HDV':
            return (vel < g_next_AL+v_next_AL) and (vel > v_beh_AL-g_beh_AL) and (g_next_AL>0)
        elif agent_type == 'CAV':
            return (vel < g_next_AL+v_next_AL) and (vel > v_beh_AL-g_beh_AL) and (g_next_AL>0)
        elif agent_type == 'HAV':
            return True #((v[i,lane,1,0] + v[i,lane,1,1] > v[i,present_lane,1,1] + v[i,present_lane,0,1]) and (g[i,lane,1,0] <= 15) and (vel[i] > v[i,lane,1,0]-g[i,lane,1,0]) and (g[i,lane,0,0]>0))

    elif model == 'W184':
        if agent_type == 'HDV':
            return (g_next_AL > 2 and g_beh_AL > 1)
        elif agent_type == 'CAV':
            return (g_next_AL > 2 and g_beh_AL > 1)
        elif agent_type == 'HAV':
            return (g_beh_AL == 1 and g_next_AL > 0) #g[i,present_lane,1,0] > 1 


cdef change_lane(int[:,:] cars_list, int[:,:] sites, int n_hdv, double P_lc, int num_change_lane, str model):
    cdef int[:] vel = cars_list[:,2]
    cdef int num_cars = len(cars_list)
    cdef int num_lanes = sites.shape[0]
    cdef int i
    cdef int lane
    cdef int present_lane
    cdef str agent_type

    cdef int [:,:,:,:] g
    cdef int [:,:,:,:] v
    g, v = dist_and_vels(cars_list, sites)

    cdef double[:] r = np.random.random(num_cars)
    
    sites = empy_car_sites(cars_list, sites)
    
    for i in range(num_cars):
        if i < n_hdv:
            agent_type = 'HDV'
        else:
            agent_type = 'CAV'

        if r[i] < P_lc:
            present_lane = cars_list[i,0]
            if CL_incentive_criterion(vel[i], g[i,present_lane,0,0], v[i,present_lane,0,0], model, agent_type): #CL_incentive_criterion(vel[i], 0, 0, model, agent_type):
                for lane in range(num_lanes):
                    if lane != present_lane:
                        if CL_safety_criterion(vel[i], g[i,lane,0,0], v[i,lane,0,0], g[i,lane,1,0], v[i,lane,1,0], model, agent_type): #CL_safety_criterion(vel[i], 0, 0, 0, 0, model, agent_type):
                            cars_list[i,0] = lane
                            num_change_lane += 1
                            break

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
        if (rand_array2[i] < 1-p[i] and v[i,3] > 1):
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
    cdef double r = 0.5
    cdef int V_max = 5
    cdef double[:] P_rbrake = np.array([0.999, 0.99, 0.98, 0.01]) #np.array([1.0, 1.0, 1.0, 1.0])
    
    cdef int[:] v_new
    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int i

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


cdef move_W184(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c, double P_lc, int num_change_lane):
    '''
    Single move action: update system from t to t+1
    '''
    cdef int num_cells = sites.shape[1]
    cdef int num_places = sites.shape[2]
    cdef int num_cars = len(cars_list)
    cdef double[:] p = np.array([0.3, 0.7, 0.99])   #p = np.array([0.3, 0.7, 0.99])
    cdef double[:] r_1 = np.random.random(num_cars)
    cdef double[:] r_2 = np.random.random(num_cars)
    cdef double[:] r_3 = np.random.random(num_cars)
    cdef int[:] v = np.zeros(num_cars, dtype = int)
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int[:] arranged_ind = np.asarray(np.argsort(cars_list[:,1]), dtype = int)
    cdef int cur_ind
    cdef int ind
    cdef int s
    cdef int i

    if num_cars == num_cells*num_places:
        return sites, cars_list, num_change_lane

    # Change lane step
    sites, cars_list, num_change_lane = change_lane(cars_list, sites, n_hdv, P_lc, num_change_lane, 'W184') # try to change lane
    g, v_next = g_v_next(cars_list, sites, 1)

    # Acceleration and random stop for human-driven vehicles
    for i in range(n_hdv):
        #if (g[i,0] > 2 and r_3[i] < p[2]) or (g[i,0] == 2 and r_2[i] < p[1]) or (g[i,0] == 1 and r_1[i] < p[0]):
        if (g[i,0] > 3) or (g[i,0] == 3 and r_3[i] < p[2]) or (g[i,0] == 2 and r_2[i] < p[1]) or (g[i,0] == 1 and r_1[i] < p[0]):
            v[i] = 1
        else:
            v[i] = 0

    # For automated vehicles
    for i in range(n_hdv, num_cars):
        if (g[i,0] > 0):
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
                    if (g[arranged_ind[ind],0] > 0):
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
    cdef int k = 2
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


cdef macroparameters(int [:,:] sites):
    '''
    Returns: density (float) and flux (float)
 
    Input: 
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    cdef int num_lanes = sites.shape[0] 
    cdef int num_places = sites.shape[1]
    cdef double average_vel = 0
    cdef int num_cars = 0
    cdef int V_max = 5
    cdef int veh_flow = 0
    cdef int i
    cdef int j
    
    for i in range(num_lanes):
        for j in range(num_places):
            if sites[i,j] > -1:
                average_vel += sites[i,j]
                num_cars += 1

    for j in range(num_lanes):
        for i in range(V_max):
            if sites[j, i+1] >= V_max - i:
                veh_flow =+ 1
    return num_cars/(num_lanes*num_places), veh_flow # average_vel/num_places

cdef double average_velocity(int [:,:] cars_list, int N_c):
    '''
    Calculates average velocity of all vehicles except counteracting ones
    '''
    cdef int num_cars = len(cars_list)
    cdef double av_vel = 0
    cdef int i
    
    for i in range(num_cars - N_c):
        av_vel += cars_list[i,2]
    
    if (num_cars > 0)&(num_cars != N_c):
        av_vel = av_vel / (num_cars - N_c)
    elif num_cars == N_c:
        av_vel = 0
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

cpdef run_simulation(int num_lanes, int num_places, double density, double P_lc, int time_steady, int num_iters, double R_hdv, int max_platoon_size, int N_c, bint visualise):
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
    cdef double rho
    cdef double q
    cdef double v_av = 0
    cdef double change_lane_rate = 0
    cdef int num_change_lane = 0
    cdef int[:,:] X
    cdef int[:,:] C
    cdef int i
    cdef int place
    cdef int [:,:,:] episodes = np.zeros((num_lanes, time_steady+num_iters+1, num_places), dtype = int)
    cdef int n_hdv = int(R_hdv*num_cars)

    X, C = initialize_random_config(num_lanes, num_places, num_cars)

    if visualise == True:
        fill_episode(X, 0, episodes)

    for i in range(time_steady):
        X, C, num_change_lane = move_SNFS(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        #X, C, num_change_lane = move_W184(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        #X, C, num_change_lane =  move_KKW(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)

        if visualise == True:
            fill_episode(X, i+1, episodes)

    for i in range(num_iters):
        X, C, num_change_lane = move_SNFS(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        #X, C, num_change_lane = move_W184(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)
        #X, C, num_change_lane =  move_KKW(C, X, n_hdv, max_platoon_size, N_c, P_lc, num_change_lane)

        if visualise == True:
            fill_episode(X, time_steady + i + 1, episodes)

        rho, q = macroparameters(X)
        flux += q
        v_av += average_velocity(C, N_c)
    
    flux /= num_iters
    v_av /= num_iters
    change_lane_rate = num_change_lane/(num_iters*num_places)

    if visualise == True:
        return rho, flux, change_lane_rate, v_av, episodes
    else:
        return rho, flux, change_lane_rate, v_av
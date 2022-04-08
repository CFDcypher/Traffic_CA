#cython: boundscheck = False
#cython: language_level = 3

import numpy as np

cdef int[:,:] update_car_sites(int[:,:] cars_list, int[:,:] sites): #duplicated function from main file
    '''
    updates sites: place cars in cells according to cars_list array
    cars_list - a list of cars, each element is [number of lane, number of place, velocity]
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    
    cdef int num_cars = len(cars_list)
    cdef int i
    cdef int lane
    cdef int place
    
    for lane in range(sites.shape[0]):
        for place in range(sites.shape[1]):
            sites[lane, place] = -1

    for i in range(num_cars):
        lane = cars_list[i, 0]
        place = cars_list[i,1]
        sites[lane, place] = cars_list[i, 2]
    return sites

cdef int neighb_car_type(int lane, int position, int distance, int max_distance, int n_hdv, int [:,:] cars_list):
    cdef int i
    cdef int t
    cdef int num_cars = len(cars_list)
    
    if distance >= max_distance:
        t = 0
    else:
        for i in range(num_cars):
            if (cars_list[i, 0] == lane and cars_list[i, 1] == position): #(current_position + dist[k] + 1)%num_cells):
                if i > n_hdv:
                    t = 2
                else:
                    t = 1
    return t

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
                v[i,lane,0,j] = sites[lane, (current_position + dist[k] + 1)%num_cells]
                t[i,lane,0,j] = neighb_car_type(lane, (current_position + dist[k] + 1)%num_cells, dist[k], g_max, n_hdv, cars_list)
                dist[k] += 1
                k += 1
                
                # backward cars positions (g_1) and speed (v_1) in the lane
                while sites[lane, (current_position - (dist[k] + 1))] < 0:
                    dist[k] += 1
                    if dist[k] > g_max:
                        break
                g[i,lane,1,j] += dist[k]
                v[i,lane,1,j] = sites[lane, (current_position - (dist[k] + 1))]
                t[i,lane,0,j] = neighb_car_type(lane, (current_position - (dist[k] + 1)), dist[k], g_max, n_hdv, cars_list)
                dist[k] += 1
                k += 1
    return g, v, t


cdef bint CL_incentive_criterion(int vel, int g_f_PL, int v_f_PL, int g_f_AL, int v_f_AL, int t_f_PL, int t_f_AL, str model, str agent_type):
    cdef bint CAV_change
    cdef double P_CAV_CL = 0.15

    if (model == 'S-NFS') or (model == 'KKW'):
        if agent_type == 'HDV':
            return (vel > g_f_PL + v_f_PL) and (vel < g_f_AL + v_f_AL)
        elif agent_type == 'CAV':
            return (vel > g_f_PL + v_f_PL) and (vel < g_f_AL + v_f_AL)
        elif agent_type == 'HAV':
            return True

    elif model == 'W184':
        if agent_type == 'HDV':
            return (g_f_PL <  g_f_AL) and (vel > 0) and (g_f_PL > 4)
        elif agent_type == 'CAV':
            CAV_change = (t_f_AL == 2) # or (np.random.random() < P_CAV_CL)  #(t_f_AL == 2 and t_f_PL == 1) or (t_f_AL == 2 and t_f_PL == 2) or ((t_f_AL == 1 and t_f_PL == 1) and (np.random.random()<0.1)
            #return ((g_f_PL <  g_f_AL) and (vel > 0) and (g_f_PL > 4) and (np.random.random() < P_CAV_CL)) or CAV_change
            return (g_f_PL <  g_f_AL) and (vel > 0) and (g_f_PL > 4)
        elif agent_type == 'HAV':
            return True

cdef bint CL_safety_criterion(int vel, int g_b_AL, int v_b_AL, str model, str agent_type):
    if (model == 'S-NFS') or (model == 'KKW'):
        return vel > v_b_AL - g_b_AL
    elif model == 'W184':
        return vel > v_b_AL - g_b_AL + 2

cpdef change_lane(int[:,:] cars_list, int[:,:] sites, int n_hdv, double P_lc, int num_change_lane, str model):
    cdef int[:] vel = cars_list[:,2]
    cdef int num_cars = len(cars_list)
    cdef int num_lanes = sites.shape[0]
    cdef int i
    cdef int lane
    cdef int pres_lane
    cdef int adj_lane
    cdef int cur_ind
    cdef str agent_type

    cdef int [:,:,:,:] g
    cdef int [:,:,:,:] v
    cdef int [:,:,:,:] t
    g, v, t = dist_and_vels(cars_list, sites, n_hdv)

    cdef double[:] r = np.random.random(num_cars)
    cdef int[:] arranged_ind = np.asarray(np.argsort(cars_list[:,1]), dtype = int)

    cdef int[:] g_nearest
      
    for i in range(num_cars):
        if i < n_hdv:
            agent_type = 'HDV'
        else:
            agent_type = 'CAV'

        if (r[i] < P_lc or i >= n_hdv):
            pres_lane = cars_list[i,0]
            adj_lane = (pres_lane + 1)%num_lanes
            if CL_incentive_criterion(vel[i], g[i,pres_lane,0,0], v[i,pres_lane,0,0], g[i,adj_lane,0,0], v[i,adj_lane,0,0], t[i,pres_lane,0,0], t[i,adj_lane,0,0], model, agent_type):
                if CL_safety_criterion(vel[i], g[i,adj_lane,1,0], v[i,adj_lane,1,0], model, agent_type):
                    cars_list[i,0] = adj_lane
                    num_change_lane += 1

    sites = update_car_sites(cars_list, sites)
    return sites, cars_list, num_change_lane
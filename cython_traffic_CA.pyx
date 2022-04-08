#cython: boundscheck = False
#cython: language_level = 3

import numpy as np
import models
import lane_change

cdef initialize_random_config(int num_lanes, int num_places, int num_cars):
    '''
    create initial random configuration with num_lanes lines and
    num_places in each line, and num_cars placed with velocity = 0

    input:
    num_lanes - number of lanes
    num_places - number of cells in the lane
    num_cars - number of cars in the system

    returns sites[num_lanes,num_places] - int matrix with exactly num_cars randomly distributed zeros.
    fills the cars array with a values from sites matrix
    Other fields are equal to -1 (which represents absence of the car)
    '''
    cdef int i
    cdef int j
    cdef int k = 0

    a = np.full(num_lanes*num_places, -1, dtype = int)
    random_places = np.random.choice(num_lanes*num_places, num_cars, replace = False)
    a[random_places] = 0
    
    cdef int[:,:] sites = a.reshape(num_lanes, num_places)
    cdef int[:,:] cars_list = np.zeros((num_cars, 3), dtype = int)
    cdef int[:] random_index = np.asarray(np.random.default_rng().choice(num_cars, size=num_cars, replace=False), dtype = int)

    for i in range(num_lanes):
        for j in range(num_places):
            if sites[i,j] > -1:
                cars_list[random_index[k],0] = i
                cars_list[random_index[k],1] = j
                cars_list[random_index[k],2] = sites[i,j]
                k += 1
    return sites, cars_list

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
    
    for lane in range(sites.shape[0]):
        for place in range(sites.shape[1]):
            sites[lane, place] = -1

    for i in range(num_cars):
        lane = cars_list[i, 0]
        place = cars_list[i,1]
        sites[lane, place] = cars_list[i, 2]
    return sites

cdef int[:,:] update_cars_list(int[:,:] cars_list, int [:] v, int num_cells):
    cdef int i

    for i in range(cars_list.shape[0]):
        cars_list[i,2] = v[i]
        cars_list[i,1] = (cars_list[i,1] + v[i])%num_cells
    return cars_list

cdef int[:] update_velocity(C, X, n_hdv, max_platoon_size, N_c, model, red_light = False):
        if model == 'S-NFS':
            return models.velocity_SNFS(C, X, n_hdv, max_platoon_size, N_c)
        elif model == 'W184':
            return models.velocity_W184(C, X, n_hdv, max_platoon_size, N_c, red_light = False)
        elif model == 'KKW':
            return models.velocity_KKW(C, X, n_hdv, max_platoon_size, N_c)
        else:
            print('ERROR. No model available called ', model)

cdef double flux_calc(int [:,:] sites): # it might be the same function as average velocity
    cdef int num_lanes = sites.shape[0] 
    cdef int num_places = sites.shape[1]
    cdef int i
    cdef int j
    cdef int average_vel = 0

    for i in range(num_lanes):
        for j in range(num_places):
            if sites[i,j] > -1:
                average_vel += sites[i,j]
    return average_vel/num_places

cdef double average_velocity(int [:,:] cars_list):
    '''
    Calculates average velocity of all vehicles
    '''
    cdef int num_cars = len(cars_list)
    cdef double av_vel = 0
    cdef int i
    
    if num_cars == 0:
        return 0
    for i in range(num_cars):
        av_vel += cars_list[i,2]
    return av_vel / num_cars

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
    cdef int[:] v
    cdef int episode
    cdef int [:,:,:] episodes = np.zeros((num_lanes, time_steady+num_iters+1, num_places), dtype = int)
    cdef int n_hdv = int(R_hdv*num_cars)

    X, C = initialize_random_config(num_lanes, num_places, num_cars)

    if visualise == True:
        fill_episode(X, 0, episodes)

    for episode in range(time_steady + num_iters):
        X, C, num_change_lane = lane_change.change_lane(C, X, n_hdv, P_lc, num_change_lane, model)
        #T_last_switch, red_light = traffic_lights(i = episode, T_last_switch = 0, T_g = 300, T_r = 200, red_light = False)
        v = update_velocity(C, X, n_hdv, max_platoon_size, N_c, model, red_light = False)
        C = update_cars_list(C, v, num_places)
        X = update_car_sites(C, X)

        if episode > time_steady:
            flux += flux_calc(X)
            v_av += average_velocity(C)
        
        if visualise == True:
            fill_episode(X, episode + 1, episodes)

    flux /= num_iters
    v_av /= num_iters
    change_lane_rate = num_change_lane/(num_iters*num_lanes*num_places)

    if visualise == True:
        return flux, change_lane_rate, v_av, episodes
    else:
        return flux, change_lane_rate, v_av
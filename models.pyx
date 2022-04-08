#cython: boundscheck = False
#cython: language_level = 3

import numpy as np

cdef int current_index(int[:] arranged_ind, int list_length, int i):
    cdef int j
    cdef int cur_ind = 0
    
    for j in range(list_length):
        if arranged_ind[j] == i:
            cur_ind = j
            break
    return cur_ind

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

cpdef int[:] velocity_SNFS(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c):
    '''
    Single move action: update system from t to t+1 with revised S-NFS model
    '''
    # Revised S-NFS model parameters
    cdef int V_max = 5
    cdef int S = 2
    cdef int G = 15
    cdef double q = 0.99
    cdef double r = 0.99

    cdef int num_cells = sites.shape[1]
    cdef int num_cars = len(cars_list)
    cdef int i
    cdef double[:] P_rbrake = np.array([1.0, 1.0, 1.0, 1.0]) #np.array([0.999, 0.99, 0.98, 0.01])
    cdef int[:,:] g
    cdef int[:,:] v_next
    cdef int[:,:] v = np.zeros((num_cars, 6), dtype = int)
    cdef int[:] s = np.random.choice((1, S), num_cars, p = [1-r, r])
    cdef int[:] v_new = np.zeros(num_cars, dtype = int)
    cdef int cur_ind
    cdef int ind
    cdef int v4_next
    cdef double[:] rand_array = np.random.random(num_cars)
    cdef int[:] arranged_ind = np.asarray(np.argsort(cars_list[:,1]), dtype = int)
    
    g, v_next = g_v_next(cars_list, sites, S)
    p = prob_slow_down(cars_list, g, v_next, G, P_rbrake)    

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
        v_new[i] = v[i,5]
    
    return v_new


cpdef int[:] velocity_W184(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c, bint red_light):
    '''
    Single move action: update system from t to t+1
    '''
    cdef int num_cells = sites.shape[1]
    cdef int num_places = sites.shape[2]
    cdef int num_cars = len(cars_list)
    cdef double[:] p = np.array([0.1, 0.3, 0.95])
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

    cdef tr_light_pos = 5000

    g, v_next = g_v_next(cars_list, sites, 1)

    # Acceleration and random brake for human-driven vehicles
    for i in range(n_hdv):
        if (cars_list[i,1] == tr_light_pos and red_light == True and cars_list[i,0] == 0):
            v[i] = 0
        elif ((g[i,0] > 4) or (g[i,0] >= 3 and r_3[i] < p[2]) or (g[i,0] == 2 and r_2[i] < p[1]) or (g[i,0] == 1 and r_1[i] < p[0])):
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
                    if (g[arranged_ind[ind],0] > 0):
                        v[i] = 1
                        break
                    else:
                        ind = (ind + 1)%num_cars
                        s += 1
                else: #next vehicle - human-driven
                    v[i] = 0
                    break
    return v

cpdef int[:] velocity_KKW(int[:,:] cars_list, int[:,:] sites, int n_hdv, int max_platoon_size, int N_c):
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
    return v
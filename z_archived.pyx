cdef double flux_calc(int [:,:] sites, str method, str model):
    '''
    Returns: density (float) and flux (float)
 
    Input: 
    sites - a (number of lanes)x(number of places) array with values are the cars velocities
    '''
    cdef int num_lanes = sites.shape[0] 
    cdef int num_places = sites.shape[1]
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
        flux = average_vel/num_places

    return flux  
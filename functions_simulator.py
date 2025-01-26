import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.special as spe
import torch


def inA(object):
    return object[0]==0

def inB(object):
    return not inA(object)

def ObjLearningTransfer(n):
    """
    Returns the list of the n objects used for the learning phase and the ntot - n objects used for the transfer phase (here n = 10 and 6 objects for the transfer phase)
    """
    obj = np.array([
        [1,0,1,0,1,0,1,0],
        [1,0,1,0,1,0,0,1],
        [1,0,1,0,0,1,1,0],
        [1,0,1,0,0,1,0,1],
        [1,0,0,1,1,0,1,0],
        [1,0,0,1,1,0,0,1],
        [1,0,0,1,0,1,1,0],
        [1,0,0,1,0,1,0,1],
        [0,1,1,0,1,0,1,0],
        [0,1,1,0,1,0,0,1],
        [0,1,1,0,0,1,1,0],
        [0,1,1,0,0,1,0,1],
        [0,1,0,1,1,0,1,0],
        [0,1,0,1,1,0,0,1],
        [0,1,0,1,0,1,1,0],
        [0,1,0,1,0,1,0,1],
    ])
    #8 first in B, 8 last in A

    A = obj[8:, :]
    B = obj[:8, :]

    A2 = np.copy(A)
    np.random.shuffle(A2)
    B2 = np.copy(B)
    np.random.shuffle(B2)
    n2 = int(n/2)
    learning = np.concatenate((A2[:n2,:],B2[:n2,:])) # 8,10
    transfer = np.concatenate((A2[n2:,:], B2[n2:,:])) # 8, 4
    np.random.shuffle(learning)
    np.random.shuffle(transfer)
    return transfer, learning

def GainOutput(P_mid, obj): #target = label of current image
    gain = np.zeros((2, 8))
    if inA(obj):
        gain[0,:] = 2 * P_mid
        gain[1,:] = - gain[0,:]
    else:
        gain[1,:] = 2 * P_mid
        gain[0,:] = - gain[1,:]
    return gain

def EWA(W_not_renorm, eta, cred):
    """
    Updates the weights `W_not_renorm` using the Exponentially Weighted Average (EWA) method.
    """
    res = W_not_renorm.copy()
    res[:, :] *= np.exp(eta * cred[:, :])
    return res

def Poisson(lamb, T):
    nb_points = stats.poisson.rvs(lamb*T)
    points = stats.uniform.rvs(0,T, size = nb_points)
    return points

        

def discrete_Hawkes(input_neurons, W, g): 
    """
    input_neurons: np matrix (nI, N) with N = T/dt
    W: matrix (nJ, nI)
    g: np array (A)
    """
    nJ, nI = W.shape
    A = g.shape[0]
    N = input_neurons.shape[1]

    # Initialize the probability output matrix
    prob_output = np.zeros((nJ, N))

    # Accumulate the weighted sums over the lag values in g
    for s in range(A):
        if s < N:
            prob_output[:, s:] += g[s] * np.dot(W, input_neurons[:, :N-s])


def simulator_vectorized(eta_output, s, f = 300, N = 5000, nb_obj_max = 100): #s = threshold (proportion to reach, between 0 and 1), f = firing rate (freq of spikes, nb by second)

    K_input = 8
    K_output = 2
    s = N*s
    p = 5 * f / N #0.3
    

    transfer, learning = ObjLearningTransfer(10) #gives the 10 learning objects and 6 transfer objects
    cond = True #when a bloc is correctly classified it becomes false
    answers = []
    accuracy = []
    times = np.zeros(nb_obj_max)
    W_output = np.zeros((K_output, K_input)) + 1/K_input
    cum_gain = np.zeros((K_output, K_input))
    nb_obj = 0
    bloc = np.copy(learning)

    #Learning phase
    while cond:
        m = 0
        condbis = True #becomes false when there is a mistake
        np.random.shuffle(bloc)
        if np.sum(accuracy[-10:])==10:
            num = 5
        else:
            num = 10
        while condbis and m < num:
            #Simulation input neurons
            prob_input = np.transpose(np.array([p*bloc[m,:] for i in range(N)])) #proba until max time
            input_neurons_out = stats.bernoulli.rvs(prob_input) #simu until max time (K_input, N)
            
            #Simulation output neurons
            prob_output = np.sum(W_output[:,None,:] * input_neurons_out.T, axis = 2)
            output_neurons = stats.bernoulli.rvs(prob_output)

            #Approximated BM
            S = np.cumsum(output_neurons, axis = 1)

            #reac time and classif
            SA = S[0,:]
            SB = S[1,:]
            timesA = np.where(SA >= s)[0]
            timesB = np.where(SB >= s)[0]
            if np.shape(timesA)[0]==0 and np.shape(timesB)[0] == 0:
                times[nb_obj] = N
                answers.append(-1)
            elif np.shape(timesA)[0]==0: #B reaches threshold first
                times[nb_obj] = timesB[0]
                answers.append(1)
            elif np.shape(timesB)[0]==0:
                times[nb_obj] = timesA[0]
                answers.append(0)
            else:
                if timesA[0] < timesB[0]:
                    times[nb_obj] = timesA[0]
                    answers.append(0)
                else:
                    times[nb_obj] = timesB[0]
                    answers.append(1)
            if answers[-1] == bloc[m,0] :
                accuracy.append(1)
            else:
                accuracy.append(0)
            
            N_time_steps = int(times[nb_obj])
            P_input = np.sum(input_neurons_out[:,:N_time_steps], axis = 1) / N_time_steps
            #Weights update
            gain_output = GainOutput(P_input, bloc[m,:]) 
            cum_gain += gain_output
            W_output = spe.softmax(eta_output * cum_gain, axis = 1)
            #W_output_tot[:,:,nb_obj] = W_output
   
            nb_obj += 1
            m+=1
            if nb_obj == nb_obj_max - 18 :
                return nb_obj_max + 1, times, accuracy
            condbis = condbis and (accuracy[-1] == 1) 
            
        #condition update
        if np.sum(accuracy[-15:])==15 or nb_obj == nb_obj_max - 18:
            cond = False

    #Transfer phase
    transfer_objects = np.zeros((18,8))
    bloc_transfer = np.copy(transfer)
    np.random.shuffle(bloc_transfer)
    transfer_objects[:6,:] = bloc_transfer 
    transfer_objects[6:12,:] = bloc_transfer 
    transfer_objects[12:,:] = bloc_transfer 

    prob_input = np.transpose(np.array([p*transfer_objects for i in range(N)])) 
    input_neurons_out = stats.bernoulli.rvs(prob_input) #(K_input, 18, N)

    prob_output = np.einsum('imt, ki -> kmt', input_neurons_out, W_output)
    output_neurons = stats.bernoulli.rvs(prob_output) #(K_output, 18, N)
    S = np.cumsum(output_neurons, axis = 2)  #(K_output, 18, N)
    #reac time and classif
    
    for m in range(18):
        SA = S[0,m,:]
        SB = S[1,m,:]
        timesA = np.where(SA >= s)[0]
        timesB = np.where(SB >= s)[0]
        if np.shape(timesA)[0]==0 and np.shape(timesB)[0] == 0:
            times[nb_obj] = N
            answers.append(-1)
        elif np.shape(timesA)[0]==0: #B reaches threshold first
            times[nb_obj] = timesB[0]
            answers.append(1)
        elif np.shape(timesB)[0]==0:
            times[nb_obj] = timesA[0]
            answers.append(0)
        else:
            if timesA[0] < timesB[0]:
                times[nb_obj] = timesA[0]
                answers.append(0)
            else:
                times[nb_obj] = timesB[0]
                answers.append(1)
        nb_obj +=1

    n = nb_obj
    fill_time = np.sum(times[n-18:n])/18 #average time transfer phase
    times[n:]+=fill_time

    #conversion in seconds (max time = 5s)
    times = 5 * times / N
    return n, times, accuracy

def simulator_vectorized_fix_nb_obj(eta_output, s, nb_obj_learning, f = 300, N = 5000): #s = threshold (proportion to reach, between 0 and 1), f = firing rate (freq of spikes)
    """
    nb_obj_learning: objects before the transfer phase
    """
    K_input = 8
    K_output = 2
    s = N*s
    p = 5 * f / N #0.3
    
    #W_output_tot = np.zeros((K_output, K_input, nb_obj_max))

    transfer, learning = ObjLearningTransfer(10) #gives the 10 learning objects and 6 transfer objects
    cond = True #when a bloc is correctly classified it becomes false
    answers = []
    accuracy = []
    times = np.zeros(nb_obj_learning + 18)
    W_output = np.zeros((K_output, K_input)) + 1/K_input
    cum_gain = np.zeros((K_output, K_input))
    nb_obj = 0
    bloc = np.copy(learning)

    #Learning phase
    while cond:
        m = 0
        condbis = True #becomes false when there is a mistake
        np.random.shuffle(bloc)
        if np.sum(accuracy[-10:])==10:
            num = 5
        else:
            num = 10
        while condbis and m < num:
            #Simulation input neurons
            prob_input = np.transpose(np.array([p*bloc[m,:] for i in range(N)])) #proba until max time
            input_neurons_out = stats.bernoulli.rvs(prob_input) #simu until max time (K_input, N)
            
            #Simulation output neurons
            prob_output = np.sum(W_output[:,None,:] * input_neurons_out.T, axis = 2)
            output_neurons = stats.bernoulli.rvs(prob_output)

            #Approximated BM
            S = np.cumsum(output_neurons, axis = 1)

            #reac time and classif
            SA = S[0,:]
            SB = S[1,:]
            timesA = np.where(SA >= s)[0]
            timesB = np.where(SB >= s)[0]
            if np.shape(timesA)[0]==0 and np.shape(timesB)[0] == 0:
                times[nb_obj] = N
                answers.append(-1)
            elif np.shape(timesA)[0]==0: #B reaches threshold first
                times[nb_obj] = timesB[0]
                answers.append(1)
            elif np.shape(timesB)[0]==0:
                times[nb_obj] = timesA[0]
                answers.append(0)
            else:
                if timesA[0] < timesB[0]:
                    times[nb_obj] = timesA[0]
                    answers.append(0)
                else:
                    times[nb_obj] = timesB[0]
                    answers.append(1)
            if answers[-1] == bloc[m,0] :
                accuracy.append(1)
            else:
                accuracy.append(0)
            
            N_time_steps = int(times[nb_obj])
            P_input = np.sum(input_neurons_out[:,:N_time_steps], axis = 1) / N_time_steps
            #Weights update
            gain_output = GainOutput(P_input, bloc[m,:]) 
            cum_gain += gain_output
            W_output = spe.softmax(eta_output * cum_gain, axis = 1)
            #W_output_tot[:,:,nb_obj] = W_output
   
            nb_obj += 1
            m+=1
            condbis = condbis and (accuracy[-1] == 1) 
            
            #print(nb_obj)
            if nb_obj == nb_obj_learning:
                condbis = False
        #condition update
        if nb_obj == nb_obj_learning :
            cond = False

    #Transfer phase
    transfer_objects = np.zeros((18,8))
    bloc_transfer = np.copy(transfer)
    np.random.shuffle(bloc_transfer)
    transfer_objects[:6,:] = bloc_transfer 
    transfer_objects[6:12,:] = bloc_transfer 
    transfer_objects[12:,:] = bloc_transfer 

    prob_input = np.transpose(np.array([p*transfer_objects for i in range(N)])) 
    input_neurons_out = stats.bernoulli.rvs(prob_input) #(K_input, 18, N)

    prob_output = np.einsum('imt, ki -> kmt', input_neurons_out, W_output)
    output_neurons = stats.bernoulli.rvs(prob_output) #(K_output, 18, N)
    S = np.cumsum(output_neurons, axis = 2)  #(K_output, 18, N)
    #reac time and classif
    
    for m in range(18):
        SA = S[0,m,:]
        SB = S[1,m,:]
        timesA = np.where(SA >= s)[0]
        timesB = np.where(SB >= s)[0]
        if np.shape(timesA)[0]==0 and np.shape(timesB)[0] == 0:
            times[nb_obj] = N
            answers.append(-1)
        elif np.shape(timesA)[0]==0: #B reaches threshold first
            times[nb_obj] = timesB[0]
            answers.append(1)
        elif np.shape(timesB)[0]==0:
            times[nb_obj] = timesA[0]
            answers.append(0)
        else:
            if timesA[0] < timesB[0]:
                times[nb_obj] = timesA[0]
                answers.append(0)
            else:
                times[nb_obj] = timesB[0]
                answers.append(1)
        nb_obj +=1

    n = nb_obj
    fill_time = np.sum(times[n-18:n])/18 #average time transfer phase
    times[n:]+=fill_time

    #conversion in seconds (max time = 5s)
    times = 5 * times / N
    return times, accuracy


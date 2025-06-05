import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Simulation.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import torch




# define rate limit handler function
if __name__ == '__main__':
    
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 50

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.9,
                  "beta_max": 1.2,
                  "beta_min": 0.1}
  
    
    
    
    Generator=gg.CreateGraphFrmDB()
    g = Generator.CreateGraph(parameters,graphModel='FB') 
    g1=g
    g2=g
    g3 = g
    g4 = g
    g5 = g
    Simulator = sim.RumorSimulator()
    # Run the simulation
    
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
     
    typeOfSim=2
    k=int(0.05*g.number_of_nodes())
    i=0
    blockPeriod=10
    Tdet = 2
    
    #aux0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='none')

    #aux_1 = Simulator.runSimulation(g1, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_PR',k=k,blockPeriod=blockPeriod,Tdet=Tdet)
    
    #aux_2 = Simulator.runSimulation(g2, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_BCN',k=k,blockPeriod=blockPeriod,Tdet=Tdet)
    
    aux_3 = Simulator.runSimulation(g3, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_MINJUGBN',k=k,blockPeriod=blockPeriod,Tdet=Tdet)
    
    #aux_4 = Simulator.runSimulation(g4, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_BMDBj',k=k,blockPeriod=blockPeriod,Tdet=2)
    
    aux_5 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MDCTCS',k=k,Tdet=Tdet)
    
    aux_6 = Simulator.runSimulation(g5, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_ARM',k=k,blockPeriod=blockPeriod,Tdet=Tdet)
    
        
    
    
    l=[aux_3,aux_5,aux_6]
    
    
    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
   
    
    Simulator.DisplyResults( l,resultType=typeOfSim,save=False,imageName="")
    """ # Iterate through the DataFrame and print the 'Age' attribute
    for index, row in df.iterrows():
        AccpR = row['AccpR']
        print(f"{index} has acceptance of {AccpR}.")
    """
    
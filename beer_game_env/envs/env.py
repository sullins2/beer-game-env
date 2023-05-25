from pickle import FALSE
import cloudpickle
import gym
from gym import error, spaces
from gym.utils import seeding
import itertools
from collections import deque
import numpy as np
from gym.wrappers import FilterObservation
import random

from numpy.core.multiarray import datetime_as_string
from numpy.core.numeric import ones

# Here we want to define the agent class for the BeerGame
class Agent(object):
	
	# initializes the agents with initial values for IL, OO and saves self.agentNum for recognizing the agents.
  def __init__(self, agentNum, IL, AO, AS, c_h, c_p, eta, compuType, config):
    self.agentNum = agentNum
    self.IL = IL		# Inventory level of each agent - changes during the game
    self.OO = 0		# Open order of each agent - changes during the game
    self.ASInitial = AS # the initial arriving shipment.
    self.ILInitial = IL	# IL at which we start each game with this number
    self.AOInitial = AO	# OO at which we start each game with this number
    self.config = config	# an instance of config is stored inside the class
    self.curState = []  # this function gets the current state of the game
    self.nextState = []
    self.curReward = 0 	# the reward observed at the current step
    self.cumReward = 0 	# cumulative reward; reset at the begining of each episode
    self.totRew = 0    	# it is reward of all players obtained for the current player.
    self.c_h=c_h		# holding cost
    self.c_p = c_p		# backorder cost
    self.eta = eta		# the total cost regulazer
    self.AS = np.zeros((1,1)) 	# arriced shipment 
    self.AO = np.zeros((1,1))	# arrived order
    self.action=0		# the action at time t
    self.totalR = 0

    self.TTT = 0
    self.srdqnBaseStock = []	# this holds the base stock levels that srdqn has came up with. added on Nov 8, 2017
    self.T = 0
    self.bsBaseStock = 0  
    self.init_bsBaseStock = 0 
    self.nextObservation = []

	# reset player information
  def resetPlayer(self, T):
    self.IL = self.ILInitial
    self.OO = 0
    self.AS = np.squeeze(np.zeros((1,T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10 ))) 	# arriced shipment 
    self.AO = np.squeeze(np.zeros((1,T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10 )))	# arrived order	
    if self.agentNum != 0:
      for i in range(self.config.leadRecOrderUp_aux[self.agentNum - 1]):
        self.AO[i] = self.AOInitial[self.agentNum - 1]
    for i in range(self.config.leadRecItemUp[self.agentNum]):
      self.AS[i] = self.ASInitial
    self.curReward = 0 # the reward observed at the current step
    self.cumReward = 0 # cumulative reward; reset at the begining of each episode	
    self.action= [] 
    self.srdqnBaseStock = []	# this holds the base stock levels that srdqn has came up with. added on Nov 8, 2017
    self.T = T
    self.curObservation = self.getCurState(1)  # this function gets the current state of the game
    self.nextObservation = []
    self.totalR = 0

	# updates the IL and OO at time t, after recieving "rec" number of items 
  def recieveItems(self, time):
    self.IL = self.IL + self.AS[time] # inverntory level update
    self.OO = self.OO - self.AS[time] # invertory in transient update
		
	
	# find action Value associated with the action list
  def actionValue(self,curTime,playType, BS):
    
    if not BS:
      actionList = [-2,-1,0,1,2]
    else:
      actionList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
    
    if not BS: #SRDQN
      a = max(0,actionList[np.argmax(self.action)] + self.AO[curTime])
    else:
      a = max(0,actionList[np.argmax(self.action)])
    
    return a
			
  # getReward returns the reward at the current state 
  def getReward(self):
    # cost (holding + backorder) for one time unit
    self.curReward= (self.c_p * max(0,-self.IL) + self.c_h * max(0,self.IL))/200.0 # self.config.Ttest # 
    self.curReward = -self.curReward;		# make reward negative, because it is the cost
    
    # sum total reward of each agent
    self.cumReward = self.config.gamma*self.cumReward + self.curReward		

  # This function returns a np.array of the current state of the agent	
  def getCurState(self,t):
    if self.config.ifUseASAO:
      if self.config.if_use_AS_t_plus_1:
        curState= np.array([-1*(self.IL<0)*self.IL,1*(self.IL>0)*self.IL,self.OO,self.AS[t],self.AO[t]])
      else:
        curState= np.array([-1*(self.IL<0)*self.IL,1*(self.IL>0)*self.IL,self.OO,self.AS[t-1],self.AO[t]])
    else:
      curState= np.array([-1*(self.IL<0)*self.IL,1*(self.IL>0)*self.IL,self.OO])

    if self.config.ifUseActionInD:
      a = self.config.actionList[np.argmax(self.action)]
      curState= np.concatenate((curState, np.array([a])))

    return curState

###################################################################
###################################################################
## BEER GAME ######################################################
###################################################################
class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=4, env_type='classical', n_turns_per_game=100,
                 add_noise_initialization=False, seed=None, test_mode=False):
        super().__init__()
        c = Config()
        config, unparsed = c.get_config()
        self.config = config
        # print("TEST MODE: ", test_mode)
        self.test_mode = test_mode
        if self.test_mode:
          self.init_test_demand()
        self.curGame = 1 # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.m = 10
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents 
        self.T = 0
        self.demand = []
        self.playType = []  # "train" or "test"
        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()
        self.totRew = 0    # it is reward of all players obtained for the current player.
        self.totalReward = 0
        self.n_agents = n_agents
        if test_mode == False:
          self.n_turns = n_turns_per_game
        else:
          self.n_turns = n_turns_per_game
        seed  = random.randint(0,1000000)
        self.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.totalTotal = 0

        # Agent 0 has 5 (-2, ..., 2) + AO       
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))

        # Create observation space = m
        # spaces = {}
        # for i in range(self.m):
        #     spaces[f'current_stock_minus{i}'] = gym.spaces.Box(low=np.array([0]), high=np.array([30]), shape=(1,))
        #     spaces[f'current_stock_plus{i}'] = gym.spaces.Box(low=np.array([0]), high=np.array([30]), shape=(1,))
        #     spaces[f'OO{i}'] = gym.spaces.Box(low=np.array([0]), high=np.array([20]), shape=(1,))
        #     spaces[f'AS{i}'] = gym.spaces.Box(low=np.array([0]), high=np.array([5]), shape=(1,))
        #     spaces[f'AO{i}'] = gym.spaces.Box(low=np.array([0]), high=np.array([5]), shape=(1,))
        
        # Create observation space = m
        ob_spaces = {}
        for i in range(self.m):
            ob_spaces[f'current_stock_minus{i}'] = spaces.Discrete(5)
            ob_spaces[f'current_stock_plus{i}'] = spaces.Discrete(5)
            ob_spaces[f'OO{i}'] = spaces.Discrete(5)
            ob_spaces[f'AS{i}'] = spaces.Discrete(5)
            ob_spaces[f'AO{i}'] = spaces.Discrete(5)
        

        # print(spaces)
        # self.deques = []
        # for i in range(self.n_agents):
        #   deques = {}
        #   deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
        #   deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
        #   deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
        #   deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
        #   deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
        #   self.deques.append(deques)

        # dict_space = gym.spaces.Dict(ob_spaces)
        # self.observation_space = gym.spaces.Tuple(tuple([dict_space] * 4))
        x = [150, 150, 70, 15, 15]
        oob = []
        for _ in range(self.m):
          for ii in range(len(x)):
            oob.append(x[ii])
        self.observation_space = gym.spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * 4))

        # obs_space = [30 for _ in range(self.m*5*4)]
        # obs_space = [(0, 30) for _ in range(self.m*5*4)]
        # obs_space = []
        # for x in range(self.m*5):
        #   obs_space.append(gym.spaces.Discrete(30))
        # self.observation_space = gym.spaces.Tuple(tuple([obs_space,obs_space,obs_space,obs_space]))
        # spaces.MultiDiscrete([30,30,30,30,30])
        # self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))
        #self.observation_space = spaces.Tuple(tuple([spaces.Discrete(30) * self.m * 5,spaces.Discrete(30) * self.m * 5, spaces.Discrete(30) * self.m * 5, spaces.Discrete(30) * self.m * 5]))
        # f = gym.spaces.Discrete(30)
        # self.observation_space = gym.spaces.Tuple(tuple([f]*self.m*5) * 4)
        # low = np.array([-5, -5, -5, -5, -5])
        # high = -np.array([-5, -5, -5, -5, -5])
        # low = np.zeros(5 * self.m)
        # high = np.ones(5 * self.m)*100
        # print(low)
        # print(high)
        # r = [gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100)]
        # self.observation_space = gym.spaces.Tuple([gym.spaces.Box(low, high, dtype=np.float32)]*4)
        # self.observation_space = []
        # for ii in range(4):
        #   self.observation_space.append(r)
        # self.observation_space = gym.spaces.Tuple(tuple(self.observation_space))
        # self.observation_space = r*4
        # os = gym.spaces.MultiDiscrete([100,100,100,100,100])
        # self.observation_space = gym.spaces.Tuple(tuple([os] * 4))
        
        # x = [gym.spaces.Discrete(5),
        #                       gym.spaces.Discrete(5),
        #                       gym.spaces.Discrete(5),
        #                       gym.spaces.Discrete(5),
        #                       gym.spaces.Discrete(5)]
        # x = []
        # for _ in range(1):
        #   for _ in range(5):
        #     x.append(gym.spaces.Discrete(30))


        # # self.observation_space = gym.spaces.Tuple(tuple([self._get_ob()] * 4))
        # self.observation_space = gym.spaces.Tuple(x * 4)

        # self.observation_space = gym.spaces.Tuple(tuple([self._get_ob] * 4))

        # sa_action_space = [100, 100, 10, 10, 10]
        # if len(sa_action_space) == 1:
        #     sa_action_space = spaces.Discrete(sa_action_space[0])
        # else:
        #     sa_action_space = spaces.MultiDiscrete(sa_action_space)
        # self.observation_space = spaces.Tuple(spaces.Discrete(5)) #spaces.Tuple(tuple(4 * [sa_action_space]))
        # self._obs_length = 5


        # self.observation_space = [spaces.MultiDiscrete([100,100,11,11,11]),spaces.MultiDiscrete([100,100,11,11,11]),spaces.MultiDiscrete([100,100,11,11,11]),spaces.MultiDiscrete([100,100,11,11,11])]
        print("OBS SPACE")
        print(self.observation_space)
        # print(obs_space)

    def _get_ob(self):
      obs = []
      for _ in range(5):
        obs.append(spaces.Discrete(101))  # Values from 0 to 100 (inclusive)
      return obs
      # return gym.spaces.Tuple([gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100),gym.spaces.Discrete(100)])

    # createAgent : Create agent objects (agentNum,IL,OO,c_h,c_p,type,config)
    def createAgent(self): 	
      agentTypes = self.config.agentTypes 
      return [Agent(i,self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i], 
                                self.config.c_h[i], self.config.c_p[i], self.config.eta[i], 
                                agentTypes[i],self.config) for i in range(self.config.NoAgent)]
      
    # planHorizon : Find a random planning horizon
    def planHorizon(self):
      # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
      #output: The planning horizon which is chosen randomly.
      return random.randint(self.n_turns, self.n_turns)# self.config.TLow,self.config.TUp)

    # this function resets the game for start of the new game
    def resetGame(self, demand, playType):
      self.playType = playType  #"train" or "test"
      self.demand = demand
      self.curTime = 0
      self.curGame += 1
      self.totIterPlayed += self.T
      self.T = self.planHorizon() #+ 1 # Hacky but was a problem
      self.totalReward = 0
      
      self.deques = [] # make this all a function
      for i in range(self.n_agents):
        deques = {}
        deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
        deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
        deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
        deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
        deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
        self.deques.append(deques) 

      # reset the required information of player for each episode
      for k in range(0,self.config.NoAgent):
        self.players[k].resetPlayer(self.T)

      # update OO when there are initial IL,AO,AS
      self.update_OO()


    # correction on cost at time T according to the cost of the other players
    def getTotRew(self):
      totRew = 0
      for i in range(self.config.NoAgent):
        # sum all rewards for the agents and make correction
        totRew += self.players[i].cumReward

      for i in range(self.config.NoAgent):
        self.players[i].curReward += self.players[i].eta*(totRew - self.players[i].cumReward) #/(self.T)

    def getAction(self, k):
      self.players[k].action = np.zeros(self.config.actionListLenOpt)
      
      # print(self.players[k].action)	
      if self.config.demandDistribution == 2:
        if self.curTime   and self.config.use_initial_BS <= 4:
          self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
              max(0,(self.players[k].int_bslBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1	
        else: 
          self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
              max(0,(self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1	
      else:
        self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
              max(0,(self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1	
      # print(self.players[k].action)	


    def next(self):
      # get a random leadtime		
      leadTimeIn = random.randint(self.config.leadRecItemLow[self.config.NoAgent-1], self.config.leadRecItemUp[self.config.NoAgent-1]) 
      
      # handle the most upstream recieved shipment
      self.players[self.config.NoAgent-1].AS[self.curTime + leadTimeIn] += self.players[self.config.NoAgent-1].actionValue(self.curTime, self.playType, BS=True)

      for k in range(self.config.NoAgent-1,-1,-1): # [3,2,1,0]
        
        # get current IL and Backorder
        current_IL = max(0, self.players[k].IL)
        current_backorder = max(0, -self.players[k].IL)

        # TODO: We have get the AS and AO from the UI and update our AS and AO, so that code update the corresponding variables
        
        # increase IL and decrease OO based on the action, for the next period 
        self.players[k].recieveItems(self.curTime)
        
        # observe the reward
        possible_shipment = min(current_IL + self.players[k].AS[self.curTime], current_backorder + self.players[k].AO[self.curTime])
        
        # plan arrivals of the items to the downstream agent
        if self.players[k].agentNum > 0:
          leadTimeIn = random.randint(self.config.leadRecItemLow[k-1], self.config.leadRecItemUp[k-1])
          self.players[k-1].AS[self.curTime + leadTimeIn] += possible_shipment

        # update IL
        self.players[k].IL -= self.players[k].AO[self.curTime]
        # observe the reward
        self.players[k].getReward()
        rewards = [-1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]
        
        # update next observation 
        self.players[k].nextObservation = self.players[k].getCurState(self.curTime+1)
      
      if self.config.ifUseTotalReward: # default is false
        # correction on cost at time T
        if self.curTime == self.T:
          self.getTotRew()					

      self.curTime +=1				
    
    def handelAction(self, action):
      # get random lead time 
      leadTime = random.randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])

      # set AO
      BS = False
      self.players[0].AO[self.curTime] += self.demand[self.curTime]
      for k in range(0,self.config.NoAgent): 
      
        if k == 0:
          self.players[k].action = np.zeros(5)#self.config.actionListLenOpt)
          # a = int(max(0, (action[k] - 0) + self.players[k].AO[self.curTime]))
          # if self.test_mode:
          #   print(action, self.test_mode)
          
          self.players[k].action[action[0]] = 1
          BS = False
          # a = max(0,self.config.actionList[np.argmax(self.action)] * self.config.action_step + self.AO[curTime])
        else:
          self.getAction(k)
          BS = True
        
        # self.players[k].srdqnBaseStock += [self.players[k].actionValue( \
        #   self.curTime, self.playType) + self.players[k].IL + self.players[k].OO]
        
        # updates OO and AO at time t+1
        self.players[k].OO += self.players[k].actionValue(self.curTime, self.playType, BS=BS) # open order level update
        leadTime = random.randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])
        if self.players[k].agentNum < self.config.NoAgent-1:
          if k == 0:
            self.players[k+1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime, self.playType, BS=False) # open order level update
          else:
            self.players[k+1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime, self.playType, BS=True) # open order level update



    # check the Shang and Song (2003) condition, and if it works, obtains the base stock policy values for each agent
    def getOptimalSol(self):
      # if self.config.NoAgent !=1:
      if self.config.NoAgent !=1 and 1 == 2:
        # check the Shang and Song (2003) condition.
        for k in range(self.config.NoAgent-1):
          if not (self.players[k].c_h == self.players[k+1].c_h and self.players[k+1].c_p == 0):
            self.ifOptimalSolExist = False
          
        # if the Shang and Song (2003) condition satisfied, it runs the algorithm
        if self.ifOptimalSolExist == True:
          calculations = np.zeros((7,self.config.NoAgent))
          for k in range(self.config.NoAgent):
            # DL_high
            calculations[0][k] = ((self.config.leadRecItemLow +self.config.leadRecItemUp + 2)/2 \
                      + (self.config.leadRecOrderLow+self.config.leadRecOrderUp + 2)/2)* \
                      (self.config.demandUp - self.config.demandLow- 1)
            if k > 0:
              calculations[0][k] += calculations[0][k-1]
            # probability_high
            nominator_ch = 0
            low_denominator_ch = 0				
            for j in range(k,self.config.NoAgent):
              if j < self.config.NoAgent-1:
                nominator_ch += self.players[j+1].c_h
              low_denominator_ch += self.players[j].c_h 
            if k == 0:
              high_denominator_ch = low_denominator_ch
            calculations[2][k] = (self.players[0].c_p + nominator_ch)/(self.players[0].c_p + low_denominator_ch + 0.0)
            # probability_low
            calculations[3][k] = (self.players[0].c_p + nominator_ch)/(self.players[0].c_p + high_denominator_ch + 0.0)
          # S_high
          calculations[4] = np.round(np.multiply(calculations[0],calculations[2]))
          # S_low
          calculations[5] = np.round(np.multiply(calculations[0],calculations[3]))
          # S_avg
          calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
          # S', set the base stock values into each agent.
          for k in range(self.config.NoAgent):
            if k == 0:
              self.players[k].bsBaseStock = calculations[6][k]
              
            else:
              self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k-1]
              if self.players[k].bsBaseStock < 0:
                self.players[k].bsBaseStock = 0
      elif self.config.NoAgent ==1:				
        if self.config.demandDistribution==0:
          self.players[0].bsBaseStock = np.ceil(self.config.c_h[0]/(self.config.c_h[0]+self.config.c_p[0]+ 0.0))*((self.config.demandUp-self.config.demandLow-1)/2)*self.config.leadRecItemUp
      elif 1 == 1:
        f = self.config.f
        f_init = self.config.f_init
        for k in range(self.config.NoAgent):
          self.players[k].bsBaseStock = f[k]
          self.players[k].int_bslBaseStock = f_init[k]
	

     
    def update_OO(self):
      for k in range(0,self.config.NoAgent):
        if k < self.config.NoAgent - 1:
          self.players[k].OO = sum(self.players[k+1].AO) + sum(self.players[k].AS)
      else:
        self.players[k].OO = sum(self.players[k].AS)


    def _get_observations(self):
        # keep a data structure here with fixed size of m
        # current observation should be getCurState + m-1 last ones
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            observations[i] = {
                                'current_stock_minus': int(curState[0]),
                                'current_stock_plus': int(curState[1]), 
                               'OO': int(curState[2]), 
                               'AS': int(curState[3]), 
                               'AO': int(curState[4])}
        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    # TODO: Incorporate testing here
    def reset(self):
        if self.test_mode:
          demand = self.test_deq.popleft()
          # print("Test:", demand)
          # print("test")
          # Reset test demands for next tests
          if not self.test_deq:
            # print("reset")
            self.init_test_demand()
        else:
          #self.init_test_demand() # added so that it resets for next test without commenting out unused that will be left over
          demand = [random.randint(0, 2) for _ in range(102)] 
          
        # This resets self.deque
        self.resetGame(demand, "train")

        observations = [None] * self.n_agents

        # This does it again
        self.deques = []
        for i in range(self.n_agents):
          deques = {}
          deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
          deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
          deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
          deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
          deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
          self.deques.append(deques)
            
        # prepend current observation
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[],[],[],[]]
        for i in range(self.n_agents):
          spaces = {}
          for j in range(self.m):
              obs[i].append(self.deques[i]['current_stock_minus'][j])
              obs[i].append(self.deques[i]['current_stock_plus'][j])
              obs[i].append(self.deques[i]['OO'][j])
              obs[i].append(self.deques[i]['AS'][j])
              obs[i].append(self.deques[i]['AO'][j])
              spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
              spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
              spaces[f'OO{j}'] = self.deques[i]['OO'][j]
              spaces[f'AS{j}'] = self.deques[i]['AS'][j]
              spaces[f'AO{j}'] = self.deques[i]['AO'][j]
          
          observations[i] = spaces
      
        obs_array = np.array([np.array(row) for row in obs])
        # print("E",obs_array)
        return obs_array#observations #self._get_observations()

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        print("")
        # print('\n' + '=' * 20)
        # print('Turn:     ', self.turn)
        # print('Stocks:   ', ", ".join([str(x) for x in self.stocks]))
        # print('Orders:   ', [list(x) for x in self.orders])
        # print('Shipments:', [list(x) for x in self.shipments])
        # print('Last incoming orders:  ', self.next_incoming_orders)
        # print('Cum holding cost:  ', self.cum_stockout_cost)
        # print('Cum stockout cost: ', self.cum_holding_cost)
        # print('Last holding cost: ', self.holding_cost)
        # print('Last stockout cost:', self.stockout_cost)

    def step(self, action: list):
        # sanity checks
        # if self.done:
        #     raise error.ResetNeeded('Environment is finished, please run env.reset() before taking actions')
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")
        
        self.handelAction(action)
        self.next()
           
        if self.curTime == self.T+1:
            
            for i in range(self.n_agents):
              self.players[i].getReward()
            rewards = [1*self.players[i].curReward for i in range(0,self.config.NoAgent)]
            self.done = [True] * 4
          
            # get current observation, prepend to deque
            for i in range(self.n_agents):
                curState = self.players[i].getCurState(self.curTime)
                self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
                self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
                self.deques[i]['OO'].appendleft(int(curState[2]))
                self.deques[i]['AS'].appendleft(int(curState[3]))
                self.deques[i]['AO'].appendleft(int(curState[4]))

            # return entire m observations
            obs = [[],[],[],[]]
            observations = [None] * self.n_agents
            for i in range(self.n_agents):
              spaces = {}
              for j in range(self.m):
                  obs[i].append(self.deques[i]['current_stock_minus'][j])
                  obs[i].append(self.deques[i]['current_stock_plus'][j])
                  obs[i].append(self.deques[i]['OO'][j])
                  obs[i].append(self.deques[i]['AS'][j])
                  obs[i].append(self.deques[i]['AO'][j])
                  spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                  spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                  spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                  spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                  spaces[f'AO{j}'] = self.deques[i]['AO'][j]
              
              observations[i] = spaces          
            
            obs_array = np.array([np.array(row) for row in obs])
            state = obs_array#observations #self._get_observations()
            return state, rewards, self.done, {}
        else:
            
            # get current observation, prepend to deque
            for i in range(self.n_agents):
                curState = self.players[i].getCurState(self.curTime)
                self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
                self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
                self.deques[i]['OO'].appendleft(int(curState[2]))
                self.deques[i]['AS'].appendleft(int(curState[3]))
                self.deques[i]['AO'].appendleft(int(curState[4]))
            
            # print(curState[0],curState[1],curState[2],curState[3],curState[4])
            # # return entire m observations
            obs = [[],[],[],[]]
            observations = [None] * self.n_agents
            for i in range(self.n_agents):
              spaces = {}
              for j in range(self.m):
                  obs[i].append(self.deques[i]['current_stock_minus'][j])
                  obs[i].append(self.deques[i]['current_stock_plus'][j])
                  obs[i].append(self.deques[i]['OO'][j])
                  obs[i].append(self.deques[i]['AS'][j])
                  obs[i].append(self.deques[i]['AO'][j])
                  spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                  spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                  spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                  spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                  spaces[f'AO{j}'] = self.deques[i]['AO'][j]
              
              observations[i] = spaces

            for i in range(self.n_agents):
              self.players[i].getReward()
            rewards = [1*self.players[i].curReward for i in range(0,self.config.NoAgent)]
            self.done = [False] * 4
            # print(obs, self.test_mode)
            obs_array = np.array([np.array(row) for row in obs])
            state = obs_array#observations #self._get_observations()
            # todo flatten observation dict
            #state = FlattenObservation(state)
            return state, rewards, self.done, {}

    def init_test_demand(self):
      self.test_deq = deque()
      demand = [0,0,1,1,1,0,2,1,1,1,1,0,2,2,1,1,0,0,1,2,2,1,0,0,2,0,2,1,2,1,1,1,2,1,1,0,1
,0,0,2,1,2,0,2,2,2,1,1,1,1,0,2,0,1,2,0,2,2,0,1,2,2,0,0,0,0,2,0,2,2,1,2,1,1
,0,1,2,1,2,1,0,2,2,1,2,0,0,0,2,2,0,1,1,1,0,1,0,0,1,1,0,0]
      self.test_deq.append(demand)
      demand = [1,0,0,0,0,1,2,0,2,1,0,1,1,2,1,1,0,2,1,1,0,0,0,1,2,0,2,2,2,0,0,2,0,0,1,1,0
,2,1,0,0,1,0,0,0,2,1,0,2,0,1,0,0,1,0,0,0,0,1,2,1,1,0,2,1,0,2,2,0,2,1,0,1,2
,0,2,2,0,0,1,2,1,0,0,0,0,2,1,0,2,2,1,2,1,1,0,1,0,2,1,0,1]
      self.test_deq.append(demand)
      demand = [1,2,0,2,1,2,1,2,1,1,0,2,1,2,1,2,0,2,0,1,1,2,0,1,1,0,1,1,1,2,1,2,1,2,2,0,1
,1,1,0,0,2,2,1,2,2,1,2,1,1,0,2,0,2,2,1,0,0,1,0,2,1,1,0,1,2,0,1,2,0,0,2,1,0
,0,0,2,0,2,1,1,0,2,2,1,2,1,1,2,0,2,0,1,1,1,1,1,2,0,2,0,0]
      self.test_deq.append(demand)
      demand = [1,0,1,2,0,2,2,1,2,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,2,0,1,0,0,0,2,0,1,2,0,1
,2,0,1,2,2,2,2,0,0,0,2,0,0,0,2,1,1,0,1,1,0,1,1,2,1,1,2,2,2,1,1,0,2,0,2,2,1
,2,1,2,2,0,2,0,2,1,2,2,1,1,1,1,2,0,2,1,1,2,0,2,2,2,2,0,2]
      self.test_deq.append(demand)
      demand = [0,2,2,1,0,2,1,2,2,1,0,0,1,1,1,2,0,1,0,2,0,2,0,1,1,2,1,2,0,2,1,1,2,2,0,0,1
,0,0,2,2,1,1,1,0,0,2,1,0,2,1,0,2,1,0,1,0,2,2,2,2,0,1,0,1,0,1,1,2,2,0,0,0,2
,0,0,0,1,0,2,0,0,2,2,1,1,0,1,0,1,0,0,2,1,0,0,0,0,1,1,1,0]
      self.test_deq.append(demand)
      demand = [0,2,1,2,2,0,0,0,2,0,1,2,0,2,2,0,0,1,0,0,2,2,2,0,2,2,0,1,0,2,0,2,2,1,0,0,2
,1,1,0,0,1,1,2,1,0,2,2,0,2,2,2,2,2,1,1,1,2,2,0,2,1,1,1,0,2,0,2,1,2,1,0,2,0
,1,1,2,2,0,0,1,0,0,1,2,1,0,1,1,1,1,2,2,2,0,0,2,2,2,0,2,0]
      self.test_deq.append(demand)
      demand = [0,1,1,0,2,2,0,1,0,0,0,2,1,1,0,2,1,0,1,2,1,0,2,2,0,0,0,2,0,0,1,0,1,1,2,0,2
,0,2,1,0,2,2,2,0,2,1,2,0,2,2,1,0,2,0,2,1,2,2,2,2,2,0,1,0,2,1,0,1,2,0,2,2,2
,1,0,2,2,2,1,0,2,1,2,1,2,1,0,2,2,2,2,0,1,1,1,2,2,0,2,1,0]
      self.test_deq.append(demand)
      demand = [1,2,1,1,0,2,1,1,0,2,1,2,2,2,1,1,2,2,2,2,1,2,1,1,0,1,0,2,0,0,2,0,1,2,0,0,0
,1,0,1,0,2,2,1,2,0,2,2,1,1,0,1,0,0,1,1,0,1,1,0,1,2,2,2,0,1,0,2,0,1,1,1,0,0
,2,1,1,0,2,0,0,1,0,0,0,2,0,0,2,0,1,0,2,1,1,0,0,1,1,2,1,1]
      self.test_deq.append(demand)
      demand = [1,1,0,1,1,0,1,0,0,1,1,2,1,0,1,2,1,2,0,2,2,0,1,1,0,1,0,0,2,2,0,1,0,2,2,2,0
,2,1,2,0,2,2,0,2,1,1,1,1,0,2,2,2,0,0,2,2,0,1,2,2,0,2,1,1,2,2,0,0,0,2,1,2,2
,1,2,0,0,0,2,1,1,2,0,2,2,0,2,1,0,0,1,1,0,0,1,1,1,2,0,0,1]
      self.test_deq.append(demand)
      demand = [0,1,2,1,1,1,1,1,0,2,1,0,2,0,0,1,0,1,1,2,2,2,2,0,1,1,2,0,0,1,2,1,1,2,1,1,0
,1,2,1,2,1,0,1,0,1,0,2,1,0,1,1,1,1,1,1,1,2,2,0,2,1,2,1,0,0,1,2,0,1,2,1,0,0
,1,2,1,2,2,0,0,0,2,1,1,1,1,1,1,2,1,0,0,2,0,2,0,2,2,1,1,1]
      self.test_deq.append(demand)
      demand = [1,2,2,0,0,1,0,0,1,2,2,1,1,0,1,1,1,1,1,2,1,1,2,1,0,2,1,0,2,2,1,0,0,0,0,1,2
,2,1,2,1,1,2,2,2,0,0,1,2,2,0,1,2,1,1,2,1,0,1,1,0,2,1,2,1,2,2,0,1,1,1,2,2,0
,2,0,1,1,1,0,1,2,1,2,2,0,2,2,1,1,0,1,1,1,0,0,2,2,1,0,2,0]
      self.test_deq.append(demand)
      demand = [2,1,1,2,0,0,0,0,2,1,0,0,2,0,0,0,1,1,0,1,0,1,2,0,0,1,0,1,2,2,2,1,0,1,0,2,0
,2,0,1,0,1,1,1,0,2,2,0,0,0,1,1,0,2,1,2,2,1,1,2,2,1,0,2,1,0,2,1,0,2,1,1,2,0
,1,0,0,0,2,2,0,1,2,2,0,1,2,0,2,1,1,2,1,2,1,1,2,2,2,1,2,2]
      self.test_deq.append(demand)
      demand = [0,0,1,1,1,2,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,1,2,0,1,1,1,0,0,1,2,0,1,1,0,0,2
,1,1,0,2,2,2,2,2,1,2,2,1,1,0,2,1,1,0,1,1,1,1,1,1,0,0,2,2,2,2,1,1,0,0,1,2,0
,2,0,0,1,2,0,0,1,2,2,2,0,2,2,1,1,0,1,0,1,2,1,1,1,1,2,0,1]
      self.test_deq.append(demand)
      demand = [2,1,0,1,0,0,0,2,0,1,1,0,0,0,1,1,0,1,0,2,1,1,2,1,0,2,0,0,1,0,0,1,0,0,1,1,2
,2,1,0,2,2,1,2,1,1,2,2,2,1,2,0,2,0,2,0,1,1,2,2,0,0,0,0,1,1,1,2,0,0,0,2,1,0
,1,2,2,1,2,0,0,2,1,1,2,0,0,2,1,2,0,2,2,1,2,2,2,0,0,1,0,0]
      self.test_deq.append(demand)
      demand = [0,0,0,2,1,2,2,0,1,2,0,2,0,1,1,2,0,1,2,1,1,2,2,1,1,1,2,0,2,2,2,1,2,2,1,2,2
,2,1,1,1,0,1,2,2,2,2,2,0,1,1,0,2,0,1,2,1,2,0,2,0,0,0,0,1,0,2,2,2,1,1,0,1,1
,1,2,0,0,2,0,0,1,2,2,1,2,1,2,2,0,0,1,0,0,2,0,1,0,0,2,1,0]
      self.test_deq.append(demand)
      demand = [2,0,1,0,0,2,2,1,1,1,0,1,0,2,1,0,0,2,0,2,0,1,2,0,1,0,1,2,1,2,2,0,2,1,0,1,2
,1,1,0,0,2,2,1,0,2,1,2,0,2,2,2,0,2,2,0,0,0,0,0,2,1,1,1,2,2,0,0,0,1,1,2,2,2
,2,1,2,2,0,2,0,1,2,0,2,1,1,1,2,2,2,0,2,2,1,1,2,0,0,0,1,1]
      self.test_deq.append(demand)
      demand = [1,0,0,0,1,0,0,1,1,2,0,2,0,2,1,0,0,2,1,0,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,0,2
,2,0,2,2,1,0,0,0,0,1,2,1,0,0,2,1,2,1,0,1,1,1,0,0,0,2,0,2,0,2,0,0,2,0,2,1,1
,0,0,2,2,1,2,2,2,2,1,0,2,2,1,2,0,0,0,0,1,0,1,1,2,1,1,1,0]
      self.test_deq.append(demand)
      demand = [2,0,1,1,0,0,1,0,1,1,1,2,1,1,2,1,0,2,0,2,0,0,2,1,0,2,2,0,2,0,1,1,2,2,0,1,0
,1,0,1,0,0,0,2,1,1,1,1,1,0,2,2,0,0,1,2,0,1,2,2,0,0,1,0,2,2,2,1,1,1,2,1,2,0
,2,2,2,1,1,1,0,2,2,0,0,1,1,1,1,0,0,2,1,0,0,2,0,1,0,2,0,1]
      self.test_deq.append(demand)
      demand = [0,0,0,2,1,0,0,2,1,0,1,2,1,1,0,2,2,1,0,2,0,2,0,1,0,0,2,0,2,0,0,0,2,0,2,1,0
,2,2,2,1,0,1,0,2,2,0,1,1,1,0,2,1,1,2,1,2,2,0,0,0,1,2,2,0,1,2,1,0,1,2,2,2,0
,2,2,1,2,1,0,1,2,2,0,2,2,0,1,1,2,2,2,2,0,1,0,0,0,1,2,1,1]
      self.test_deq.append(demand)
      demand = [2,2,2,1,2,2,0,1,0,0,2,2,1,0,2,0,1,0,1,1,1,1,0,2,1,2,2,1,1,1,2,2,2,2,0,0,2
,0,1,1,2,1,2,0,0,1,2,1,0,0,1,2,0,1,0,1,2,1,1,1,2,1,2,2,2,2,0,2,2,1,1,2,0,1
,0,0,0,2,1,2,0,1,2,1,0,2,2,2,2,0,0,0,0,1,1,2,0,2,1,1,1,2]
      self.test_deq.append(demand)
      demand = [2,2,0,1,2,1,0,2,1,2,1,1,2,1,0,1,2,0,1,2,1,2,0,2,0,2,1,1,2,0,0,0,0,0,1,2,1
,1,0,2,1,2,2,1,2,2,0,1,2,0,2,1,2,0,2,0,2,2,1,1,0,0,1,0,1,0,2,2,2,1,0,1,0,1
,1,1,0,1,2,0,0,1,1,2,0,2,0,0,2,1,1,0,0,2,0,0,1,0,0,1,0,1]
      self.test_deq.append(demand)
      demand = [1,2,0,2,1,1,2,1,0,0,2,2,2,0,0,0,1,0,2,2,2,0,2,2,0,1,1,2,0,0,2,1,0,2,2,1,2
,2,2,0,2,0,1,2,1,2,1,0,1,1,1,0,2,2,0,2,1,0,2,1,1,1,0,2,1,1,0,0,1,0,0,0,0,0
,2,1,0,1,2,1,2,0,0,0,0,2,2,0,1,1,2,1,0,1,2,0,2,2,1,1,0,1]
      self.test_deq.append(demand)
      demand = [1,1,1,0,1,2,0,0,0,2,2,0,2,0,0,2,0,2,1,0,2,1,0,0,1,1,1,0,1,2,1,2,1,2,1,2,1
,0,1,0,0,2,2,2,1,0,1,1,1,1,1,2,2,2,0,1,0,0,0,2,2,0,1,2,0,2,2,1,0,2,0,0,1,0
,1,0,1,1,0,1,1,0,2,1,0,2,0,0,1,0,1,1,1,2,1,2,1,0,2,2,0,2]
      self.test_deq.append(demand)
      demand = [1,0,0,1,1,0,0,0,2,0,2,1,2,1,2,2,2,1,2,1,1,2,1,0,2,1,0,0,2,2,2,0,2,1,1,1,2
,2,0,0,0,0,2,1,0,0,2,2,1,1,2,0,2,0,0,0,1,0,0,2,1,1,2,2,2,0,1,0,2,2,2,1,0,1
,1,1,2,0,0,1,1,2,2,2,0,2,0,0,2,0,1,1,0,1,2,2,1,2,1,0,0,1]
      self.test_deq.append(demand)
      demand = [2,0,1,1,1,1,2,1,2,2,1,1,1,2,1,1,1,2,1,2,0,2,2,0,2,2,0,1,1,0,1,2,2,1,1,0,1
,2,0,0,0,1,2,0,2,0,2,2,2,2,2,2,2,1,2,1,0,0,1,0,1,0,1,0,2,2,1,1,1,2,2,2,2,2
,1,2,0,1,2,2,1,2,1,1,1,0,1,1,2,0,1,1,0,1,0,0,2,0,1,2,0,2]
      self.test_deq.append(demand)
      demand = [2,1,0,2,0,0,0,0,1,2,0,2,0,1,0,0,0,0,2,1,1,1,0,0,0,2,2,0,1,0,0,1,0,2,2,1,2
,1,2,0,1,2,1,0,1,1,1,2,2,0,2,1,0,1,1,2,2,0,1,1,2,0,2,1,2,0,0,0,2,0,0,2,0,1
,1,1,2,0,1,0,0,2,1,2,0,0,0,2,2,2,2,1,2,1,2,1,0,2,0,0,2,0]
      self.test_deq.append(demand)
      demand = [1,1,2,0,1,2,1,0,0,0,0,1,2,2,2,0,0,1,2,2,2,1,0,0,1,2,0,2,1,1,1,2,2,1,0,0,1
,1,0,2,2,2,2,1,1,2,1,0,1,2,2,2,1,2,1,2,1,2,1,0,1,2,1,1,1,2,2,2,2,0,0,0,1,2
,1,1,1,0,1,0,0,0,1,0,1,0,1,1,2,1,0,0,2,0,0,2,1,0,0,1,0,0]
      self.test_deq.append(demand)
      demand = [2,2,2,0,2,1,2,1,2,1,2,0,0,0,1,1,1,0,1,1,0,1,0,0,2,2,2,2,1,1,2,2,0,1,1,0,0
,2,0,1,2,2,2,2,2,2,2,0,2,0,1,1,1,2,0,0,0,0,2,2,0,1,0,0,2,0,2,0,2,0,2,1,0,2
,1,0,0,1,2,1,0,2,2,0,1,1,0,0,1,1,1,0,1,1,1,1,2,2,1,2,0,2]
      self.test_deq.append(demand)
      demand = [0,2,1,0,2,0,2,1,1,2,2,0,2,0,2,2,2,1,2,2,0,1,2,1,1,1,2,0,2,0,2,0,2,1,2,2,2
,2,0,2,2,1,0,1,2,0,0,1,2,2,2,2,1,2,2,0,1,1,0,0,0,1,2,1,0,0,2,0,2,2,2,1,2,2
,2,1,2,1,2,0,2,2,2,1,1,1,0,0,2,0,1,2,1,2,0,2,0,2,2,1,0,2]
      self.test_deq.append(demand)
      demand = [2,1,0,2,0,1,2,0,2,0,1,2,1,1,2,0,1,1,1,0,0,0,2,0,2,0,0,2,2,1,2,1,2,0,2,2,1
,0,1,0,0,1,2,2,2,2,2,1,1,0,2,1,2,1,0,0,0,0,0,1,0,2,1,2,2,2,2,2,1,1,0,0,1,2
,1,1,0,2,2,1,0,0,0,1,2,0,1,1,0,1,1,1,2,1,2,2,0,1,2,0,1,1]
      self.test_deq.append(demand)
      demand = [2,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,0,2,0,1,2,0,1,1,0,0,0,2,0,1,2,0,1,0,0,2,2
,2,2,2,1,0,1,0,0,0,0,0,0,0,1,2,2,1,2,1,2,0,1,2,0,1,1,1,2,2,1,1,2,0,2,2,2,1
,0,1,0,2,2,1,2,1,1,2,0,0,2,2,0,2,1,1,0,1,1,0,0,2,1,2,2,0]
      self.test_deq.append(demand)
      demand = [1,2,2,1,1,1,2,1,2,2,1,1,2,0,0,2,0,0,0,1,2,2,0,2,1,2,1,0,2,0,0,1,2,1,2,1,2
,0,0,0,2,1,1,0,1,1,0,2,0,2,1,1,0,2,0,0,1,2,2,0,1,1,2,0,0,2,2,1,1,0,1,2,0,0
,1,2,0,2,2,0,0,1,2,1,0,2,0,0,2,2,0,0,2,0,1,1,2,2,1,0,0,2]
      self.test_deq.append(demand)
      demand = [0,1,0,0,2,0,1,1,1,2,0,1,0,0,2,0,1,0,1,1,2,0,2,1,1,2,1,2,0,1,0,0,1,0,2,2,0
,2,0,1,2,0,1,1,2,0,0,1,1,0,1,2,1,0,1,1,1,0,0,1,2,0,1,1,1,2,2,2,1,0,0,1,2,1
,2,0,1,0,0,2,2,0,0,2,2,2,1,0,2,2,2,1,2,0,2,2,1,2,2,2,0,0]
      self.test_deq.append(demand)
      demand = [1,0,0,1,2,2,1,2,1,2,2,2,1,0,2,2,1,1,1,2,1,2,2,0,1,0,0,0,2,1,2,0,2,0,0,1,0
,2,1,0,0,0,2,1,0,0,2,2,2,1,0,2,1,0,2,1,2,1,1,2,0,1,0,1,1,2,0,2,0,1,1,2,0,1
,1,2,1,2,2,2,2,2,1,1,1,2,0,0,2,1,0,1,1,2,2,2,2,0,0,1,2,1]
      self.test_deq.append(demand)
      demand = [1,0,0,0,1,1,0,1,1,0,2,2,0,0,2,0,1,1,1,0,0,2,0,1,0,0,0,2,1,1,0,2,0,0,1,2,1
,0,2,0,1,2,2,1,0,1,2,2,2,0,0,1,1,0,0,1,2,1,2,0,0,1,2,2,0,2,0,0,1,0,0,1,1,2
,2,0,0,0,2,2,1,0,0,1,0,2,2,1,0,0,2,2,0,1,1,0,0,1,1,1,0,0]
      self.test_deq.append(demand)
      demand = [2,0,2,1,0,0,1,0,0,1,0,2,0,1,2,2,2,0,2,2,2,2,1,2,0,0,0,0,0,0,2,2,2,2,2,2,2
,1,1,2,0,0,1,1,2,2,2,2,1,2,2,1,2,0,2,1,2,0,2,0,1,0,1,2,2,1,1,2,2,0,2,1,1,2
,2,1,2,1,2,1,1,1,1,2,2,0,2,0,0,0,0,0,0,0,0,0,0,0,2,0,2,2]
      self.test_deq.append(demand)
      demand = [0,2,0,2,0,1,1,2,0,2,1,1,2,2,0,0,1,2,0,2,2,2,0,0,2,2,2,1,1,1,1,2,2,2,1,0,2
,2,0,0,2,2,2,1,2,1,0,2,0,2,0,0,2,0,0,0,2,2,1,1,2,2,1,0,1,1,0,2,1,0,2,2,2,1
,2,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0,1,2,1,1,1,1,0,1,1,2,2]
      self.test_deq.append(demand)
      demand = [1,0,2,0,2,1,0,2,0,2,1,0,1,1,0,1,0,0,0,0,2,2,0,0,1,0,0,2,1,2,0,2,2,1,1,1,2
,0,1,2,2,1,1,1,1,2,0,1,0,0,2,1,0,1,1,0,0,2,1,1,0,0,1,0,1,0,0,0,0,0,2,2,1,0
,2,0,2,0,0,0,0,0,1,0,0,2,2,2,2,1,1,0,0,0,1,1,2,0,0,0,2,0]
      self.test_deq.append(demand)
      demand = [1,1,1,0,1,2,1,2,0,1,0,1,0,0,2,2,1,1,1,2,2,0,0,1,1,2,0,2,0,1,0,1,2,2,2,1,1
,0,1,0,2,1,0,0,0,2,2,0,0,1,1,1,0,0,0,1,2,1,0,1,0,0,0,1,1,1,1,1,0,0,2,0,0,0
,2,0,2,1,0,1,2,1,0,2,2,0,2,1,0,0,1,2,2,2,0,2,2,0,1,0,1,2]
      self.test_deq.append(demand)
      demand = [0,2,1,1,2,0,1,2,0,2,0,1,2,1,1,2,0,1,2,0,2,0,0,0,0,0,1,1,2,2,1,1,0,2,0,0,0
,0,1,1,1,2,1,0,2,1,2,1,1,1,1,2,2,1,0,0,0,1,0,2,0,0,1,2,2,2,1,1,1,0,2,2,1,2
,2,2,2,2,1,0,2,0,2,1,0,0,0,1,0,1,1,1,2,2,1,1,0,0,2,0,2,1]
      self.test_deq.append(demand)
      demand = [2,0,2,1,0,0,1,1,2,1,0,0,2,1,0,1,0,1,2,1,2,2,0,0,1,0,1,2,1,0,0,0,0,1,0,2,0
,1,0,1,1,1,0,0,0,0,2,0,0,2,0,2,0,1,0,0,1,2,0,2,0,2,1,0,1,2,2,0,2,1,0,1,0,0
,1,2,0,0,1,1,1,0,1,0,1,1,0,0,1,1,0,2,0,0,2,2,0,2,1,0,2,0]
      self.test_deq.append(demand)
      demand = [0,2,1,0,0,0,2,0,2,0,2,2,0,0,2,1,0,1,0,1,1,1,2,1,0,1,2,0,1,2,1,0,1,2,0,0,1
,0,0,1,1,1,0,1,0,2,1,2,1,0,2,2,0,2,2,2,2,1,2,1,0,2,0,0,1,2,0,2,0,2,1,1,2,2
,0,2,2,2,1,1,1,2,2,1,2,1,1,0,2,2,1,2,0,0,2,2,2,2,2,0,0,0]
      self.test_deq.append(demand)
      demand = [2,2,1,0,1,2,2,2,1,1,1,0,2,2,1,1,1,0,1,2,0,2,2,2,1,0,0,1,2,0,2,0,0,0,0,0,0
,1,0,0,2,2,1,1,0,2,0,1,2,1,2,2,1,0,0,1,1,0,2,0,2,0,1,0,0,0,0,0,1,1,1,2,1,1
,0,2,0,1,2,0,2,2,1,1,1,2,1,1,2,2,0,0,2,2,0,2,0,2,2,0,0,0]
      self.test_deq.append(demand) 
      demand = [2,0,1,1,1,1,2,1,0,2,1,1,0,0,1,2,1,0,1,2,2,0,0,2,1,1,2,1,2,0,1,2,1,1,2,1,0
,0,1,2,2,1,2,2,2,2,1,0,1,0,1,1,2,1,1,0,0,0,0,0,2,1,0,2,1,1,0,2,1,1,0,1,2,0
,1,1,1,2,0,2,2,0,2,0,0,0,2,2,1,2,0,2,0,2,2,1,2,2,2,0,0,1]
      self.test_deq.append(demand)
      demand = [2,1,1,0,0,1,2,0,1,2,2,2,0,0,1,2,1,0,0,2,0,1,1,1,1,1,2,2,1,1,0,0,0,1,1,1,0
,0,0,0,2,1,2,1,0,0,1,2,0,2,0,2,0,1,0,1,2,0,1,1,2,0,1,1,0,0,2,2,1,0,0,1,2,1
,2,2,1,2,1,2,1,0,2,1,0,2,1,2,2,2,1,1,0,2,0,2,1,1,2,1,1,0]
      self.test_deq.append(demand) 
      demand = [2,1,2,1,0,2,2,1,0,0,2,2,1,1,0,0,0,0,2,2,0,2,2,1,1,1,2,2,0,2,1,1,1,1,1,1,0
,0,0,1,1,2,1,2,1,0,0,1,1,1,0,0,2,1,1,0,1,0,2,0,1,2,0,1,0,1,0,2,2,2,2,0,1,0
,0,0,1,2,0,1,0,2,2,2,1,2,0,2,1,0,1,0,0,0,0,2,0,0,1,1,2,0]
      self.test_deq.append(demand) 
      demand = [0,1,1,2,1,1,0,1,2,2,2,0,0,2,0,0,0,1,2,2,1,0,0,0,0,2,0,2,1,1,1,1,2,0,1,1,0
,0,0,1,2,2,1,2,1,2,2,1,0,2,2,0,2,0,0,2,0,0,1,0,0,0,0,0,1,2,2,2,1,2,0,1,0,0
,0,1,0,0,2,0,2,1,1,2,2,0,1,0,0,1,2,2,0,1,2,0,2,1,2,1,1,2]
      self.test_deq.append(demand)
      demand = [1,0,0,0,2,2,2,0,2,0,2,1,1,0,0,0,1,0,1,1,2,1,2,1,1,0,1,1,0,1,0,0,2,0,2,2,1
,1,1,2,1,1,0,0,0,0,0,0,0,0,2,2,2,1,1,0,2,1,2,1,2,0,2,1,1,0,2,2,2,1,1,0,0,2
,2,2,1,0,2,2,1,0,0,1,0,0,1,0,1,0,2,1,0,0,1,1,2,2,0,2,2,1]
      self.test_deq.append(demand)
      demand = [2,1,0,0,2,2,2,2,1,1,1,2,1,2,0,2,2,2,2,1,2,0,1,1,1,2,0,1,0,2,2,0,0,1,1,1,1
,2,1,1,0,1,2,1,0,2,2,0,1,0,2,2,2,1,1,2,2,1,2,1,2,0,1,0,0,1,0,1,1,0,0,0,2,0
,2,2,2,1,0,1,1,0,2,0,0,0,1,0,0,1,0,0,1,2,1,1,1,0,2,1,2,1]
      self.test_deq.append(demand)
      demand = [0,2,1,2,0,2,0,2,0,2,0,2,1,0,0,1,0,2,1,2,2,1,1,0,2,2,1,0,1,0,2,2,2,1,1,1,1
,2,1,1,2,0,0,2,2,1,2,1,0,0,1,1,2,1,1,2,2,2,1,1,2,1,1,2,1,1,1,1,2,0,2,1,1,0
,1,0,0,2,1,2,1,0,1,2,0,1,1,0,0,0,1,0,2,0,2,1,1,1,0,0,0,0]
      self.test_deq.append(demand)



def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len



###################################################################
###################################################################
##################### CONFIG ######################################
###################################################################

import argparse
import os
import numpy as np 

def str2bool(v):
    return v.lower() in ('true', '1')

class Config(object):
  def __init__(self):
    # crm
    self.arg_lists = []
    self.parser = argparse.ArgumentParser()
    game_arg = self.add_argument_group('BeerGame')
    game_arg.add_argument('--task', type=str, default='bg')
    game_arg.add_argument('--fixedAction', type=str2bool, default='False', help='if you want to have actions in [0,actionMax] set it to True. with False it will set it [actionLow, actionUp]')
    game_arg.add_argument('--observation_data', type=str2bool, default=False, help='if it is True, then it uses the data that is generated by based on few real world observation')
    game_arg.add_argument('--data_id', type=int, default=22, help='the default item id for the basket dataset')
    game_arg.add_argument('--TLow', type=int, default=100, help='duration of one GAME (lower bound)')
    game_arg.add_argument('--TUp', type=int, default=100, help='duration of one GAME (upper bound)')
    game_arg.add_argument('--demandDistribution', type=int, default=0, help='0=uniform, 1=normal distribution, 2=the sequence of 4,4,4,4,8,..., 3= basket data, 4= forecast data')
    game_arg.add_argument('--scaled', type=str2bool, default=False, help='if true it uses the (if) existing scaled parameters')
    game_arg.add_argument('--demandLow', type=int, default=0, help='the lower bound of random demand')
    game_arg.add_argument('--demandUp', type=int, default=3, help='the upper bound of random demand')
    game_arg.add_argument('--demandMu', type=float, default=10, help='the mu of the normal distribution for demand ')
    game_arg.add_argument('--demandSigma', type=float, default=2, help='the sigma of the normal distribution for demand ')
    game_arg.add_argument('--actionMax', type=int, default=2, help='it works when fixedAction is True')
    game_arg.add_argument('--actionUp', type=int, default=2, help='bounds on my decision (upper bound), it works when fixedAction is True')
    game_arg.add_argument('--actionLow', type=int, default=-2, help='bounds on my decision (lower bound), it works when fixedAction is True')
    game_arg.add_argument('--action_step', type=int, default=1, help='The obtained action value by dnn is multiplied by this value')
    game_arg.add_argument('--actionList', type=list, default=[],  help='The list of the available actions')
    game_arg.add_argument('--actionListLen', type=int, default=0, help='the length of the action list')
    game_arg.add_argument('--actionListOpt', type=int, default=0 , help='the action list which is used in optimal and sterman')
    game_arg.add_argument('--actionListLenOpt', type=int, default=0, help='the length of the actionlistopt')
    game_arg.add_argument('--agentTypes', type=list, default=['dnn','dnn','dnn','dnn'], help='the player types')
    game_arg.add_argument('--agent_type1', type=str, default='dnn', help='the player types for agent 1, it can be dnn, Strm, bs, rnd')
    game_arg.add_argument('--agent_type2', type=str, default='dnn', help='the player types for agent 2, it can be dnn, Strm, bs, rnd')
    game_arg.add_argument('--agent_type3', type=str, default='dnn', help='the player types for agent 3, it can be dnn, Strm, bs, rnd')
    game_arg.add_argument('--agent_type4', type=str, default='dnn', help='the player types for agent 4, it can be dnn, Strm, bs, rnd')
    game_arg.add_argument('--NoAgent', type=int, default=1, help='number of agents, currently it should be in {1,2,3,4}')
    game_arg.add_argument('--cp1', type=float, default=2.0, help='shortage cost of player 1')
    game_arg.add_argument('--cp2', type=float, default=0.0, help='shortage cost of player 2')
    game_arg.add_argument('--cp3', type=float, default=0.0, help='shortage cost of player 3')
    game_arg.add_argument('--cp4', type=float, default=0.0, help='shortage cost of player 4')
    game_arg.add_argument('--ch1', type=float, default=2.0, help='holding cost of player 1')
    game_arg.add_argument('--ch2', type=float, default=2.0, help='holding cost of player 2')
    game_arg.add_argument('--ch3', type=float, default=2.0, help='holding cost of player 3')
    game_arg.add_argument('--ch4', type=float, default=2.0, help='holding cost of player 4')
    game_arg.add_argument('--alpha_b1', type=float, default=-0.5, help='alpha of Sterman formula parameter for player 1')
    game_arg.add_argument('--alpha_b2', type=float, default=-0.5, help='alpha of Sterman formula parameter for player 2')
    game_arg.add_argument('--alpha_b3', type=float, default=-0.5, help='alpha of Sterman formula parameter for player 3')
    game_arg.add_argument('--alpha_b4', type=float, default=-0.5, help='alpha of Sterman formula parameter for player 4')
    game_arg.add_argument('--betta_b1', type=float, default=-0.2, help='beta of Sterman formula parameter for player 1')
    game_arg.add_argument('--betta_b2', type=float, default=-0.2, help='beta of Sterman formula parameter for player 2')
    game_arg.add_argument('--betta_b3', type=float, default=-0.2, help='beta of Sterman formula parameter for player 3')
    game_arg.add_argument('--betta_b4', type=float, default=-0.2, help='beta of Sterman formula parameter for player 4')
    game_arg.add_argument('--eta', type=list, default=[0,4,4,4], help='the total cost regulazer')
    game_arg.add_argument('--distCoeff', type=int, default=20, help='the total cost regulazer')
    game_arg.add_argument('--gameConfig', type=int, default=3, help='if it is "0", it uses the current "agentType", otherwise sets agent types according to the function setAgentType() in this file.')
    game_arg.add_argument('--ifUseTotalReward', type=str2bool, default='False', help='if you want to have the total rewards in the experience replay, set it to true.')
    game_arg.add_argument('--ifUsedistTotReward', type=str2bool, default='True', help='If use correction to the rewards in the experience replay for all iterations of current game')
    game_arg.add_argument('--ifUseASAO', type=str2bool, default='True', help='if use AS and AO, i.e., received shipment and received orders in the input of DNN')
    game_arg.add_argument('--ifUseActionInD', type=str2bool, default='False', help='if use action in the input of DNN')
    game_arg.add_argument('--stateDim', type=int, default=5, help='Number of elements in the state desciptor - Depends on ifUseASAO')
    game_arg.add_argument('--iftl', type=str2bool, default=False, help='if apply transfer learning')
    game_arg.add_argument('--ifTransferFromSmallerActionSpace', type=str2bool, default=False, help='if want to transfer knowledge from a network with different action space size.')
    game_arg.add_argument('--baseActionSize', type=int, default=5, help='if ifTransferFromSmallerActionSpace is true, this determines the size of action space of saved network')
    game_arg.add_argument('--tlBaseBrain', type=int, default=3, help='the gameConfig of the base network for re-training with transfer-learning')
    game_arg.add_argument('--baseDemandDistribution', type=int, default=0, help='same as the demandDistribution')
    game_arg.add_argument('--MultiAgent', type=str2bool, default=False, help='if run multi-agent RL model, not fully operational')
    game_arg.add_argument('--MultiAgentRun', type=list, default=[True, True, True, True], help='In the multi-RL setting, it determines which agent should get training.')
    game_arg.add_argument('--if_use_AS_t_plus_1', type=str2bool, default='False', help='if use AS[t+1], not AS[t] in the input of DNN')
    game_arg.add_argument('--ifSinglePathExist', type=str2bool, default=False, help='If true it uses the predefined path in pre_model_dir and does not merge it with demandDistribution.')
    game_arg.add_argument('--ifPlaySavedData', type=str2bool, default=False, help='If true it uses the saved actions which are read from file.')

    #################### parameters of the leadtimes ########################
    leadtimes_arg = self.add_argument_group('leadtimes')
    leadtimes_arg.add_argument('--leadRecItemLow', type=list, default=[2,2,2,4], help='the min lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecItemUp', type=list, default=[2,2,2,4], help='the max lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecOrderLow', type=int, default=[2,2,2,0], help='the min lead time for receiving orders')
    leadtimes_arg.add_argument('--leadRecOrderUp', type=int, default=[2,2,2,0], help='the max lead time for receiving orders')
    leadtimes_arg.add_argument('--ILInit', type=list, default=[0,0,0,0], help='')
    leadtimes_arg.add_argument('--AOInit', type=list, default=[0,0,0,0], help='')
    leadtimes_arg.add_argument('--ASInit', type=list, default=[0,0,0,0], help='the initial shipment of each agent')
    leadtimes_arg.add_argument('--leadRecItem1', type=int, default=2, help='the min lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecItem2', type=int, default=2, help='the min lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecItem3', type=int, default=2, help='the min lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecItem4', type=int, default=2, help='the min lead time for receiving items')
    leadtimes_arg.add_argument('--leadRecOrder1', type=int, default=2, help='the min lead time for receiving order')
    leadtimes_arg.add_argument('--leadRecOrder2', type=int, default=2, help='the min lead time for receiving order')
    leadtimes_arg.add_argument('--leadRecOrder3', type=int, default=2, help='the min lead time for receiving order')
    leadtimes_arg.add_argument('--leadRecOrder4', type=int, default=2, help='the min lead time for receiving order')
    leadtimes_arg.add_argument('--ILInit1', type=int, default=0, help='the initial inventory level of the agent')
    leadtimes_arg.add_argument('--ILInit2', type=int, default=0, help='the initial inventory level of the agent')
    leadtimes_arg.add_argument('--ILInit3', type=int, default=0, help='the initial inventory level of the agent')
    leadtimes_arg.add_argument('--ILInit4', type=int, default=0, help='the initial inventory level of the agent')
    leadtimes_arg.add_argument('--AOInit1', type=int, default=0, help='the initial arriving order of the agent')
    leadtimes_arg.add_argument('--AOInit2', type=int, default=0, help='the initial arriving order of the agent')
    leadtimes_arg.add_argument('--AOInit3', type=int, default=0, help='the initial arriving order of the agent')
    leadtimes_arg.add_argument('--AOInit4', type=int, default=0, help='the initial arriving order of the agent')
    leadtimes_arg.add_argument('--ASInit1', type=int, default=0, help='the initial arriving shipment of the agent')
    leadtimes_arg.add_argument('--ASInit2', type=int, default=0, help='the initial arriving shipment of the agent')
    leadtimes_arg.add_argument('--ASInit3', type=int, default=0, help='the initial arriving shipment of the agent')
    leadtimes_arg.add_argument('--ASInit4', type=int, default=0, help='the initial arriving shipment of the agent')


    ####################	DQN setting		####################	
    DQN_arg = self.add_argument_group('DQN')
    DQN_arg.add_argument('--maxEpisodesTrain', type=int, default=60100, help='number of GAMES to be trained')
    DQN_arg.add_argument('--NoHiLayer', type=int, default=3, help='number of hidden layers')
    DQN_arg.add_argument('--NoFixedLayer', type=int, default=1, help='number of hidden layers')
    DQN_arg.add_argument('--node1', type=int, default=180, help='the number of nodes in the first hidden layer')
    DQN_arg.add_argument('--node2', type=int, default=130, help='the number of nodes in the second hidden layer')
    DQN_arg.add_argument('--node3', type=int, default=61, help='the number of nodes in the third hidden layer')
    DQN_arg.add_argument('--nodes', type=list, default=[], help='')

    DQN_arg.add_argument('--seed', type=int, default=40, help='the seed for DNN stuff')
    DQN_arg.add_argument('--batchSize', type=int, default=64, help='the batch size which is used to obtain')
    DQN_arg.add_argument('--minReplayMem', type=int, default=50000, help='the minimum of experience reply size to start dnn')
    DQN_arg.add_argument('--maxReplayMem', type=int, default=1000000, help='the maximum size of the replay memory')
    DQN_arg.add_argument('--alpha', type=float, default=.97, help='learning rate for total reward distribution ')
    DQN_arg.add_argument('--gamma', type=float, default=.99, help='discount factor for Q-learning')
    DQN_arg.add_argument('--saveInterval', type=int, default=10000, help='every xx training iteration, saves the games network')
    DQN_arg.add_argument('--epsilonBeg', type=float, default=0.9, help='')
    DQN_arg.add_argument('--epsilonEnd', type=float, default=0.1, help='')
            
    DQN_arg.add_argument('--lr0', type=float, default=0.00025 , help='the learning rate')
    DQN_arg.add_argument('--Minlr', type=float, default=1e-8, help='the minimum learning rate, if it drops below it, fix it there ')
    DQN_arg.add_argument('--ifDecayAdam', type=str2bool, default=True, help='decays the learning rate of the adam optimizer')
    DQN_arg.add_argument('--decayStep', type=int, default=10000, help='the decay step of the learning rate')
    DQN_arg.add_argument('--decayRate', type=float, default=0.98, help='the rate to reduce the lr at every decayStep')

    DQN_arg.add_argument('--display', type=int, default=1000, help='the number of iterations between two display of results.')
    DQN_arg.add_argument('--momentum', type=float, default=0.9, help='the momentum value')
    DQN_arg.add_argument('--dnnUpCnt', type=int, default=10000, help='the number of iterations that updates the dnn weights')
    DQN_arg.add_argument('--multPerdInpt', type=int, default=10, help='Number of history records which we feed into DNN')


    ####################	Utilities			####################	
    utility_arg = self.add_argument_group('Utilities')
    utility_arg.add_argument('--address', type=str, default="", help='the address which is used to save the model files')
    utility_arg.add_argument('--ifUsePreviousModel', type=str2bool, default='False', help='if there is a saved model, then False value of this parameter will overwrite.')
    utility_arg.add_argument('--number_cpu_active', type=int, default=5, help='number of cpu cores')
    utility_arg.add_argument('--gpu_memory_fraction', type=float, default=0.1, help='the fraction of gpu memory which we are gonna use')
    # Dirs
    utility_arg.add_argument('--load_path', type=str, default='', help='The directory to load the models')
    utility_arg.add_argument('--log_dir', type=str, default=os.path.expanduser('./logs/'), help='')
    utility_arg.add_argument('--pre_model_dir', type=str, default=os.path.expanduser('./pre_model'),help='')
    utility_arg.add_argument('--action_dir', type=str, default=os.path.expanduser('./'),help='if ifPlaySavedData is true, it uses this path to load actions')
    utility_arg.add_argument('--model_dir', type=str, default='./',help='')
    utility_arg.add_argument('--TB', type=str2bool, default=False, help='set to True if use tensor board and save the required data for TB.')
    utility_arg.add_argument('--INFO_print', type=str2bool, default=True, help='if true, it does not print anything all.')
    utility_arg.add_argument('--tbLogInterval', type=int, default=80000, help='number of GAMES for testing')
        
    ####################	testing			####################	
    test_arg = self.add_argument_group('testing')
    test_arg.add_argument('--testRepeatMid', type=int, default=1, help='it is number of episodes which is going to be used for testing in the middle of training')
    test_arg.add_argument('--testInterval', type=int, default=100, help='every xx games compute "test error"')
    test_arg.add_argument('--ifSaveFigure', type=str2bool, default=True, help='if is it True, save the figures in each testing.')
    test_arg.add_argument('--if_titled_figure', type=str2bool, default='True', help='if is it True, save the figures with details in the title.')
    test_arg.add_argument('--saveFigInt', type=list, default=[10000,60000], help='')
    test_arg.add_argument('--saveFigIntLow', type=int, default=10000, help='')
    test_arg.add_argument('--saveFigIntUp', type=int, default=60000, help='')
    test_arg.add_argument('--ifsaveHistInterval', type=str2bool, default=False, help='if every xx games save details of the episode')
    test_arg.add_argument('--saveHistInterval', type=int, default=5000, help='every xx games save details of the play')
    test_arg.add_argument('--Ttest', type=int, default=100, help='it defines the number of periods in the test cases')
    test_arg.add_argument('--ifOptimalSolExist', type=str2bool, default=True, help='if the instance has optimal base stock policy, set it to True, otherwise it should be False.')
    test_arg.add_argument('--f1', type=float, default=8, help='base stock policy decision of player 1')
    test_arg.add_argument('--f2', type=float, default=8, help='base stock policy decision of player 2')
    test_arg.add_argument('--f3', type=float, default=0, help='base stock policy decision of player 3')
    test_arg.add_argument('--f4', type=float, default=0, help='base stock policy decision of player 4')
    test_arg.add_argument('--f_init', type=list, default=[32,32,32,24], help='base stock policy decision for 4 time-steps on the C(4,8) demand distribution')
    test_arg.add_argument('--use_initial_BS', type=str2bool, default=False, help='If use f_init set it to True')
    test_arg.add_argument('--ifSaveHist', type=str2bool, default='False', help='if it is true, saves history, prediction, and the randBatch in each period, WARNING: just make it True in small runs, it saves huge amount of files.')

    # DQN_arg = self.add_argument_group('DQN')
    # DQN_arg.add_argument('--gamma', type=float, default=.99, help='discount factor for Q-learning')


  def str2bool(self, v):
    return v.lower() in ('true', '1')


  def add_argument_group(self, name):
    arg = self.parser.add_argument_group(name)
    self.arg_lists.append(arg)
    return arg


		
  #buildActionList: actions for the beer game problem	
  def buildActionList(self, config):
    aDiv = 1  # difference in the action list
    if config.fixedAction:
      actions = list(range(0,config.actionMax+1,aDiv)) # If you put the second argument =11, creates an actionlist from 0..xx
    else:
      actions = list(range(config.actionLow,config.actionUp+1,aDiv) )
    return actions	
	
  # specify the  dimension of the state of the game	
  def getStateDim(self, config):
    if config.ifUseASAO:
      stateDim=5
    else:
      stateDim=3

    if config.ifUseActionInD:
      stateDim += 1

    return stateDim	

  # agents 1=[dnn,dnn,dnn,dnn]; 2=[dnn,Strm,Strm,Strm]; 3=[dnn,bs,bs,bs]
  def setAgentType(self, config):
      config.agentTypes = ["bs", "bs","bs","bs"]

  def set_optimal(self, config):
    if config.demandDistribution == 0:
      if config.cp1==2 and config.ch1==2 and config.ch2==2 and config.ch3==2 and config.ch4==2 :
        config.f1 = 8.
        config.f2 = 8.
        config.f3 = 0.
        config.f4 = 0.

  def get_config(self):
    config, unparsed = self.parser.parse_known_args()
    config = self.update_config(config)
    return config, unparsed

  def fill_leadtime_initial_values(self,config):
    config.leadRecItemLow = [config.leadRecItem1, config.leadRecItem2, config.leadRecItem3, config.leadRecItem4]
    config.leadRecItemUp = [config.leadRecItem1, config.leadRecItem2, config.leadRecItem3, config.leadRecItem4]
    config.leadRecOrderLow = [config.leadRecOrder1, config.leadRecOrder2, config.leadRecOrder3, config.leadRecOrder4]
    config.leadRecOrderUp = [config.leadRecOrder1, config.leadRecOrder2, config.leadRecOrder3, config.leadRecOrder4]
    config.ILInit = [config.ILInit1, config.ILInit2, config.ILInit3, config.ILInit4]
    config.AOInit = [config.AOInit1, config.AOInit2, config.AOInit3, config.AOInit4]
    config.ASInit = [config.ASInit1, config.ASInit2, config.ASInit3, config.ASInit4]

  def get_auxuliary_leadtime_initial_values(self, config):
    config.leadRecOrderUp_aux = [config.leadRecOrder1, config.leadRecOrder2, config.leadRecOrder3, config.leadRecOrder4]
    config.leadRecItemUp_aux = [config.leadRecItem1, config.leadRecItem2, config.leadRecItem3, config.leadRecItem4]

  def fix_lead_time_manufacturer(self, config):
    if config.leadRecOrder4 > 0:
      config.leadRecItem4 += config.leadRecOrder4
      config.leadRecOrder4 = 0 

  def set_sterman_parameters(self, config):
    config.alpha_b =[config.alpha_b1,config.alpha_b2,config.alpha_b3,config.alpha_b4]
    config.betta_b =[config.betta_b1,config.betta_b2,config.betta_b3,config.betta_b4]	


  def update_config(self,config):
    config.actionList = self.buildActionList(config)		# The list of the available actions
    config.actionListLen = len(config.actionList)		# the length of the action list
		
    # set_optimal(config)
    config.f = [config.f1, config.f2, config.f3, config.f4] # [6.4, 2.88, 2.08, 0.8]

    config.actionListLen=len(config.actionList)
    if config.demandDistribution == 0:
      config.actionListOpt=list(range(0,int(max(config.actionUp*30+1, 3*sum(config.f))),1))
    else:
      config.actionListOpt=list(range(0,int(max(config.actionUp*30+1, 7*sum(config.f))),1))
    config.actionListLenOpt=len(config.actionListOpt)
    config.agentTypes=['dnn','dnn','dnn','dnn']
    config.saveFigInt = [config.saveFigIntLow, config.saveFigIntUp]
    
    if config.gameConfig == 0:
      config.NoAgent=min(config.NoAgent,len(config.agentTypes))
      config.agentTypes=[config.agent_type1,config.agent_type2,config.agent_type3,config.agent_type4]
    else:
      config.NoAgent=4
      self.setAgentType(config)					# set the agent brain types according to ifFourDNNtrain, ...

    config.c_h =[config.ch1, config.ch2, config.ch3, config.ch4]
    config.c_p =[config.cp1, config.cp2, config.cp3, config.cp4]

    config.stateDim= self.getStateDim(config) # Number of elements in the state description - Depends on ifUseASAO		
    #np.random.seed(seed = config.seed)
    #self.setSavedDimentionPerBrain(config) # set the parameters of pre_trained model. 
    #self.fillnodes(config)			# create the structure of network nodes 	
    self.get_auxuliary_leadtime_initial_values(config)
    self.fix_lead_time_manufacturer(config)
    self.fill_leadtime_initial_values(config)
    self.set_sterman_parameters(config)

    return config





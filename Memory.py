import numpy as np
import random

class Memory:
        
    capacity = None
    
    
    def __init__(
            self,
            capacity,
    ):
        self.capacity = capacity
        self.length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states2 = []

    def push(self, s, a, r, done, s2):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(done)
        self.states2.append(s2)
        
        self.length = self.length + 1
            
        if (self.length > self.capacity): 
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.states2.pop(0)
            self.length = self.length - 1
            
        
    def sample(self,batch_size):
        if (self.length >= batch_size):
            idx = random.sample(range(0,self.length),batch_size)
            s = [self.states[i] for i in idx]
            a = [self.actions[i] for i in idx]
            r = [self.rewards[i] for i in idx]
            d = [self.dones[i] for i in idx]
            s2 = [self.states2[i] for i in idx]
                
            return [s, a, r, s2, d]
                    

# Author: Vibhakar Mohta (vmohta@cs.cmu.edu)

import cv2
import numpy as np

class GridWorld:
    ## Initialise starting data
    def __init__(self, rho=0.01):
        # Set information about the gridworld
        self.rho = rho
        self.height = 11
        self.width = 11
        self.grid = np.zeros(( self.height, self.width)) - 1
        
        
        # Set locations for the bomb and the gold
        self.bomb_locations = [(1,2), (1,3), (1,4), (6,3), (5,6), (5,0), (1,5), (2,5), (3,6), (4,6)]
        self.gold_location = (0,3)
        self.terminal_states = [*self.bomb_locations, self.gold_location]
        
        self.reset()
        
        # TODO: Set grid rewards for special cells (bomb and gold)
        for bomb_location in self.bomb_locations:
            self.grid[] = ...
        self.grid[self.gold_location[0], self.gold_location[1]] = ...
        
        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        self.display_queue = []
        self.titles = []
        self.SCALE = None

    def reset(self):
        self.current_location = (np.random.randint(0,self.height),
                                 np.random.randint(0,self.width))
        if self.current_location in self.terminal_states:
            self.reset()
        return self.current_location
        
    def clear_display(self):
        self.display_queue = []
        self.titles = []
        self.SCALE = None

    def save_display(self, filename):
        # Save all frames in display queue as a mp4 video
        print("Saving {} frames to {}".format(len(self.display_queue), filename))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        IMG_WIDTH = self.width*self.SCALE+self.SCALE
        IMG_HEIGHT = self.height*self.SCALE+self.SCALE*2
        out = cv2.VideoWriter(filename, fourcc, 10, (IMG_WIDTH, IMG_HEIGHT))

        for i, orig_frame in enumerate(self.display_queue):
            frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            frame[self.SCALE+20:-self.SCALE+20, self.SCALE//2:-self.SCALE//2] = np.array(orig_frame*255, dtype=np.uint8) # copy frame to center
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add title text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = self.titles[i]
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (frame.shape[1] - textsize[0]) // 2
            cv2.putText(frame, text, (textX, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            for i in range(self.height):
                cv2.putText(frame, str(i), (10, 3*self.SCALE//2 + i*self.SCALE), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            for i in range(self.width):
                cv2.putText(frame, str(i), (3*self.SCALE//4+i*self.SCALE, self.SCALE), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Write the frame to video
            out.write(frame)
            
        # Release everything when job is finished
        out.release()
        
    def render_grid(self, title=None, SCALE=100):
        """
        Renders the current gridworld state, stores in buffer for saving as video
        Call save_display to save the buffer as a video
        """
        grid_img = np.zeros((self.height*SCALE, self.width*SCALE, 3)) + 0.1
        for bomb_location in self.bomb_locations:
            grid_img[bomb_location[0]*SCALE:bomb_location[0]*SCALE+SCALE, bomb_location[1]*SCALE:bomb_location[1]*SCALE+SCALE] = [1, 0, 0]
        grid_img[self.gold_location[0]*SCALE:self.gold_location[0]*SCALE+SCALE, self.gold_location[1]*SCALE:self.gold_location[1]*SCALE+SCALE] = [1, 1, 0]
        grid_img[self.current_location[0]*SCALE:self.current_location[0]*SCALE+SCALE, self.current_location[1]*SCALE:self.current_location[1]*SCALE+SCALE] = [0, 0, 1]
        
        self.display_queue.append(grid_img)
        self.titles.append(title)
        assert self.SCALE==None or self.SCALE==SCALE, "SCALE cannot be changed during rendering"
        if self.SCALE==None:
            self.SCALE = SCALE
    
    def get_reward(self, new_location):
        """
        Returns the reward for an input position
        """
        return self.grid[ new_location[0], new_location[1]]

    def check_state(self):
        """
        Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'
        """
        if self.current_location in self.terminal_states:
            return 'TERMINAL'
        
    def step(self, action):
        """
        Moves the agent in the specified direction. 
        The action is stochastic with epsilon chance of moving in a random direction.
        If agent is at a border, agent stays still but takes negative reward. 
        Function returns a tuple (obs, reward, done)
        """        
        # TODO: Implement this function
        # With epsilon chance, move in a random direction
        if np.random.uniform(0,1) < self.rho:
            action = ...
        
        # UP
        last_location = self.current_location
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        
        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            ...
            
        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            ...

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            ...
        
        # check if reached a terminal state
        done = ...
        
        return self.current_location, reward, done
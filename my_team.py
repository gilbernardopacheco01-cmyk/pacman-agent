# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers. Â isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.previous_food_defending = self.get_food_you_are_defending(game_state).as_list()
        self.lost_food = None
        self.previous_capsules = self.get_capsules(game_state)
        self.capsule_timer = 0
        self.dead_ends = self.get_dead_ends(game_state.data.layout, game_state)
        self.frontier_list = self.get_frontier(game_state) # all walkable positions on the frontier
        self.frontier_patrol = self.get_frontier_patrol(game_state)
        self.current_frontier_index_patrol_point = 0
        self.flanking_target = None


    def get_frontier(self, game_state):
            """
            Returns a list of valid (x, y) positions on the boundary of our territory.
            We use this to find the closest point to return home.
            """
            layout = game_state.data.layout
            height = layout.height
            width = layout.width
            
            # Calculate the middle x-coordinate
            # Note: We use int() because coordinates must be integers
            mid_x = int(width / 2) 
            
            # If we are the Red team, the boundary is the column just to the left of the center.
            # If we are Blue, the boundary is the center column.
            if self.red:
                mid_x -= 1
                
            # Create a list of all y-coordinates in that column that represent valid paths (no walls)
            frontier_list = []
            for y in range(0, height - 1):
                if not game_state.has_wall(mid_x, y):
                    frontier_list.append((mid_x, y))
                    
            return frontier_list

    def get_frontier_patrol(self, game_state):
        """
            Returns a list of valid (x, y) positions on the boundary of our territory.
            We use this to find the closest point to return home.
        """
        layout = game_state.data.layout
        height = layout.height
        width = layout.width
        
        # Calculate the middle x-coordinate
        # Note: We use int() because coordinates must be integers
        mid_x = int(width / 2)
        
        patrol_x = None
        # If we are the Red team, the boundary is the column just to the left of the center.
        # If we are Blue, the boundary is the center column.
        if self.red:
            patrol_x = mid_x -3
        else :
            patrol_x = mid_x + 2
            
        # Create a list of all y-coordinates in that column that represent valid paths (no walls)
        frontier_patrol_list = []
        for y in range(0, height - 1):
            if not game_state.has_wall(patrol_x, y):
                frontier_patrol_list.append((patrol_x, y))
                
        return frontier_patrol_list


    def get_dead_ends(self, layout, game_state) :
        dead_ends = set()
        height = layout.height
        width = layout.width
        
        # Calculate the middle x-coordinate
        # Note: We use int() because coordinates must be integers
        mid_x = int(width / 2) 
        
        # If we are the Red team, the boundary is the column just to the left of the center.
        # If we are Blue, the boundary is the center column.
        if self.red:
            walkable_pos = set()
            for x in range(mid_x, width - 1) :
                for y in range(0, height - 1) :
                    if not game_state.has_wall(x, y) :
                        walkable_pos.add((x,y))
            continue_loop = True
            while(continue_loop) :
                continue_loop = False
                to_remove = []
                for (x,y) in walkable_pos :
                    neighbours = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                    walkable_neighbours = 0
                    for n in neighbours :
                        if not game_state.has_wall(n[0], n[1]) and n not in dead_ends :
                            walkable_neighbours += 1
                    if walkable_neighbours == 1 :
                        dead_ends.add((x,y))
                        to_remove.append((x,y))
                        continue_loop = True
                for (x,y) in to_remove :
                    walkable_pos.remove((x,y))
        else :
            mid_x -= 1
            walkable_pos = set()
            for x in range(0, mid_x) :
                for y in range(0, height - 1) :
                    if not game_state.has_wall(x, y) :
                        walkable_pos.add((x,y))
            continue_loop = True
            while(continue_loop) :
                continue_loop = False
                to_remove = []
                for (x,y) in walkable_pos :
                    neighbours = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                    walkable_neighbours = 0
                    for n in neighbours :
                        if not game_state.has_wall(n[0], n[1]) and n not in dead_ends :
                            walkable_neighbours += 1
                    if walkable_neighbours == 1 :
                        dead_ends.add((x,y))
                        to_remove.append((x,y))
                        continue_loop = True
                for (x,y) in to_remove :
                    walkable_pos.remove((x,y))
        return dead_ends


    def choose_action(self, game_state):
            """
            Manages the memory (States) and chooses the best action.
            """
            my_pos = game_state.get_agent_position(self.index)
            
            # --- STATE MANAGEMENT: DEFENSE (Shadowing) ---
            current_food = self.get_food_you_are_defending(game_state).as_list()
            eaten_food = set(self.previous_food_defending) - set(current_food)
            if len(eaten_food) > 0:
                self.lost_food = eaten_food.pop()
            self.previous_food_defending = current_food
            
            if self.lost_food is not None:
                if self.get_maze_distance(my_pos, self.lost_food) <= 1:
                    self.lost_food = None # Reached the lost food

        
            # --- STATE MANAGEMENT: DEFENSE (Patrol) ---
            # (Only for the Defender to change the random target if reached)
            if self.lost_food is None and hasattr(self, 'patrol_target') and self.patrol_target is not None:
                if self.get_maze_distance(my_pos, self.patrol_target) <= 1:
                    # Choose new point if the previous one was reached
                    self.patrol_target = random.choice(self.frontier_patrol)

            # --- STATE MANAGEMENT: OFFENSE (Flanking) --- <--- NEW PLACE HERE
            # If we have a flank target and we've reached it (or we are already Pacman), we clear it.
            if self.flanking_target is not None:
                # Note: You can include 'or my_state.is_pacman' if you want it to stop when crossing the line
                # But to ensure it goes deep, we only keep the distance.
                dist = self.get_maze_distance(my_pos, self.flanking_target)
                if dist <= 1:
                    self.flanking_target = None
                    print(f"DEBUG: Flank complete! Returning to normal mode.")

            # --- ACTION SELECTION (Standard) ---
            actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]

            # Time fail-safe (Forced return if running out of food)
            food_left = len(self.get_food(game_state).as_list())
            if food_left <= 2:
                best_dist = 9999
                best_action = None
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    pos2 = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(self.start, pos2)
                    if dist < best_dist:
                        best_action = action
                        best_dist = dist
                return best_action

            return random.choice(best_actions)


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state. Â They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action) # successor game state after taking the action
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position() # position after taking the action
        my_pos_int = (int(my_pos[0]), int(my_pos[1]))
        food_carrying = game_state.get_agent_state(self.index).num_carrying # how many food pellets in the current agent
        food_list = self.get_food(successor).as_list() # all food pellets on the enemy side
        frontier_list = self.frontier_list
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)] # enemies' states
        defenders = [enemie for enemie in enemies if not enemie.is_pacman and enemie.get_position() is not None and enemie.scared_timer == 0] #(enemy)defenders' states
        scared_defenders = [enemie for enemie in enemies if not enemie.is_pacman and enemie.get_position() is not None and enemie.scared_timer > 10] #(enemy)defenders' states
        attackers = [enemie for enemie in enemies if enemie.is_pacman]
        power_capsules = self.get_capsules(successor)
        is_flanking = False

        # ---------------------------------------------------------
        # 1. ACTIVATE FLANKING (If necessary)
        # ---------------------------------------------------------
        # Note: Target clearing (reset) was already done in the Parent class's choose_action.
        # Here we only check if we need to START a new flank.
        
        current_is_pacman = game_state.get_agent_state(self.index).is_pacman
        
        # We only start flanking if: We have no target, we are at base, and we see a blockade
        if self.flanking_target is None and not current_is_pacman and len(defenders) > 0:
            dists = [self.get_maze_distance(my_pos, d.get_position()) for d in defenders]
            min_dist = min(dists)
            
            if min_dist < 8:
                defender = defenders[dists.index(min_dist)]
                defender_pos = defender.get_position()
                
                if len(self.frontier_list) > 0:
                    dist_from_def = [self.get_maze_distance(defender_pos, f) for f in self.frontier_list]
                    best_entry = self.frontier_list[dist_from_def.index(max(dist_from_def))]
                    self.flanking_target = best_entry 

        # ---------------------------------------------------------
        # 2. FEATURE CALCULATION (Exclusive Logic)
        # ---------------------------------------------------------

        # RETURN MODE (Maximum Priority if full)
        if food_carrying >= 4 and self.capsule_timer == 0 :
            print('nigga')
            features['distance_to_home'] = min([self.get_maze_distance(my_pos, f) for f in self.frontier_list])
            features['successor_score'] = 0
            features['distance_to_food'] = 0
            features['flank_entry'] = 0
            # If we have to escape with food, we cancel any flanking plan
            self.flanking_target = None 

        # FLANKING MODE (High Priority - Persistent)
        elif self.flanking_target is not None:
            features['flank_entry'] = self.get_maze_distance(my_pos, self.flanking_target)
            
            # TOTAL BLOCKADE OF DISTRACTIONS
            features['successor_score'] = 0
            features['distance_to_food'] = 0
            features['distance_to_home'] = 0 

        # NORMAL MODE (Food)
        else:
            features['successor_score'] = -len(food_list)
            if len(food_list) > 0:
                features['distance_to_food'] = min([self.get_maze_distance(my_pos, f) for f in food_list])
            
            features['distance_to_home'] = 0

        # ---------------------------------------------------------
        # 3. SAFETY AND DEAD ENDS
        # ---------------------------------------------------------
        
        # Danger
        if len(defenders) > 0:
            dists = [self.get_maze_distance(my_pos, d.get_position()) for d in defenders]
            min_dist = min(dists)
            
            # "Courage" adjustment:
            # If we are at base (not Pacman), we tolerate the enemy closer (2 steps)
            # If we are Pacman, we keep the safe distance (5 steps)
            threshold = 5
            if not my_state.is_pacman: threshold = 2
                
            if min_dist < threshold:
                features['danger'] = (threshold - min_dist)

        # Dead Ends
        if my_pos_int in self.dead_ends:
            nearby_defenders = False
            if len(defenders) > 0:
                dists_to_def = [self.get_maze_distance(my_pos, d.get_position()) for d in defenders]
                if min(dists_to_def) < 6:
                    nearby_defenders = True

            if nearby_defenders and my_pos_int not in power_capsules:
                food_set = set(food_list)
                only_food_in_dead_ends = food_set.issubset(self.dead_ends)
                
                if only_food_in_dead_ends:
                    if len(power_capsules) > 0:
                        features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in power_capsules])
                        features['successor_score'] = 0
                        features['distance_to_food'] = 0
                        features['distance_to_home'] = 0
                        features['flank_entry'] = 0
                    else:
                        features['distance_to_home'] = min([self.get_maze_distance(my_pos, f) for f in self.frontier_list])
                        features['successor_score'] = 0
                        features['distance_to_food'] = 0
                        features['flank_entry'] = 0
                else:
                    features['dead_end'] = 1

      


        if action == Directions.STOP: features['stop'] = 1
            
        return features


    def get_weights(self, game_state, action):
        return {'successor_score': 100, 
                'distance_to_food': -1, 
                'distance_to_home': -2, 
                'danger' : -10000, 
                'dead_end' : -1000, 
                'distance_to_capsule' : -1000, 
                'flank_entry' : -5, 
                'emergency_escape' : -1000, 
                'stop' : -100}
    
    
    


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like. Â It is not the best or only way to make
    such an agent.
    """
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        current_defending_food = self.get_food_you_are_defending(game_state).as_list()
        successor_defending_food = self.get_food_you_are_defending(successor).as_list()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)] # enemies' states
        defenders = [enemie for enemie in enemies if not enemie.is_pacman and enemie.get_position() is not None and enemie.scared_timer == 0] #(enemy)defenders' states
        invaders_unseen = [enemie for enemie in enemies if enemie.is_pacman]
        frontier_patrol = self.frontier_patrol
        
        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_dist = min(dists)
            if my_state.scared_timer > 0 :
                if closest_dist < 6 :
                    features['scared'] = min(dists)
                    features['invader_distance'] = 0
                    features['distance_to_lost_food'] = 0
                    features['distance_to_patrol_point'] = 0
                else :
                    features['invader_distance'] = min(dists)
                    features['scared'] = 0

        elif self.lost_food is not None :
            features['distance_to_lost_food'] = self.get_maze_distance(my_pos, self.lost_food)
        else :
            patrol_points = frontier_patrol
            target_indexes = [0, int(len(patrol_points)/2), len(patrol_points) - 1]
            target_pos = patrol_points[target_indexes[self.current_frontier_index_patrol_point]]
            dist = self.get_maze_distance(my_pos, target_pos)
            features['distance_to_patrol_point'] = dist
        
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features


    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 
                'on_defense': 10000, 
                'invader_distance': -10, 
                'stop': -100, 
                'reverse': -2, 
                'distance_to_lost_food' : -5, 
                'distance_to_patrol_point' : -5, 
                'scared' : 100}
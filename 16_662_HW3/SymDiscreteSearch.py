import numpy as np
import heapq

def CheckCondition(state, condition):
    """
    Check if the condition is satisfied in the state
    """
    if (np.sum(np.multiply(state, condition))-np.sum(np.multiply(condition, condition)))==0:
        return True
    else:
        return False

def CheckVisited(state,vertices):
    """
    Check if the state is already visited
    """
    for i in range(len(vertices)):
        if np.linalg.norm(np.subtract(state,vertices[i]))==0:
            return True
    return False

def ComputeNextState(state, effect):
    """
    Compute the next state by applying the effect
    """
    newstate=np.add(state, effect)
    return newstate


def Heuristic(state, GoalIndicesOneStep, GoalIndicesTwoStep): 
    """
    Compute the heuristic value for the state
    """
    score=0
    for idx in GoalIndicesOneStep:
        if state[idx[0]][idx[1]]==-1:
            score+=1

    for idx in GoalIndicesTwoStep:
        if state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==-1:
            score+=2
        elif state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==1:
            score+=1	
            
    return score

def InitializePreconditionsAndEffects(nrObjects, nrPredicates, Predicates, Objects):
    ActionPre=[]
    ActionEff=[]
    ActionDesc=[]

    ### Move to hallway
    for i in range(1,5,1):
        Precond=np.zeros([nrObjects, nrPredicates])
        Precond[0][0]=-1 #Robot not in hallway
        Precond[0][i]=1  #Robot in i-th room

        Effect=np.zeros([nrObjects, nrPredicates])
        Effect[0][0]=2.  # Robot in the hallway
        Effect[0][i]=-2. # Robot not in the i-th room
    
        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Move to InHallway from "+Predicates[i])

    ### Move to room
    for i in range(1,5,1):
        Precond=np.zeros([nrObjects, nrPredicates])
        Precond[0][0]=1  # Robot in the hallway
        Precond[0][i]=-1 # Robot not in the ith room

        Effect=np.zeros([nrObjects, nrPredicates])
        Effect[0][0]=-2. # Robot not in the hallway
        Effect[0][i]=2.  # Robot in the ith room

        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Move to "+Predicates[i]+" from InHallway")

    ### Move to Pantry 
    Precond=np.zeros([nrObjects, nrPredicates])
    Precond[0][1]=1  # Robot in the kitchen
    Precond[0][5]=-1 # Robot not in the pantry

    Effect=np.zeros([nrObjects, nrPredicates])
    Effect[0][1]=-2. # Robot not in the kitchen
    Effect[0][5]=2.  # Robot in the the pantry

    ActionPre.append(Precond)
    ActionEff.append(Effect)
    ActionDesc.append("Move to InPantry from InKitchen")

    ### Move from Pantry 
    Precond=np.zeros([nrObjects, nrPredicates])
    Precond[0][5]=1  # Robot in the pantry
    Precond[0][1]=-1 # Robot not in the kitchen

    Effect=np.zeros([nrObjects, nrPredicates])
    Effect[0][5]=-2. # Robot not in the pantry
    Effect[0][1]=2.  # Robot in the the kitchen

    ActionPre.append(Precond)
    ActionEff.append(Effect)
    ActionDesc.append("Move to InKitchen from InPantry")


    ###Cut fruit in kitchen
    for j in [1,2]:
        Precond=np.zeros([nrObjects, nrPredicates])
        Precond[0][1]=1  # Robot in the kitchen
        Precond[j][1]=1  # Fruit j in the kitchen
        Precond[4][1]=1  # Knife in the kitchen
        Precond[j][6]=-1  # Fruit is not chopped
        
        Effect=np.zeros([nrObjects, nrPredicates])
        Effect[j][6]=2  # Fruit is chopped

        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Cut "+Objects[j]+" in the kitchen")

    ###Pickup object
    for i in range(1,6,1):
        for j in range(1,5,1):
            Precond=np.zeros([nrObjects, nrPredicates])
            Precond[0][i]=1 #Robot in ith room
            Precond[j][i]=1 #Object j in ith room
            Precond[j][-1]=-1 #Object j not on robot

            Effect=np.zeros([nrObjects, nrPredicates])
            Effect[j][i]=-2 #Object j not in ith room
            Effect[j][-1]=2 # Object j on robot

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Pick up "+Objects[j]+" from "+Predicates[i])
        
    ###Place object
    for i in range(1,6,1):
        for j in range(1,5,1):
            Precond=np.zeros([nrObjects, nrPredicates])
            Precond[0][i]=1 #Robot in ith room
            Precond[j][i]=-1 #Object j not in ith room
            Precond[j][-1]=1 #Object j on robot

            Effect=np.zeros([nrObjects, nrPredicates])
            Effect[j][i]=2.  #Object j in ith room
            Effect[j][-1]=-2 #Object j not on robot

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Place "+Objects[j]+" at "+Predicates[i])
            
    return ActionPre, ActionEff, ActionDesc

def InitializeStateAndGoal(nrObjects, nrPredicates):
    InitialState=-1*np.ones([nrObjects, nrPredicates])
    InitialState[0][0]=1 # Robot is in the hallway
    InitialState[1][4]=1 # Strawberry is in the garden
    InitialState[2][5]=1 # Lemon is in the pantry
    InitialState[3][2]=1 # Paper is in the office
    InitialState[4][2]=1 # Knife is in the office
    
    GoalState=np.zeros([nrObjects, nrPredicates])
    GoalState[0][1]=1 # Robot is in the kitchen
    GoalState[1][1]=1 # Strawberry is in the kitchen
    GoalState[2][4]=1 # Lemon is in the Garden
    GoalState[1][6]=1 # Strawberry is chopped
    
    return InitialState, GoalState

np.random.seed(13)

########### Construct the Initial State and Goal State ###########
Predicates=['InHallway', 'InKitchen', 'InOffice', 'InLivingRoom', 'InGarden','InPantry','Chopped','OnRobot']
Objects=['Robot','Strawberry','Lemon', 'Paper', 'Knife'] 
nrPredicates=len(Predicates)
nrObjects=len(Objects)

ActionPre, ActionEff, ActionDesc = InitializePreconditionsAndEffects(nrObjects, nrPredicates, Predicates, Objects)
InitialState, GoalState = InitializeStateAndGoal(nrObjects, nrPredicates)

# For Heuristic
GoalIndicesOneStep=[[0,1],[1,6]]
GoalIndicesTwoStep=[[1,1],[2,4]]

########### Search ###########
vertices=[] # List of Visited States
parent=[]
action=[]
cost2come=[]
pq = [] # Use heapq to implement priority queue

# Insert the initial state
heapq.heappush(pq, (0, 0)) # (cost, vertex_id)
vertices.append(InitialState)
parent.append(0)
action.append(-1)
cost2come.append(0)

x = -1

FoundPath=False
while len(pq)>0:
    cost, vertex_id = heapq.heappop(pq)

    for i in range(len(ActionPre)):
        if CheckCondition(vertices[vertex_id], ActionPre[i]):
            next_state = ComputeNextState(vertices[vertex_id], ActionEff[i])
            # print(next_state)
            if not CheckVisited(next_state, vertices):
                vertices.append(next_state)
                parent.append(vertex_id)
                action.append(i)
                cost2come.append(cost2come[vertex_id] + 1)
                # next_cost = cost2come[-1]
                next_cost = cost2come[-1] + Heuristic(next_state, GoalIndicesOneStep, GoalIndicesTwoStep)
                heapq.heappush(pq, (next_cost, len(vertices)-1))
 
                if CheckCondition(next_state, GoalState):
                    FoundPath = True
                    x = len(vertices) - 1
                    break

    if x != -1:
        break

print("Path Found: ", FoundPath)

# Extract Plan
Plan=[]
if FoundPath:
    while not x==0:
        Plan.insert(0,action[x])
        x=parent[x]
        
    # Print Plan
    print("States Explored: ", len(vertices))
    print("Plan Length: ", len(Plan))
    print()
    print("Plan:")
    for i in range(len(Plan)):
        print(ActionDesc[Plan[i]])
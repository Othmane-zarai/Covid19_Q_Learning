from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
#A better view for our model
def plot_Model(t, S ,E, I , R,D):
  f , ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
  ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
  ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
  ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')
  ax.set_xlabel('Time (days)')
  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, c='w',lw=2,ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top','right','bottom','left'):
    ax.spines[spine].set_visible(False)
  plt.show()
#function for defining our propagation model equations
def deriv(y , t , N , beta , gamma,delta,alpha , rho):
  S , E,I,R,D= y
  dSdt = - beta * S * I / N
  dEdt = beta * S * I / N - delta * E
  dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
  dRdt = (1 - alpha) * gamma * I
  dDdt = alpha * rho * I
  return dSdt , dEdt , dIdt , dRdt , dDdt
#Each action has some benefits on R0 : the total number of people an infected person can infect
def R_0(action):
    R0 = 5.0
    if action == "Close groceries and urgent services":
        return 0.25*R0
    if action =="Ask to remain home except for food and urgent services":
        return 0.22*R0
    if action =="Most services close":
        return 0.07*R0
    if action =="Schools and universities close":
        return  0.15*R0
    if action =="Travel Restrictions":
        return 0.15*R0
    if action =="Temperatures Checkpoints":
        return 0.08*R0
    if action =="Sports close":
        return 0.08*R0
    if action =="Bars and restaurants close":
        return 0.24*R0
    if action =="Conferences close":
        return 0.04*R0
    if action =="Airgaps with food delivery":
        return 0.02*R0
    if action=="Everything open":
        return R0
#Defining states on day t depending on R0 value
def states(R_0 , day):
    N=1_000_000
    D = 4.0 # infections lasts four days
    gamma = 1.0 / D
    delta = 1.0 / 5.0  # incubation period of five days
    beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma
    alpha = 0.2  # 10% death rate + we need to add ressource and age dependency
    rho = 1/9  # 9 days from infection until death
    S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed
    t = np.linspace(0,day , day)
    y0 = S0 ,E0, I0 , R0 , D0
    ret = odeint(deriv , y0 , t , args=(N, beta , gamma,delta , alpha , rho ))
    S ,E , I , R , D= ret.T
    return S,E,I,R,D
#Total number of people
N=1_000_000
Reward = np.matrix([[-(max(states(R_0("Close groceries and urgent services"),day)[2]+0.25*N)) for day in range(10,110,10) ],
               [-(max(states(R_0("Ask to remain home except for food and urgent services"),day)[2]+0.2*N)) for day in range(10,110,10) ],
               [   -(max(states(R_0("Most services close"),day)[2]+0.4*N)) for day in range(10,110,10)],
               [-(max(states(R_0("Schools and universities close"),day)[2]+0.4*N)) for day in range(10,110,10)],
               [-(max(states(R_0("Travel Restrictions"),day)[2]+0.3*N)) for day in range(10,110,10) ],
               [-(max(states(R_0("Temperatures Checkpoints"),day)[2]+0.1*N)) for day in range(10,110,10)  ],
               [-(max(states(R_0("Sports close"),day)[2]+0.01*N)) for day in range(10,110,10)],
               [-(max(states(R_0("Bars and restaurants close"),day)[2]+0.15*N)) for day in range(10,110,10)],
               [-(max(states(R_0("Conferences close"),day)[2]+0.001*N)) for day in range(10,110,10)],
               [-(max(states(R_0("Airgaps with food delivery"),day)[2]+0.02*N)) for day in range(10,110,10)]
               ])
#Defining Q-table
Q = np.matrix(np.zeros([10,10]))
gamma = 0.8
initial_state = 0
#Available actions for a chosen state
def available_actions(state):
  current_state_row = Reward[state,]
  av_act = np.where(current_state_row < 0)[1]
  return av_act
available_act = available_actions(initial_state)
#Choose random action for training
def sample_next_action(available_actions_range):
  next_action = int(np.random.choice(available_act,1))
  return next_action
action = sample_next_action(available_act)
#Updating q- table with bellman function
def update(current_state, action, gamma):
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_index.shape[0] > 1:
    max_index = int(np.random.choice(max_index, size = 1))
  else:
    max_index=int(max_index)
  max_value=Q[action,max_index]
  Q[current_state, action] = Reward[current_state, action] + gamma * max_value
update(initial_state,action,gamma)
# Next, we train our algorithm so that it self-learns based on
# a long sequence of trials (actions and rewards/punishments)
for i in range(100):
  current_state = np.random.randint(0, int(Q.shape[0]))
  available_act = available_actions(current_state)
  action = sample_next_action(available_act)
  update(current_state,action,gamma)
print("Trained Q matrix:")
#Normalize the "trained" Q matrix
print(Q/np.max(-Q)*100)
Positive_Q = abs(Q)
current_state = 0
steps = [current_state]
#Choosing the optimal actions that we can take regarding people infected and people confined
while current_state != 9:
  next_step_index = np.where(Positive_Q[current_state,] == np.min(Positive_Q[current_state,]))[1]
  if next_step_index.shape[0] > 1:
    next_step_index = int(np.random.choice(next_step_index, size = 1))
  else:
    next_step_index = int(next_step_index)
  steps.append(next_step_index)
  current_state=next_step_index
print("**********************************************************************")
print("Selected path:")
print(steps)
#Associating steps to actions
def steps_to_action(steps):
    L = []
    actions = ["Close groceries and urgent services",
               "Ask to remain home except for food and urgent services",
               "Most services close",
               "Schools and universities close",
               "Travel Restrictions",
               "Temperatures Checkpoints",
               "Sports close",
               "Bars and restaurants close",
               "Conferences close",
               "Airgaps with food delivery"
               ]
    steps = list(dict.fromkeys(steps))
    for i in steps:
        L.append(actions[i])
    return L
Actions=steps_to_action(steps)
print(Actions)
print("Before taking Actions:")
t = np.linspace(0,99,100)
S , E , I , R ,D = states(R_0("Everything open"),100)
plot_Model(t,S,E,I,R,D)
print("After Taking Action:")
R_0_Sum = 5.0
for i in range(len(Actions)):
    R_0_Sum += R_0(Actions[i])
S , E , I , R ,D = states(R_0_Sum,100)
plot_Model(t,S,E,I,R,D)
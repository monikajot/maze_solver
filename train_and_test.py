import time
import numpy as np

from random_environment import Environment
from agent import Agent
import matplotlib.pyplot as plt

lost = [1606610773, 1606610171, 1606609971, 1606609953, 1606610153,
        1606610353, 1606610753, 1606610953, 1606611753, 1606612354,
        1606610550, 1606611350, 1606611751, 1606612351, 1606613151,
        1606640292, 1606640492, 1606640892, 1606641092, 1606638023,
        1606638623, 1606639824, 1606640024]

# Main entry point
for i in range(1):

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    colab_wins = [1606609020, 1606609023]
    wins = [1606639629, 1606609370, 1606609570,1606609770, 1606609770, 1606610372,1606610572,1606610572, 1606610973,1606611174, 1606609148,1606420347, 1606599639, 1606604170, 1606604495, 1606604225, 1606436201, 1606604762,1606437508, 1606415217, 1606606352,1606608679, 1606608443, 1606608390]
    random_seed = 1606704921
    """
    while random_seed in wins:
        random_seed =int(time.time())"""
    np.random.seed(random_seed)
    print("RANDOM SEEED")
    print(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time +200

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on and agent.greedy_path:
            environment.show(state, agent.greedy_path)
        elif display_on:
            environment.show(state)

    # my graph

    print("number of episodes " +str(agent.episode_number))
    print("RANDOM SEEED")
    print(random_seed)
    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False

    path = [state]
    for step_num in range(100):
        action = agent.get_greedy_action(state)

        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        path.append(next_state)
        #environment.show(state, path)
        if distance_to_goal < 0.03:
            has_reached_goal = True
            wins.append(random_seed)
            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        lost.append(random_seed)
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
print('wins')
print(wins)
print('losses')
print(lost)
    #plt.plot(list(range(len(agent.loss))), agent.loss)
    #plt.show()

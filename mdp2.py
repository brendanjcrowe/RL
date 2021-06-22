import numpy as np
import random

MAX_POPULATION = 10
GAMMA = 0.01
THETA = 1

OPEN_STATE = 0
CLOSED_STATE = 1


class State:

    '''
        Representations of our states

            case_count (int): number of people currently infected
            lockdown (boolean): Whether locked down or not
    '''

    def __repr__(self):

        return f"({self.case_count}, {'CLOSED' if self.lockdown else 'OPEN'})"

    def next_states(self):
        return [State(self.case_count+1, OPEN_STATE),
                State(self.case_count, OPEN_STATE),
                State(self.case_count, CLOSED_STATE),
                State(self.case_count-1, CLOSED_STATE)]

    def __init__(self, case_count, lockdown):
        self.case_count = case_count
        self.lockdown = lockdown
        

def initialize_state_space():

    '''
        An enumerated vector representing the state space

            return (numpy.array((number_of_states, ), dtype=State)): our state space
    '''

    state_space = []

    for case_count in range(MAX_POPULATION+1):
        for state in [OPEN_STATE, CLOSED_STATE]:
            state_space.append(State(case_count, state))

    return np.array(state_space)

def initialize_action_space():

    '''
        An enumerated vector representing the actions space

            return (numpy.array((number_of_actions,) dtype=np.int64)): our actions space
    '''

    return np.array([OPEN_STATE, CLOSED_STATE])


def p(s, action, s_prime):

    '''
        Probability of a transition

            s (State): The current state
            action (boolean): The action to take
            s_prime (State): Potential next state

            return (float([0,1])): probability of this transition 
    '''

    # There is a 0% chance of entering a lockdown state that doesn't match the action you
    #   are taking (e.g. you can't enter an open state if you choose to lock down)
    if action != s_prime.lockdown:
        return 0.

    # There is a 100% chance of staying at 0 cases if currently at 0 cases
    if s.case_count == 0 and s_prime.case_count == 0:
        return 1.

    # There is a 100% chance of staying at your current case count if you are at the max.
    #   number of cases and choose to stay open
    if s.case_count == MAX_POPULATION and action == OPEN_STATE:
        return 1.

    # There is a 50% chance of staying at your current case count no matter the action
    if s.case_count == s_prime.case_count:
        return 0.5

    # Defining these transition probabilities might be redundant since we limit the next
    #   possible states in state itself
    # Closing
    if action == CLOSED_STATE:

        # There is a 50% chance of losing one case
        if s_prime.case_count == s.case_count-1:
            return 0.5

        # There is a 0% chance of losing more than one case or going up in cases
        # (We already assigned a 50% chance of staying at the same case count above)
        else:
            return 0.

    # Opening
    if action == OPEN_STATE:

        # There is a 50% chance of gaining one case
        if s_prime.case_count == s.case_count + 1:
            return 0.5

        # There is a 0% chance of gaining more than one case or going down in cases
        # (We already assigned a 50% chance of staying at the same case count above)
        else:
            return 0.



def r(s):

    '''
        Reward for a transition

            s (State): The current state
            action (boolean): The action to take
            s_prime (State): Potential next state

            return (int([-2:1])): reward for this transition
    '''

    return (s.lockdown * -1) + (s.case_count * -1)

def transition_probabilities(state_space, action_space):

    '''
        Tensor representation of our transition probabilities

            state_space (np.array((number_of_states, ), dtype=State)): All possible states
            action_space (np.array((number_of_actions, ), dtype=np.int64)): All possible actions

            return (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.float)):
            Order 3 tensor of transition probabilities
    '''

    return np.array(
        [
            [
                [
                    p(state, action, state_next)
                    for action in action_space
                ]
                for state_next in state_space
            ]
            for state in state_space
        ]
    )


def rewards_(state_space):

    '''
        Tensor representation of our transition probabilities

            state_space (np.array((number_of_states, ), dtype=State)): All possible states
            action_space (np.array((number_of_actions, ), dtype=np.int64)): All possible actions

            return (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.int64)):
            Order 3 tensor of transition rewards
    '''

    return np.array([r(s) for s in state_space])


def policy_evaluation(v, pi, state_space, pt, rewards):

    '''
        Finds the expected reward of the current policy pi

            v (np.array((number_of_states, ), dtype=np.float)): Current values for each state
            pi (np.array((number_of_states, ), dtype=np.int64)): Current policy for each state
            state_space (np.array((number_of_states, ), dtype=State)): All possible states
            pt (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.float)):
            Order 3 tensor of transition probabilities
            rewards (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.int64)):
            Order 3 tensor of transition rewards

            return n(np.array((number_of_states, ), dtype=np.float)): Our new value for each state
                (np.array((number_of_states, ) Our policy pi -- is not altered in this function
    ''' 
    
    print(pi)
    print(pt[np.arange(state_space.shape[0]), :, pi])
    return np.linalg.solve(
        1 - GAMMA * pt[np.arange(state_space.shape[0]), :, pi],
        rewards 
    )



def policy_improvement(v, pi, state_space, pt, rewards):

    ''' 
        Improves our policy based upon new values for states

            v (np.array((number_of_states, ), dtype=np.float)): Current values for each state
            pi (np.array((number_of_states, ), dtype=np.int64)): Current policy for each state
            state_space (np.array((number_of_states, ), dtype=State)): All possible states
            pt (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.float)):
            Order 3 tensor of transition probabilities
            rewards (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.int64)):
            Order 3 tensor of transition rewards

            return n(np.array((number_of_states, ), dtype=np.float)): Value for each state -- not altered in this function
                (np.array((number_of_states, ) Our updated policy pi
                (boolean): Whether our old policy matches our new policy
    '''

    policy_stable = True

    pi_prev = pi.copy()  

    pi = np.argmax(
        np.einsum(
            'ijk, ijk -> ik',
            pt,
            rewards + GAMMA * np.expand_dims(v, axis=1)
        ),
        axis=1
    )
    

    policy_stable = np.array_equal(pi, pi_prev)

    return v, pi, policy_stable



def policy_iteration(v, pi, state_space, pt, rewards):

    ''' 
        Improves our policy based upon new values for states

            v (np.array((number_of_states, ), dtype=np.float)): Current values for each state
            pi (np.array((number_of_states, ), dtype=np.int64)): Current policy for each state
            state_space (np.array((number_of_states, ), dtype=State)): All possible states
            pt (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.float)):
            Order 3 tensor of transition probabilities
            rewards (np.array((number_of_states, number_of_states, number_of_actions), dtype=np.int64)):
            Order 3 tensor of transition rewards

            return n(np.array((number_of_states, ), dtype=np.float)): Value for each state
                (np.array((number_of_states, ) Optimal policy
    '''

    while True:
        print("Evaluation")
        v, pi = policy_evaluation(v, pi, state_space, pt, rewards)
        print(v)
        print("Improvement")
        v, pi, policy_stable = policy_improvement(v, pi, state_space, pt, rewards)
        print(pi)
        if policy_stable:
            return v, pi


def main():
    state_space = initialize_state_space()
    action_space = initialize_action_space()

    rewards = rewards_(state_space)
    pt = transition_probabilities(state_space, action_space)

    v = np.array(
        [
            0 
            for state in state_space
        ]
    )
    pi = np.random.randint(0, 2, 22)
    v, pi = policy_iteration(v, pi, state_space, pt, rewards)

    print(pi)
    print(v)


if __name__ == '__main__':
    main()
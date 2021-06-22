import numpy as np

MAX_POPULATION = 10
GAMMA = 0.01
THETA = 1

OPEN_STATE = 0
CLOSED_STATE = 1


# Define our state by current case count and whether or not we're open
class State:
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
        

# Initialize our state space
def initialize_state_space():
    state_space = []

    for case_count in range(MAX_POPULATION+1):
        for state in [OPEN_STATE, CLOSED_STATE]:
            state_space.append(State(case_count, state))

    return state_space


# Return a transition probability given a state, an action, and the next state
def p(s, action, s_prime):

    # There is a 0% chance of entering a lockdown state that doesn't match the action you
    #   are taking (e.g. you can't enter an open state if you choose to lock down)
    if action != s_prime.lockdown:
        return 0

    # There is a 100% chance of staying at 0 cases if currently at 0 cases
    if s.case_count == 0:
        return 1

    # There is a 100% chance of staying at your current case count if you are at the max.
    #   number of cases and choose to stay open
    if s.case_count == MAX_POPULATION and action == OPEN_STATE:
        return 1

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
            return 0

    # Opening
    if action == OPEN_STATE:

        # There is a 50% chance of gaining one case
        if s_prime.case_count == s.case_count + 1:
            return 0.5

        # There is a 0% chance of gaining more than one case or going down in cases
        # (We already assigned a 50% chance of staying at the same case count above)
        else:
            return 0


# Return for being in a current state, taking an action, and ending up in another state
def r(s, action, s_prime):

    # Choose to close
    if action == CLOSED_STATE:

        # We are already locked down, -1
        if s.lockdown == CLOSED_STATE:
            return -1

        # We are currently open, -2
        else:
            return -2

    # Choose to open
    else:

        # We have a new case, -1
        if s_prime.case_count > s.case_count:
            return -1

        # Our case count stays the same
        else:

            # We are currently locked down, 2
            if s.lockdown == CLOSED_STATE:
                return 2

            # We are already open, 0
            else:
                return 0


def policy_evaluation(v, pi, state_space):
    while True:
        delta = 0
        for state in state_space:
            temp = v[state]
            v[state] = np.sum([
                p(state, pi[state], next_state) * (r(state, pi[state], next_state) + GAMMA*v[state])
                for next_state in state.next_states()
            ])
            delta = max(delta, abs(temp - v[state]))

        if delta < THETA:
            return v, pi


def policy_improvement(v, pi, state_space):
    policy_stable = True

    for state in state_space:
        temp = pi[state]
        pi[state] = np.argmax([
            np.sum([
                p(state, action, next_state) * (r(state, action, next_state) + GAMMA*v[state])
                for next_state in state.next_states()])
            for action in [OPEN_STATE, CLOSED_STATE]
        ])

        if temp != pi[state]:
            policy_stable = False

    return v, pi, policy_stable


def policy_iteration(v, pi , state_space):
    while True:
        print("Evaluation")
        v, pi = policy_evaluation(v, pi, state_space)
        print(v)
        print("Improvement")
        v, pi, policy_stable = policy_improvement(v, pi, state_space)
        print(pi)
        if policy_stable:
            return v, pi


def main():
    state_space = initialize_state_space()
    v = {
        state: 0
        for state in state_space
    }

    pi = {
        state: 0
        for state in state_space
    }

    v, pi = policy_iteration(v, pi, state_space)

    print(pi)
    print(v)


main()

import numpy as np
from csv_loader import load_csv

GAMMA = 0.01
THETA = 1


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

    while True:
        delta = 0

        v_prev = v.copy()

        v = np.einsum(
            'ij, ij -> i',
            pt[np.arange(state_space.shape[0]), :, pi],
            rewards[np.arange(state_space.shape[0]), :, pi] + GAMMA * v
        )

        if max(delta, np.max(np.abs(v_prev - v))) < THETA:
            return v, pi



def policy_improvement(v, pi, pt, rewards):

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
        v, pi, policy_stable = policy_improvement(v, pi, pt, rewards)
        print(pi)
        if policy_stable:
            return v, pi


def value_iteration(v, pt, rewards):

    while True:
        delta = 0

        v_prev = v.copy()

        v = np.max(
        np.einsum(
            'ijk, ijk -> ik',
            pt,
            rewards + GAMMA * np.expand_dims(v_prev, axis=1)
        ),
        axis=1)

        if max(delta, np.max(np.abs(v_prev - v))) < THETA:

            pi = np.argmax(np.einsum(
            'ijk, ijk -> ik',
            pt, rewards + GAMMA * np.expand_dims(v_prev, axis=1)), axis=1)
            return v, pi


def main(path='transitions_singular.csv'):
    pt, rewards, state_space, action_space = load_csv(path)

    print(pt)

    v = np.array(
        [
            0 
            for state in state_space
        ]
    )
    pi = np.random.randint(0, 2, state_space.shape[0])

    print("Value Iteration")
    v, pi = value_iteration(v, pt, rewards)

    print(pi)
    print(v)

    v = np.array(
        [
            0
            for state in state_space
        ]
    )
    pi = np.random.randint(0, 2, state_space.shape[0])

    print("Policy Iteration")
    v, pi = policy_iteration(v, pi, state_space, pt, rewards)

    print(pi)
    print(v)


if __name__ == '__main__':
    main()

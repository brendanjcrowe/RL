# Loads probability and reward matrices from .CSVs

import csv, re
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_csv(path):
    '''
        Ingests a well-formatted CSV file to produce our transition probability and reward matrices.

        Assuming the CSV file is well formatted, the file will include the number of actions possible in our MDL, a description
            of our state space as a list of features and the values that those features can be, and a list of transitions from
            states including the action taken to make that transition, the probability of that transition given that action, and
            the reward for that transition given the action.

        See transitions.csv for a full example.

        return (numpy.array((number_of_actions, number_of_states, number_of_states) dtype=np.int64)): our transition probability matrices for each action
        return (numpy.array((number_of_actions, number_of_states, number_of_states) dtype=np.float64)): our reward matrices for each action

    '''

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        subsample = next(reader)

        # Read in the number of possible actions as defined in the CSV
        actions = int(next(reader)[1])

        # Read in the description of features as defined in the CSV
        features_description = next(reader)[0]

        # Parse the description of features as defined in the CSV
        feature_ranges = []
        features = features_description.split(';')
        for feature in features:
            start, end = re.search('(\d:\d)', feature).group(1).split(':')
            feature_ranges.append(list(range(int(start), int(end)+1)))

        # Create our state space representing all possible combinations of our features
        state_space = np.array(np.meshgrid(*feature_ranges)).T.reshape(-1, len(feature_ranges)).tolist()

        # Create blank (zero) matrices for our state and reward space
        p_matrices = np.zeros((len(state_space), len(state_space), actions))
        r_matrices = np.zeros((len(state_space), len(state_space), actions))

        # Skip the header row
        next(reader)

        # Parse the defined transition probabilities and rewards as defined in the CSV
        for row in reader:
            from_state = list(map(lambda x: int(x), row[0].split(',')))
            to_state = list(map(lambda x: int(x), row[1].split(',')))
            action = int(row[2])
            prob = float(row[3])
            reward = int(row[4])

            from_index = state_space.index(from_state)
            to_index = state_space.index(to_state)
            p_matrices[from_index][to_index][action] = prob
            r_matrices[from_index][to_index][action] = reward

        # TODO: Rewrite checks with new indexing scheme
        # # Check if the transition probability matrices are stochastic and invertible
        # for action in range(actions):
        #     p_matrix =
        #     if not all([np.sum(row) == 1 for row in p]):         # Check if P matrices are stochastic
        #         print(f"WARNING: Transition probability matrix for action {i} is NOT stochastic.")
        #     try:
        #         np.linalg.inv(p)
        #     except np.linalg.LinAlgError:                     # Check if P matrix is invertible
        #         print(f"WARNING: Transition probability matrix for action {i} is NOT invertible.")

        return np.array(p_matrices), np.array(r_matrices), np.array(state_space), np.array(list(range(actions)))


def load_marek_csv(path):
    transitions = pd.read_csv(path)
    transitions = transitions[transitions['idstate_to'] <= max(transitions['idstate_from'])]
    state_space = range(min(transitions['idstate_from']), max(transitions['idstate_from']) + 1)
    action_space = range(min(transitions['idaction']), max(transitions['idaction']) + 1)
    p_matrices = np.zeros((len(state_space), len(state_space), len(action_space)))
    r_matrices = np.zeros((len(state_space), len(state_space), len(action_space)))

    print("Loading Marek's CSV")
    for index, row in tqdm(transitions.iterrows(), position=0, leave=True):
        from_index = int(row['idstate_from']) - 1
        to_index = int(row['idstate_to']) - 1
        prob = row['probability']
        reward = row['reward']
        action = int(row['idaction'])

        p_matrices[from_index][to_index][action] = prob
        r_matrices[from_index][to_index][action] = reward

    return np.array(p_matrices), np.array(r_matrices), np.array(list(state_space)), np.array(list(action_space))


if __name__ == '__main__':
    p, r, state_space, action_space = load_csv('/Users/colinrsmall/Documents/GitHub/aml-spring2021/mdp/transitions.csv')
    # print(p[0])
    # print(r[0])

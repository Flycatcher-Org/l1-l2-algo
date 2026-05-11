from dataclasses import dataclass
from typing import Optional
import random

import numpy as np

from tree import Node, PlayerNode, TerminalNode, build_node, print_tree

HIGHEST_NUMBER = 20

tree = build_node()
random.seed(0)

# This function initializes strategy tables to 0.
# The memory layout for a strategy table is a np.array
# (public_number, private_number, action_index).
# Nodes before the public number is revealed are just
# (private_number, action_index).
# Action indices align with their order in the children
# array.
def init_table(tree: Node):
    strategy = {}
    for n in tree:
        if isinstance(n, TerminalNode):
            continue

        assert isinstance(n, PlayerNode)
        shape = [HIGHEST_NUMBER + 1] * (n.round + 1)
        shape = (*shape, len(n.children))
        strategy[n] = np.zeros(shape)
    return strategy

# Represents a probabilistic sample of an outcome
# (i.e. a private number for each player and a
# public number).
@dataclass(slots=True, frozen=True)
class Sample:
    players: tuple[int, int]
    public: tuple[int, ...]

    @classmethod
    def create(cls) -> 'Sample':
        numbers = tuple(random.randint(0, HIGHEST_NUMBER) for _ in range(3))
        return Sample((numbers[0], numbers[1]), numbers[2:])

# This function is denoted "u" in the pseudo-code.
# Calculates the utility (reward) of a player reaching a terminal
# node given the current sample.
def utility(node: TerminalNode, player: int, sample: Sample) -> float:
    winner = node.winner
    if winner is None:
        # showdown
        p1_score = abs(sample.public[0] - sample.players[0])
        p2_score = abs(sample.public[0] - sample.players[1])
        if p1_score == p2_score:
            return 0
        elif p1_score < p2_score:
            winner = 0
        else:
            winner = 1
    return node.pot / 2 if winner == player else -node.pot / 2

# Applies RegretMatching over the last dimension of the np array.
# I.e. this function normalizes the last dimension according to the
# RegretMatching algorithm.
# This function can be used to create a strategy from the regrets
# table.
def regret_matching(table: np.ndarray) -> np.ndarray:
    pos = np.maximum(table, 0)
    s = np.sum(pos, axis=-1, keepdims=True)

    # Avoid division by 0
    np.divide(pos, s, out=pos, where=(s != 0))

    # Choose uniform random for empty entries.
    zero_rows = (s[..., 0] == 0)
    if np.any(zero_rows):
        pos[zero_rows] = 1 / pos.shape[-1]

    return pos

# Returns a key that can be used to index a strategy table's np.ndarray.
# Where the strategy information is stored depends on the game node
# as well as the current sample. The action index can be ommitted
# to return the data for all actions. See `init_table` for more on
# table layout.
def strategy_key(node: PlayerNode, sample: Sample, ix: Optional[int] = None) -> tuple[int, ...]:
    public_chance = list(reversed(sample.public))
    visible_public_chance = public_chance[:node.round]
    key = visible_public_chance + [sample.players[node.player]]
    if ix is not None:
        return (*key, ix)
    return tuple(key)


# See init_table for table layout and use strategy_key for lookup.
current:    dict[PlayerNode, np.ndarray] = init_table(tree)
regrets:    dict[PlayerNode, np.ndarray] = init_table(tree)
cumulative: dict[PlayerNode, np.ndarray] = init_table(tree)

# This function is denoted "\pi" in the pseudo-code.
def probability_of_reaching(n: Node, sample: Sample, *, from_node: Optional[Node] = None, only_from_player: Optional[int] = None, aside_from_player: Optional[int] = None) -> float:
    if n == from_node:
        return 1

    prob = 1
    action = n.previous_action
    parent = n.parent
    while parent is not None and parent != from_node:
        assert isinstance(parent, PlayerNode)

        if (aside_from_player is None or parent.player != aside_from_player) and (only_from_player is None or parent.player == only_from_player):
            for ix, (a, _) in enumerate(parent.children):
                if a == action:
                    break
            else:
                assert False
            key = strategy_key(parent, sample, ix)
            prob *= current[parent][key]

        action = parent.previous_action
        parent = parent.parent

    if parent is None:
        assert from_node is None, f'Could not find parent node {from_node} of {n}'
    return prob

# -------- END OF STARTER CODE --------

# This function is denoted "v" in the pseudo-code.
def counterfactual_value(node: Node, player: int, sample: Sample) -> float:
    return NotImplemented # TODO

# Ananlyze the running time of this function
def train(T: int, nsamples: int) -> dict[PlayerNode, np.ndarray]:
    # Index into the strategy tables like so:
    # current[node][strategy_key(node, sample, action_ix)]
    # If action_ix is ommited, an ndarray is returned with
    # the data for each action.
    return NotImplemented # TODO

def main():
    print_tree(tree)
    train(100, 10)

if __name__ == '__main__':
    main()

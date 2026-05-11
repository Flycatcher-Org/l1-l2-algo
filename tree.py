from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy
from typing import Callable, Iterator, Optional

class Node(ABC):
    def __init__(self, previous_action: str, round_number: int):
        self.previous_action = previous_action
        self.round = round_number
        self.parent: Optional['Node'] = None

    def __str__(self):
        return f'node at path {self.path()} round#{self.round}'

    def __repr__(self):
        return str(self)

    def path(self) -> str:
        ppath = ''
        end = ''
        if self.parent is not None:
            ppath = self.parent.path()
            if self.parent.round != self.round:
                end = '/'
        return ppath + self.previous_action + end

    @abstractmethod
    def __iter__(self) -> Iterator['Node']:
        return NotImplemented

class PlayerNode(Node):
    def __init__(self, previous_action: str, round_number: int, player: int, children: list[tuple[str, Node]]):
        super().__init__(previous_action, round_number)
        self.player = player
        self.children = children
        for _, c in self.children:
            c.parent = self

    def __str__(self):
        return super().__str__() + f' player {self.player}\'s turn'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield self
        for _, c in self.children:
            yield from c

class TerminalNode(Node):
    def __init__(self, previous_action: str, round_number: int, pot: int, winner: Optional[int] = None):
        super().__init__(previous_action, round_number)
        self.pot = pot
        self.winner = winner

    def __str__(self):
        return super().__str__() + f' ({self.pot} chips)'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield self

@dataclass
class Round:
    nraises: int = 0
    contrib: list[int] = field(default_factory=lambda: [0, 0])
    player: int = 0
    must_act: list[bool] = field(default_factory=lambda: [True, True])

@dataclass
class Construction:
    # This stores contributions between rounds.
    pot: int = 0
    round_number: int = 0
    game_over: bool = False
    round: Round = field(default_factory=Round)

def build_fold(c: Construction):
    c.round.must_act = [False, False]
    c.round.player = 1 - c.round.player
    c.game_over = True
    return build_node('f', c)

def build_raise(c: Construction):
    c.round.nraises += 1
    c.round.contrib[c.round.player] += 1
    c.round.must_act = [True, True]
    c.round.must_act[c.round.player] = False
    c.round.player = 1 - c.round.player
    return build_node('r', c)

def build_call(c: Construction):
    c.round.must_act[c.round.player] = False
    if c.round.nraises > 0:
        c.round.contrib[c.round.player] += 1
    c.round.player = 1 - c.round.player
    return build_node('c', c)

def build_node(previous_action: str = '', c: Optional[Construction] = None) -> Node:
    if c is None:
        c = Construction()

    if c.game_over:
        return TerminalNode(previous_action, c.round_number, c.pot, c.round.player)

    if not any(c.round.must_act):
        # Round over!
        # Avoid putting money into the pot until the round is complete (the money is matched).
        c.pot += sum(c.round.contrib)
        c.round_number += 1
        c.round = Round()

        if c.round_number > 1:
            # Terminal!
            return TerminalNode(previous_action, c.round_number, c.pot)

    assert c.round.must_act[c.round.player]

    children = []
    if c.round.nraises > 0:
        children.append(('f', build_fold(copy.deepcopy(c))))
    else:
        children.append(('r', build_raise(copy.deepcopy(c))))

    children.append(('c', build_call(copy.deepcopy(c))))

    return PlayerNode(previous_action, c.round_number, c.round.player, children)

def print_tree(n: Node, custom: Optional[Callable[[Node], str]] = None, action: Optional[str] = None, indent: int = 0):
    if action is None:
        print(n, end='')
    else:
        print(f'{"| " * indent}{action}: {n}', end='')
    if custom is not None:
        print('', custom(n))
    else:
        print()

    if isinstance(n, PlayerNode):
        for (a, c) in n.children:
            print_tree(c, custom, a, indent + 1)

if __name__ == '__main__':
    print_tree(build_node())

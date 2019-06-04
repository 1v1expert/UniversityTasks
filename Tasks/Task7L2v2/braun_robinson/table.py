from collections import defaultdict

from sympy import Matrix, Rational

from lab7.braun_robinson.algorithm_step import BraunRobinsonAlgorithmStep


class BraunRobinsonTable:
    def __init__(self, payoff_matrix: Matrix, first_a: int, first_b: int):
        self.payoff_matrix = payoff_matrix
        self.steps = []

        self.make_step(first_a, first_b)

    def solve(self, threshold: float = 0.01, max_steps: int = 2**64):
        while threshold < self.steps[-1].epsilon \
                and max_steps > len(self.steps):
            self.make_step()

    def make_step(self, a_strategy: int = None, b_strategy: int = None):
        if all([a_strategy is None, b_strategy is None]):
            a_strategy = self._get_next_a_strategy()
            b_strategy = self._get_next_b_strategy()
        elif any([a_strategy is None, b_strategy is None]):
            raise Exception(
                'Both strategy should be defined or should be None.'
                ' Got strategies: %s',
                [a_strategy, b_strategy]
            )

        step = BraunRobinsonAlgorithmStep(
            a_choice=a_strategy,
            b_choice=b_strategy,
            payoff_matrix=self.payoff_matrix,
            previous_step=self.steps[-1] if self.steps else None
        )
        self.steps.append(step)

    def get_a_mixed_strategy(self):
        counter = defaultdict(int)
        for step in self.steps:
            counter[step.a_choice] += 1

        return [Rational(counter[choice], len(self.steps))
                for choice in range(self.payoff_matrix.cols)]

    def get_b_mixed_strategy(self):
        counter = defaultdict(int)
        for step in self.steps:
            counter[step.b_choice] += 1

        return [Rational(counter[choice], len(self.steps))
                for choice in range(self.payoff_matrix.cols)]
    
    @staticmethod
    def annotate_table():
        return '{step_number:^5} {a_choice:^3} {b_choice:^3} {a_gain:^25}'\
               ' {b_gain:^25} {lower_game_cost:^12} {upper_game_cost:^12}'\
               ' {epsilon:^8}'.format(
            step_number='step',
            a_choice='A',
            b_choice='B',
            a_gain='A gain',
            b_gain='B gain',
            lower_game_cost='v lower',
            upper_game_cost='v upper',
            epsilon='E',
        )

    def _get_next_a_strategy(self) -> int:
        max_from_previous_a_gain = max(self.steps[-1].a_gain)

        if len(self.steps) > 1 and self.steps[-2].a_gain[
                    self.steps[-1].a_choice] == max_from_previous_a_gain:
            return self.steps[-1].a_choice

        return self.steps[-1].a_gain.index(max_from_previous_a_gain)

    def _get_next_b_strategy(self) -> int:
        min_from_previous_b_gain = min(self.steps[-1].b_gain)

        if len(self.steps) > 1 and self.steps[-2].b_gain[
            self.steps[-1].b_choice] == min_from_previous_b_gain:
            return self.steps[-1].b_choice

        return self.steps[-1].b_gain.index(min_from_previous_b_gain)

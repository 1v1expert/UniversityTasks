from typing import Optional

from sympy import Matrix, Rational


class BraunRobinsonAlgorithmStep:
    def __init__(
            self,
            a_choice: int,
            b_choice: int,
            payoff_matrix: Matrix,
            previous_step: Optional):
        self.step_number = 1
        if previous_step:
            self.step_number = previous_step.step_number + 1

        self.a_choice = a_choice
        self.b_choice = b_choice

        self.a_gain = payoff_matrix.col(self.b_choice).T.tolist()[0]
        self.b_gain = payoff_matrix.row(self.a_choice).tolist()[0]
        self.min_upper_game_cost = self.upper_game_cost
        self.max_lower_game_cost = self.lower_game_cost

        if previous_step:
            self.a_gain = [sum(x) for x in
                           zip(self.a_gain, previous_step.a_gain)]
            self.b_gain = [sum(x) for x in
                           zip(self.b_gain, previous_step.b_gain)]
            self.min_upper_game_cost = min(
                (self.upper_game_cost, previous_step.min_upper_game_cost))
            self.max_lower_game_cost = max(
                (self.lower_game_cost, previous_step.max_lower_game_cost))

    @property
    def upper_game_cost(self) -> Rational:
        return max(self.a_gain) / Rational(self.step_number)

    @property
    def lower_game_cost(self) -> Rational:
        return min(self.b_gain) / Rational(self.step_number)

    @property
    def epsilon(self) -> Rational:
        return self.min_upper_game_cost - self.max_lower_game_cost

    def __str__(self):
        return '{step_number:^5}|{a_choice:^3}|{b_choice:^3}|{a_gain:^25}'\
               '|{b_gain:^25}|{lower_game_cost:^12}|{upper_game_cost:^12}'\
               '|{epsilon:^8}'.format(
                step_number=self.step_number,
                a_choice=self.a_choice,
                b_choice=self.b_choice,
                a_gain=str(self.a_gain),
                b_gain=str(self.b_gain),
                lower_game_cost=str(self.lower_game_cost),
                upper_game_cost=str(self.upper_game_cost),
                epsilon=str(float(self.epsilon))[:8],
            )

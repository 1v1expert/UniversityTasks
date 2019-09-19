import numpy as np


def print_matrix(matrix, matrix_name="Матрица:"):
    print(matrix_name)
    print("\n".join(", ".join("{0:.3f}".format(x) for x in row) for row in matrix))


class InformationalConfrontationGame(object):
    def __init__(self, dim=10, epsilon=10e-6, opinion_range=(0, 100), initial_opinions=None, trust_matrix=None):
        self.dim = dim
        self.epsilon = epsilon
        self.opinion_range = opinion_range
        self.initial_opinions = initial_opinions
        if not initial_opinions:
            self.gen_initial_opinions()
        
        self.trust_matrix = trust_matrix
        if not trust_matrix:
            self.gen_trust_matrix()
    
    def gen_initial_opinions(self):
        self.initial_opinions = np.random.randint(self.opinion_range[0], self.opinion_range[1], self.dim)
    
    def gen_trust_matrix(self):
        matrix = []
        for _ in np.arange(self.dim):
            row = np.random.sample(self.dim)
            matrix.append(row / row.sum())
        self.trust_matrix = np.array(matrix)
    
    def reach_accuracy(self, opinions):
        _iter = 0
        accuracy_reached = True
        while accuracy_reached:
            _iter += 1
            
            new_opinions = self.trust_matrix.dot(opinions).transpose()
            if all(x <= self.epsilon for x in np.abs(opinions - new_opinions)):
                accuracy_reached = False
            opinions = new_opinions
        return opinions, _iter
    
    def solve(self):
        result_opinions, iter_count = self.reach_accuracy(self.initial_opinions)
        print_matrix(self.trust_matrix, "Матрица доверия:")
        print("Изначальные мнения агентов:")
        print("X(0) =", self.initial_opinions)
        print("Потребовалось итераций:", iter_count)
        print("Результирующее мнение агентов (без влияния):")
        print("X(t->inf) =", ", ".join("{0:.3f}".format(x) for x in result_opinions))
        print()
    
    def solve_with_info_influence(self):
        agents = np.arange(self.dim)
        np.random.shuffle(agents)
        u_size, v_size = len(agents), len(agents)
        while u_size + v_size > len(agents):
            u_size = np.random.randint(1, len(agents))
            v_size = np.random.randint(1, len(agents))
        u_agents = agents[:u_size]
        v_agents = agents[u_size:u_size + v_size]
        print("Агенты первого игрока: {0}, агенты второго игрока: {1}".format(sorted(u_agents), sorted(v_agents)))
        opinions_with_infl = self.initial_opinions
        u_influence_value = np.random.randint(self.opinion_range[0], self.opinion_range[1])
        v_influence_value = -np.random.randint(self.opinion_range[0], self.opinion_range[1])
        print("Сформированное начальное мнение первого игрока: {0:.0f}".format(u_influence_value))
        print("Сформированное начальное мнение второго игрока: {0:.0f}".format(v_influence_value))
        for number in np.hstack((v_agents, u_agents)):
            opinions_with_infl[number] = u_influence_value if number in u_agents else v_influence_value
        
        print("Изначальные мнения с учетом сформированных:")
        print("X(0) =", opinions_with_infl)
        result_opinions, iter_count = self.reach_accuracy(opinions_with_infl)
        print("Потребовалось итераций:", iter_count)
        print("Результирующее мнение:")
        print("X(t->inf) =", ", ".join("{0:.3f}".format(x) for x in result_opinions))
        # вывод матрицы в степени бескоечность
        accur_matrix = self.trust_matrix
        for _ in range(iter_count - 1):
            accur_matrix = accur_matrix.dot(self.trust_matrix)
        print_matrix(accur_matrix)


if __name__ == '__main__':
    game = InformationalConfrontationGame(10)
    game.solve()
    game.solve_with_info_influence()

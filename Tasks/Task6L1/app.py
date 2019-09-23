import numpy as np
import json

import contextlib

EPSILONE_CONST = 0.1


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class AnalyticalGame(object):
    ''' Аналитический(матричный) метод решения матричной игры с нулевой суммой '''
    
    def __init__(self, **args):
        self.matrix = args.get('matrix', None)
        self.calculate = args.get('calculate', None)
        self.print = args.get('print', True)
        
        self.n = len(self.matrix)  # lines
        self.ones = np.ones(self.n)
        self.np_matrix = np.array(self.matrix, dtype=np.float)
        self.m = self.np_matrix.size / self.n  # columns
        self.inv_matrix = np.linalg.inv(self.np_matrix)
        self.optimal_strategy_x, self.optimal_strategy_y = None, None
        
        if self.calculate:
            self.optimal_strategy_x, self.optimal_strategy_y, self.price = self._get_decision_game()
        
        self.print_result()
    
    def _get_decision_game(self):
        self.denominator = np.dot(np.dot(self.ones, self.inv_matrix), self.ones.transpose())
        x = np.dot(self.inv_matrix, self.ones.transpose()) / self.denominator
        y = np.dot(self.ones, self.inv_matrix) / self.denominator
        return np.around(x, decimals=2), np.around(y, decimals=2), 1 / self.denominator
    
    def print_result(self):
        if self.print:
            print('Optimal strategy X: {},' \
                  '\nOptimal strategy Y: {},'
                  '\nPrice game = {}'.format(self.optimal_strategy_x,
                                             self.optimal_strategy_y,
                                             1 / self.denominator)
                  )


class BraunRobinsGame(AnalyticalGame):
    
    def __init__(self, **args):
        super().__init__(**args)
        self.top_game_price = None
        self.lower_game_price = None
        self.result_matrix = None
        self.winnings_ab = None
        # self.get_price_game()
        # TODO: maybe not transpose matrix
        self.np_matrix = self.np_matrix  # .transpose()  # for test
        
        self.static_matrix = None  # np.array([[1, 1, 0, 0]])
        self.winnings_a = np.array([np.ones(self.n)])
        self.winnings_b = np.array([np.ones(self.n)])
        
        self.building_payment_matrix()
    
    # ll = self.np_matrix.tolist()
    # print(ll.index([3, 0, 1]))
    
    def building_payment_matrix(self):
        for cycle in range(100):
            if not cycle:
                posI, posJ = self.get_price_game()
                self.static_matrix = np.array([[
                    posI + 1,
                    posJ + 1,
                    self.np_matrix[posI, :].min() / (cycle + 1),
                    self.np_matrix[:, posJ].max() / (cycle + 1),
                    (self.np_matrix[posI, :].min() / (cycle + 1) + self.np_matrix[:, posJ].max() / (cycle + 1)) / 2]])
                self.winnings_a = np.vstack([self.winnings_a, self.np_matrix[:, posJ]])
                self.winnings_b = np.vstack([self.winnings_b, self.np_matrix[posI, :]])
                self.winnings_a = np.delete(self.winnings_a, (0), axis=0)  # remove first line
                self.winnings_b = np.delete(self.winnings_b, (0), axis=0)  # remove first line
            else:
                rowB, posI, rowA, posJ = self.iteration(cycle - 1)
                self.static_matrix = np.vstack([
                    self.static_matrix, [
                        posI + 1,
                        posJ + 1,
                        rowB.min() / (cycle + 1),
                        rowA.max() / (cycle + 1),
                        (rowB.min() / (cycle + 1) + rowA.max() / (cycle + 1)) / 2]])
                self.winnings_a = np.vstack([self.winnings_a, rowA])
                self.winnings_b = np.vstack([self.winnings_b, rowB])
            epsilone = self.static_matrix[:, 3].min() - self.static_matrix[:, 2].max()
            print('Epsilon {} ->> {}'.format(cycle + 1, epsilone))
            if epsilone <= EPSILONE_CONST:
                print('BREAK with epsilon = {}'.format(epsilone))
                break
        # print(epsilone)
        # for i, row in enumerate(self.static_matrix):
        # 	#if not i:
        # 	#	posI, posJ = get_price_game()
        #
        # 	#print(i, row)
        # 	x1, y1 = row[0]-1, row[1]-1
        # 	#print(self.np_matrix[:, x1])
        # 	#print()
        # 	self.winnings_a = np.vstack([self.winnings_a, self.np_matrix[:, x1]])
        # 	self.winnings_b = np.vstack([self.winnings_b, self.np_matrix[y1, :]])
        # 	if not i:
        # 		self.winnings_a = np.delete(self.winnings_a, (0), axis=0)  # remove first line
        # 		self.winnings_b = np.delete(self.winnings_b, (0), axis=0)  # remove first line
        
        self.pprint_payment_matrix()
    
    # self.winnings_a.put(self.np_matrix.max(axis=0))
    # print(self.winnings_a, self.winnings_b)
    
    def iteration(self, iterr):
        max_a = self.winnings_a[iterr].max()  # нахождение макисмального элемента у играка А
        pos_max_a = np.nonzero(self.winnings_a[iterr] == max_a)[0][0]  # получение позиции максимального элемента
        line = self.np_matrix[pos_max_a, :]  # взятие соответствующей строки матрица игрока А
        new_row_winnings_b = self.winnings_b[iterr] + line  # сложение результатов
        # ======  другой игрок
        min_b = new_row_winnings_b.min()
        pos_min_b = np.nonzero(new_row_winnings_b == min_b)[0][0]
        row = self.np_matrix[:, pos_min_b]
        new_row_winnings_a = self.winnings_a[iterr] + row
        # print(pos_max_a+1, new_row_winnings_b, pos_min_b+1, new_row_winnings_a)
        # print(self.winnings_b[iterr], self.winnings_a[iterr], max_a, pos_max_a, self.np_matrix[pos_max_a, :], new_row_winnings_b)
        return new_row_winnings_b, pos_max_a, new_row_winnings_a, pos_min_b
    
    def pprint_payment_matrix(self):
        self.winnings_ab = np.column_stack((self.winnings_b, self.winnings_a))
        self.result_matrix = np.column_stack((self.winnings_ab, self.static_matrix))
        
        pre_line = 'K\tBy1\tBy2\tBy3\tAx1\tAx2\tAx3\tI\tJ\tVmin\tVmax\tVsr'
        print(pre_line)
        for i, row in enumerate(self.result_matrix):
            line = '{}\t{}'.format(
                i + 1, '\t'.join(
                    [
                        '{0:.2f}'.format(elem) if j not in (6, 7) else '{}'.format(int(elem)) for j, elem in
                        enumerate(row)
                    ]
                )
            )
            print(line)
        print('Цена игры, W = {0:.2f}'.format(self.static_matrix[len(self.static_matrix[:, 4]) - 1, 4]))
        br_rob_strategy1 = [round(len(np.where(self.static_matrix[:, 0] == i)[0]) / len(self.result_matrix), 2) for i in
                            range(1, 4, 1)]
        br_rob_strategy2 = [round(len(np.where(self.static_matrix[:, 1] == i)[0]) / len(self.result_matrix), 2) for i in
                            range(1, 4, 1)]
        print('Стратегия игрока 1: {}'.format(br_rob_strategy1))
        print('Стратегия игрока 2: {}'.format(br_rob_strategy2))
    
    # print(br_rob_strategy1)
    # maybe fix
    # strategy_1 = [np.nonzero(len(list(self.static_matrix[:, 0] == i)[0])) for i in range(1, 4) ]
    # print(strategy_1)
    # print('Стратегия игрока 1: '.format(self.static_matrix[:, 2]))
    # print('Стратегия игрока 2: '.format())
    
    def get_price_game(self):
        def check_point(a, b):  # проверка наличия седловой точки
            if a == b:
                return True
            else:
                return False
        
        self.top_game_price = self.np_matrix.max(axis=0).min()  # нахождение верхней цены игры
        self.lower_game_price = self.np_matrix.min(axis=1).max()  # -//- нижней
        
        if check_point(self.top_game_price, self.lower_game_price):
            print('Есть седловая точка, {} == {}'.format(self.top_game_price, self.lower_game_price))
        else:
            print('Нет седловой точки, {} <> {}'.format(self.top_game_price, self.lower_game_price))
        
        print('Игроки\tB1\tB2\tB3    min(Ai)')
        for i, row in enumerate(self.np_matrix):
            line = 'A{}\t{}\t{}'.format(i + 1, '\t'.join(['{}'.format(elem) for elem in row]),
                                        self.np_matrix.min(axis=1)[i])
            print(line)
        print('max(Bi)\t{}   {}\{}'.format(
            '\t'.join(['{}'.format(elem) for elem in self.np_matrix.max(axis=0)]),
            self.np_matrix.max(axis=0).min(),
            self.np_matrix.min(axis=1).max())
        )
        print('=============================')
        
        posA = np.nonzero(self.np_matrix.max(axis=0) == self.top_game_price)[0][0]  # чистая стратегия игрока B на линии
        posB = np.nonzero(self.np_matrix.min(axis=1) == self.lower_game_price)[0][
            0]  # чистая стратегия игрока A на линии
        # print(posA, posB)
        return posB, posA
    # print(price_game_b, price_game_a, self.top_game_price, self.lower_game_price)

# with printoptions(threshold=4, edgeitems=20, linewidth=80, suppress=True):
# print(x)

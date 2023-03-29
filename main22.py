import time
import numpy as np
import math
import ast
import functools
import itertools
import copy

from numpy import ndarray

NUMBER_OF_CALLS = 0
LOG_FILE = open('logfile.txt', 'w')
DATA_CACHE = dict()
USED = []


def advent1_1():
    file = open('input1.txt')
    arr = []
    cal_sum = 0
    for line in file:
        if line in ['\n', '\r\n']:
            arr.append(cal_sum)
            cal_sum = 0
        else:
            cal_sum += int(line)

    arr.sort()
    l = len(arr)
    print('top cal: ', arr[l - 1])
    print('top3 cals: ', arr[l - 1] + arr[l - 2] + arr[l - 3])


def advent2_1():
    file = open('input2.txt')
    score = 0
    for line in file:
        game = line.split()
        if game[0] == 'A' and game[1] == 'Z':
            score += 0 + 3
        elif game[0] == 'A' and game[1] == 'Y':
            score += 6 + 2
        elif game[0] == 'A' and game[1] == 'X':
            score += 3 + 1
        elif game[0] == 'B' and game[1] == 'Z':
            score += 6 + 3
        elif game[0] == 'B' and game[1] == 'Y':
            score += 3 + 2
        elif game[0] == 'B' and game[1] == 'X':
            score += 0 + 1
        elif game[0] == 'C' and game[1] == 'Z':
            score += 3 + 3
        elif game[0] == 'C' and game[1] == 'Y':
            score += 0 + 2
        elif game[0] == 'C' and game[1] == 'X':
            score += 6 + 1

    print('Score: ', score)


def advent2_2():
    file = open('input2.txt')
    score = 0
    for line in file:
        game = line.split()
        if game[0] == 'A' and game[1] == 'Z':  # choose B
            score += 6 + 2
        elif game[0] == 'A' and game[1] == 'Y':  # choose A
            score += 3 + 1
        elif game[0] == 'A' and game[1] == 'X':  # choose C
            score += 0 + 3
        elif game[0] == 'B' and game[1] == 'Z':  # choose C
            score += 6 + 3
        elif game[0] == 'B' and game[1] == 'Y':  # choose B
            score += 3 + 2
        elif game[0] == 'B' and game[1] == 'X':  # choose A
            score += 0 + 1
        elif game[0] == 'C' and game[1] == 'Z':  # choose A
            score += 6 + 1
        elif game[0] == 'C' and game[1] == 'Y':  # choose C
            score += 3 + 3
        elif game[0] == 'C' and game[1] == 'X':  # choose B
            score += 0 + 2

    print('Score: ', score)


def item_prio(item):
    if item.islower():
        return ord(item) - 96
    else:
        return ord(item) - 64 + 26


def priority(items):
    h_n_items = int(len(items)/2)
    comp_1 = set(items[:h_n_items])
    comp_2 = set(items[h_n_items:])
    common = comp_1.intersection(comp_2)
    if not len(common) == 1:
        print('common: ', common)
        raise Exception("wrong intersect")

    ret_val = 0
    for e in common:
        ret_val += item_prio(e)

    return ret_val


def advent3_1():
    file = open('input3.txt')
    psum = 0
    line_no = 0
    for line in file:
        psum += priority(line.strip('\n'))
        line_no += 1

    print('Sum of prioritys: ', psum)


def badge_value(triple):
    r_1 = set(triple[0])
    r_2 = set(triple[1])
    r_3 = set(triple[2])
    c_1 = r_1.intersection(r_2)
    c_2 = c_1.intersection(r_3)
    if not len(c_2) == 1:
        raise Exception("Wrong no badges")

    ret_val = 0
    for e in c_2:
        ret_val += item_prio(e)

    return ret_val


def advent3_2():
    file = open('input3.txt')
    psum = 0
    line_no = 1
    triple = []
    for line in file:
        triple.append(line.strip('\n'))
        if len(triple) == 3:
            psum += badge_value(triple)
            triple = []
        line_no += 1

    print('Sum of badges: ', psum)


def advent4_1():
    file = open('input4.txt')
    no_complete_overlaps = 0
    for line in file:
        assignments = line.strip('\n').split(',')

        b_1 = assignments[0].split('-')
        b_2 = assignments[1].split('-')

        if (int(b_1[0]) >= int(b_2[0])) and (int(b_1[1]) <= int(b_2[1])):
            no_complete_overlaps += 1
        elif (int(b_2[0]) >= int(b_1[0])) and (int(b_2[1]) <= int(b_1[1])):
            no_complete_overlaps += 1

    print('Nof complete overlaps: ', no_complete_overlaps)


def advent4_2():
    file = open('input4.txt')
    no_overlaps = 0
    for line in file:
        assignments = line.strip('\n').split(',')

        b_1 = assignments[0].split('-')
        b_2 = assignments[1].split('-')

        if (int(b_1[0]) >= int(b_2[0])) and (int(b_1[1]) <= int(b_2[1])):
            no_overlaps += 1
        elif (int(b_2[0]) >= int(b_1[0])) and (int(b_2[1]) <= int(b_1[1])):
            no_overlaps += 1
        elif (int(b_1[0]) >= int(b_2[0])) and (int(b_1[0]) <= int(b_2[1])):
            no_overlaps += 1
        elif (int(b_2[0]) >= int(b_1[0])) and (int(b_2[0]) <= int(b_1[1])):
            no_overlaps += 1

    print('Nof overlaps: ', no_overlaps)


def advent5_1(inorder=False):
    file = open('input5.txt')

    commands = []
    heap_lines = []
    for line in file:
        if line[0] == 'm':  # command
            commands.append(line.strip('\n'))
        elif '[' in line:
            heap_lines.append(line.strip('\n'))

    nof_heaps = (len(heap_lines[0]) + 1) / 4

    heaps = []
    for i in range(0, int(nof_heaps)):
        heaps.append([])

    for h in heap_lines:
        heap_no = 0
        for i in range(1, len(h), 4):
            if h[i] != ' ':
                heaps[heap_no].append(h[i])
            heap_no += 1

    for h in heaps:
        h.reverse()

    nof_c = 0
    for c in commands:

        nos_s = c.strip('move ').replace('from ', '').replace('to ', '').split(' ')
        nos = []
        for n in nos_s:
            nos.append(int(n))
        nos[1] -= 1
        nos[2] -= 1
        t_store = []
        if not inorder:
            for i in range(0, nos[0]):
                heaps[nos[2]].append(heaps[nos[1]][-1:][0])
                heaps[nos[1]].pop()
        else:
            t_store = []
            for i in range(0, nos[0]):
                t_store.append(heaps[nos[1]][-1:][0])
                heaps[nos[1]].pop()
            for i in range(0, nos[0]):
                heaps[nos[2]].append(t_store[-1:][0])
                t_store.pop()
        nof_c += 1

    for h in heaps:
        print(h.pop())


def four_different(four_letters):
    if four_letters[0] == four_letters[1] or four_letters[0] == four_letters[2] or four_letters[0] == four_letters[3]:
        return False
    if four_letters[1] == four_letters[2] or four_letters[1] == four_letters[3]:
        return False
    if four_letters[2] == four_letters[3]:
        return False

    return True


def fourteen_different(fourteen_letters):
    letter_set = set(fourteen_letters)
    if len(letter_set) == 14:
        return True
    return False


def advent6_1():
    file = open('input6.txt')

    for line in file:
        message = line.strip('\n')
        letter_count = 0
        four_found = False
        for l in message:
            if not four_found:
                if four_different(message[letter_count:letter_count + 4]):
                    print('Found 4 different: ', message[letter_count:letter_count + 4])
                    print(letter_count + 4)
                    four_found = True

            letter_count += 1
            if fourteen_different(message[letter_count:letter_count + 14]):
                print('Found 14 different: ', message[letter_count:letter_count + 14])
                print(letter_count + 14)
                break


class FileNode:
    def __init__(self, name, size):
        self.name = name
        self.size = size


class FolderNode:
    def __init__(self, name):
        self.name = name
        self.files = []
        self.folders = []
        self.parent = None
        self.depth = 0

    def print_large(self):
        if self.size() > 7870454:
            print(self.size())
        for f in self.folders:
            #print(2*self.depth*' ', '-', f.name, '(dir)')
            f.print_large()
        #for f in self.files:
            #print(2*self.depth*' ', '-', f.name, '(file, size=', f.size, ')')

    def print_small(self):
        if self.size() <= 100000:
            print(self.size())
        for f in self.folders:
            f.print_small()

    def print(self):
        for f in self.folders:
            print(2*self.depth*' ', '-', f.name, '(dir)')
            f.print()
        for f in self.files:
            print(2*self.depth*' ', '-', f.name, '(file, size=', f.size, ')')

    def size(self):
        sum = 0
        for f in self.files:
            sum += int(f.size)
        for f in self.folders:
            sum += f.size()
        return sum


def advent7_1(print_small=True):
    file = open('input7.txt')

    root_folder = FolderNode('/')
    curr_folder = root_folder
    file.readline()  # skip first line, it is cd /
    for line in file:
        if line[0] == '$':
            cmd = line.strip('\n').replace('$ ', '')
            if cmd[0:2] == 'cd':
                dest = cmd[3:]
                if dest != '..':
                    curr_folder.folders.append(FolderNode(dest))
                    prev_folder = curr_folder
                    curr_folder = prev_folder.folders[-1]
                    curr_folder.parent = prev_folder
                    curr_folder.depth = prev_folder.depth + 1
                else:
                    curr_folder = curr_folder.parent
        elif line[0].isdigit():
            file_size, name = line.strip('\n').split(' ')
            curr_folder.files.append(FileNode(name, file_size))

    if print_small:
        root_folder.print_small()
    else:
        root_folder.print_large()


def calc_scenic_score(i, j, dim, A):
    down = 0
    for r in range(i - 1, -1, -1):
        if A[r, j] < A[i, j]:
            down += 1
        else:
            down += 1
            break
    up = 0
    for r in range(i + 1, dim):
        if A[r, j] < A[i, j]:
            up += 1
        else:
            up += 1
            break
    left = 0
    for c in range(j - 1, -1, -1):
        if A[i, c] < A[i, j]:
            left += 1
        else:
            left += 1
            break
    right = 0
    for c in range(j + 1, dim):
        if A[i, c] < A[i, j]:
            right += 1
        else:
            right += 1
            break
    score = up*down*right*left
    return score


def advent8_1(dim=5):
    file = open('input8.txt')

    forest = np.zeros((dim, dim))
    row_no = 0
    for line in file:
        row = line.strip('\n')
        col_no = 0
        for e in row:
            forest[row_no, col_no] = e
            col_no += 1
        row_no += 1

    nof_visible = 2*dim + 2*(dim - 2)
    for i in range(1, dim - 1):
        for j in range(1, dim - 1):
            if (forest[i, j] > forest[:i, j]).all() or (forest[i, j] > forest[i+1:, j]).all() or (forest[i, j] > forest[i, :j]).all() or (forest[i, j] > forest[i, j+1:]).all():
                nof_visible += 1

    print('No of visible trees:', nof_visible)

    max_scenic_score = 0
    for i in range(1, dim - 1):
        for j in range(1, dim - 1):
            scenic_score = calc_scenic_score(i, j, dim, forest)
            max_scenic_score = max(scenic_score, max_scenic_score)

    print('Scenic score: ', max_scenic_score)


def advent9_1():
    file = open('input9.txt')
    h_pos = [0, 0]
    t_pos = [0, 0]

    incr = [0, 0]
    t_positions = set()
    t_positions.add((t_pos[0], t_pos[1]))
    for line in file:
        move = line.strip('\n').split(' ')
        if move[0] == 'R':
            incr = [1, 0]
        elif move[0] == 'L':
            incr = [-1, 0]
        elif move[0] == 'U':
            incr = [0, 1]
        elif move[0] == 'D':
            incr = [0, -1]

        for m in range(0, int(move[1])):
            h_pos[0] += incr[0]
            h_pos[1] += incr[1]
            if t_pos[0] == h_pos[0] and abs(t_pos[1] - h_pos[1]) > 1:
                t_pos[1] -= (t_pos[1] - h_pos[1])/abs(t_pos[1] - h_pos[1])
            elif t_pos[1] == h_pos[1] and abs(t_pos[0] - h_pos[0]) > 1:
                t_pos[0] -= (t_pos[0] - h_pos[0])/abs(t_pos[0] - h_pos[0])
            elif abs(t_pos[0] - h_pos[0]) > 1 or abs(t_pos[1] - h_pos[1]) > 1:
                t_pos[1] -= (t_pos[1] - h_pos[1]) / abs(t_pos[1] - h_pos[1])
                t_pos[0] -= (t_pos[0] - h_pos[0]) / abs(t_pos[0] - h_pos[0])
            t_positions.add((t_pos[0], t_pos[1]))

    print('Head pos: ', h_pos)
    print('Tail pos: ', t_pos)
    print('Visited: ', len(t_positions))


def advent9_2():
    file = open('input9.txt')
    h_pos = [0, 0]
    rope = []
    for i in range(0, 10):
        rope.append(h_pos.copy())

    incr = [0, 0]
    t_positions = set()
    t_positions.add((rope[9][0], rope[9][1]))
    for line in file:
        move = line.strip('\n').split(' ')
        if move[0] == 'R':
            incr = [1, 0]
        elif move[0] == 'L':
            incr = [-1, 0]
        elif move[0] == 'U':
            incr = [0, 1]
        elif move[0] == 'D':
            incr = [0, -1]

        for m in range(0, int(move[1])):
            rope[0][0] += incr[0]
            rope[0][1] += incr[1]
            for i in range(1, 10):
                if rope[i][0] == rope[i-1][0] and abs(rope[i][1] - rope[i-1][1]) > 1:
                    rope[i][1] -= (rope[i][1] - rope[i-1][1])/abs(rope[i][1] - rope[i-1][1])
                elif rope[i][1] == rope[i-1][1] and abs(rope[i][0] - rope[i-1][0]) > 1:
                    rope[i][0] -= (rope[i][0] - rope[i-1][0])/abs(rope[i][0] - rope[i-1][0])
                elif abs(rope[i][0] - rope[i-1][0]) > 1 or abs(rope[i][1] - rope[i-1][1]) > 1:
                    rope[i][1] -= (rope[i][1] - rope[i-1][1]) / abs(rope[i][1] - rope[i-1][1])
                    rope[i][0] -= (rope[i][0] - rope[i-1][0]) / abs(rope[i][0] - rope[i-1][0])
                t_positions.add((rope[9][0], rope[9][1]))

    print('Visited: ', len(t_positions))


def advent10_1():
    file = open('input10.txt')

    screen = np.full((6, 40), '.', dtype=str)
    x = 1
    cycle = 0
    sign_str = 0
    sum_sign_strs = 0
    cycle_chkpts = [20, 60, 100, 140, 180, 220]
    cmd = 'noop'
    v = 0
    for line in file:
        instr = line.strip('\n')
        if instr[0] == 'n':  # noop
            cmd = 'noop'
            cycle += 1
            if cycle in cycle_chkpts:
                sign_str = cycle*x
                print('Cycle(noop): ', cycle, ', x =', x, ', Str: ', sign_str)
                sum_sign_strs += sign_str
        else:  # addx v
            cmd = 'addx'
            v = int(line.split(' ')[1])
            cycle += 2
            x += v
            if cycle in cycle_chkpts:
                sign_str = cycle*(x - v)
                print('Cycle(addx): ', cycle, ', x =', x - v, ', Str: ', sign_str)
                sum_sign_strs += sign_str
            elif (cycle - 1) in cycle_chkpts:
                sign_str = (cycle - 1)*(x - v)
                print('Cycle(addx): ', cycle, ', x =', x - v, ', Str: ', sign_str)
                sum_sign_strs += sign_str
        row = (cycle - 1) // 40
        col = (cycle - 1) % 40
        crt_col = 0
        if cmd == 'noop':
            crt_col = x
            if col in [crt_col - 1, crt_col, crt_col + 1]:
                screen[row, col] = '#'
        else:
            crt_col = x - v
            print(crt_col, col)
            if col in [crt_col - 1, crt_col, crt_col + 1]:
                screen[row, col] = '#'
            row = (cycle - 2) // 40
            col = (cycle - 2) % 40
            print(crt_col, col)
            if col in [crt_col - 1, crt_col, crt_col + 1]:
                screen[row, col] = '#'

    print('Sum of signal strs: ', sum_sign_strs)
    print(screen)


class Monkey:
    def __init__(self, start_items, divisor, true_monkey, false_monkey, op, element, lcm=0):
        self.items = start_items
        self.divisor = divisor
        self.true_monkey = true_monkey
        self.false_monkey = false_monkey
        self.op = op
        self.element = element
        self.handle_times = 0
        self.cm = lcm

    def operation(self, item):
        if self.element == 'old':
            element = item
        else:
            element = int(self.element)
        if self.op == '+':
            return item + element
        else:
            return item * element

    def handle_item(self, item, divideby):
        self.handle_times += 1
        item = self.operation(item)
        if not divideby == 1:
            item /= divideby
            item = math.floor(item)
        return item, item % self.divisor == 0


def digit_sum(n):
    dig_sum = 0
    for digit in str(n):
        dig_sum += int(digit)
    return dig_sum


def is_divisible_by_3(item):
    dig_sum = digit_sum(item)
    return dig_sum % 3 == 0


def is_divisible_by_5(item):
    return str(item)[-1] == 0 or str(item)[-1] == 5


def is_divisible_by_7(item):
    if item == 0 or item == 7 or item == -7:
        return True
    if item < 10:
        return False
    return is_divisible_by_7(abs(item // 10 - 2 * (item - item // 10 * 10)))


def is_divisible_by_11(item):
    st = str(item)
    n = len(st)
    odd_dig_sum = 0
    even_dig_sum = 0
    for i in range(0, n):
        if i % 2 == 0:
            odd_dig_sum = odd_dig_sum + (int(st[i]))
        else:
            even_dig_sum = even_dig_sum + (int(st[i]))
    return (odd_dig_sum - even_dig_sum) % 11 == 0


def is_divisible_by_13(item):
    while abs(item) // 100:
        d = item % 10
        item //= 10
        item += d * 4
    return item % 13 == 0


def is_divisible_by_17(item):
    while abs(item) // 100:
        d = item % 10
        item //= 10
        item -= d * 5
    return item % 17 == 0


def is_divisible_by_19(item):
    while item // 100:
        d = item % 10
        item //= 10
        item += d * 2
    return item % 19 == 0


def is_divisible_by_23(item):
    while item // 100:
        d = item % 10
        item //= 10
        item += d * 7
    return item % 23 == 0


class Monkey2(Monkey):
    def __init__(self, start_items, divisor, true_monkey, false_monkey, op, element, lcm=1):
        Monkey.__init__(self, start_items, divisor, true_monkey, false_monkey, op, element)

    def handle_item(self, item, divideby):
        self.handle_times += 1
        item = self.operation(item)
        if not divideby == 1:
            item /= divideby
            item = math.floor(item)
        item %= self.cm
        return item, item % self.divisor == 0

    def is_dividable(self, item):
        if self.divisor == 2:
            return not (item & 1)
        elif self.divisor == 3:
            return is_divisible_by_3(item)
        elif self.divisor == 5:
            return is_divisible_by_5(item)
        elif self.divisor == 7:
            return is_divisible_by_7(item)
        elif self.divisor == 11:
            return is_divisible_by_11(item)
        elif self.divisor == 13:
            return is_divisible_by_13(item)
        elif self.divisor == 17:
            return is_divisible_by_17(item)
        elif self.divisor == 19:
            return is_divisible_by_19(item)
        elif self.divisor == 23:
            return is_divisible_by_23(item)
        else:
            raise "this number is not handled!"


def advent11_1(divideby, no_rounds):
    file = open('input11.txt')

    monkeys = []
    lcm = 1
    while True:
        monkey_row = file.readline().strip('\n')
        if not monkey_row:
            break
        item_row = file.readline().strip('\n')
        operation_row = file.readline().strip('\n')
        divisor_row = file.readline().strip('\n')
        true_row = file.readline().strip('\n')
        false_row = file.readline().strip('\n')
        empty_row = file.readline().strip('\n')
        items = item_row.split(':')[1].split(', ')
        items = [int(i) for i in items]
        op = operation_row.split('= old')[1].split(' ')[1]
        element = operation_row.split('= old')[1].split(' ')[2]
        divisor = int(divisor_row.split(' ').pop())
        lcm *= divisor
        true_monkey = int(true_row.split(' ').pop())
        false_monkey = int(false_row.split(' ').pop())
        monkeys.append(Monkey2(items, divisor, true_monkey, false_monkey, op, element))

    for monkey in monkeys:
        monkey.cm = lcm

    prev_time = 0
    for rounds in range(0, no_rounds):
        #print('Round: ', rounds)
        for monkey in monkeys:
            for item in monkey.items:
                item, res = monkey.handle_item(item, divideby)
                if res:
                    monkeys[monkey.true_monkey].items.append(item)
                else:
                    monkeys[monkey.false_monkey].items.append(item)
            monkey.items = []

        #print(monkeys[0].handle_times, ',', end='')
        #print(len(str(monkeys[1].items[0])))
        #prev_time = monkeys[0].handle_times

    for monkey in monkeys:
        print('==================')
        print(monkey.handle_times)


class Node:
    def __init__(self, r=0, c=0, area=np.full((1, 1), 'a', dtype=str)):
        self.neighbors = []
        area_size = area.shape
        max_r = area_size[0]
        max_c = area_size[1]
        me_val = ord(area[r, c])
        if r > 0 and ord(area[r - 1, c]) - me_val <= 1:
            self.neighbors.append((r - 1, c))
        if c > 0 and ord(area[r, c - 1]) - me_val <= 1:
            self.neighbors.append((r, c - 1))
        if r + 1 < max_r and ord(area[r + 1, c]) - me_val <= 1:
            self.neighbors.append((r + 1, c))
        if c + 1 < max_c and ord(area[r, c + 1]) - me_val <= 1:
            self.neighbors.append((r, c + 1))


def NotReallyDijkstra(graph, start):
    dist: ndarray = np.ones([np.size(graph, 0), np.size(graph, 1)])
    visited = (dist != 1)
    dist[:, :] = 2**32
    dist[start] = 0
    visited[start] = True
    start_node = start
    value_set = dict()

    while not visited.all():
        neighbors = graph[start_node].neighbors
        for node in neighbors:
            if not visited[node]:
                dist[node] = min(dist[node], 1 + dist[start_node])
                value_set[node] = dist[node]
        if len(value_set) == 0:
            break
        temp = min(value_set.values())
        min_nodes = [key for key in value_set if value_set[key] == temp]
        start_node = min_nodes[0]
        value_set.pop(start_node)
        visited[start_node] = True

    return dist


def advent12_1():
    file = open('input12.txt')

    area = np.full((41, 179), '.', dtype=str)
    r = 0
    for line in file:
        row = line.strip('\n')
        for c in range(0, 179):
            area[r, c] = line[c]
        r += 1

    start = (0, 0)
    end = (0, 0)
    for r in range(0, area.shape[0]):
        for c in range(0, area.shape[1]):
            if area[r, c] == 'S':
                start = (r, c)
                area[r, c] = 'a'
            elif area[r, c] == 'E':
                end = (r, c)
                area[r, c] = 'z'

    g_area = np.full(area.shape, Node(), dtype=Node)
    for r in range(0, g_area.shape[0]):
        for c in range(0, g_area.shape[1]):
            g_area[r, c] = Node(r, c, area)

    dist = NotReallyDijkstra(g_area, start)
    print('Shortest dist from starting a: ', dist[end])

    min_a_dist = dist[end]
    for start_r in range(0, g_area.shape[0]):  # cheat by looking at input data, all b's in single column (1)
        start = (start_r, 0)
        dist = NotReallyDijkstra(g_area, start)
        min_a_dist = min(min_a_dist, dist[end])
    print('Shortest distance from all possible a: ', min_a_dist)


def compare_values(v_1, v_2):
    t_1 = type(v_1)
    t_2 = type(v_2)
    if t_1 == int and t_2 == int:
        if v_1 < v_2:
            return 'right'
        elif v_2 < v_1:
            return 'wrong'
        else:
            return 'equal'
    if t_1 == list and t_2 == list:
        for i in range(0, max(len(v_1), len(v_2))):
            if i == len(v_1):
                return 'right'
            if i == len(v_2):
                return 'wrong'
            result = compare_values(v_1[i], v_2[i])
            if result in ['right', 'wrong']:
                return result
    if t_1 == int and t_2 == list:
        result = compare_values([v_1], v_2)
        return result
    if t_1 == list and t_2 == int:
        result = compare_values(v_1, [v_2])
        return result


def p_less_than(v_1, v_2):
    result = compare_values(v_1, v_2)
    if result == 'right':
        return -1
    elif result == 'wrong':
        return 1
    return 0


def advent13_1():
    file = open('input13.txt')

    index = 1
    sum_indices = 0
    all = []
    while True:
        pair1 = file.readline().strip('\n')
        if not pair1:
            break
        pair2 = file.readline().strip('\n')
        empty_line = file.readline()
        p_1 = ast.literal_eval(pair1)
        p_2 = ast.literal_eval(pair2)

        comp = compare_values(p_1, p_2)
        if comp == 'right':
            sum_indices += index
        index += 1

        all.append(p_1)
        all.append(p_2)

    print('Sum of correct pair indices: ', sum_indices)

    all.append([[2]])
    all.append([[6]])
    s_all = sorted(all, key=functools.cmp_to_key(p_less_than))
    for s in range(0, len(s_all)):
        if s_all[s] == [[2]] or s_all[s] == [[6]]:
            print(s_all[s], 'index= ', s + 1)


def sand_fall(cave):
    cave_size = cave.shape
    pos = (0, 0)
    for c in range(0, cave_size[1]):
        if cave[0, c] == 'o':
            pos = (0, c)
            break
    new_pos = pos
    while True:
        pos = new_pos
        if cave[pos[0] + 1, pos[1]] == '.':
            new_pos = (pos[0] + 1, pos[1])
            cave[pos] = '.'
            if new_pos[0] >= cave_size[0] - 1 or new_pos[1] < 0 or new_pos[1] >= cave_size[1]:
                return False
            cave[new_pos] = 'o'
        elif cave[pos[0] + 1, pos[1] - 1] == '.':
            new_pos = (pos[0] + 1, pos[1] - 1)
            cave[pos] = '.'
            if new_pos[0] >= cave_size[0] - 1 or new_pos[1] < 0 or new_pos[1] >= cave_size[1]:
                return False
            cave[new_pos] = 'o'
        elif cave[pos[0] + 1, pos[1] + 1] == '.':
            new_pos = (pos[0] + 1, pos[1] + 1)
            cave[pos] = '.'
            if new_pos[0] >= cave_size[0] - 1 or new_pos[1] < 0 or new_pos[1] >= cave_size[1]:
                return False
            cave[new_pos] = 'o'
        else:
            break
    return True


def advent14_1():
    file = open('input14.txt')

    paths = []
    min_x = 0
    max_x = 0
    min_y = 2**32
    max_y = 0
    for line in file:
        path_row = line.strip('\n')
        nodes = path_row.replace(' ', '').split('->')
        for n in nodes:
            y, x = n.split(',')
            min_x = min(int(x), min_x)
            max_x = max(int(x), max_x)
            min_y = min(int(y), min_y)
            max_y = max(int(y), max_y)
        paths.append(nodes)

    cave = np.full((max_x - min_x + 1, max_y - min_y + 1), '.', dtype=str)
    sand_start = (0, 500 - min_y)
    cave[sand_start] = '+'
    for p in paths:
        for i in range(0, len(p) - 1):
            ys, xs = p[i].split(',')
            ye, xe = p[i + 1].split(',')
            xs = int(xs) - min_x
            xe = int(xe) - min_x
            ys = int(ys) - min_y
            ye = int(ye) - min_y
            if xs > xe:
                xs, xe = xe, xs
            if ys > ye:
                ys, ye = ye, ys
            cave[xs:xe+1, ys:ye+1] = '#'

    while True:
        cave[sand_start] = 'o'
        result = sand_fall(cave)
        if not result:
            break

    sum_arr = cave == 'o'
    print('Sum of sand: ', sum(sum(sum_arr)))
    cave[sand_start] = '+'
    #ofile = open('cave.txt', 'w')
    #for r in range(0, cave.shape[0]):
    #    for c in range(0, cave.shape[1]):
    #        ofile.write(cave[r, c])
    #    ofile.write('\n')

    big_cave = np.full((max_x - min_x + 1 + 2, max_y - min_y + 1 + 2*(max_x - min_x)), '.', dtype=str)
    for r in range(0, cave.shape[0]):
        for c in range(0, cave.shape[1]):
            big_cave[r, c + (max_x - min_x) - int(sand_start[1] - (max_y - min_y)/2)] = cave[r, c]
    big_cave[-1, :] = '#'

    for c in range(0, big_cave.shape[1]):
        if big_cave[0, c] == '+':
            sand_start = (0, c)
            break
    while True:
        big_cave[sand_start] = 'o'
        sand_fall(big_cave)
        if big_cave[sand_start] == 'o':
            break
    big_sum_arr = big_cave == 'o'
    print('Sum of sand (2): ', sum(sum(big_sum_arr)))

    b_o_file = open('big_cave.txt', 'w')
    for r in range(0, big_cave.shape[0]):
        for c in range(0, big_cave.shape[1]):
            b_o_file.write(big_cave[r, c])
        b_o_file.write('\n')


def intersection_with_line(center, distance, line):
    dist_to_closest = abs(center[1] - line)
    overlap = distance - dist_to_closest
    if overlap >= 0:
        left_point = center[0] - overlap
        right_point = center[0] + overlap
        return left_point, right_point
    return 1, -1


def covered_by_sensor(point, sensors_with_radius):
    for s_w_r in sensors_with_radius:
        dist = abs(point[0] - s_w_r[0][0]) + abs(point[1] - s_w_r[0][1])
        if dist <= s_w_r[1]:
            return True
    return False


def advent15_1():
    file = open('input15.txt')

    sensors_with_beacons = []
    for line in file:
        sensor_with_beacon = []
        sensor_str, beacon_str = line.strip('\n').split(': ')
        x, y = sensor_str.replace('Sensor at ', '').replace('x=', '').replace('y=', '').split(', ')
        sensor_with_beacon.append((int(x), int(y)))
        x, y = beacon_str.replace('closest beacon is at ', '').replace('x=', '').replace('y=', '').split(', ')
        sensor_with_beacon.append((int(x), int(y)))
        sensors_with_beacons.append(sensor_with_beacon)

    coverage = set()
    beacons_on_line = set()
    line = 2000000
    for s_w_b in sensors_with_beacons:
        sensor = s_w_b[0]
        beacon = s_w_b[1]
        if beacon[1] == line:
            beacons_on_line.add(beacon)
        manhattan_dist = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])
        left, right = intersection_with_line(sensor, manhattan_dist, line)
        if left <= right:
            for p in range(left, right + 1):
                coverage.add(p)

    print('Coverage: ', len(coverage) - len(beacons_on_line))

    max_coord = (4000000, 4000000)
    for y in range(0, max_coord[1] + 1):
        coverage = []
        for s_w_b in sensors_with_beacons:
            sensor = s_w_b[0]
            beacon = s_w_b[1]
            manhattan_dist = abs(sensor[0] - beacon[0]) + abs(sensor[1] - beacon[1])
            left, right = intersection_with_line(sensor, manhattan_dist, y)
            if left <= right:
                coverage.append([max(0, left), min(max_coord[0], right)])
        # union of intervals
        coverage.sort(key=lambda e: e[0])
        intervals = [coverage[0]]
        for x in coverage[1:]:
            if intervals[-1][1] < x[0]:
                intervals.append(x)
            elif intervals[-1][1] == x[0]:
                intervals[-1][1] = x[1]
            if x[1] > intervals[-1][1]:
                intervals[-1][1] = x[1]
        if len(intervals) > 1:
            # ==> not covering the whole line [0 ... 4000000] (should have checked min|max
            # values as well, but this worked here)
            print('Tuning frequency found: ', 4000000*(intervals[0][1] + 1) + y)
            break


class Valve:
    def __init__(self, flow_rate, tunnels):
        self.flow_rate = flow_rate
        self.tunnels = [t.replace(' ', '') for t in tunnels]
        self.open = False


def shortest_distance(graph, start):
    dist = dict()
    visited = dict()
    for k in graph:
        visited[k] = False
        dist[k] = 2**32
    dist[start] = 0
    visited[start] = True
    start_node = start
    value_set = dict()

    while not functools.reduce(lambda a, b: a and b, list(visited.values())):
        neighbors = graph[start_node].tunnels
        for node in neighbors:
            if not visited[node]:
                dist[node] = min(dist[node], 1 + dist[start_node])
                value_set[node] = dist[node]
        if len(value_set) == 0:
            break
        temp = min(value_set.values())
        min_nodes = [key for key in value_set if value_set[key] == temp]
        start_node = min_nodes[0]
        value_set.pop(start_node)
        visited[start_node] = True

    return dist


def calc_acc_pressure(valves, actions):
    acc_p = 0
    #print(len(actions))
    #ofile = open('actions.txt', 'a')
    #for a in actions:
    #    ofile.write(a)
    #    ofile.write(', ')
    #ofile.write('\n')
    for no in range(0, len(actions)):
        if actions[no] == 'Open':
            valve = actions[no - 1].replace('Go to ', '')
            flow_rate = valves[valve].flow_rate
            acc_p += flow_rate*(31 - no - 1)
    #print(actions, acc_p)
    return acc_p


def visit_valves(distances, rem_time, action, valves_to_visit, from_valve):
    for next_valve in valves_to_visit:
        dist = distances[from_valve][next_valve]
        for it in range(0, dist):
            action.append('Go to %s' % next_valve)
            rem_time -= 1
            if rem_time == 0:
                return
        action.append('Open')
        rem_time -= 1
        if rem_time == 0:
            return
        from_valve = next_valve
    for it in range(0, 31 - len(action)):
        action.append('all open')
    return


def explore_valve(valves, valves_with_pressure, actions, distances, rem_time, val, from_valve, elephant=False):
    e_val = 0
    if elephant:
        elephant_valves = valves_with_pressure.copy()
        e_val = explore_valve(valves, elephant_valves, [], distances, 26, 0, 'AA')
    ret_vals = [val + e_val]
    for next_valve in valves_with_pressure:
        new_rem_time = rem_time
        new_actions = actions.copy()

        dist = distances[from_valve][next_valve]
        for it in range(0, dist):
            new_actions.append('Go to %s' % next_valve)
            new_rem_time -= 1
            if new_rem_time == 0:
                break
        if new_rem_time == 0:
            ret_vals.append(val)
            continue
        new_actions.append('Open')
        new_rem_time -= 1
        if new_rem_time == 0:
            ret_vals.append(val)
            continue
        new_val = val + valves[next_valve].flow_rate * new_rem_time
        new_valves_with_pressure = valves_with_pressure.copy()
        new_valves_with_pressure.remove(next_valve)

        ret_vals.append(explore_valve(valves, new_valves_with_pressure, new_actions, distances, new_rem_time, new_val, next_valve, elephant))

    return max(ret_vals)


def advent16_1(phase='I'):
    file = open('input16.txt')

    valves = dict()
    valves_with_flow = []
    for line in file:
        valve_str, tunnels_str = line.strip('\n').split('; ')
        valve_key = valve_str[6:8]
        flow_rate = int(valve_str[23:])
        tunnels = tunnels_str[22:].split(',')
        valves[valve_key] = Valve(flow_rate, tunnels)
        if flow_rate > 0:
            valves_with_flow.append(valve_key)
    distances = dict()
    for k in valves.keys():
        distances[k] = shortest_distance(valves, k)

    if phase == 'I':
        actions = []
        print('Max released pressure: ', explore_valve(valves, valves_with_flow, actions, distances, 30, 0, 'AA'))

    elif phase == 'II':
        actions = []
        print('Max released pressure: ', explore_valve(valves, valves_with_flow, actions, distances, 26, 0, 'AA', True))

    elif phase == 'oldI':
        max_press = 0
        for no_valves in range(1, len(valves_with_flow) + 1):
            press = 0
            permuted_valves = itertools.permutations(valves_with_flow, no_valves)
            it = 0
            opt_act = []
            for p in permuted_valves:
                actions = []
                actions.append('Go to AA')
                visit_valves(distances, 30, actions, p, 'AA')
                new_press = max(press, calc_acc_pressure(valves, actions))
                if new_press > press:
                    press = new_press
                    opt_act = actions
                it += 1
                if it % 1000000 == 0:
                    print('#', end='', flush=True)
                if it % 10000000 == 0:
                    print(' ==> ', press, flush=True)

            print(press)
            print(opt_act)
            if press > max_press:
                max_press = press
            else:
                break
        print('Max acc. press: ', max_press)


def find_highest_rock(cave):
    for y in range(0, 3*2022):
        for x in range(0, 7):
            if cave[y, x] == '#':
                return 3*2022 - y
    return 0


def place_rock(rock, h, cave):
    for coord in rock:
        cave[3*2022 - 1 - h - 3 - 4 + coord[0], 2 + coord[1]] = '@'


def move_rock_sideways(wind, rock, origo, cave):
    if wind == '<':
        for coord in rock:
            if origo[1] + coord[1] - 1 < 0 or cave[origo[0] - 4 + coord[0], origo[1] + coord[1] - 1] == '#':
                return origo
        for coord in rock:
            cave[origo[0] - 4 + coord[0], origo[1] + coord[1]] = '.'
        for coord in rock:
            cave[origo[0] - 4 + coord[0], origo[1] + coord[1] - 1] = '@'
        return (origo[0], origo[1] - 1)
    elif wind == '>':
        for coord in rock:
            if origo[1] + coord[1] + 1 > 6 or cave[origo[0] - 4 + coord[0], origo[1] + coord[1] + 1] == '#':
                return origo
        for coord in rock:
            cave[origo[0] - 4 + coord[0], origo[1] + coord[1]] = '.'
        for coord in rock:
            cave[origo[0] - 4 + coord[0], origo[1] + coord[1] + 1] = '@'
        return (origo[0], origo[1] + 1)


def move_rock_down(rock, origo, cave):
    stop = False
    if origo[0] == 3*2022 - 1:
        stop = True
        return origo, stop
    for coord in rock:
        if cave[origo[0] - 4 + coord[0] + 1, origo[1] + coord[1]] == '#':
            stop = True
            return origo, stop
    for coord in rock:
        cave[origo[0] - 4 + coord[0], origo[1] + coord[1]] = '.'
    for coord in rock:
        cave[origo[0] - 4 + coord[0] + 1, origo[1] + coord[1]] = '@'
    return (origo[0] + 1, origo[1]), stop


def solidify_rock(rock, origo, cave):
    for coord in rock:
        cave[origo[0] - 4 + coord[0], origo[1] + coord[1]] = '#'


def advent17_1():
    file = open('input17.txt')

    wind = file.readline().strip('\n')
    nof_rocks = 2022
    cave = np.full((nof_rocks*3, 7), '.', dtype=str)

    rocks = []
    rocks.append([(4, 0), (4, 1), (4, 2), (4, 3)])
    rocks.append([(2, 1), (3, 0), (3, 1), (3, 2), (4, 1)])
    rocks.append([(2, 2), (3, 2), (4, 0), (4, 1), (4, 2)])
    rocks.append([(1, 0), (2, 0), (3, 0), (4, 0)])
    rocks.append([(3, 0), (3, 1), (4, 0), (4, 1)])

    wind_counter = 0
    prev_height = 0
    for r in range(0, nof_rocks):
        height = find_highest_rock(cave)
        origo = (3*nof_rocks - 1 - height - 3, 2)
        rock = rocks[r % 5]
        place_rock(rock, height, cave)
        if r % 5 == 0:
            print(height - prev_height, ', ', end='')
            prev_height = height
        #print(cave[3*2017:3*2022, :])
        while True:
            current_wind = wind[wind_counter % len(wind)]
            wind_counter += 1
            origo = move_rock_sideways(current_wind, rock, origo, cave)
            origo, stop = move_rock_down(rock, origo, cave)
            #print(cave[3 * 2017:3 * 2022, :])
            if stop:
                solidify_rock(rock, origo, cave)
                #print(cave[3*2017:3*2022, :])
                break

    print(wind_counter)
    h = find_highest_rock(cave)
    print('Height of rocks:', h)
    print(cave[3*2022 - h:3*2022 - h + 10, :])

    # The solution for part II (nof_rocks = 1000000000000)
    # is (1000000000000 - 19*5) // (349*5) * 2738 + 141 + 1426
    # 349 is the number of 5 rock intervals before it starts to repeat (found by experiment)
    # 2738 is the height of the accumulated rocks during the 349 intervals
    # it tool 19 intervals before the first interval start
    # during those 19 intervals the height rose to 141
    # (1000000000000 - 19*5) % (349*5) is 915, or 183 5 rock intervals
    # during those another 1426 was added to the height


def m_index_ident(a, b):
    if a[0] == b[0] and a[1] == b[1] and a[2] == b[2]:
        return 0
    return 1


def outside(coord, volume):
    if coord[0] < 0 or coord[1] < 0 or coord[2] < 0:
        return True
    if coord[0] >= volume.shape[0] or coord[1] >= volume.shape[1] or coord[2] >= volume.shape[2]:
        return True
    return False


def explore_domain(volume, domains, no, start):
    coords = [start]
    real_no = no
    while True:
        new_coords = []
        for c in coords:
            for n in [[c[0] + 1, c[1], c[2]], [c[0] - 1, c[1], c[2]],
                      [c[0], c[1] + 1, c[2]], [c[0], c[1] - 1, c[2]],
                      [c[0], c[1], c[2] + 1], [c[0], c[1], c[2] - 1]]:
                if n in coords:
                    continue
                if n in new_coords:
                    continue
                if outside(n, volume):
                    real_no = -1
                    continue
                if volume[tuple(n)] == -1:
                    continue
                if domains[tuple(n)] != 0:
                    continue
                new_coords.append(n)
        #print(len(new_coords))
        if len(new_coords) == 0:
            break
        for new_c in new_coords:
            coords.append(new_c)
    for c in coords:
        domains[tuple(c)] = real_no


def advent18_1():
    file = open('input18.txt')

    cube_normals = []
    cubes = []
    for line in file:
        cube_line = line.strip('\n').split(',')
        cube = [int(s) for s in cube_line]
        cubes.append(cube)
        n_x_p = [cube[0] + 0.5, cube[1], cube[2], +1, 0, 0]
        n_x_m = [cube[0] - 0.5, cube[1], cube[2], -1, 0, 0]
        n_y_p = [cube[0], cube[1] + 0.5, cube[2], 0, +1, 0]
        n_y_m = [cube[0], cube[1] - 0.5, cube[2], 0, -1, 0]
        n_z_p = [cube[0], cube[1], cube[2] + 0.5, 0, 0, +1]
        n_z_m = [cube[0], cube[1], cube[2] - 0.5, 0, 0, -1]
        cube_normals.append(n_x_p)
        cube_normals.append(n_x_m)
        cube_normals.append(n_y_p)
        cube_normals.append(n_y_m)
        cube_normals.append(n_z_p)
        cube_normals.append(n_z_m)

    #print(cube_normals)
    #print(sorted_normals)
    nof_normals = len(cube_normals)
    nof_pairs = 0
    free_normals = cube_normals.copy()
    for i in range(0, len(cube_normals)):
        n = cube_normals[i]
        for j in range(i + 1, len(cube_normals)):
            if m_index_ident(n, cube_normals[j]) == 0:
                nof_pairs += 1
                free_normals.remove(n)
                free_normals.remove(cube_normals[j])
    print('Tot. nof normals: ', nof_normals)
    print('Exposed normals: ', nof_normals - 2*nof_pairs)

    max_i = 0
    max_j = 0
    max_k = 0
    for n in free_normals:
        max_i = max(max_i, n[0])
        max_j = max(max_j, n[1])
        max_k = max(max_k, n[2])

    volume = np.zeros((round(max_i + 2), round(max_j + 2), round(max_k + 2)), dtype=int)
    domains = volume.copy()
    for n in free_normals:
        volume[round(n[0] - 0.5*n[3]), round(n[1] - 0.5*n[4]), round(n[2] - 0.5*n[5])] = -1
        volume[round(n[0] + 0.5*n[3]), round(n[1] + 0.5*n[4]), round(n[2] + 0.5*n[5])] = 1

    domain_no = 1
    for z in range(0, volume.shape[2]):
        for y in range(0, volume.shape[1]):
            for x in range(0, volume.shape[0]):
                if volume[x, y, z] == 1 and domains[x, y, z] == 0:
                    explore_domain(volume, domains, domain_no, [x, y, z])
                    print(domain_no)
                    domain_no += 1

    total = 0
    for no in range(1, domain_no + 1):
        no_count = np.count_nonzero(domains == no)
        #print('Count of ', no, ': ', no_count)
        area = 0
        for z in range(0, volume.shape[2]):
            for y in range(0, volume.shape[1]):
                for x in range(0, volume.shape[0]):
                    if domains[x, y, z] == no:
                        if volume[x - 1, y, z] == -1:
                            area += 1
                        if volume[x + 1, y, z] == -1:
                            area += 1
                        if volume[x, y - 1, z] == -1:
                            area += 1
                        if volume[x, y + 1, z] == -1:
                            area += 1
                        if volume[x, y, z - 1] == -1:
                            area += 1
                        if volume[x, y, z + 1] == -1:
                            area += 1
        #print('Area of ', no, ': ', area)
        total += area
    print('Total area : ', len(free_normals) - total)


def simulate_remaining_time(rem_time, decisions, amounts, nof_robots, blueprint, mtrl_types, ftime, forbidden=set()):
    global NUMBER_OF_CALLS
    NUMBER_OF_CALLS += 1

    base_decision = {'ore': 0, 'clay': 0, 'obsidian': 0, 'geode': 0}
    new_decisions = [base_decision]
    possible_decisions = base_decision.copy()
    for robot in mtrl_types:
        if blueprint[robot]['ore'] <= amounts['ore'] and \
                blueprint[robot]['clay'] <= amounts['clay'] and \
                blueprint[robot]['obsidian'] <= amounts['obsidian'] \
                and (robot not in forbidden or rem_time > ftime):
            new_decision = base_decision.copy()
            new_decision[robot] = 1
            new_decisions.append(new_decision)
            possible_decisions[robot] = 1

    if possible_decisions['geode'] == 1 and rem_time < ftime:  # alway prioritize
        new_decision = [{'ore': 0, 'clay': 0, 'obsidian': 0, 'geode': 1}]
    for m in mtrl_types:
        nof_robots[m] += decisions[m]
        amounts[m] += nof_robots[m]

    geode_harvests = [amounts['geode']]

    new_rem_time = rem_time - 1
    if new_rem_time > 0:
        new_amounts = amounts.copy()
        new_nof_robots = nof_robots.copy()
        saved_nof_geodes = new_amounts['geode']
        saved_nof_geode_robots = new_nof_robots['geode']
        saved_rem_time = new_rem_time

        key = str(new_rem_time) + ',' + str(new_amounts['ore']) + ',' + str(new_amounts['clay']) \
              + ',' + str(new_amounts['obsidian']) + ',' + str(new_nof_robots['ore']) \
              + ',' + str(new_nof_robots['clay']) + ',' + str(new_nof_robots['obsidian'])

        if key in DATA_CACHE.keys():
            num_geodes = DATA_CACHE[key] + saved_nof_geodes + saved_nof_geode_robots * saved_rem_time
            return num_geodes

        for d in new_decisions:
            new_amounts = amounts.copy()
            new_forbidden = set()
            for r in d.keys():
                if d[r] == 0 and possible_decisions[r] == 1 and r not in forbidden:
                    new_forbidden.add(r)
                if new_amounts['ore'] > 40 or new_rem_time <= ftime:
                    new_forbidden.add('ore')
                if new_amounts['clay'] > 60 or new_rem_time <= ftime:
                    new_forbidden.add('clay')
                if new_rem_time <= 1:
                    new_forbidden.add('obsidian')
                new_amounts['ore'] -= blueprint[r]['ore']*d[r]
                new_amounts['clay'] -= blueprint[r]['clay']*d[r]
                new_amounts['obsidian'] -= blueprint[r]['obsidian']*d[r]
            new_nof_robots = nof_robots.copy()
            num_geodes = simulate_remaining_time(new_rem_time, d, new_amounts, new_nof_robots,
                                                 blueprint, mtrl_types, ftime, new_forbidden)
            geode_harvests.append(num_geodes)

        increase = max(geode_harvests) - saved_nof_geodes - saved_nof_geode_robots * saved_rem_time
        DATA_CACHE[key] = increase
    return max(geode_harvests)


def advent19_1(phase='I'):
    file = open('input19.txt')

    blueprints = []
    for line in file:
        blueprint_line = line.strip('\n').split(':')[1].split('.')
        #print(blueprint)
        blueprint = dict()
        blueprint['ore'] = {"ore": int(blueprint_line[0].split(' ')[-2]), "clay": 0, "obsidian": 0}
        blueprint['clay'] = {"ore": int(blueprint_line[1].split(' ')[-2]), "clay": 0, "obsidian": 0}
        blueprint['obsidian'] = {"ore": int(blueprint_line[2].split(' ')[-5]), "clay": int(blueprint_line[2].split(' ')[-2]), "obsidian": 0}
        blueprint['geode'] = {"ore": int(blueprint_line[3].split(' ')[-5]), "clay": 0, "obsidian": int(blueprint_line[3].split(' ')[-2])}
        blueprints.append(blueprint)

    blueprint_no = 1
    quality_levels = []
    for blueprint in blueprints:
        DATA_CACHE.clear()
        rem_time = 24
        ftime = 2
        if phase == 'II':
            rem_time = 32
            ftime = 10
        robot_decisions = {'ore': 0, 'clay': 0, 'obsidian': 0, 'geode': 0}
        amounts = {"ore": 0, "clay": 0, "obsidian": 0, "geode": 0}
        nof_robots = {"ore": 1, "clay": 0, "obsidian": 0, "geode": 0}
        robot_types = ["ore", "clay", "obsidian", "geode"]
        print(blueprint)
        quality_levels.append((simulate_remaining_time(rem_time, robot_decisions,
                                                       amounts, nof_robots, blueprint,
                                                       robot_types, ftime),
                               blueprint_no))
        print(quality_levels[-1], NUMBER_OF_CALLS, len(DATA_CACHE))
        blueprint_no += 1
        if blueprint_no == 4 and phase == 'II':
            break
    if phase == 'I':
        score = 0
        for q in quality_levels:
            score += q[0]*q[1]
        print('Acc. quality levels: ', score)
    elif phase == 'II':
        prod = 1
        for q in quality_levels:
            prod *= q[0]
        print('Muliplied Q\'s: ', prod)


def pos(x):
    if x >= 0:
        return 1
    else:
        return 0

class NumberNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        if left:
            self.left = left
        else:
            self.left = self
        if right:
            self.right = right
        else:
            self.right = self


def print_circle(c):
    cycle = len(c)
    n = c[0]
    cnt = 0
    while cnt < cycle:
        print(n.value, ', ', end='')
        n = n.right
        cnt += 1
    print()


def mix(node, cycle):
    val = node.value % (cycle - 1)
    #print(val)
    if val < 0:
        node.left.right = node.right
        node.right.left = node.left
        for m in range(0, abs(val)):
            node.right = node.left
            node.left = node.right.left
        node.right.left = node
        node.left.right = node
    if val > 0:
        node.left.right = node.right
        node.right.left = node.left
        for m in range(0, val):
            node.left = node.right
            node.right = node.left.right
        node.right.left = node
        node.left.right = node


def advent20_1(multiplier=1, repeat=1):
    file = open('input20.txt')

    nodes = []
    for line in file:
        number = int(line.strip('\n'))*multiplier
        nodes.append(NumberNode(number))

    cycle = len(nodes)
    for i in range(0, cycle):
        nodes[i].left = nodes[i-1]
        nodes[i].right = nodes[(i+1) % cycle]

    print_circle(nodes)
    print('----------------------')

    for r in range(0, repeat):
        for mover in nodes:
            mix(mover, cycle)
            #print_circle(nodes)

    z_index = 0
    nums = []
    while True:
        if nodes[z_index].value == 0:
            break
        z_index += 1

    n = nodes[z_index]

    for iter in [1, 2, 3]:
        for cnt in range(0, 1000):
            n = n.right
        print(n.value)
        nums.append(n.value)
    print('Sum: ', sum(nums))


class YellingMonkey:
    def __init__(self, name, value, operation=None, left_monkey=None, right_monkey=None):
        self.name = name
        self.value = value
        self.operation = operation
        self.left_monkey = left_monkey
        self.right_monkey = right_monkey


def monkey_iteration(monkeys):
    for name in monkeys.keys():
        m = monkeys[name]
        if m.value:
            continue
        if monkeys[m.left_monkey].value and monkeys[m.right_monkey].value:
            #print(monkeys[m.left_monkey].value, m.operation, monkeys[m.right_monkey].value)
            m.value = str(eval(monkeys[m.left_monkey].value + m.operation + monkeys[m.right_monkey].value))
    if monkeys['root'].value:
        #print(eval(monkeys['lsbv'].value + '-' + monkeys['bsgz'].value))
        return True
    else:
        return False


def advent21_1(phase='I'):
    file = open('input21.txt')

    monkeys = dict()
    for line in file:
        monkey, yell = line.strip('\n').split(': ')
        expression = yell.split(' ')
        #print(expression)
        if len(expression) == 1:
            value = expression[0]
            monkeys[monkey] = YellingMonkey(monkey, value)
        elif len(expression) == 3:
            left_monkey = expression[0]
            operation = expression[1]
            right_monkey = expression[2]
            monkeys[monkey] = YellingMonkey(monkey, None, operation, left_monkey, right_monkey)

    if phase == 'I':
        solved = False
        while not solved:
            solved = monkey_iteration(monkeys)

        print('Root\'s value: ', monkeys['root'].value)
    elif phase == 'II':
        monkeys['root'].operation = '=='

        for guess in [3848301405790]: # this number was found by succesive interval tests (printing the difference of 'root')
            new_monkeys = copy.deepcopy(monkeys)
            new_monkeys['humn'].value = str(guess)
            solved = False
            while not solved:
                solved = monkey_iteration(new_monkeys)
            #print(guess, new_monkeys['root'].value, ', ', end='')
            if new_monkeys['root'].value == 'True':
                print('Yell: ', guess)
                break


def next_move(path, start_index):
    index = start_index
    steps = 0
    while True:
        tok = path[index]
        if tok in ['L', 'R']:
            steps = int(path[start_index:index])
            return tok, steps, index + 1
        index += 1
        if index == len(path):
            steps = int(path[start_index:])
            return None, steps, index


def search_row_from_left(board, r):
    c = 0
    while True:
        if board[r, c] != ' ':
            return c
        c += 1


def search_row_from_right(board, r):
    c = board.shape[1] - 1
    while True:
        if board[r, c] != ' ':
            return c
        c -= 1


def search_column_from_bottom(board, c):
    r = board.shape[0] - 1
    while True:
        if board[r, c] != ' ':
            return r
        r -= 1


def search_column_from_top(board, c):
    r = 0
    while True:
        if board[r, c] != ' ':
            return r
        r += 1

def make_move(board, coord, steps):
    height, width = board.shape
    r, c = coord[0], coord[1]
    passable = ['.', '>', '<', '^', 'v']
    if board[coord] == '>':
        for s in range(0, steps):
            if (c + 1) < width and board[r, c + 1] in passable:
                c += 1
                board[r, c] = '>'
            elif (c + 1) == width or board[r, c + 1] == ' ':
                nc = search_row_from_left(board, r)
                if board[r, nc] in passable:
                    board[r, nc] = '>'
                    c = nc
                else:
                    break
            else:
                break
    elif board[coord] == '<':
        for s in range(0, steps):
            if c > 0 and board[r, c - 1] in passable:
                c -= 1
                board[r, c] = '<'
            elif c == 0 or board[r, c - 1] == ' ':
                nc = search_row_from_right(board, r)
                if board[r, nc] in passable:
                    board[r, nc] = '<'
                    c = nc
                else:
                    break
            else:
                break
    elif board[coord] == '^':
        for s in range(0, steps):
            if r > 0 and board[r - 1, c] in passable:
                r -= 1
                board[r, c] = '^'
            elif r == 0 or board[r - 1, c] == ' ':
                nr = search_column_from_bottom(board, c)
                if board[nr, c] in passable:
                    board[nr, c] = '^'
                    r = nr
                else:
                    break
            else:
                break
    elif board[coord] == 'v':
        for s in range(0, steps):
            if (r + 1) < height and board[r + 1, c] in passable:
                r += 1
                board[r, c] = 'v'
            elif (r + 1) == height or board[r + 1, c] == ' ':
                nr = search_column_from_top(board, c)
                if board[nr, c] in passable:
                    board[nr, c] = 'v'
                    r = nr
                else:
                    break
            else:
                break
    return r, c


def onboard(r, c, height, width):
    if r < 0:
        return False
    if c < 0:
        return False
    if r >= height:
        return False
    if c >= width:
        return False
    return True


def make_move_on_cube(board, coord, steps, face_corners):
    height, width = board.shape
    r, c = coord[0], coord[1]
    passable = ['.', '>', '<', '^', 'v']
    for s in range(0, steps):
        tok = board[r, c]
        if tok == '>':
            nr = r
            nc = c + 1
        elif tok == '<':
            nr = r
            nc = c - 1
        elif tok == '^':
            nr = r - 1
            nc = c
        elif tok == 'v':
            nr = r + 1
            nc = c
        if onboard(nr, nc, height, width) and board[nr, nc] == '#':
            break
        elif onboard(nr, nc, height, width) and board[nr, nc] in passable:
            board[nr, nc] = tok
            r, c = nr, nc
        else:
            nr, nc, tok = cross_edge(board, r, c, steps - s, face_corners, tok)
            if board[nr, nc] in passable:
                board[nr, nc] = tok
                r, c = nr, nc
            else:
                break

    return r, c


def cross_edge(board, r, c, rem_steps, face_corners, tok):
    emap = {(1, '^'): (6, '>'), (6, '<'): (1, 'v'), (1, '<'): (4, '>'), (4, '<'): (1, '>'),
            (2, '^'): (6, '^'), (6, 'v'): (2, 'v'), (2, '>'): (5, '<'), (5, '>'): (2, '<'),
            (3, '<'): (4, 'v'), (4, '^'): (3, '>'), (2, 'v'): (3, '<'), (3, '>'): (2, '^'),
            (5, 'v'): (6, '<'), (6, '>'): (5, '^')}
    side = max(face_corners[2][0] - face_corners[1][0], face_corners[2][1] - face_corners[1][1])
    face = 0
    for f in face_corners.keys():
        if r >= face_corners[f][0] and r < (face_corners[f][0] + side) and c >= face_corners[f][1] and c < (face_corners[f][1] + side):
            face = f
            break
    new_edge = emap[(face, tok)]
    nr, nc = coord_map(r, c, face, new_edge[0], face_corners)

    return nr, nc, new_edge[1]


def coord_map(r, c, e, ne, face_corners):
    side = max(face_corners[2][0] - face_corners[1][0], face_corners[2][1] - face_corners[1][1])
    cr_e, cc_e = face_corners[e]
    cr_ne, cc_ne = face_corners[ne]
    roff = r - cr_e
    coff = c - cc_e
    if (e == 2 and ne == 6):
        nr = cr_ne + side - 1
        nc = cc_ne + coff
    elif (e == 6 and ne == 2):
        nr = cr_ne
        nc = cc_ne + coff
    elif (e == 1 and ne == 4) or (e == 4 and ne == 1):
        nr = cr_ne + side - 1 - roff
        nc = cc_ne
    elif (e == 1 and ne == 6):
        nr = cr_ne + coff
        nc = cc_ne
    elif (e == 6 and ne == 1):
        nr = cr_ne
        nc = cc_ne + roff
    elif (e == 3 and ne == 4):
        nr = cr_ne
        nc = cc_ne + roff
    elif (e == 4 and ne == 3):
        nr = cr_ne + coff
        nc = cc_ne
    elif (e == 2 and ne == 3):
        nr = cr_ne + coff
        nc = cc_ne + side - 1
    elif (e == 3 and ne == 2):
        nr = cr_ne + side - 1
        nc = cc_ne + roff
    elif (e == 2 and ne == 5) or (e == 5 and ne == 2):
        nr = cr_ne + side - 1 - roff
        nc = cc_ne + side - 1
    elif (e == 5 and ne == 6):
        nr = cr_ne + coff
        nc = cc_ne + side - 1
    elif (e == 6 and ne == 5):
        nr = cr_ne + side - 1
        nc = cc_ne + roff
    return nr, nc


def make_turn(board, coord, direction):
    if board[coord] == '>' and direction == 'L':
        board[coord] = '^'
    elif board[coord] == '>' and direction == 'R':
        board[coord] = 'v'
    elif board[coord] == '<' and direction == 'L':
        board[coord] = 'v'
    elif board[coord] == '<' and direction == 'R':
        board[coord] = '^'
    elif board[coord] == '^' and direction == 'L':
        board[coord] = '<'
    elif board[coord] == '^' and direction == 'R':
        board[coord] = '>'
    elif board[coord] == 'v' and direction == 'L':
        board[coord] = '>'
    elif board[coord] == 'v' and direction == 'R':
        board[coord] = '<'


def advent22_1(phase='I'):
    file = open('input22.txt')

    width = 0
    height = 0
    data = []
    for line in file:
        data.append(line.strip('\n'))
        width = max(width, len(data[-1]))
        height += 1
        #print(line)
    height -= 2
    board = np.full((height, width), ' ', dtype=str)

    path = []
    for r in range(0, len(data)):
        if data[r] != '':
            for c in range(0, len(data[r])):
                board[r, c] = data[r][c]
        else:
            path = data[r + 1]
            break

    #find start
    start_coord = (0, 0)
    for i in range(0, width):
        if board[0, i] == '.':
            start_coord = (0, i)
            break

    board[start_coord] = '>'
    coord = start_coord
    index = 0
    if phase == 'I':
        while index < len(path):
            direction, steps, index = next_move(path, index)
            coord = make_move(board, coord, steps)
            if direction:
                make_turn(board, coord, direction)

        facing = {'>': 0, 'v': 1, '<': 2, '^': 3}
        password = (coord[0] + 1)*1000 + (coord[1] + 1)*4 + facing[board[coord]]
        print('Password: ', password)
    elif phase == 'II':
        face_corners = {1: (0, 50), 2: (0, 100), 3: (50, 50), 4: (100, 0), 5: (100, 50), 6: (150, 0)}
        while index < len(path):
            direction, steps, index = next_move(path, index)
            coord = make_move_on_cube(board, coord, steps, face_corners)
            if direction:
                make_turn(board, coord, direction)
            #break
            #print(board)
        facing = {'>': 0, 'v': 1, '<': 2, '^': 3}
        password = (coord[0] + 1) * 1000 + (coord[1] + 1) * 4 + facing[board[coord]]
        print('Password: ', password)


class Elf:
    def __init__(self, coord):
        self.coord = coord
        self.first_direction_to_consider = 0
        self.to_consider = [[(-1, -1), (-1, 0), (-1, 1)], [(1, -1), (1, 0), (1, 1)],
                            [(-1, -1), (0, -1), (1, -1)], [(-1, 1), (0, 1), (1, 1)]]
        self.proposal = None

    def close_other_elf(self, area):
        for r in range(-1, 2):
            for c in range(-1, 2):
                if area[self.coord[0] + r, self.coord[1] + c] == '#' and not (r == 0 and c == 0):
                    return True
        return False

    def do_round(self, area):
        if not self.close_other_elf(area):
            self.proposal = None
            self.first_direction_to_consider = (self.first_direction_to_consider + 1) % 4
            return
        r, c = self.coord
        for direction in range(self.first_direction_to_consider, self.first_direction_to_consider + 4):
            d = direction % 4
            if area[r + self.to_consider[d][0][0], c + self.to_consider[d][0][1]] == '.' and \
                    area[r + self.to_consider[d][1][0], c + self.to_consider[d][1][1]] == '.' and \
                    area[r + self.to_consider[d][2][0], c + self.to_consider[d][2][1]] == '.':
                self.proposal = (r + self.to_consider[d][1][0], c + self.to_consider[d][1][1])
                break
        self.first_direction_to_consider = (self.first_direction_to_consider + 1) % 4

    def finish_round(self, area, elves):
        if self.proposal:
            for elf in elves:
                if elf == self:
                    continue
                if elf.proposal == self.proposal:
                    self.proposal = None
                    elf.proposal = None
                    return
            if self.proposal:
                area[self.coord] = '.'
                area[self.proposal] = '#'
                self.coord = self.proposal
                self.proposal = None


def init_area(area, elves):
    for elf in elves:
        area[elf.coord] = '#'


def find_bbox(elves):
    rmin = 100000
    cmin = 100000
    rmax = 0
    cmax = 0
    for elf in elves:
        rmin = min(rmin, elf.coord[0])
        cmin = min(cmin, elf.coord[1])
        rmax = max(rmax, elf.coord[0])
        cmax = max(cmax, elf.coord[1])
    return rmin, cmin ,rmax, cmax


def advent23_1(phase='I'):
    file = open('input23.txt')

    elf_location_data = []
    nof_rows = 0
    for line in file:
        elf_location_data.append(line.strip('\n'))
        nof_rows += 1
    nof_cols = len(elf_location_data[-1])
    margin = 100
    area = np.full((nof_rows + 2*margin, nof_cols + 2*margin), '.', dtype=str)
    elves = []
    r = margin
    for row in elf_location_data:
        c = margin
        for e in row:
            if e == '#':
                elves.append(Elf((r, c)))
            c += 1
        r += 1
    init_area(area, elves)
    #print(area)

    if phase == 'I':
        for elf_round in range(0, 10):
            for elf in elves:
                elf.do_round(area)
            for elf in elves:
                elf.finish_round(area, elves)
            #print('After round ', elf_round + 1)
            #print(area)
        rmin, cmin, rmax, cmax = find_bbox(elves)
        #print(area)
        print('Free area: ', sum(sum(area[rmin:rmax+1, cmin:cmax+1] == '.')))
    elif phase == 'II':
        ctr = 0
        while True:
            for elf in elves:
                elf.do_round(area)
            for elf in elves:
                elf.finish_round(area, elves)
            ctr += 1
            done = True
            for elf in elves:
                if elf.close_other_elf(area):
                    done = False
            if done:
                break
            if ctr % 20 == 0:
                rmin, cmin, rmax, cmax = find_bbox(elves)
                print('Bbox after round ', ctr + 1, ' : ', rmin, cmin, rmax, cmax)
        print('First no-move round: ', ctr + 1)


class Blizzard:
    def __init__(self, coord, direction):
        self.coord = coord
        self.glyph = direction
        direction_map = {'^': (-1, 0), 'v': (1, 0), '<': (0, -1), '>': (0, 1)}
        self.direction = direction_map[self.glyph]
        self.time = 0

    def update(self, board):
        h, w = board.shape
        if self.glyph == '^':
            self.coord = (self.coord[0] - 1, self.coord[1])
        elif self.glyph == 'v':
            self.coord = (self.coord[0] + 1, self.coord[1])
        elif self.glyph == '>':
            self.coord = (self.coord[0], self.coord[1] + 1)
        elif self.glyph == '<':
            self.coord = (self.coord[0], self.coord[1] - 1)
        self.border_wrap(h, w)
        if board[self.coord] == '.':
            board[self.coord] = self.glyph
        elif board[self.coord] in ['^', 'v', '<', '>']:
            board[self.coord] = '2'
        else:
            board[self.coord] = str(int(board[self.coord]) + 1)
        self.time += 1

    def border_wrap(self, h, w):
        if self.coord[0] == 0:
            self.coord = (h - 2, self.coord[1])
        elif self.coord[0] == h - 1:
            self.coord = (1, self.coord[1])
        elif self.coord[1] == 0:
            self.coord = (self.coord[0], w - 2)
        elif self.coord[1] == w - 1:
            self.coord = (self.coord[0], 1)


def possible_moves(coord, valley):
    r, c = coord[0], coord[1]
    forbidden = ['v', '^', '<', '>', '#', '2', '3', '4']
    moves = []
    if r < np.size(valley, 0) - 1 and valley[r + 1, c] not in forbidden:
            moves.append('S')
    if r > 0 and valley[r - 1, c] not in forbidden:
            moves.append('N')
    if valley[r, c - 1] not in forbidden:
        moves.append('W')
    if valley[r, c + 1] not in forbidden:
        moves.append('E')
    if valley[r, c] not in forbidden:
        moves.append('WAIT')
    return moves


def simulate_journey(start, valley, blizzards, path=[]):
    #print('Start:, ', start)
    goal = (valley.shape[0] - 1, valley.shape[1] - 2)
    move_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1), 'WAIT': (0, 0)}
    new_valley = valley.copy()
    new_valley[1:-1, 1:-1] = '.'
    show = new_valley.copy()
    for blizzard in blizzards:
        blizzard.update(new_valley)
    visited = [e[1] for e in path]
    coord = start
    forks = []
    game_over = False
    while True:
        possible = possible_moves(coord, new_valley)
        #print(possible)
        if len(possible) == 0:
            #path.append(('WAIT', coord))
            #if new_valley[coord] in ['v', '^', '<', '>', '2', '3', '4']:
            game_over = True
            print(', GAME OVER (at ', coord, '), goal = ', goal)
            break
        else:
            move = 'Q'
            for m in possible:
                mc = move_map[m]
                new_coord = (coord[0] + mc[0], coord[1] + mc[1])
                if new_coord == goal:
                    move = m
                    break
                if new_coord not in visited:
                    move = m
                    break
            if move == 'Q':
                move = possible[0]
            path.append((move, coord))
            if len(possible) > 1:
                possible.remove(move)
                remaining = []
                for p in possible:
                    #print('rem: ', coord, (p, coord))
                    remaining.append((p, coord))
                forks.append((len(path), remaining))
            move_c = move_map[move]
            coord = (coord[0] + move_c[0], coord[1] + move_c[1])
        if coord == goal:
            break
        new_valley = valley.copy()
        new_valley[1:-1, 1:-1] = '.'
        for blizzard in blizzards:
            blizzard.update(new_valley)
        show[coord] = str(len(path))
        visited.append(coord)

            #print(path)
            #print(show)
            #print(new_valley)

    return path, forks, coord == goal


def explore_paths(start, valley, blizzards, path=[], indent=''):
    origo = (0, 1)
    #print('path length: ', len(path), end='')

    global MIN_LEN
    move_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1), 'WAIT': (0, 0)}
    path_lengths = []
    blizzards_new = copy.deepcopy(blizzards)
    new_path, forks, success = simulate_journey(start, valley, blizzards_new, path)
    if success:
        path_lengths.append(len(new_path))
        MIN_LEN = min(path_lengths)
        print('Success:', MIN_LEN, new_path)
    indent += ' '
    #print(indent, 'all forks: ', forks)
    for fork in forks:
        #print(indent, '- fork: ', fork)
        for f in fork[1]:
            fork_string = str(fork[0]) + ',' + str(f[0]) + ',' + str(f[1])
            if fork_string in USED:
                continue
            #print(indent, '-- f:', fork_string)
            start_path = new_path[:fork[0] - 1]
            start_path.append(f)
            if len(start_path) >= 650:
                continue
            new_start = f[1]
            dist = math.sqrt((new_start[0] - origo[0])**2 + (new_start[1] - origo[1])**2)
            if (dist + 2) < math.sqrt(len(start_path)):
                continue
            USED.append(fork_string)
            fm = move_map[start_path[-1][0]]
            #print(start_path[-1][0:])
            #print('fm:', fm)
            #print(new_start)
            new_start = (new_start[0] + fm[0], new_start[1] + fm[1])
            blizzards_newer = copy.deepcopy(blizzards)
            new_valley = valley.copy()
            for i in range(0, len(start_path) - blizzards_newer[0].time):
                for b in blizzards_newer:
                    b.update(new_valley)
            extra_path_lengths = explore_paths(new_start, new_valley, blizzards_newer, start_path, indent)
            for e in extra_path_lengths:
                path_lengths.append(e)
    return path_lengths


def find_shortest_path_through_valley(start, end, valley, blizzards):
    move_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1), 'WAIT': (0, 0)}
    dist = np.ones([np.size(valley, 0), np.size(valley, 1)])
    dist[:, :] = -1
    dist[start] = 0
    start_node = start
    valley[1:-1, 1:-1] = '.'
    for blizzard in blizzards:
        blizzard.update(valley)
    neighbors = possible_moves(start_node, valley)
    for node in neighbors:
        mc = move_map[node]
        coord = (start_node[0] + mc[0], start_node[1] + mc[1])
        dist[coord] = dist[start_node] + 1

    #print(valley)
    #print(dist)

    iteration = 1
    while True:
        valley[1:-1, 1:-1] = '.'
        for blizzard in blizzards:
            blizzard.update(valley)
        new_dist = dist.copy()
        for i in range(0, np.size(valley, 0)):
            for j in range(0, np.size(valley, 1)):
                if dist[i, j] == iteration:
                    neighbors = possible_moves((i, j), valley)
                    for node in neighbors:
                        mc = move_map[node]
                        coord = (i + mc[0], j + mc[1])
                        new_dist[coord] = dist[i, j] + 1
        dist = new_dist
        iteration += 1
        #print(valley[0:, 0:])
        #print(new_dist[0:, 0:])
        if new_dist[end] != -1:
            break
    return dist, blizzards


def advent24_I(phase='I'):
    file = open('input24.txt')

    valley_data = []
    nof_rows = 0
    for line in file:
        valley_data.append(line.strip('\n'))
        nof_rows += 1
    nof_cols = len(valley_data[-1])
    valley = np.full((nof_rows, nof_cols), '.', dtype=str)
    blizzards = []
    for r in range(0, nof_rows):
        for c in range(0, nof_cols):
            valley[r, c] = valley_data[r][c]
            if valley[r, c] in ['^', 'v', '<', '>']:
                blizzards.append(Blizzard((r, c), valley[r, c]))

    #print(valley)
    #valley_history = [valley]
    #for t in range(0, 5):
    #    new_valley = valley.copy()
    #    new_valley[1:-1, 1:-1] = '.'
    #    for blizzard in blizzards:
    #        blizzard.update(new_valley)
    #    valley_history.append(new_valley)
    #for v in valley_history:
    #    print(v)
    move_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
    path_lengths = []
    start = (0, 1)
    goal = (valley.shape[0] - 1, valley.shape[1] - 2)
    blizzards_c = copy.deepcopy(blizzards)
    global MIN_LEN
    MIN_LEN = 2 ** 32
    #path_lengths = explore_paths(start, valley, blizzards_c)
    #print(min(path_lengths))
    dist, new_blizzards = find_shortest_path_through_valley(start, goal, valley, blizzards_c)
    print('Start to end took: ', dist[goal])

    dist, newer_blizzards = find_shortest_path_through_valley(goal, start, valley, new_blizzards)
    print('End to start took: ', dist[goal])

    dist, newest_blizzards = find_shortest_path_through_valley(start, goal, valley, newer_blizzards)
    print('End to start took: ', dist[goal])


def snafu_to_decimal(snafu):
    digit_mapping = {'0': 0, '1': 1, '2': 2, '-': -1, '=': -2}
    nof_digits = len(snafu)
    sum = 0
    for d in range(0, nof_digits):
        sum += digit_mapping[snafu[d]]*5**(nof_digits - d - 1)

    return sum


def decimal_to_snafu(decimal):
    digit_mapping = {0: '0', 1: '1', 2: '2', -1: '-', -2: '='}
    nof_digits = 0
    while True:
        nof_digits += 1
        if 5**(nof_digits - 1)*2 > decimal:
            break
    snafu = ['0']*nof_digits

    while True:
        #print(nof_digits)
        #if decimal > 0:
        if nof_digits != 1:
            digit = round(decimal / 5**(nof_digits - 1))
        else:
            digit = math.floor(decimal / 5 ** (nof_digits - 1))
        #print(digit)
        snafu[-nof_digits] = digit_mapping[digit]
        #print(snafu, digit)
        decimal -= 5**(nof_digits - 1)*digit
        #print(decimal)
        if decimal == 0:
            break
        nof_digits -= 1

    ret_str = ''
    for d in snafu:
        ret_str += d

    return ret_str


def advent25_I():
    file = open('input25.txt')

    snafu_numbers = []
    nof_rows = 0
    dec_sum = 0
    for line in file:
        snafu_numbers.append(line.strip('\n'))
        nof_rows += 1
        print(snafu_numbers[-1], snafu_to_decimal(snafu_numbers[-1]))
        dec_sum += snafu_to_decimal(snafu_numbers[-1])

    print('Sun of snumbers: ', dec_sum)
    snafu_sum = decimal_to_snafu(dec_sum)
    #print(snafu_to_decimal(snafu_sum))
    print('Snafu sum: ', snafu_sum)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    start_time = time.time()
    print('Advent 1')
    advent1_1()
    end_time_1 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_1 - start_time))
    print('Advent 2')
    advent2_1()
    advent2_2()
    end_time_2 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_2 - end_time_1))
    print('Advent 3')
    advent3_1()
    advent3_2()
    end_time_3 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_3 - end_time_2))
    print('Advent 4')
    advent4_1()
    advent4_2()
    end_time_4 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_4 - end_time_3))
    print('Advent 5')
    advent5_1(False)
    advent5_1(True)
    end_time_5 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_5 - end_time_4))
    print('Advent 6')
    advent6_1()
    end_time_6 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_6 - end_time_5))
    print('Advent 7')
    advent7_1()
    advent7_1(False)
    end_time_7 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_7 - end_time_6))
    print('Advent 7')
    advent8_1(99)
    end_time_8 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_8 - end_time_7))
    print('Advent 9')
    advent9_1()
    advent9_2()
    end_time_9 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_9 - end_time_8))
    print('Advent 9')
    advent10_1()
    end_time_10 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_10 - end_time_9))
    print('Advent 11')
    advent11_1(3, 20)
    advent11_1(1, 10000)
    end_time_11 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_11 - end_time_10))
    print('Advent 12')
    advent12_1()
    end_time_12 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_12 - end_time_11))
    print('Advent 13')
    advent13_1()
    end_time_13 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_13 - end_time_12))
    print('Advent 14')
    advent14_1()
    end_time_14 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_14 - end_time_13))
    print('Advent 15')
    advent15_1()
    end_time_15 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_15 - end_time_14))
    print('Advent 16')
    advent16_1('I')
    advent16_1('II')
    end_time_16 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_16 - end_time_15))
    print('Advent 17')
    advent17_1()
    end_time_17 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_17 - end_time_16))
    print('Advent 18')
    advent18_1()
    end_time_18 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_18 - end_time_17))
    print('Advent 19')
    advent19_1('I')
    advent19_1('II')
    end_time_19 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_19 - end_time_18))
    print('Advent 20')
    advent20_1()
    advent20_1(811589153, 10)
    end_time_20 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_20 - end_time_19))
    print('Advent 21')
    advent21_1('I')
    advent21_1('II')
    end_time_21 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_21 - end_time_20))
    print('Advent 22')
    advent22_1('I')
    advent22_1('II')
    end_time_22 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_22 - end_time_21))
    print('Advent 23')
    advent23_1('I')
    advent23_1('II')
    end_time_23 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_23 - end_time_22))
    print('Advent 24')
    advent24_I()
    end_time_24 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_24 - end_time_23))
    print('Advent 25')
    advent25_I()
    end_time_25 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_25 - end_time_24))

    print("Accumulated time elapsed: {:.2f}s".format(end_time_25 - start_time))


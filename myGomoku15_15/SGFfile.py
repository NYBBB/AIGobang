
import time
import os
import numpy as np
from mcts_alphaZero import MCTS


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def tabulator_probs(move_probs, board, move, max_prob, length=7):
    column = board.width

    def aligns(string, index, length=7):
        difference = length - len(string)  # 计算限定长度为20时需要补齐多少个空格
        if difference == 0:  # 若差值为0则不需要补
            return string
        elif difference < 0:
            print('错误：限定的对齐长度小于字符串长度!')
            return None
        new_string = string
        space = ' '
        return_string = None
        if index == move:
            return_string = "\033[33m{}\033[0m".format(new_string) + space * difference  # 返回补齐空格后的字符串
        if index == max_prob:
            return_string = "\033[32m{}\033[0m".format(new_string) + space * difference
        if index == move and index == max_prob:
            return_string = "\033[31m{}\033[0m".format(new_string) + space * difference
        if board.states.get(index) == 1:
            return_string = "\033[34m{}\033[0m".format("X  ") + space * difference
        if board.states.get(index) == 2:
            return_string = "\033[35m{}\033[0m".format("O  ") + space * difference
        if not index == move and not index == max_prob and index in board.availables:
            return_string = new_string + space * difference
        return return_string
    # 将每个位置的概率显示出来
    move_probs_ = move_probs
    # 将里面的每一项都变成保留一定小数的然后转换成字符串
    for i in range(len(move_probs)):
        move_probs_[i] = round(move_probs[i], 4)  # 保留n位小数并转换成字符串
    move_str = []
    for i in range(len(move_probs_)):
        move_str.append(str(move_probs_[i]))

    print("\n")
    p = ''
    num = 0
    sum = len(move_str)
    index = 0
    for i in move_str:
        p = p + aligns(i, index, length)
        num = num + 1
        sum = sum - 1
        if num >= column:
            print(p)
            p = ''
            num = 0
        elif sum <= 0:
            print(p)
        index += 1


class SGFflie():
    def __init__(self):
        """
        初始化：
        POS：棋盘坐标的对应字母顺序
        savepath:保存路径
        trainpath:训练数据的路径
        """
        self.POS = 'abcdefghijklmno'
        self.savepath = 'save qiju\qiju'
        self.trainpath = '棋谱库\sgf'

        self.mcts = None

    def openfile(self, filepath):
        """打开文件,读取棋谱"""
        f = open(filepath, 'r')
        data = f.read()
        f.close()

        # 分割数据
        effective_data = data.split(';')
        s = effective_data[2:-1]

        board = []
        step = 0
        for point in s:
            x = self.POS.index(point[2])
            y = self.POS.index(point[3])
            color = step % 2
            step += 1
            board.append([x, y, color, step])

        return board

    def get_file_step(self, board, filepath, index, mcts, temp=1e-3, n_playout=2000):
        move_probs = np.zeros(225)
        # 先将这一步的棋盘传入神经网络预测出每个落点的概率
        actions, probs = mcts.get_move_probs(board, temp=temp)
        move_probs[list(actions)] = probs
        # 将棋谱里面读取出来的棋子落点位置的概率改成n
        state = self.openfile(filepath)
        action = (state[index][0] - 1) * 15 + state[index][1]
        color = state[index][2]
        # 显示出每个位置的概率
        tabulator_probs(move_probs, board, action, np.argmax(move_probs))
        is_last_step = False
        if len(state)-1 == index:
            is_last_step = True

        mcts.update_with_move(action)

        print(action, move_probs[action], np.argmax(move_probs), move_probs[np.argmax(move_probs)])

        return action, move_probs, is_last_step

    @staticmethod
    def allFileFromDir(Dirpath):
        """获取文件夹中所有文件的路径"""
        pathDir = os.listdir(Dirpath)
        pathfile = []
        for allDir in pathDir:
            child = os.path.join('%s%s' % (Dirpath, allDir))
            pathfile.append(child)
        return pathfile

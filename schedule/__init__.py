import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly as py
from colorama import init, Fore
from matplotlib import colors as mcolors

deepcopy = copy.deepcopy
pyplt = py.offline.plot
COLORS = list(mcolors.CSS4_COLORS)
LEN_COLORS = len(COLORS)
init(autoreset=True)
np.set_printoptions(precision=2)


class Utils:
    @staticmethod
    def make_dir(dir_name):
        try:
            os.mkdir("./%s" % dir_name)
        except FileExistsError:
            pass

    @staticmethod
    def print(msg, fore=Fore.LIGHTCYAN_EX, ):
        print(fore + msg)

    @staticmethod
    def calculate_fitness(obj):
        return 1 / (1 + obj)

    @staticmethod
    def update_info(old_obj, new_obj):
        if new_obj < old_obj:
            return True
        return False

    @staticmethod
    def figure_png(file_name="Figure", dpi=200, ):
        ax = plt.gca()
        [ax.spines[name].set_color('none') for name in ["top", "right", "bottom", "left"]]
        Utils.make_dir("Figure")
        plt.savefig("./Figure/%s" % file_name, dpi=dpi, )
        plt.clf()

    @staticmethod
    def crt_data_dfsp(w, n, m, low, high, ):
        ops, prt = [], []
        for i in range(w):
            a = np.tile(range(m), (n, 1))
            b = np.random.uniform(low, high + 1, [n, m])
            ops.append(a)
            prt.append(b)
        for i in range(w):
            prt[i] = prt[0] * (1 + 0.02 * np.random.random((n, m)) * i)
        return ops, prt


class Info:
    """
    调度信息
    """

    def __init__(self, w, n, m, length, job, wkc, osc, mac, start, finish, duration, job_index, machine_index, ):
        """

        :param w:
        :param n:
        :param m:
        :param length:
        :param job:
        :param wkc:
        :param osc:
        :param mac:
        :param start:
        :param finish:
        :param duration:
        :param job_index:
        :param machine_index:
        """
        self.w = w
        self.n = n
        self.m = m
        self.length = length
        self.job = deepcopy(job)
        self.wkc = deepcopy(wkc)
        self.osc = deepcopy(osc)
        self.mac_info = deepcopy(mac)
        self.start = deepcopy(start)
        self.finish = deepcopy(finish)
        self.duration = deepcopy(duration)
        self.job_index = deepcopy(job_index)
        self.machine_index = deepcopy(machine_index)
        self.wkc_info = np.tile(self.wkc, self.m)
        self.makespan = max(self.finish)

    def crossover_pmx_job(self, info2, ):
        job1 = deepcopy(self.job)
        job2 = deepcopy(info2.job)
        a, b = np.random.choice(self.n, 2, replace=False)
        min_a_b, max_a_b = min([a, b]), max([a, b])
        r_a_b = range(min_a_b, max_a_b)
        r_left = np.delete(range(self.n), r_a_b)
        left_1, left_2 = job2[r_left], job1[r_left]
        middle_1, middle_2 = job2[r_a_b], job1[r_a_b]
        job1[r_a_b], job2[r_a_b] = middle_2, middle_1
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = np.argwhere(middle_1 == j)[0, 0]
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = np.argwhere(middle_1 == value)[0, 0]
                        value = middle_2[index]
                    else:
                        break
                mapping[0].append(i)
                mapping[1].append(value)
            elif i in middle_2:
                pass
            else:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[np.argwhere(left_1 == i)[0, 0]] = j
            elif i in left_2:
                left_2[np.argwhere(left_2 == i)[0, 0]] = j
            if j in left_1:
                left_1[np.argwhere(left_1 == j)[0, 0]] = i
            elif j in left_2:
                left_2[np.argwhere(left_2 == j)[0, 0]] = i
        job1[r_left], job2[r_left] = left_1, left_2
        return job1, job2

    def crossover_wkc_random(self, info2, ):
        wkc1 = deepcopy(self.wkc)
        wkc2 = deepcopy(info2.wkc)
        for i, (u, v) in enumerate(zip(wkc1, wkc2)):
            if np.random.random() < 0.5:
                wkc1[i], wkc2[i] = v, u
        return wkc1, wkc2

    def mutation_tpe_job(self, ):
        job = deepcopy(self.job)
        a = np.random.choice(self.n, 2, replace=False)
        job[a] = job[a[::-1]]
        return job

    def mutation_wkc_random(self, ):
        wkc = deepcopy(self.wkc)
        for i in range(self.n):
            if np.random.random() < 0.5:
                a = np.delete(range(self.w), wkc[i])
                wkc[i] = np.random.choice(a, 1, replace=False)[0]
        return wkc

    def ganttChart_png(self, file_name="GanttChart", dpi=200, ):
        np.random.shuffle(COLORS)
        plt.figure(figsize=[9, 5])
        plt.margins()
        plt.tight_layout()
        ymin = -0.5
        ymax = self.w * self.m + ymin
        plt.vlines(self.makespan, ymin, ymax, colors="red", linestyles="--")
        plt.text(self.makespan, ymin, round(self.makespan, 2))
        for i in range(self.length):
            color_bar = COLORS[self.osc[i] % LEN_COLORS]
            edgecolor, text_color = color_bar, "black"
            width = self.finish[i] - self.start[i]  # self.duration[i]
            y_position = self.wkc_info[i] * self.m + self.mac_info[i]
            plt.barh(y=y_position, width=width, left=self.start[i],
                     color=color_bar, edgecolor=edgecolor)
            text = r"${%s}$" % (self.osc[i] + 1)
            plt.text(x=self.start[i] + 0.5 * width, y=y_position,
                     s=text, c=text_color,
                     ha="center", va="center", )
        yticks = []
        for i in range(self.w):
            for j in range(self.m):
                msg = "${C}_{%s}^{%s}$" % (i + 1, j + 1)
                yticks.append(msg)
        plt.yticks(range(self.m * self.w), yticks)
        plt.xticks([], [])
        Utils.figure_png(file_name=file_name, dpi=dpi)


class Objective:
    """
    目标函数
    :return:
    """

    @staticmethod
    def makespan(info):
        return info.makespan


class DFsp:
    """
    Distribute flow shop scheduling problem（分布式流水车间调度问题）
    """

    def __init__(self, w, n, m, ops, prt, ):
        self.w = w
        self.n = n
        self.m = m
        self.ops = ops
        self.prt = prt
        self.length = n * m

    def code_job_dfsp(self, ):
        return np.random.permutation(self.n)

    def code_wkc_dfsp(self, ):
        return np.random.choice(self.w, self.n, replace=True)

    def decode_dfsp(self, job, wkc, ):
        osc = np.tile(job, self.m)
        mac = np.repeat(range(self.m), self.n)
        job_index = []
        for i in range(self.n):
            j = np.argwhere(osc == i)[0, 0]
            k = [j + index * self.n for index in range(self.m)]
            job_index.append(k)
        machine_index = []
        for i in range(self.m):
            j = i * self.n
            machine_index.append(np.arange(j, j + self.n).tolist())
        duration = np.zeros(self.length)
        start = np.zeros(self.length)
        finish = np.zeros(self.length)
        for w in range(self.w):
            prt = self.prt[w]
            index = np.argwhere(wkc == w)[:, 0]
            n_job = index.shape[0]
            job_w = job[index]
            osc_w = np.tile(job_w, self.m)
            opr_w = np.repeat(range(self.m), n_job)
            jpt = prt[osc_w, opr_w]
            jst = np.zeros([n_job, self.m])
            jft = np.zeros([n_job, self.m])
            index_w = []
            job_index_w = []
            for i in job_w:
                j = np.argwhere(osc_w == i)[0, 0]
                k = [j + index * n_job for index in range(self.m)]
                job_index_w.extend(k)
            for i in range(n_job):
                index_w.extend(job_index[job_w[i]])
                for j in range(self.m):
                    jst[i, j] = max([jft[i, j - 1], jft[i - 1, j]])
                    jft[i, j] = jst[i, j] + jpt[i + j * n_job]
            duration[index_w] = jpt.T.flatten()[job_index_w]
            start[index_w] = jst.T.flatten()[job_index_w]
            finish[index_w] = jft.T.flatten()[job_index_w]
        return Info(self.w, self.n, self.m, self.length,
                    job, wkc, osc, mac, start, finish, duration,
                    job_index, machine_index, )

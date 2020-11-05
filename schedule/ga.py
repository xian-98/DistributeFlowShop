import logging
import time

import numpy as np

from . import DFsp, Utils, deepcopy, plt
from .name import DataName, GaName

Utils.make_dir("log")
logging.basicConfig(filename='./log/ga.log', level=logging.INFO)


class Ga:
    """
    遗传算法
    """

    def __init__(self, pop_size, rate_crossover, rate_mutation, operator_crossover, operator_mutation,
                 operator_selection, stop_max_generation, stop_max_stay, function_objective, ):
        self.pop_size = pop_size
        self.rate_crossover = rate_crossover
        self.rate_mutation = rate_mutation
        self.operator_crossover = operator_crossover
        self.operator_mutation = operator_mutation
        self.operator_selection = operator_selection
        self.stop_max_generation = stop_max_generation
        self.stop_max_stay = stop_max_stay
        self.function_objective = function_objective
        # 定义
        self.pop_info = []
        self.pop_objective = np.zeros([self.pop_size, 1])
        self.pop_fitness = np.zeros([self.pop_size, 1])

        self.generation_objective = np.zeros([self.stop_max_generation + 1, 3])  # 最优、最差、平均
        self.generation_runtime = np.zeros([self.stop_max_generation + 1, 3])  # 开始、结束、计时

        self.global_best_info = None
        self.global_best_job = None
        self.global_best_wkc = None
        self.global_best_obj = None
        self.global_best_fitness = None

    # 【选择操作】轮盘赌选
    def selection_roulette(self, ):
        a = self.pop_fitness / sum(self.pop_fitness[:, 0])
        b = np.zeros([self.pop_size, 1])
        for i in range(self.pop_size):  # 计算个体的被选择概率
            b[i, 0] = sum(a[:i + 1])
        return b

    # 更新
    def update_generation(self, i, ):
        index_min = int(np.argmin(self.pop_objective[:, 0]))
        index_max = int(np.argmax(self.pop_objective[:, 0]))
        self.generation_objective[i, 0] = self.pop_objective[index_min]
        self.generation_objective[i, 1] = self.pop_objective[index_max]
        self.generation_objective[i, 2] = np.mean(self.pop_objective[:, 0])
        self.generation_runtime[i, 2] = self.generation_runtime[i, 1] - self.generation_runtime[i, 0]
        if self.global_best_info is None or Utils.update_info(self.global_best_obj, self.generation_objective[i, 0]):
            self.global_best_info = self.pop_info[index_min]
            self.global_best_job = self.pop_info[index_min].job
            self.global_best_wkc = self.pop_info[index_min].wkc
            self.global_best_obj = self.pop_objective[index_min][0]
            self.global_best_fitness = self.pop_fitness[index_min][0]
        # 记录
        msg = "Generation{:>3}:Runtime:{:<.2f},Best:{:<.2f},Worst:{:<.2f},Mean:{:<.2f}".format(
            i,
            self.generation_runtime[i, 2],
            self.global_best_obj,  # 不等于generation_objective[i, 0]
            self.generation_objective[i, 1],
            self.generation_objective[i, 2],
        )
        logging.info(msg)
        Utils.print(msg)

    def objective_png(self, file_name="ObjectiveTrace", dpi=200, ):
        plt.figure(figsize=[9, 5])
        plt.margins()
        plt.tight_layout()
        marker = ["v", "^", "o"]
        line = ['--', '-.', ':']
        label = ["Best", "Worst", "Mean"]
        for i in range(self.generation_objective.shape[1]):
            plt.plot(self.generation_objective[:, i], marker=marker[i], linestyle=line[i], label=label[i])
        plt.legend()
        Utils.figure_png(file_name=file_name, dpi=dpi)

    def runtime_png(self, file_name="Runtime", dpi=200, ):
        plt.figure(figsize=[9, 5])
        plt.margins()
        plt.tight_layout()
        plt.plot(self.generation_runtime[:, 2], marker="o", linestyle="--", label="Runtime")
        plt.legend()
        Utils.figure_png(file_name=file_name, dpi=dpi)


class GaDFsp(Ga, DFsp):
    """
    流水车间调度的遗传算法
    """

    def __init__(self, para, data, ):
        pop_size = para[GaName.pop_size]
        rate_crossover = para[GaName.rate_crossover]
        rate_mutation = para[GaName.rate_mutation]
        operator_crossover = para[GaName.operator_crossover]
        operator_mutation = para[GaName.operator_mutation]
        operator_selection = para[GaName.operator_selection]
        stop_max_generation = para[GaName.stop_max_generation]
        stop_max_stay = para[GaName.stop_max_stay]
        function_objective = para[GaName.function_objective]
        Ga.__init__(self, pop_size, rate_crossover, rate_mutation, operator_crossover, operator_mutation,
                    operator_selection, stop_max_generation, stop_max_stay, function_objective, )
        DFsp.__init__(self, data[DataName.w], data[DataName.n], data[DataName.m],
                      data[DataName.ops], data[DataName.prt], )
        # 定义
        self.pop_job = np.zeros([self.pop_size, self.n], dtype=int)
        self.pop_wkc = np.zeros([self.pop_size, self.n], dtype=int)

    def update_info_job(self, i, obj_new, info_new, job_new, ):
        if Utils.update_info(self.pop_objective[i], obj_new):
            self.pop_info[i] = info_new
            self.pop_job[i] = job_new
            self.pop_objective[i] = obj_new
            self.pop_fitness[i] = Utils.calculate_fitness(obj_new)

    def update_info_wkc(self, i, obj_new, info_new, wkc_new, ):
        if Utils.update_info(self.pop_objective[i], obj_new):
            self.pop_info[i] = info_new
            self.pop_wkc[i] = wkc_new
            self.pop_objective[i] = obj_new
            self.pop_fitness[i] = Utils.calculate_fitness(obj_new)

    def do_init(self, ):
        self.generation_runtime[0, 0] = time.perf_counter()
        for i in range(self.pop_size):
            self.pop_job[i] = self.code_job_dfsp()
            self.pop_wkc[i] = self.code_wkc_dfsp()
            info = self.decode_dfsp(self.pop_job[i], self.pop_wkc[i])
            self.pop_info.append(info)
            self.pop_objective[i] = self.function_objective(info)
            self.pop_fitness[i] = Utils.calculate_fitness(self.pop_objective[i])
        self.generation_runtime[0, 1] = time.perf_counter()
        self.update_generation(0)

    def do_selection_roulette(self, ):
        a = self.selection_roulette()
        pop_osc = deepcopy(self.pop_job)
        pop_wkc = deepcopy(self.pop_wkc)
        pop_objective = deepcopy(self.pop_objective)
        pop_fitness = deepcopy(self.pop_fitness)
        pop_info = deepcopy(self.pop_info)
        for i in range(self.pop_size):
            j = np.argwhere(a[:, 0] > np.random.random())[0, 0]
            self.pop_job[i] = pop_osc[j]
            self.pop_wkc[i] = pop_wkc[j]
            self.pop_objective[i] = pop_objective[j]
            self.pop_fitness[i] = pop_fitness[j]
            self.pop_info[i] = pop_info[j]
        index = int(np.argmax(self.pop_fitness[:, 0]))
        self.pop_info[index] = self.global_best_info
        self.pop_job[index] = self.global_best_job
        self.pop_wkc[index] = self.global_best_wkc
        self.pop_objective[index] = self.global_best_obj
        self.pop_fitness[index] = self.global_best_fitness

    def do_crossover_pmx(self, i, j, ):
        job1, job2 = self.pop_info[i].crossover_pmx_job(self.pop_info[j])
        info1 = self.decode_dfsp(job1, self.pop_wkc[i])
        obj1 = self.function_objective(info1)
        self.update_info_job(i, obj1, info1, job1)
        info2 = self.decode_dfsp(job2, self.pop_wkc[j])
        obj2 = self.function_objective(info2)
        self.update_info_job(j, obj2, info2, job2)

    def do_mutation_tpe(self, i, ):
        job1 = self.pop_info[i].mutation_tpe_job()
        info1 = self.decode_dfsp(job1, self.pop_wkc[i])
        obj1 = self.function_objective(info1)
        self.update_info_job(i, obj1, info1, job1)

    def do_crossover_wkc(self, i, j, ):
        wkc1, wkc2 = self.pop_info[i].crossover_wkc_random(self.pop_info[j])
        info1 = self.decode_dfsp(self.pop_job[i], wkc1)
        obj1 = self.function_objective(info1)
        self.update_info_wkc(i, obj1, info1, wkc1)
        info2 = self.decode_dfsp(self.pop_job[j], wkc2)
        obj2 = self.function_objective(info2)
        self.update_info_wkc(j, obj2, info2, wkc2)

    def do_mutation_wkc(self, i, ):
        wkc1 = self.pop_info[i].mutation_wkc_random()
        info1 = self.decode_dfsp(self.pop_job[i], wkc1)
        obj1 = self.function_objective(info1)
        self.update_info_wkc(i, obj1, info1, wkc1)

    def start_generation(self, ):
        self.do_init()
        for g in range(1, self.stop_max_generation + 1):
            self.generation_runtime[g, 0] = time.perf_counter()
            for i in range(self.pop_size):
                if np.random.random() < self.rate_crossover:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover_pmx(i, j)
                    self.do_crossover_wkc(i, j)
                if np.random.random() < self.rate_mutation:
                    self.do_mutation_tpe(i)
                    self.do_mutation_wkc(i)
            self.do_selection_roulette()
            self.generation_runtime[g, 1] = time.perf_counter()
            self.update_generation(g)
            if g >= self.stop_max_stay and np.std(self.generation_objective[g - self.stop_max_stay + 1:g + 1, 0]) == 0:
                k = np.arange(g + 1, self.stop_max_generation + 1)
                self.generation_objective = np.delete(self.generation_objective, k, axis=0)
                self.generation_runtime = np.delete(self.generation_runtime, k, axis=0)
                break

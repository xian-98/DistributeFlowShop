from schedule import Utils, Objective, DFsp
from schedule.ga import GaDFsp
from schedule.name import GaName, DataName


def main_dfsp():
    w, n, m, low, high = 3, 10, 4, 5, 30
    ops, prt = Utils.crt_data_dfsp(w, n, m, low, high)
    # dfsp = DFsp(w, n, m, ops, prt)
    # job = dfsp.code_job_dfsp()
    # wkc = dfsp.code_wkc_dfsp()
    # info = dfsp.decode_dfsp(job, wkc)
    # info.ganttChart_png(file_name="GanttChart-dfsp-1")
    para = {
        GaName.pop_size: 40,
        GaName.rate_crossover: 0.65,
        GaName.rate_mutation: 0.35,
        GaName.operator_crossover: "pmx",
        GaName.operator_mutation: "tpe",
        GaName.operator_selection: "roullete",
        GaName.stop_max_generation: 50,
        GaName.stop_max_stay: 30,
        GaName.function_objective: Objective.makespan
    }
    data = {
        DataName.w: w,
        DataName.n: n,
        DataName.m: m,
        DataName.ops: ops,
        DataName.prt: prt
    }
    ga_fsp = GaDFsp(para, data)
    ga_fsp.start_generation()
    ga_fsp.global_best_info.ganttChart_png(file_name="GanttChart-dfsp-ga-1")
    ga_fsp.objective_png(file_name="Objective-dfsp-ga-1")
    ga_fsp.runtime_png(file_name="Runtime-dfsp-ga-1")


def run():
    main_dfsp()


if __name__ == "__main__":
    run()

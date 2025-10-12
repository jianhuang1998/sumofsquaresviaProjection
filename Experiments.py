import tables as tb
import numpy as np
import time

class ExperimentResult(tb.IsDescription):
    is_success = tb.BoolCol()
    CPU_time = tb.Float64Col()
    End_iterater = tb.Int32Col()


def Experiment_dimension_rank(filename, poly_set, projection, Ex_times=100, maxiter=500):
    file = tb.open_file(filename, mode="w")
    kwargs = {}
    for poly in poly_set:
        poly_var, poly_deg, poly_term = poly
        group_name = f"poly_{poly_var}_{poly_deg}_{poly_term}"  # variable, degree, number of terms
        group = file.create_group("/", group_name, f"Polynomial {poly}")
        poly_group_name = f"poly_{poly_var}_{poly_deg}_{poly_term}coeff_sos"  # variable, degree, number of terms
        poly_group = file.create_group("/", poly_group_name, f"Polynomial {poly}")

        basis1, basis2, d1, d2, indices_matrix, nb_parts = auxiliary(poly_var, poly_deg)
        dim = len(indices_matrix)
        proj_vector_space_V = make_proj_vector_space_SOS(indices_matrix, nb_parts)

        all_g = [general_sos(poly_var, poly_deg, poly_term, indices_matrix) for _ in range(Ex_times)]  # 生成 100 个不同的数据
        coeff_SOS = file.create_vlarray(poly_group, f"coeff_SOS{poly_var}_{poly_deg}_{poly_term}", tb.Float64Atom())
        for proj in projection:
            table = file.create_table(group, f"{proj}_table_results", ExperimentResult, f"{proj} table result")
            if proj == "MOSEK":
                pass
            else:
                list_neg_least_ev = file.create_vlarray(group, f"{proj}_list_neg_least_ev", tb.Float64Atom())
                list_data = file.create_earray(group, f"{proj}_list_data", tb.Float64Atom(),
                                               shape=(0, len(basis1), len(basis1)))
            algo_dict = {
                # "AP": AP, "Dykstra": Dykstra,
                "oneHIP_AP": oneHIP_AP,
                "HIPswitch": HIPswitch, "pureHIP": pure_HIP,
                # "AP_inf": AP_inf, "ex_AP_inf": lazy_HIP, "MOSEK":mosek,
                "APTR": AP_rank_nlev, "lazyAPTR": lazy_AP_rank_nlev,
                "APFR": AP_fixed_rank_beta, "lazyAPFR": lazy_AP_fixed_rank_beta
            }
            alg_proj = algo_dict.get(proj, None)
            if poly == (10, 4, 20) and proj == "MOSEK":
                break
            if proj == "APFR" or proj == "lazyAPFR":
                kwargs['fixed_rank'] = poly_term
            else:
                kwargs = {}
            if alg_proj is None:
                raise ValueError(f"Unknow projection algorithem: {proj}")
            for i in range(Ex_times):
                g = all_g[i]
                if len(coeff_SOS) <= Ex_times:
                    coeff_SOS.append(np.array(g))
                proj_0_on_V = project_to_linear_space(zeros((dim, dim)), g, indices_matrix, nb_parts)
                p = np.eye(len(basis1)) * 0.1
                if alg_proj == mosek:
                    st = time.process_time()
                    mosek_result = mosek(g, poly_deg, poly_var)
                    alg_time = time.process_time() - st
                    row = table.row
                    row["CPU_time"] = alg_time
                    row["is_success"] = mosek_result
                    row.append()
                else:
                    try:
                        st = time.process_time()
                        data, neg_least_ev = alg_proj(p, proj_vector_space_V, proj_0_on_V, maxiter=maxiter, **kwargs)
                        alg_time = time.process_time() - st

                        row = table.row
                        row["CPU_time"] = alg_time
                        row["End_iterater"] = len(neg_least_ev)
                        row["is_success"] = (neg_least_ev[-1] < 1e-8)
                        row.append()

                        list_neg_least_ev.append(np.array(neg_least_ev))
                        list_data.append((data[-1])[np.newaxis, :, :])
                    except Exception as e:
                        row = table.row
                        row["CPU_time"] = -1  # 代表失败
                        row["End_iterater"] = -1
                        row["is_success"] = False
                        row.append()

                        # 用空的或无效的数据占位
                        list_neg_least_ev.append(np.array([]))
                        list_data.append(np.zeros((1, len(basis1), len(basis1)))[np.newaxis, :, :])
            table.flush()

    file.close()

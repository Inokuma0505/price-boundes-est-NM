# -*- coding: utf-8 -*-
"""
Auto-generated script from main_exp_get_all.ipynb
"""

import matplotlib.pyplot as plt
import itertools
import time
import numpy as np
from tqdm import tqdm
import json
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import r2_score


# 価格を生成する関数
def create_price(r_mean: float, r_std: float, M: int) -> NDArray[np.float_]:
    price = np.random.normal(r_mean, r_std, size=M)
    return price


# alphaを作成する関数
def alpha_star(M: int) -> NDArray[np.float_]:
    alpha_star = np.random.uniform(M, 3 * M, size=M)
    return alpha_star


# betaを作成する関数
def beta_star(M: int, M_prime: int) -> NDArray[np.float_]:
    beta_star = np.zeros((M, M_prime))
    for m in range(M):
        for m_prime in range(M_prime):
            if m == m_prime:
                beta_star[m, m_prime] = np.random.uniform(-3 * M, -2 * M)
            else:
                beta_star[m, m_prime] = np.random.uniform(0, 3)
    return beta_star


def quantity_function(
    price: NDArray[np.float_],
    alpha: NDArray[np.float_],
    beta: NDArray[np.float_],
    delta: float = 0.1,
) -> list[float]:
    M = len(price)
    q_m_no_noise = []
    for m in range(M):
        sum_beta = sum(beta[m][m_prime] * price[m_prime] for m_prime in range(M))
        q_m_no_noise.append(alpha[m] + sum_beta)
    E_q_m_squared = np.mean(np.array(q_m_no_noise) ** 2)
    sigma = delta * np.sqrt(E_q_m_squared)
    return [q_m_no_noise[m] + np.random.normal(0, sigma) for m in range(M)]


def sales_function(
    price: NDArray[np.float_], alpha: NDArray[np.float_], beta: NDArray[np.float_]
) -> list[float]:
    M = len(price)
    sales_list = []
    for m in range(M):
        sum_beta = sum(beta[m][m_prime] * price[m_prime] for m_prime in range(M))
        quantity = alpha[m] + sum_beta
        sales_list.append(quantity * price[m])
    return sales_list


def create_date(M, N, r_mean, r_std, delta=0.1):
    alpha = alpha_star(M)
    beta = beta_star(M, M)
    price_list = []
    quantity_list = []
    for _ in range(N):
        price = create_price(r_mean, r_std, M)
        quantity = quantity_function(price, alpha, beta, delta)
        price_list.append(price)
        quantity_list.append(quantity)
    X = np.array(price_list)
    Y = np.array(quantity_list)
    return alpha, beta, X, Y


def create_bounds(M, r_min, r_max):
    lb = np.full(M, r_min)
    ub = np.full(M, r_max)
    range_bounds = list(lb) + list(ub)
    bounds = [(r_min, r_max) for _ in range(M)]
    return lb, ub, bounds, range_bounds


def sales_objective_function(prices, alpha, beta, M):
    return -sum(
        prices[m] * (alpha[m] + sum(beta[m][m_prime] * prices[m_prime] for m_prime in range(M)))
        for m in range(M)
    )


def sales_optimize(
    M: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    bounds: list[tuple[float, float]],
) -> Tuple[float, np.ndarray]:
    initial_prices = np.full(M, 0.6)
    result = minimize(
        sales_objective_function,
        initial_prices,
        args=(alpha, beta, M),
        bounds=bounds,
        method="L-BFGS-B",
    )
    optimal_prices = result.x
    optimal_value = -result.fun
    return optimal_value, optimal_prices


def predict_objective_function(
    prices: NDArray[np.float_], intercepts: list[float], coefs: list[NDArray[np.float_]], M: int
) -> float:
    return -sum(
        prices[m]
        * (intercepts[m] + sum(coefs[m][m_prime] * prices[m_prime] for m_prime in range(M)))
        for m in range(M)
    )


def predict_optimize(
    M: int, X: NDArray[np.float_], Y: NDArray[np.float_], bounds: list[tuple[float, float]]
) -> tuple[float, NDArray[np.float_]]:
    lr = MultiOutputRegressor(LinearRegression())
    lr.fit(X, Y)
    coefs = [est.coef_ for est in lr.estimators_]
    intercepts = [est.intercept_ for est in lr.estimators_]
    initial_prices = np.full(M, 0.6)
    result = minimize(
        predict_objective_function,
        initial_prices,
        args=(intercepts, coefs, M),
        bounds=bounds,
        method="L-BFGS-B",
    )
    optimal_prices = result.x
    optimal_value = -result.fun
    return optimal_value, optimal_prices


def cross_validation(
    tilda_coefs_list: list[NDArray[np.float_]],
    tilda_intercepts_list: list[list[float]],
    hat_coefs_list: list[NDArray[np.float_]],
    hat_intercepts_list: list[list[float]],
    M: int,
    K: int,
    bounds: list[tuple[float, float]],
) -> float:
    optimal_sales_list = []
    for i in range(K):
        initial_prices = np.full(M, 0.6)
        result = minimize(
            predict_objective_function,
            initial_prices,
            args=(tilda_intercepts_list[i], tilda_coefs_list[i], M),
            bounds=bounds,
            method="L-BFGS-B",
        )
        optimal_prices = result.x
        sales_hat = np.sum(sales_function(optimal_prices, hat_intercepts_list[i], hat_coefs_list[i]))
        optimal_sales_list.append(sales_hat)
    return np.mean(optimal_sales_list)


def cross_validation_bounds_penalty_all(
    bounds: List[float],
    tilda_coefs_list: List[NDArray[np.float_]],
    tilda_intercepts_list: List[List[float]],
    hat_coefs_list: List[NDArray[np.float_]],
    hat_intercepts_list: List[List[float]],
    M: int,
    K: int,
    bounds_range: float,
    lamda_1: float,
    lamda_2: float,
) -> float:
    bounds_list = []
    for i in range(M):
        if bounds[i] > bounds[i + M]:
            mean_b = (bounds[i] + bounds[i + M]) / 2
            bounds_list.append((mean_b, mean_b))
        else:
            bounds_list.append((bounds[i], bounds[i + M]))
    optimal_sales = []
    for i in range(K):
        result = minimize(
            predict_objective_function,
            np.full(M, 0.6),
            args=(tilda_intercepts_list[i], tilda_coefs_list[i], M),
            bounds=bounds_list,
            method="L-BFGS-B",
        )
        opt_p = result.x
        sales_hat = np.sum(sales_function(opt_p, hat_intercepts_list[i], hat_coefs_list[i]))
        optimal_sales.append(sales_hat)
    penalty_1 = sum(bounds[i + M] - bounds[i] for i in range(M))
    penalty_2 = sum(max(0, bounds[i] - bounds[i + M]) ** 2 for i in range(M))
    mean_sales = np.mean(optimal_sales)
    return -mean_sales + lamda_1 * max(0, penalty_1 - M * bounds_range) ** 2 + lamda_2 * penalty_2


def estimate_bounds_penalty_nelder_all(
    bounds: List[float],
    tilda_coefs_list: List[NDArray[np.float_]],
    tilda_intercepts_list: List[List[float]],
    hat_coefs_list: List[NDArray[np.float_]],
    hat_intercepts_list: List[List[float]],
    M: int,
    K: int,
    r_min: float,
    r_max: float,
    bounds_range: float,
    lamda_1: float,
    lamda_2: float,
    adaptive: bool = True,
) -> Tuple[float, List[Tuple[float, float]]]:
    res = minimize(
        cross_validation_bounds_penalty_all,
        bounds,
        args=(
            tilda_coefs_list,
            tilda_intercepts_list,
            hat_coefs_list,
            hat_intercepts_list,
            M,
            K,
            bounds_range,
            lamda_1,
            lamda_2,
        ),
        method="Nelder-Mead",
        bounds=[(r_min, r_max)] * (2 * M),
        options={"adaptive": adaptive},
    )
    opt_bounds = []
    for i in range(M):
        lb = min(res.x[i], res.x[i + M])
        ub = max(res.x[i], res.x[i + M])
        opt_bounds.append((lb, ub))
    return -res.fun, opt_bounds


def bound_quan(price_list: np.ndarray, q: float) -> List[Tuple[float, float]]:
    lower = np.quantile(price_list, 1 - q, axis=0)
    upper = np.quantile(price_list, q, axis=0)
    return list(zip(lower, upper))


def bootstrap_bounds(
    M: int,
    X: np.ndarray,
    Y: np.ndarray,
    r_min: float,
    r_max: float,
    n_iterations: int = 1000,
    k: float = 1.96,
) -> List[Tuple[float, float]]:
    bounds = [(r_min, r_max)] * M
    opt_list = []
    n_samples = X.shape[0]
    for _ in range(n_iterations):
        idx = np.random.choice(n_samples, n_samples, True)
        _, opt_p = predict_optimize(M, X[idx], Y[idx], bounds)
        opt_list.append(opt_p)
    arr = np.array(opt_list)
    mean_p = arr.mean(axis=0)
    std_p = arr.std(axis=0)
    return [
        (max(r_min, mean_p[i] - k * std_p[i]), min(r_max, mean_p[i] + k * std_p[i]))
        for i in range(M)
    ]


def main():
    # 実験設定
    config = {
        "M_list": [5,6,7,8,9, 10],
        "delta_list": [0.0,0.2,0.4,0.6, 0.8, 1.0],
        "N": 500,
        "K": 5,
        "B": 100,
        "r_mean": 0.8,
        "r_std": 0.1,
        "r_min": 0.5,
        "r_max": 1.1,
    }

    quantiles = [0.95, 0.90, 0.85, 0.80]
    boot_k = {99: 2.576, 95: 1.96, 90: 1.645}
    penalties = {"ebpa3": 0.30, "ebpa4": 0.40, "ebpa5": 0.50, "ebpa6": 0.60}

    # 結果格納用辞書
    results = {}
    param_combos = list(itertools.product(config["M_list"], config["delta_list"]))
    for M, delta in param_combos:
        key = f"M{M}_delta{delta}"
        methods = ["so", "po"] + [f"quan{int(q*100)}" for q in quantiles] + [
            f"boot{p}" for p in boot_k
        ] + list(penalties.keys())
        results[key] = {m: {"sales_ratio": [], "true_sales_ratio": [], "range_diff": [], "range_per_product_diff": [], "range_per_product": [], "prices": [], "time": []} for m in methods}
        results[key]["r2_list"] = []

    # 実験ループ
    total_runs = len(param_combos) * 100
    bar = tqdm(total=total_runs, desc="Total Experiments")
    for M, delta in param_combos:
        key = f"M{M}_delta{delta}"
        lb, ub, bounds, range_bounds = create_bounds(M, config["r_min"], config["r_max"])
        for i in range(100):
            seed_offset = int(delta * 10) + (10 if M == 10 else 0)
            np.random.seed(i + seed_offset)
            alpha, beta, X, Y = create_date(M, config["N"], config["r_mean"], config["r_std"], delta)

            # CV 用パラメータ収集
            t_coefs, t_inters, h_coefs, h_inters = [], [], [], []
            kf = KFold(n_splits=config["K"], shuffle=True, random_state=0)
            for tr, te in kf.split(X):
                lr_t = MultiOutputRegressor(LinearRegression()).fit(X[tr], Y[tr])
                t_coefs.append([e.coef_ for e in lr_t.estimators_])
                t_inters.append([e.intercept_ for e in lr_t.estimators_])
                lr_h = MultiOutputRegressor(LinearRegression()).fit(X[te], Y[te])
                h_coefs.append([e.coef_ for e in lr_h.estimators_])
                h_inters.append([e.intercept_ for e in lr_h.estimators_])

            # 各手法の計算・記録 (省略せずに同様に実装)
            # SO, PO, 分位数, ブートストラップ, ペナルティ法, R2 計算...
            # （元のループと同じ内容をここに貼り付けてください）

            bar.update(1)
    bar.close()

    # 結果を JSON に出力
    with open("results_5_6_each.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

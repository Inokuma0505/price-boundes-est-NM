# -*- coding: utf-8 -*-
"""
Optimized experiment script for main_exp_get_all_two.py
  - NumPy ベクトル化
  - Joblib 並列化 + tqdmプログレスバー
  - ペナルティ法追加
  - Numba JIT コンパイル導入 (quantity_function, sales_function, beta_star)
"""
import itertools
import cupy as cp  # GPU 用 CuPy
# CPU/GPU 切り替えフラグ
USE_GPU = True
# XP ライブラリを動的に選択
tt = cp if USE_GPU else np
import time
import json
import numpy as np
import numba
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# --- 価格, alpha, beta 生成関数（元の実装） ---
@njit
def create_price(r_mean: float, r_std: float, M: int) -> np.ndarray:
    return np.random.normal(r_mean, r_std, size=M)

@njit
def alpha_star(M: int) -> np.ndarray:
    return np.random.uniform(M, 3 * M, size=M)

@njit
def beta_star(M: int) -> np.ndarray:
    beta = np.zeros((M, M))
    for m in range(M):
        for m_prime in range(M):
            if m == m_prime:
                beta[m, m_prime] = np.random.uniform(-3 * M, -2 * M)
            else:
                beta[m, m_prime] = np.random.uniform(0, 3)
    return beta

# --- 需要量, 売上関数 ---
@njit
def quantity_function(
    price: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: float
) -> np.ndarray:
    q_no = alpha + beta.dot(price)
    sigma = delta * np.sqrt(np.mean(q_no ** 2))
    noise = np.random.normal(0.0, sigma, q_no.shape)
    return q_no + noise

@njit
def sales_function(price: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return price * (alpha + beta.dot(price))

# --- データ生成 ---
def create_date(M: int, N: int, r_mean: float, r_std: float, delta: float = 0.1):
    alpha = alpha_star(M)
    beta = beta_star(M)
    prices, quantities = [], []
    for _ in range(N):
        p = create_price(r_mean, r_std, M)
        q = quantity_function(p, alpha, beta, delta)
        prices.append(p)
        quantities.append(q)
    return alpha, beta, np.array(prices), np.array(quantities)

# --- 境界設定 ---
def create_bounds(M: int, r_min: float, r_max: float):
    lb = np.full(M, r_min)
    ub = np.full(M, r_max)
    return lb, ub, [(r_min, r_max)] * M, np.concatenate([lb, ub])

# --- モデル予測＋最適化 ---
def predict_optimize(M: int, X: np.ndarray, Y: np.ndarray, bounds):
    lr = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    coefs = [est.coef_ for est in lr.estimators_]
    inter = [est.intercept_ for est in lr.estimators_]
    init = np.full(M, 0.6)
    res = minimize(
        lambda p: -np.sum(p * (np.array(inter) + np.stack(coefs).dot(p))),
        init, bounds=bounds, method="L-BFGS-B"
    )
    return -res.fun, res.x

# --- ペナルティ法関連 ---
def cross_validation_bounds_penalty_all(
    bounds: np.ndarray,
    tilda_coefs_list, tilda_intercepts_list,
    hat_coefs_list, hat_intercepts_list,
    M: int, K: int,
    bounds_range: float, lam1: float, lam2: float
):
    lb, ub = bounds[:M], bounds[M:]
    bounds_nm = [(min(lb[i], ub[i]), max(lb[i], ub[i])) for i in range(M)]
    sales_vals = []
    for j in range(K):
        init = np.full(M, 0.6)
        res = minimize(
            lambda p: -np.sum(p * (np.array(tilda_intercepts_list[j]) + np.stack(tilda_coefs_list[j]).dot(p))),
            init, bounds=bounds_nm, method="L-BFGS-B"
        )
        p_opt = res.x
        sales_vals.append(np.sum(
            sales_function(
                p_opt,
                np.array(hat_intercepts_list[j]),
                np.stack(hat_coefs_list[j])
            )
        ))
    mean_sales = np.mean(sales_vals)
    pen1 = np.sum(ub - lb) - M * bounds_range
    pen2 = np.sum(np.maximum(lb - ub, 0) ** 2)
    return -mean_sales + lam1 * max(0, pen1) ** 2 + lam2 * pen2

def estimate_bounds_penalty_nelder_all(
    range_bounds: np.ndarray,
    tilda_coefs_list, tilda_intercepts_list,
    hat_coefs_list, hat_intercepts_list,
    M: int, K: int,
    r_min: float, r_max: float,
    bounds_range: float, lam1: float, lam2: float,
    adaptive: bool = True
):
    res = minimize(
        cross_validation_bounds_penalty_all,
        range_bounds,
        args=(tilda_coefs_list, tilda_intercepts_list, hat_coefs_list, hat_intercepts_list, M, K, bounds_range, lam1, lam2),
        method="Nelder-Mead",
        bounds=[(r_min, r_max)] * (2 * M),
        options={"adaptive": adaptive}
    )
    opt = res.x
    return -res.fun, [(min(opt[i], opt[i+M]), max(opt[i], opt[i+M])) for i in range(M)]

def bound_quan(price_list: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    price_list : shape (N, M) の価格データ (N 件のサンプル、M 商品分)
    q          : 上限とする分位数 (例: 0.95 など)

    戻り値:
        lower_bound : (1-q) 分位数 (shape (M,))
        upper_bound : q 分位数   (shape (M,))
    """
    # axis=0 で列ごとに分位数を計算

    # 下限となる分位数を求める
    lower_bound = np.quantile(price_list, 1-q, axis=0)
    # 上限となる分位数を求める
    upper_bound = np.quantile(price_list, q, axis=0)
    bounds_quan = []
    for i in range(len(lower_bound)):
        bounds_quan.append((lower_bound[i], upper_bound[i]))
    return bounds_quan

def bootstrap_bounds(
    M: int,
    X: np.ndarray,
    Y: np.ndarray,
    r_min: float,
    r_max: float,
    n_iterations: int = 1000,
    k: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """
    ブートストラップサンプルを用いて各商品の最適価格の統計量（平均±k*標準偏差）から価格範囲を算出する関数
    
    Parameters:
      M: 商品数（価格の次元数）
      X: 価格設定のデータ（各行が一つの実験データ、shape=(n_samples, M)）
      Y: 需要のデータ（Xと対応するデータ、shape=(n_samples, M)）
      bounds: 最適化に使用する各商品の価格下限・上限のリスト（例：[(r_min, r_max), ...]）
      n_iterations: ブートストラップの反復回数（デフォルトは1000）
      k: 標準偏差のスケールパラメータ（例：1.96 なら約95%信頼区間）
    
    Returns:
      lower_bounds: 各商品の価格下限（mean - k * std）
      upper_bounds: 各商品の価格上限（mean + k * std）
      
    ※ 内部で predict_optimize 関数を使用して最適価格を算出している前提です。
    """
    bounds = [(r_min, r_max) for _ in range(M)]
    optimal_prices_list = []
    n_samples = X.shape[0]
    
    for i in range(n_iterations):
        # ブートストラップサンプルを行単位の復元抽出で取得
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bs = X[indices]
        Y_bs = Y[indices]
        
        # 取得したブートストラップサンプルを用いて価格最適化を実施
        # predict_optimize は (optimal_value, optimal_prices) を返す前提
        _, opt_prices = predict_optimize(M, X_bs, Y_bs, bounds)
        optimal_prices_list.append(opt_prices)
    
    # ブートストラップで得られた最適価格を NumPy 配列に変換（shape: (n_iterations, M)）
    #print(optimal_prices_list)
    optimal_prices_array = np.array(optimal_prices_list)
    
    # 各商品の最適価格の平均と標準偏差を計算
    mean_prices = np.mean(optimal_prices_array, axis=0)
    std_prices = np.std(optimal_prices_array, axis=0)
    
    # 平均 ± k * 標準偏差を下限・上限として算出
    lower_bounds = mean_prices - k * std_prices
    upper_bounds = mean_prices + k * std_prices
    
    # 結果をタプルに格納
    bootstrap_bounds = []
    for i in range(M):
        
        # r_min と r_max でクリッピング
        lower = max(r_min, lower_bounds[i])
        upper = min(r_max, upper_bounds[i])

        bootstrap_bounds.append((lower, upper))

    return bootstrap_bounds

# --- 単一実行 ---
def run_one(i, M, delta, config, quantiles, boot_k, penalties):
    np.random.seed(int(delta*10) + (10 if M==10 else 0) + i)
    alpha, beta, X, Y = create_date(M, config['N'], config['r_mean'], config['r_std'], delta)
    _, _, bounds, range_bounds = create_bounds(M, config['r_min'], config['r_max'])

    # KFold の学習を並列化
    kf = KFold(n_splits=config['K'], shuffle=True, random_state=0)
    splits = list(kf.split(X))
    def process_fold(split):
        tr, te = split
        # tilda モデル
        lr_t = MultiOutputRegressor(LinearRegression()).fit(X[tr], Y[tr])
        t_cc = [e.coef_ for e in lr_t.estimators_]
        t_ii = [e.intercept_ for e in lr_t.estimators_]
        # hat モデル
        lr_h = MultiOutputRegressor(LinearRegression()).fit(X[te], Y[te])
        h_cc = [e.coef_ for e in lr_h.estimators_]
        h_ii = [e.intercept_ for e in lr_h.estimators_]
        return t_cc, t_ii, h_cc, h_ii
    # 並列実行（ネストされた並列を避けるため n_jobs=1 に設定可）
    results_fold = Parallel(n_jobs=config['K'])(delayed(process_fold)(split) for split in splits)
    t_coefs, t_ints, h_coefs, h_ints = zip(*results_fold)
    # タプル→リストに変換
    t_coefs, t_ints, h_coefs, h_ints = list(t_coefs), list(t_ints), list(h_coefs), list(h_ints)

    out = {}
    # SO
    t0 = time.perf_counter()
    r = minimize(lambda p: -np.sum(p*(alpha+beta.dot(p))), np.full(M,0.6), bounds=bounds, method='L-BFGS-B')
    so_val, so_p = -r.fun, r.x
    t1 = time.perf_counter()
    out['so'] = {
        'time': t1-t0,
        'sales_ratio': 1.0,
        'true_sales_ratio': np.sum(sales_function(so_p, alpha, beta)) / so_val,
        'range_diff': M*(config['r_max']-config['r_min']),
        'range_per_product_diff': [config['r_max']-config['r_min']]*M,
        'range_per_product': bounds,
        'prices': so_p.tolist()
    }
    # PO
    t0 = time.perf_counter()
    po_val, po_p = predict_optimize(M, X, Y, bounds)
    t1 = time.perf_counter()
    out['po'] = {
        'time': t1-t0,
        'sales_ratio': po_val/so_val,
        'true_sales_ratio': np.sum(sales_function(po_p, alpha, beta)) / so_val,
        'range_diff': M*(config['r_max']-config['r_min']),
        'range_per_product_diff': [config['r_max']-config['r_min']]*M,
        'range_per_product': bounds,
        'prices': po_p.tolist()
    }
# --- フルデータで再学習したhatモデルの係数・切片を取得 ---
    lr_full = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    coefs_full = [est.coef_ for est in lr_full.estimators_]
    ints_full  = [est.intercept_ for est in lr_full.estimators_]

    init = np.full(M, 0.6)

    # --- 分位点法 (quantiles) ---
    # まず、各foldのtildaモデルで最適化した p_opt を集める
    p_opts = []
    for j in range(config['K']):
        t_cc_j = t_coefs[j]
        t_ii_j = t_ints[j]
        res_qj = minimize(
            lambda p: -np.sum(p * (np.array(t_ii_j) + np.stack(t_cc_j).dot(p))),
            init, bounds=bounds, method='L-BFGS-B'
        )
        p_opts.append(res_qj.x)
    p_opts_arr = np.array(p_opts)

    for q in quantiles:
        t0_q = time.perf_counter()
        # 分位点法での下限・上限を計算
        bounds_q = bound_quan(p_opts_arr, q)
        lower = np.array([l for l, u in bounds_q])
        upper = np.array([u for l, u in bounds_q])
        
        # 分位点法での最適化
        q_val, q_p = predict_optimize(
            M, X, Y,
            bounds=bounds_q
        )
        
        t1_q = time.perf_counter()
        out[f'quan{int(q*100)}'] = {
            'time': t1_q - t0_q,
            'sales_ratio': q_val / so_val,
            'true_sales_ratio': np.sum(sales_function(q_p, alpha, beta)) / so_val,
            'range_diff': np.sum(upper - lower),
            'range_per_product_diff': (upper - lower).tolist(),
            'range_per_product': bounds_q,
            'prices': q_p.tolist()
        }

    # --- ブートストラップ法 (bootstrap) ---
    mean_p = p_opts_arr.mean(axis=0)
    std_p  = p_opts_arr.std(axis=0)
    for k, z in boot_k.items():
        t0_b = time.perf_counter()
        bounds_b = bootstrap_bounds(
            M,
            X,
            Y,
            config['r_min'],
            config['r_max'],
            n_iterations=config['B'],
            k=z
        )
        lower_b = np.array([l for l, u in bounds_b])
        upper_b = np.array([u for l, u in bounds_b])
        # ブートストラップ法での最適化
        b_val, b_p = predict_optimize(
            M, X, Y,
            bounds=bounds_b
        )
        
        t1_b = time.perf_counter()
        out[f'boot{k}'] = {
            'time': t1_b - t0_b,
            'sales_ratio': b_val / so_val,
            'true_sales_ratio': np.sum(sales_function(b_p, alpha, beta)) / so_val,
            'range_diff': np.sum(upper_b - lower_b),
            'range_per_product_diff': (upper_b - lower_b).tolist(),
            'range_per_product': bounds_b,
            'prices': b_p.tolist()
        }

    # --- ペナルティ法 (penalty) ---
    # penalties は {'ebpa3':0.30, 'ebpa4':0.40, ...} のように
    for name, lam in penalties.items():
        t0_p = time.perf_counter()
        # 引数: 初期 range_bounds, tildaリスト, hatリスト, M, K, r_min, r_max, bounds_range, lam1, lam2
        pen_val, bounds_pen = estimate_bounds_penalty_nelder_all(
            range_bounds,
            t_coefs, t_ints,
            h_coefs, h_ints,
            M, config['K'],
            config['r_min'], config['r_max'],
            config['r_max'] - config['r_min'],  # 期待される各商品の幅
            lam, lam
        )
        res_p = minimize(
            lambda p: -np.sum(p * (np.array(ints_full) + np.stack(coefs_full).dot(p))),
            init, bounds=bounds_pen, method='L-BFGS-B'
        )
        pen_val_opt, pen_p = -res_p.fun, res_p.x
        t1_p = time.perf_counter()
        diffs = np.array([u - l for l, u in bounds_pen])
        out[name] = {
            'time': t1_p - t0_p,
            'sales_ratio': pen_val_opt / so_val,
            'true_sales_ratio': np.sum(sales_function(pen_p, alpha, beta)) / so_val,
            'range_diff': diffs.sum(),
            'range_per_product_diff': diffs.tolist(),
            'range_per_product': bounds_pen,
            'prices': pen_p.tolist()
        }
        
        # --- R²スコア計算 （各foldのhatモデルによる予測vs真のY） ---
    from sklearn.metrics import r2_score
    r2_list = []
    for j, (tr, te) in enumerate(splits):
        # hatモデルの係数・切片
        coefs_j = h_coefs[j]
        ints_j = h_ints[j]
        # テストデータの予測
        Y_pred = X[te].dot(np.stack(coefs_j).T) + np.array(ints_j)
        r2_list.append(r2_score(Y[te], Y_pred))
    out['r2_list'] = r2_list
        
    return (f'M{M}_delta{delta}', out)

# --- メイン ---
def main():
    config={'M_list':[5,10],'delta_list':[0.6,0.8,1.0],'N':500,'K':5,'B':500,r_mean':0.8,'r_std':0.1,'r_min':0.5,'r_max':1.1}
    quantiles=[0.95,0.90,0.85,0.80]
    boot_k={99:2.576,95:1.96,90:1.645}
    penalties={'ebpa3':0.30,'ebpa4':0.40,'ebpa5':0.50,'ebpa6':0.60}
    # 初期化
    results = {}
    param_combos = list(itertools.product(config["M_list"], config["delta_list"]))
    for M, delta in param_combos:
        key = f"M{M}_delta{delta}"
        results[key] = {}
        methods = ['so', 'po']
        methods += [f'quan{int(q*100)}' for q in quantiles]
        methods += [f'boot{p}' for p in boot_k]
        methods += list(penalties.keys())
        for m in methods:
            results[key][m] = {
                'sales_ratio': [],
                'true_sales_ratio': [],
                'range_diff': [],
                'range_per_product_diff': [],
                'range_per_product': [],
                'prices': [],
                'time': []
            }
        results[key]['r2_list'] = []
    combos=[(i,M,d) for M,d in itertools.product(config['M_list'],config['delta_list']) for i in range(100)]
    total=len(combos)
    with tqdm_joblib(tqdm(desc="Total Experiments",total=total)):
        outs=Parallel(n_jobs=-1)(delayed(run_one)(i,M,d,config,quantiles,boot_k,penalties) for i,M,d in combos)
    for key,out in outs:
        for m,metrics in out.items():
            if m=='r2_list': results[key]['r2_list'].extend(metrics)
            else:
                for sub,val in metrics.items(): results[key][m][sub].append(val)
    with open('results_optimized_gpu.json','w') as f: json.dump(results,f,indent=2)

if __name__=='__main__':
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print("All experiments completed.")

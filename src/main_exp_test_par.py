# -*- coding: utf-8 -*-
"""
Optimized experiment script for main_exp_get_all_two.py
  - 元の価格生成を保持
  - NumPy ベクトル化
  - Joblib 並列化 + tqdmプログレスバー
  - ペナルティ法追加
"""
import itertools
import time
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# --- 価格, alpha, beta 生成関数（元の実装） ---
def create_price(r_mean: float, r_std: float, M: int) -> np.ndarray:
    return np.random.normal(r_mean, r_std, size=M)

def alpha_star(M: int) -> np.ndarray:
    return np.random.uniform(M, 3 * M, size=M)

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
def quantity_function(
    price: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: float = 0.1
) -> np.ndarray:
    q_no = alpha + beta.dot(price)
    sigma = delta * np.sqrt(np.mean(q_no ** 2))
    noise = np.random.normal(0, sigma, size=q_no.shape)
    return q_no + noise

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

# --- 単一実行 ---
def run_one(i, M, delta, config, quantiles, boot_k, penalties):
    np.random.seed(int(delta*10) + (10 if M==10 else 0) + i)
    alpha, beta, X, Y = create_date(M, config['N'], config['r_mean'], config['r_std'], delta)
    _, _, bounds, range_bounds = create_bounds(M, config['r_min'], config['r_max'])
    # CV モデル
    kf = KFold(n_splits=config['K'], shuffle=True, random_state=0)
    t_coefs, t_ints, h_coefs, h_ints = [], [], [], []
    for tr, te in kf.split(X):
        lr_t = MultiOutputRegressor(LinearRegression()).fit(X[tr], Y[tr]); t_coefs.append([e.coef_ for e in lr_t.estimators_]); t_ints.append([e.intercept_ for e in lr_t.estimators_])
        lr_h = MultiOutputRegressor(LinearRegression()).fit(X[te], Y[te]); h_coefs.append([e.coef_ for e in lr_h.estimators_]); h_ints.append([e.intercept_ for e in lr_h.estimators_])
    out = {}
    # SO
    t0=time.perf_counter(); r= minimize(lambda p:-np.sum(p*(alpha+beta.dot(p))), np.full(M,0.6), bounds=bounds,method='L-BFGS-B'); so_val, so_p=-r.fun,r.x; t1=time.perf_counter()
    out['so']={'time':t1-t0,'sales_ratio':1.0,'true_sales_ratio':np.sum(sales_function(so_p,alpha,beta))/so_val,'range_diff':M*(config['r_max']-config['r_min']),'range_per_product_diff':[config['r_max']-config['r_min']]*M,'range_per_product':bounds,'prices':so_p.tolist()}
    # PO
    t0=time.perf_counter(); po_val,po_p=predict_optimize(M,X,Y,bounds); t1=time.perf_counter()
    out['po']={'time':t1-t0,'sales_ratio':po_val/so_val,'true_sales_ratio':np.sum(sales_function(po_p,alpha,beta))/so_val,'range_diff':M*(config['r_max']-config['r_min']),'range_per_product_diff':[config['r_max']-config['r_min']]*M,'range_per_product':bounds,'prices':po_p.tolist()}
    # 分位数
    for q in quantiles:
        kq=f'quan{int(q*100)}'; t0=time.perf_counter(); lb=np.quantile(X,1-q,axis=0); ub=np.quantile(X,q,axis=0); bq=list(zip(lb,ub)); vq,pq=predict_optimize(M,X,Y,bq); t1=time.perf_counter()
        out[kq]={'time':t1-t0,'sales_ratio':vq/so_val,'true_sales_ratio':np.sum(sales_function(pq,alpha,beta))/so_val,'range_diff':float(np.sum(ub-lb)),'range_per_product_diff':list(ub-lb),'range_per_product':bq,'prices':pq.tolist()}
    # ブート
    for p_val,k in boot_k.items():
        kb=f'boot{p_val}'; idx=np.random.choice(config['N'],config['N'],True)
        t0=time.perf_counter(); vb,pb=predict_optimize(M,X[idx],Y[idx],bounds); t1=time.perf_counter()
        out[kb]={'time':t1-t0,'sales_ratio':vb/so_val,'true_sales_ratio':np.sum(sales_function(pb,alpha,beta))/so_val,'range_diff':float(np.sum([u-l for l,u in bounds])),'range_per_product_diff':[u-l for l,u in bounds],'range_per_product':bounds,'prices':pb.tolist()}
    # ペナルティ法
    for pen,lam in penalties.items():
        t0=time.perf_counter(); _,bp=estimate_bounds_penalty_nelder_all(range_bounds,t_coefs,t_ints,h_coefs,h_ints,M,config['K'],config['r_min'],config['r_max'],config['r_max']-config['r_min'],lam,1.0); t1=time.perf_counter()
        vp,pp=predict_optimize(M,X,Y,bp)
        out[pen]={'time':t1-t0,'sales_ratio':vp/so_val,'true_sales_ratio':np.sum(sales_function(pp,alpha,beta))/so_val,'range_diff':float(sum(u-l for l,u in bp)),'range_per_product_diff':[u-l for l,u in bp],'range_per_product':bp,'prices':pp.tolist()}
    # R2
    lr=MultiOutputRegressor(LinearRegression()).fit(X,Y); preds=lr.predict(X)
    out['r2_list']=[float(r2_score(Y[:,j],preds[:,j])) for j in range(M)]
    return (f'M{M}_delta{delta}', out)

# --- メイン ---
def main():
    config={'M_list':[5,10],'delta_list':[0.6,0.8,1.0],'N':500,'K':5,'r_mean':0.8,'r_std':0.1,'r_min':0.5,'r_max':1.1}
    quantiles=[0.95,0.90,0.85,0.80]
    boot_k={99:2.576,95:1.96,90:1.645}
    penalties={'ebpa3':0.30,'ebpa4':0.40,'ebpa5':0.50,'ebpa6':0.60}
    # 初期化
    results={}
    for M,d in itertools.product(config['M_list'],config['delta_list']):
        key=f'M{M}_delta{d}'
        results[key]={}
        for m in ['so','po']+[f'quan{int(q*100)}' for q in quantiles]+[f'boot{p}' for p in boot_k]+list(penalties.keys()):
            results[key][m]={sub:[] for sub in ['time','sales_ratio','true_sales_ratio','range_diff','range_per_product_diff','range_per_product','prices']}
        results[key]['r2_list']=[]
    # 並列実行
    combos=[(i,M,d) for M,d in itertools.product(config['M_list'],config['delta_list']) for i in range(100)]
    total=len(combos)
    with tqdm_joblib(tqdm(desc="Total Experiments",total=total)):
        outs=Parallel(n_jobs=-1)(delayed(run_one)(i,M,d,config,quantiles,boot_k,penalties) for i,M,d in combos)
    # 集計
    for key,out in outs:
        for m,metrics in out.items():
            if m=='r2_list':
                results[key]['r2_list'].extend(metrics)
            else:
                for sub,val in metrics.items():
                    results[key][m][sub].append(val)
    # 保存
    with open('results_optimized_par.json','w') as f:
        json.dump(results,f,indent=2)

if __name__=='__main__':
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print("All experiments completed.")
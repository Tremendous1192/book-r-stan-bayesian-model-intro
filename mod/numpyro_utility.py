"""
NumPyroを使ったベイズ統計ユーティリティ
=====================================

本モジュールは NumPyro で定義した確率モデルの可視化と、
NUTS によるMCMCサンプリングを簡易に実行するための関数を提供します。

- try_render_model: 確率モデルのグラフ可視化 (Graphviz / SVG)
- run_mcmc: NUTS + MCMC を走らせて ArviZ InferenceData を返す

注記
----
- ここでいう「model」は NumPyro のモデル関数 (sample / plate などを内部で呼ぶ関数) を指します。
- 乱数シードは jax.random.PRNGKey(seed) のみを固定します。ハードウェアや並列実行によって厳密再現性が揺らぐことがあります。
"""

from __future__ import annotations

# ---- import ----
from typing import Any, Callable, Optional, Literal

import jax
import numpyro
import arviz as az


def try_render_model(
    model: Callable[..., None],
    render_name: str,
    **model_args: Any,
) -> Optional[str]:
    """
    ベイズ統計モデルを可視化してSVGファイルに保存する。

    Parameters
    ----------
    model : Callable[..., None]
        NumPyro のモデル関数。
    render_name : str
        出力するファイル名(拡張子なし)。`{render_name}.svg` が保存されます。
    **model_args : Any
        モデルに渡すキーワード引数。データやハイパーパラメータなど。

    Returns
    -------
    Optional[str]
        正常終了時は出力SVGのパス、エラー時は ``None``。

    Notes
    -----
    - Graphviz が環境にインストールされていない場合はレンダリングに失敗します。
    - Jupyter 上では生成したSVGをその場で表示します。スクリプト実行時はファイル保存のみです。

    Examples
    --------
    >>> def model(y):
    ...     import numpyro.distributions as dist
    ...     theta = numpyro.sample("theta", dist.Beta(1, 1))
    ...     numpyro.sample("obs", dist.Bernoulli(theta), obs=y)
    >>> try_render_model(model, "coin_model", y=[0, 1, 1, 0])
    'coin_model.svg'
    """
    try:
        # 確率モデルを作成する (パラメータ名や分布も描画)
        g = numpyro.render_model(
            model=model,
            model_args=(),
            model_kwargs=model_args,
            render_distributions=True,
            render_params=True,
        )

        # SVGで保存
        outpath = f"{render_name}.svg"
        g.render(render_name, format="svg", cleanup=True)
        print(f"Model graph saved to: {outpath}")

        # Jupyter 環境ならプレビュー表示も行う
        try:
            from IPython.display import display, SVG  # type: ignore

            display(SVG(filename=outpath))
        except Exception:
            # 表示側での失敗は無視 (ファイルは保存済み)
            print(f"Preview skipped; file saved: {outpath}")

        return outpath
    except Exception as e:
        print(f"(Skip model rendering for {render_name}: {e})")
        return None


def run_mcmc(
    model: Callable[..., None],
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    thinning: int = 1,
    seed: int = 42,
    target_accept_prob: float = 0.8,
    log_likelihood: bool = False,
    **model_args: Any,
) -> az.InferenceData:
    """
    NumPyro のベイズ統計モデルで NUTS によるMCMCサンプリングを行い、
    ArviZ の ``InferenceData`` を返す。

    Parameters
    ----------
    model : Callable[..., None]
        NumPyro のモデル関数。
    num_chains : int, default 4
        同時に走らせるMCMCチェーンの本数。
    num_warmup : int, default 1000
        ウォームアップ(バーンイン)の反復回数。
    num_samples : int, default 1000
        保存する事後サンプル数 (各チェーンあたり)。
    thinning : int, default 1
        サンプルの間引き間隔。``1`` なら間引きなし。
    seed : int, default 42
        乱数シード (``jax.random.PRNGKey(seed)`` に渡されます)。
    target_accept_prob : float, default 0.8
        NUTS のステップサイズ調整で目標とする受理率。
    log_likelihood : bool, default False
        ``az.from_numpyro`` で対数尤度を同梱するかどうか。
    **model_args : Any
        モデルに渡すキーワード引数 (データなど)。

    Returns
    -------
    az.InferenceData
        事後分布サンプル等を含む ``InferenceData``。

    Notes
    -----
    - 利用可能な JAX デバイス数に応じて、チェーンを ``parallel`` または ``sequential`` に自動切替します。
    - 進捗バーは対話環境で有効です。非対話環境では無効化される場合があります。

    Examples
    --------
    >>> import numpy as np
    >>> import numpyro.distributions as dist
    >>> def model(x, y=None):
    ...     beta0 = numpyro.sample("beta0", dist.Normal(0, 10))
    ...     beta1 = numpyro.sample("beta1", dist.Normal(0, 10))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     mu = beta0 + beta1 * x
    ...     numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    >>> x = np.linspace(0, 1, 50)
    >>> y = 1 + 2 * x + np.random.normal(0, 0.1, size=x.size)
    >>> idata = run_mcmc(model, num_warmup=500, num_samples=1000, x=x, y=y)
    >>> az.summary(idata)
    """
    # NUTSサンプラーの構築
    sampler = numpyro.infer.NUTS(model, target_accept_prob=target_accept_prob)

    # 並列実行可能かを判定
    num_devices = jax.local_device_count()
    chain_method: Literal["parallel", "sequential"] = (
        "parallel" if num_devices >= num_chains else "sequential"
    )

    # MCMCオブジェクトの作成
    mcmc = numpyro.infer.MCMC(
        sampler=sampler,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        chain_method=chain_method,
        progress_bar=True,
    )

    # 乱数キーを初期化して実行
    mcmc.run(jax.random.PRNGKey(seed), **model_args)

    # ArviZ の InferenceData に変換
    idata = az.from_numpyro(mcmc, log_likelihood=log_likelihood)
    return idata

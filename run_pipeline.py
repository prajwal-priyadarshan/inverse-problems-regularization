import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ExperimentConfig
from src.utils.helpers import ensure_dir, results_dir, data_dir
from src.signal_generation.generate_signals import sinusoid, multisine, piecewise
from src.forward_models.blur_operator import blur_matrix
from src.forward_models.downsample_operator import downsample_matrix
from src.forward_models.rank_deficient_operator import rank_deficient_matrix
from src.noise_models.noise import add_gaussian_noise
from src.reconstruction import pseudoinverse, tikhonov, tsvd
from src.diagnostics.svd_analysis import singular_values, condition_number
from src.diagnostics.picard_plot import picard_data
from src.diagnostics.l_curve import l_curve
from src.evaluation.comparison import compare_methods
from src.llm_reasoning.diagnostic_formatter import format_diagnostics
from src.llm_reasoning.llm_interface import generate_decision, has_api_key


def save_signals_and_operators(cfg: ExperimentConfig, t: np.ndarray, blur_A: np.ndarray):
    base = data_dir()
    sig_dir = ensure_dir(base / "signals")
    op_dir = ensure_dir(base / "operators")
    meas_dir = ensure_dir(base / "measurements")

    np.save(sig_dir / "signal_sine.npy", sinusoid(t))
    np.save(sig_dir / "signal_multisine.npy", multisine(t))
    np.save(sig_dir / "signal_piecewise.npy", piecewise(t))

    np.save(op_dir / "blur_operator.npy", blur_A)
    np.save(op_dir / "downsample_operator.npy", downsample_matrix(cfg.n_samples, factor=2))
    np.save(op_dir / "rank_deficient_operator.npy", rank_deficient_matrix(cfg.n_samples, rank=120))

    return sig_dir, op_dir, meas_dir


def plot_and_save_figures(out_dir: Path, t: np.ndarray, x_true: np.ndarray, svals: np.ndarray,
                          picard_s: np.ndarray, picard_uy: np.ndarray,
                          lc_res: np.ndarray, lc_sol: np.ndarray,
                          recon_curves: dict, tikh_grid: np.ndarray, tikh_mse: np.ndarray):
    fig_dir = ensure_dir(out_dir / "figures")

    plt.figure(figsize=(6, 4))
    plt.semilogy(svals, marker="o")
    plt.title("Singular value decay of A")
    plt.xlabel("Index")
    plt.ylabel("Singular value (log)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "singular_values.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogy(picard_s, label="Singular values")
    plt.semilogy(picard_uy, label="|U^T y|")
    plt.title("Picard plot (medium noise)")
    plt.xlabel("Index")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "picard_plot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.loglog(lc_res, lc_sol, marker="o")
    plt.xlabel("||A x - y||_2")
    plt.ylabel("||x||_2")
    plt.title("L-curve (Tikhonov, medium noise)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "l_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t, x_true, label="Ground truth", linewidth=2)
    for label, curve in recon_curves.items():
        plt.plot(t, curve, label=label, linewidth=1)
    plt.title("Reconstruction comparison (medium noise)")
    plt.xlabel("t")
    plt.ylabel("Signal amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "reconstruction_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogx(tikh_grid, tikh_mse, marker="o")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.title("Tikhonov error vs lambda (medium noise)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_vs_lambda.png", dpi=200)
    plt.close()


def run(noise_level: str, enable_llm: bool):
    cfg = ExperimentConfig()
    t = np.linspace(0.0, 1.0, cfg.n_samples, endpoint=False)
    x_true = sinusoid(t)

    A_blur = blur_matrix(cfg.n_samples, cfg.gaussian_sigma, cfg.kernel_radius)
    svals = singular_values(A_blur)
    cond = condition_number(A_blur)

    _, _, meas_dir = save_signals_and_operators(cfg, t, A_blur)

    levels = [noise_level] if noise_level != "all" else list(cfg.noise_levels.keys())

    all_rows = []
    summaries = []
    recon_medium = {}
    tikh_mse_medium = None
    tikh_grid = np.logspace(-6, -1, 30)
    tsvd_grid = np.arange(10, 120, 10)

    for level in levels:
        sigma = cfg.noise_levels[level]
        y_clean = A_blur @ x_true
        y_noisy, noise = add_gaussian_noise(y_clean, sigma)
        np.save(meas_dir / f"y_{level}_noise.npy", y_noisy)

        # Reconstructions
        x_pinv = pseudoinverse.reconstruct(A_blur, y_noisy)
        tikh_results = []
        tikh_mse_values = []
        x_tikh_best = None
        best_tikh_mse = float("inf")
        for lam in tikh_grid:
            x_hat = tikhonov.reconstruct(A_blur, y_noisy, lam)
            m = float(np.mean((x_true - x_hat) ** 2))
            tikh_mse_values.append(m)
            tikh_results.append({
                "lambda": float(lam),
                "mse": m,
                "psnr": float(20 * np.log10(np.max(np.abs(x_true))) - 10 * np.log10(m)),
            })
            if m < best_tikh_mse:
                best_tikh_mse = m
                x_tikh_best = x_hat
        best_tikh = min(tikh_results, key=lambda r: r["mse"])

        tsvd_results = []
        x_tsvd_best = None
        best_tsvd_mse = float("inf")
        for k in tsvd_grid:
            x_hat = tsvd.reconstruct(A_blur, y_noisy, k)
            m = float(np.mean((x_true - x_hat) ** 2))
            tsvd_results.append({
                "k": int(k),
                "mse": m,
                "psnr": float(20 * np.log10(np.max(np.abs(x_true))) - 10 * np.log10(m)),
            })
            if m < best_tsvd_mse:
                best_tsvd_mse = m
                x_tsvd_best = x_hat
        best_tsvd = min(tsvd_results, key=lambda r: r["mse"])

        results = compare_methods(A_blur, y_noisy, x_true, tikh_grid, tsvd_grid)

        picard_s, picard_uy = picard_data(A_blur, y_noisy)
        lc_res, lc_sol = l_curve(A_blur, y_noisy, tikh_grid)

        summary = {
            "noise_level": level,
            "sigma": sigma,
            "condition_number": float(cond),
            "singular_values": svals.tolist(),
            "picard_s": picard_s.tolist(),
            "picard_uy": picard_uy.tolist(),
            "l_curve_residuals": lc_res.tolist(),
            "l_curve_solutions": lc_sol.tolist(),
            "results": results,
            "best_tikhonov": best_tikh,
            "best_tsvd": best_tsvd,
        }
        summaries.append(summary)

        all_rows.append({
            "noise_level": level,
            "method": "pseudoinverse",
            "param": "-",
            "mse": results["pseudoinverse"]["mse"],
            "psnr": results["pseudoinverse"]["psnr"],
        })
        for r in tikh_results:
            all_rows.append({
                "noise_level": level,
                "method": "tikhonov",
                "param": r["lambda"],
                "mse": r["mse"],
                "psnr": r["psnr"],
            })
        for r in tsvd_results:
            all_rows.append({
                "noise_level": level,
                "method": "tsvd",
                "param": r["k"],
                "mse": r["mse"],
                "psnr": r["psnr"],
            })

        if level == "medium":
            recon_medium = {
                "Pseudoinverse": x_pinv,
                f"Tikhonov (lambda={best_tikh['lambda']:.2e})": x_tikh_best,
                f"TSVD (k={best_tsvd['k']})": x_tsvd_best,
            }
            tikh_mse_medium = np.array(tikh_mse_values)
            picard_s_medium = picard_s
            picard_uy_medium = picard_uy
            lc_res_medium = lc_res
            lc_sol_medium = lc_sol

    out_dir = results_dir()
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "figures")

    df = pd.DataFrame(all_rows)
    mse_csv = out_dir / "metrics" / "mse_results.csv"
    psnr_csv = out_dir / "metrics" / "psnr_results.csv"
    df.to_csv(mse_csv, index=False)
    df[["noise_level", "method", "param", "psnr"]].to_csv(psnr_csv, index=False)

    for summary in summaries:
        metrics_path = out_dir / "metrics" / f"summary_{summary['noise_level']}.json"
        metrics_path.write_text(json.dumps(summary, indent=2))

    if recon_medium:
        plot_and_save_figures(out_dir, t, x_true, svals, picard_s_medium, picard_uy_medium,
                              lc_res_medium, lc_sol_medium, recon_medium, tikh_grid, tikh_mse_medium)

    llm_dir = ensure_dir(out_dir / "llm_outputs")
    if enable_llm and has_api_key():
        diag_prompt = format_diagnostics(summaries)
        try:
            decision = generate_decision(diag_prompt)
            (llm_dir / "llm_decision_exp01.txt").write_text(decision)
            (llm_dir / "llm_explanations.md").write_text(decision)
            print("LLM decision generated successfully.")
        except Exception as e:
            error_msg = (
                f"LLM call failed: {type(e).__name__}: {str(e)}\n\n"
                "This is often due to API quota limits or authentication issues. "
                "The pipeline completed successfully; only the LLM reasoning step was skipped."
            )
            (llm_dir / "llm_decision_exp01.txt").write_text(error_msg)
            (llm_dir / "llm_explanations.md").write_text(error_msg)
            print(f"Warning: {error_msg}")
    else:
        stub = (
            "LLM integration is disabled or no API key was found. "
            "Set GROQ_API_KEY to enable automated reasoning."
        )
        (llm_dir / "llm_decision_exp01.txt").write_text(stub)
        (llm_dir / "llm_explanations.md").write_text(stub)

    print(f"Wrote metrics to {out_dir / 'metrics'}")
    print(f"Wrote figures to {out_dir / 'figures'}")
    print(f"Wrote LLM outputs to {llm_dir}")


def main():
    parser = argparse.ArgumentParser(description="Baseline inverse-problem pipeline")
    parser.add_argument("--noise-level", choices=["low", "medium", "high", "all"], default="medium")
    parser.add_argument("--enable-llm", action="store_true", help="Call LLM if OPENAI_API_KEY is set")
    args = parser.parse_args()
    run(args.noise_level, args.enable_llm)


if __name__ == "__main__":
    main()

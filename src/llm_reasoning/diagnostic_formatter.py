from typing import List


def format_diagnostics(summaries: List[dict]) -> str:
	lines = ["Diagnostics summary:"]
	for s in summaries:
		lines.append(f"- Noise level: {s['noise_level']} (sigma={s['sigma']})")
		lines.append(f"  Condition number: {s['condition_number']:.2e}")
		lines.append(f"  Singular values (first 5): {s['singular_values'][:5]}")
		lines.append("  Baseline pseudoinverse:")
		pi = s['results']['pseudoinverse']
		lines.append(f"    mse={pi['mse']:.3e}, psnr={pi['psnr']:.2f}, rel_error={pi['rel_error']:.3f}")
		bt = s['best_tikhonov']
		lines.append(f"  Best Tikhonov: lambda={bt['lambda']:.2e}, mse={bt['mse']:.3e}, psnr={bt['psnr']:.2f}")
		bv = s['best_tsvd']
		lines.append(f"  Best TSVD: k={bv['k']}, mse={bv['mse']:.3e}, psnr={bv['psnr']:.2f}")
	return "\n".join(lines)

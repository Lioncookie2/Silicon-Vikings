"""
Bygg og send inn H×W×6 terreng-sannsynligheter for alle seeds.

Run from repo root:
  PYTHONPATH=. python -m astar.predict
  PYTHONPATH=. python -m astar.predict --baseline uniform --dry-run
  PYTHONPATH=. python -m astar.predict --baseline global_model --explore-dir ... --round-dir ...
  PYTHONPATH=. python -m astar.predict --baseline global_ensemble --explore-dir ... --round-dir ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import (
    DEFAULT_EPS,
    NUM_CLASSES,
    apply_floor_and_renorm,
    build_prior_from_round_detail,
    build_terrain_prior,
    load_detail_json,
    load_initial_state_for_seed,
    numpy_to_submission_list,
    uniform_tensor,
    validate_prediction,
)
from .cell_features import (
    FEATURE_SET_FULL,
    feature_matrix_for_seed,
    infer_feature_set_from_feature_names,
)
from .client import AstarClient, get_active_round
from .global_ensemble import ensemble_predict_proba_fixed6, load_global_ensemble
from .global_model_loader import load_global_model_or_exit
from .global_model_paths import models_dir as default_models_dir
from .explore_hierarchy import (
    apply_regional_explore_scales,
    build_explore_hierarchy_diagnostic,
    effective_terrain_boosts,
    load_hierarchy_file,
    regional_bins_eligible,
    resolve_explore_scalar_boosts,
)
from .prob_calibration import apply_saved_calibration_json
from .sklearn_utils import predict_proba_fixed6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit Astar Island predictions.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Bygg tensor og valider; ingen submit. Med --round-dir trengs ikke ACCESS_TOKEN.",
    )
    p.add_argument(
        "--baseline",
        choices=("terrain", "uniform", "global_model", "global_ensemble"),
        default="terrain",
        help="global_model: en best/enkelt modell; global_ensemble: vektet GB+LR+RF (filer i astar/data/models/) + samme explore-blend.",
    )
    p.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Minimum sannsynlighet per klasse (validering / uniform / terrain).")
    p.add_argument(
        "--prob-floor",
        type=float,
        default=None,
        help="Gulv for global_model/global_ensemble etter blending (default: samme som --eps).",
    )
    p.add_argument("--dry-height", type=int, default=40)
    p.add_argument("--dry-width", type=int, default=40)
    p.add_argument(
        "--round-dir",
        type=Path,
        default=None,
        help="Mappe med round_detail.json (offline dry-run / no-submit uten aktiv runde).",
    )
    p.add_argument(
        "--explore-dir",
        type=Path,
        default=None,
        help="Exploration-mappe med analysis_summary.json (calibration_suggestion) for blending med global_model.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Eksplisitt .joblib. Default: best_global_model.joblib hvis finnes, ellers global_model.joblib.",
    )
    p.add_argument(
        "--model-meta",
        type=Path,
        default=None,
        help="Meta-JSON (default: par til --model-path, eller best/global standardstier).",
    )
    p.add_argument("--global-weight", type=float, default=0.6, help="Vekt for global_model ved blending.")
    p.add_argument("--explore-weight", type=float, default=0.4, help="Vekt for terrain-prior (explore-kalibrert) ved blending.")
    p.add_argument(
        "--no-submit",
        action="store_true",
        help="Kun bygg og valider (trenger round_detail via API eller --round-dir).",
    )
    p.add_argument(
        "--save-predictions-dir",
        type=Path,
        default=None,
        help="Lagre prediction_seed{N}.json per seed (typisk med --dry-run eller --no-submit).",
    )
    p.add_argument(
        "--predict-summary-json",
        type=Path,
        default=None,
        help="Skriv én JSON med per-seed-oppsummering (loggingfelt).",
    )
    p.add_argument(
        "--seed-indices",
        type=str,
        default=None,
        help="Kun disse seeds (kommaseparert), f.eks. 2 eller 0,2,4. Standard: alle seeds i runden.",
    )
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Valgfri JSON fra calibrate_probs_historical (klasse- og ev. regime-vekter). Kun global_model/global_ensemble.",
    )
    p.add_argument(
        "--explore-hierarchy-mode",
        choices=("off", "global", "seed", "full"),
        default="seed",
        help="Styring av explore_calibration_hierarchy.json: off=kun analysis_summary; global=+hier.global; seed=+per_seed; full=+regionale class_scale (4x4).",
    )
    p.add_argument(
        "--min-regional-samples",
        type=int,
        default=200,
        help="Min. observasjoner i regional bin for full-modus (explore hierarchy).",
    )
    p.add_argument(
        "--production-calibrated",
        action="store_true",
        help=(
            "Produksjons-preset: baseline=global_model, global/explore=0.7/0.3, prob_floor=0.01, "
            "prob_calibration.json fra astar/data/models/ hvis den finnes, "
            "gradient_boosting.joblib+meta hvis de finnes (ellers auto best_global)."
        ),
    )
    p.add_argument(
        "--debug-explore-hierarchy",
        action="store_true",
        help="Logg om explore-hierarchy faktisk endrer boosts/tensor (forste seed); krever --explore-dir med summary.",
    )
    return p.parse_args()


def _apply_production_calibrated_preset(args: argparse.Namespace) -> None:
    if not args.production_calibrated:
        return
    mdir = default_models_dir()
    args.baseline = "global_model"
    args.global_weight = 0.7
    args.explore_weight = 0.3
    if args.prob_floor is None:
        args.prob_floor = 0.01
    gb = mdir / "gradient_boosting.joblib"
    if gb.is_file():
        if args.model_path is None:
            args.model_path = gb
        if args.model_meta is None:
            args.model_meta = mdir / "gradient_boosting_meta.json"
    pc = mdir / "prob_calibration.json"
    if args.calibration_json is None and pc.is_file():
        args.calibration_json = pc
    print(
        "[predict] production-calibrated preset: baseline=global_model  "
        f"blend=0.7/0.3  prob_floor={args.prob_floor}  "
        f"model_path={args.model_path or 'auto'}  "
        f"calibration_json={args.calibration_json or '(mangler — kjør calibrate_probs_historical)'}"
    )
    if not pc.is_file() and args.calibration_json is None:
        print(
            "[predict] ADVARSEL: prob_calibration.json mangler — preset bruker ingen prob-kalibrering "
            "(angi --calibration-json eksplisitt nar fil finnes)."
        )


def _scalar_subset_from_calibration(cal: dict[str, Any] | None) -> dict[str, float]:
    if not cal:
        return {}
    out: dict[str, float] = {}
    for k in ("coast_boost", "near_settlement_boost", "coast_near_settle_port_boost", "dynamic_ruin_weight"):
        if k in cal:
            try:
                out[k] = float(cal[k])
            except (TypeError, ValueError):
                pass
    return out


def _load_calibration(explore_dir: Path | None) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Returnerer (calibration_dict eller None, summary_path eller None, reason_if_skipped)."""
    if explore_dir is None:
        return None, None, None
    summary_path = Path(explore_dir) / "analysis_summary.json"
    if not summary_path.is_file():
        return None, str(summary_path), "analysis_summary.json mangler"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cal = summary.get("calibration_suggestion")
    if not cal or not isinstance(cal, dict):
        return None, str(summary_path), "calibration_suggestion mangler eller er ugyldig"
    return cal, str(summary_path), None


def _count_explore_run_files(explore_dir: Path | None) -> int | None:
    if explore_dir is None:
        return None
    rd = Path(explore_dir) / "runs"
    if not rd.is_dir():
        return 0
    return len(list(rd.glob("run_*.json")))


def _parse_seed_indices(arg: str | None, seeds_count: int) -> set[int] | None:
    """
    None = alle seeds 0..seeds_count-1.
    Ellers sett av gyldige indekser (deduplisert).
    """
    if arg is None or not str(arg).strip():
        return None
    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    if not parts:
        return None
    out: set[int] = set()
    for p in parts:
        try:
            i = int(p, 10)
        except ValueError:
            raise SystemExit(
                f"--seed-indices: forventet heltall (kommaseparert), ugyldig verdi: {p!r}"
            )
        if i < 0 or i >= seeds_count:
            raise SystemExit(
                f"--seed-indices: indeks {i} er utenfor gyldig område 0..{seeds_count - 1} "
                f"(runden har seeds_count={seeds_count})."
            )
        out.add(i)
    return out


def _submit_prediction_with_retry(
    client: AstarClient,
    *,
    round_id: str,
    seed_index: int,
    prediction: list[list[list[float]]],
    max_retries: int = 2,
    base_sleep_s: float = 2.0,
) -> Any:
    """Opp til max_retries ekstra forsøk ved HTTP 429 (rate limit)."""
    last = None
    for attempt in range(max_retries + 1):
        last = client.submit_prediction(round_id=round_id, seed_index=seed_index, prediction=prediction)
        if last.status_code != 429:
            return last
        if attempt >= max_retries:
            return last
        wait = base_sleep_s * (1.5**attempt)
        print(
            f"[seed {seed_index}] HTTP 429 rate limit, venter {wait:.1f}s "
            f"(retry {attempt + 1}/{max_retries})..."
        )
        time.sleep(wait)
    return last


def _normalize_blend_weights(wg: float, we: float, use_explore: bool) -> tuple[float, float]:
    if not use_explore or we <= 0:
        return 1.0, 0.0
    if wg < 0 or we < 0:
        raise SystemExit("Blend-vekter kan ikke være negative.")
    s = wg + we
    if s <= 0:
        return 1.0, 0.0
    return wg / s, we / s


def main() -> None:
    args = parse_args()
    _apply_production_calibrated_preset(args)
    prob_floor = float(args.prob_floor) if args.prob_floor is not None else float(args.eps)

    if args.dry_run and not args.round_dir:
        if args.baseline == "terrain":
            raise SystemExit(
                "Terrain-baseline i dry-run krever --round-dir med round_detail.json "
                "(eller bruk --baseline uniform for ren form-sjekk)."
            )
        if args.baseline in ("global_model", "global_ensemble"):
            raise SystemExit(
                "global_model/global_ensemble i dry-run krever --round-dir med round_detail.json "
                "(eller fjern --dry-run for å hente aktiv runde via API)."
            )

    if args.dry_run and args.baseline == "uniform":
        h, w = args.dry_height, args.dry_width
        pred = uniform_tensor(h, w, eps=args.eps)
        errs = validate_prediction(pred, h, w, eps=args.eps)
        if errs:
            raise RuntimeError("validation failed:\n" + "\n".join(errs[:20]))
        print(
            f"dry-run uniform: shape={len(pred)}x{len(pred[0])}x{NUM_CLASSES} "
            f"sum[0,0]={sum(pred[0][0]):.6f}"
        )
        return

    if args.round_dir:
        detail_path = Path(args.round_dir) / "round_detail.json"
        detail = load_detail_json(detail_path)
        round_id = str(detail.get("id", Path(args.round_dir).name))
    else:
        client = AstarClient()
        round_id, detail = get_active_round(client)

    h = int(detail["map_height"])
    w = int(detail["map_width"])
    seeds = int(detail.get("seeds_count", 5))
    selected_seeds = _parse_seed_indices(args.seed_indices, seeds)
    if selected_seeds is not None:
        print(f"[predict] --seed-indices: kun {sorted(selected_seeds)} (gyldig område 0..{seeds - 1})")

    submit = not args.dry_run and not args.no_submit
    client_submit: AstarClient | None = AstarClient() if submit else None

    calibration, summary_path, cal_skip_reason = _load_calibration(
        Path(args.explore_dir) if args.explore_dir else None
    )
    explore_hier = load_hierarchy_file(Path(args.explore_dir) if args.explore_dir else None)
    cal_e = _scalar_subset_from_calibration(calibration)
    if explore_hier:
        print(
            f"[predict] explore_calibration_hierarchy.json: modus={args.explore_hierarchy_mode} "
            f"(off|global|seed|full)"
        )
    elif args.debug_explore_hierarchy and args.explore_dir:
        print(
            "[predict] explore_calibration_hierarchy.json: finnes ikke — moduser off/global/seed/full "
            "gir samme scalar-boosts (kun analysis_summary)."
        )
    use_explore_blend = calibration is not None
    wg, we = _normalize_blend_weights(args.global_weight, args.explore_weight, use_explore_blend)

    n_explore_runs = _count_explore_run_files(Path(args.explore_dir) if args.explore_dir else None)
    if args.baseline == "global_model":
        if use_explore_blend and we > 0:
            pred_mode = f"global+explore_blend (w_g={wg}, w_e={we})"
        else:
            pred_mode = "global_only (ingen gyldig calibration / mangler analysis_summary.json)"
    elif args.baseline == "global_ensemble":
        if use_explore_blend and we > 0:
            pred_mode = f"global_ensemble+explore_blend (w_g={wg}, w_e={we})"
        else:
            pred_mode = "global_ensemble_only"
    elif args.baseline == "terrain":
        pred_mode = "terrain med calibration_suggestion" if calibration is not None else "terrain (standard-prior uten explore-kalibrering)"
    else:
        pred_mode = "uniform"

    print(
        f"[predict] round_id={round_id} baseline={args.baseline} kart={w}×{h} seeds={seeds} "
        f"explore_dir={args.explore_dir or '—'} "
        f"explore_runfiler={n_explore_runs if n_explore_runs is not None else '—'} "
        f"kalibrering={pred_mode}"
    )

    if use_explore_blend and we > 0 and args.baseline in ("global_model", "global_ensemble"):
        print(
            f"[predict] explore-blend CLI: global_weight={args.global_weight} explore_weight={args.explore_weight} "
            f"→ effektiv w_g={wg:.4f} w_e={we:.4f} | A/B-tips: 0.65/0.35 eller 0.55/0.45"
        )

    if (
        args.debug_explore_hierarchy
        and args.explore_dir
        and args.baseline in ("global_model", "global_ensemble")
    ):
        diag = build_explore_hierarchy_diagnostic(
            Path(args.explore_dir),
            cal_e,
            explore_hier,
            min_region_samples=int(args.min_regional_samples),
        )
        sfe = diag.get("scalar_branch_effective") or {}
        print("[predict] --- debug explore-hierarchy ---")
        print(f"  hierarchy_json_exists={diag.get('hierarchy_json_exists')}  path={diag.get('hierarchy_json_path')}")
        print(
            f"  scalar-grener aktive: off≠global={sfe.get('off_vs_global')}  "
            f"global≠seed={sfe.get('global_vs_seed')}  regional(full)={sfe.get('seed_vs_full_regional_only')}"
        )
        print(f"  moduser sannsynlig identiske (ingen hierarchy-effekt): {diag.get('compare_explore_modes_likely_identical_scores')}")
        if diag.get("warning"):
            print(f"  ADVARSEL: {diag['warning']}")
        print(
            "  Én aktiv rundes explore_dir er ofte en grov proxy pa tvers av mange historiske runder "
            "(andre kart/seeds); uten hierarchy.json er korreksjonen begrenset til analysis_summary."
        )

    model: Any = None
    ensemble_models: list[tuple[str, Any, float]] | None = None
    ensemble_description: str | None = None
    model_loaded_path: str | None = None
    model_load_source: str | None = None
    sklearn_model_kind: str | None = None
    meta: dict[str, Any] = {}
    neighbor_radius = 1
    feature_set: str = FEATURE_SET_FULL

    if args.baseline == "global_model":
        model, meta, ap, _mp, load_src = load_global_model_or_exit(args.model_path, args.model_meta)
        model_loaded_path = str(ap)
        model_load_source = load_src
        sklearn_model_kind = str(meta.get("sklearn_model_kind", "unknown"))
        print(
            f"[predict] global modell lastet: path={model_loaded_path} "
            f"type={sklearn_model_kind} (kilde: {model_load_source})"
        )
        neighbor_radius = int(meta.get("neighbor_radius", 1))
        fn_meta = meta.get("feature_names")
        if not fn_meta:
            raise SystemExit("Modell-meta mangler feature_names; tren pa nytt eller oppdater *_meta.json.")
        feature_names_expected = [str(x) for x in fn_meta]
        feature_set = str(meta.get("feature_set") or infer_feature_set_from_feature_names(feature_names_expected))
        nf = getattr(model, "n_features_in_", None)
        if nf is not None and int(nf) != len(feature_names_expected):
            raise SystemExit(
                f"Modell forventer {nf} features, men meta definerer {len(feature_names_expected)} ({feature_set}). "
                "Tren modellen pa nytt med matchende build_dataset / feature_set."
            )
        print(
            f"[predict] feature_set={feature_set}  n_features={len(feature_names_expected)}"
            f"{f'  sklearn n_features_in_={int(nf)}' if nf is not None else ''}"
        )
        cal_path = Path(args.calibration_json) if args.calibration_json else None
        cal_active = cal_path.is_file() if cal_path else False
        print(
            f"[predict] prob_calibration_json={'aktiv: ' + str(cal_path) if cal_active else 'av (ingen fil)'}"
        )
        print(
            f"[predict] effektive blend-vekter: w_global={wg:.6f} w_explore={we:.6f} "
            f"(CLI {args.global_weight}/{args.explore_weight})  prob_floor={prob_floor}"
        )
        if args.production_calibrated:
            print("[predict] production-calibrated: klar for innsending (sjekk at explore-dir matcher aktiv analyse).")
    elif args.baseline == "global_ensemble":
        try:
            ensemble_models, neighbor_radius, ensemble_description = load_global_ensemble(default_models_dir())
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
        mdir = default_models_dir()
        em_loaded = False
        for cand in ("gradient_boosting_meta.json", "random_forest_meta.json", "logistic_regression_meta.json"):
            mp = mdir / cand
            if not mp.is_file():
                continue
            em = json.loads(mp.read_text(encoding="utf-8"))
            fn_e = em.get("feature_names")
            if not fn_e:
                continue
            feature_names_ens = [str(x) for x in fn_e]
            feature_set = str(em.get("feature_set") or infer_feature_set_from_feature_names(feature_names_ens))
            em_loaded = True
            break
        if not em_loaded:
            raise SystemExit(
                "global_ensemble: fant ikke meta med feature_names (gradient_boosting_meta.json / ...). "
                "Kjor train_model --compare-all pa nytt."
            )
        model_loaded_path = f"ensemble({ensemble_description})"
        model_load_source = ensemble_description or ""
        sklearn_model_kind = "global_ensemble"
        print(f"[predict] global ensemble lastet: {ensemble_description} feature_set={feature_set}")

    seed_summaries: list[dict[str, Any]] = []

    if args.explore_dir and cal_skip_reason and args.baseline in ("global_model", "global_ensemble"):
        print(f"NOTE --explore-dir: {cal_skip_reason} (kjorer uten explore-blend).")

    if args.baseline == "terrain" and calibration is not None:
        print(f"using calibration from {summary_path} (terrain baseline)")

    for seed_idx in range(seeds):
        if selected_seeds is not None and seed_idx not in selected_seeds:
            print(f"[seed {seed_idx}] hopper over (ikke valgt i --seed-indices)")
            continue

        if args.baseline == "uniform":
            pred = uniform_tensor(h, w, eps=args.eps)
            summary = {
                "round_id": round_id,
                "seed_index": seed_idx,
                "baseline": args.baseline,
                "model_path": None,
                "explore_calibration_used": False,
                "blend_global_weight": 1.0,
                "blend_explore_weight": 0.0,
                "prob_floor": float(args.eps),
            }
        elif args.baseline == "terrain":
            pred = build_prior_from_round_detail(detail, seed_idx, eps=args.eps, calibration=calibration)
            summary = {
                "round_id": round_id,
                "seed_index": seed_idx,
                "baseline": args.baseline,
                "model_path": None,
                "explore_calibration_used": calibration is not None,
                "analysis_summary_path": summary_path,
                "blend_global_weight": 1.0,
                "blend_explore_weight": 0.0,
                "prob_floor": float(args.eps),
            }
        elif args.baseline in ("global_model", "global_ensemble"):
            grid, settlements = load_initial_state_for_seed(detail["initial_states"], seed_idx)
            X = feature_matrix_for_seed(
                grid,
                settlements,
                neighbor_radius=neighbor_radius,
                feature_set=feature_set,
            )
            if args.baseline == "global_model":
                assert model is not None
                global_flat = predict_proba_fixed6(model, X)
            else:
                assert ensemble_models is not None
                global_flat = ensemble_predict_proba_fixed6(ensemble_models, X)
            global_hw = global_flat.reshape(h, w, NUM_CLASSES)
            min_g = float(np.min(global_hw))
            max_g = float(np.max(global_hw))

            explore_used = False
            min_before_floor = min_g
            max_before_floor = max_g

            if use_explore_blend and we > 0:
                c = resolve_explore_scalar_boosts(
                    dict(calibration or {}),
                    explore_hier,
                    seed_idx,
                    args.explore_hierarchy_mode,
                )
                explore_hw = build_terrain_prior(
                    grid,
                    settlements,
                    eps=DEFAULT_EPS,
                    coast_boost=float(c.get("coast_boost", 0.35)),
                    near_settlement_boost=float(c.get("near_settlement_boost", 0.5)),
                    coast_near_settle_port_boost=float(c.get("coast_near_settle_port_boost", 0.28)),
                    dynamic_ruin_weight=float(c.get("dynamic_ruin_weight", 1.0)),
                )
                explore_hw_before_reg = np.asarray(explore_hw, dtype=np.float64)
                explore_hw = apply_regional_explore_scales(
                    explore_hw_before_reg,
                    explore_hier,
                    h,
                    w,
                    seed_idx,
                    mode=args.explore_hierarchy_mode,
                    min_region_samples=int(args.min_regional_samples),
                )
                if args.debug_explore_hierarchy and seed_idx == 0:
                    eb = effective_terrain_boosts(c)
                    dmax = float(np.max(np.abs(np.asarray(explore_hw, dtype=np.float64) - explore_hw_before_reg)))
                    elig, rnotes = regional_bins_eligible(
                        explore_hier,
                        seed_idx,
                        min_region_samples=int(args.min_regional_samples),
                    )
                    print(
                        f"[predict] debug seed0 explore_hierarchy_mode={args.explore_hierarchy_mode} "
                        f"effective_scalar_boosts={eb}"
                    )
                    print(
                        f"[predict] debug seed0 explore_tensor max|delta| etter regional: {dmax} "
                        f"(0.0 => full ingen effekt eller mode!=full)"
                    )
                    print(f"[predict] debug seed0 regional_bins_eligible: {elig if elig else '(ingen)'}")
                    if rnotes:
                        print(f"[predict] debug seed0 regional_notes (utdrag): {rnotes[:5]}")
                blended = wg * global_hw + we * explore_hw
                rs = np.sum(blended, axis=-1, keepdims=True)
                rs = np.where(rs > 0, rs, 1.0)
                blended = blended / rs
                explore_used = True
                min_before_floor = float(np.min(blended))
                max_before_floor = float(np.max(blended))
                tensor_np = blended
            else:
                tensor_np = global_hw

            if args.calibration_json and Path(args.calibration_json).is_file():
                flat = tensor_np.reshape(-1, NUM_CLASSES)
                flat = apply_saved_calibration_json(flat, X, Path(args.calibration_json))
                tensor_np = flat.reshape(h, w, NUM_CLASSES)
                print(f"[predict] prob calibration applied: {args.calibration_json}")

            final_np = apply_floor_and_renorm(tensor_np, eps=prob_floor)
            min_after = float(np.min(final_np))
            max_after = float(np.max(final_np))
            pred = numpy_to_submission_list(final_np)

            summary = {
                "round_id": round_id,
                "seed_index": seed_idx,
                "baseline": args.baseline,
                "model_path": model_loaded_path,
                "sklearn_model_kind": sklearn_model_kind,
                "model_load_source": model_load_source,
                "ensemble_members": (
                    [{"name": n, "weight": float(wt)} for n, _, wt in ensemble_models]
                    if ensemble_models
                    else None
                ),
                "ensemble_description": ensemble_description,
                "neighbor_radius": neighbor_radius,
                "explore_calibration_used": explore_used,
                "analysis_summary_path": summary_path if args.explore_dir else None,
                "calibration_load_note": cal_skip_reason,
                "blend_global_weight": wg,
                "blend_explore_weight": we,
                "blend_global_weight_cli": args.global_weight,
                "blend_explore_weight_cli": args.explore_weight,
                "prob_floor": prob_floor,
                "global_proba_min": min_g,
                "global_proba_max": max_g,
                "before_floor_min": min_before_floor,
                "before_floor_max": max_before_floor,
                "after_floor_min": min_after,
                "after_floor_max": max_after,
                "prob_calibration_json": str(args.calibration_json) if args.calibration_json else None,
            }

        else:
            raise RuntimeError(f"ukjent baseline: {args.baseline}")

        val_eps = prob_floor if args.baseline in ("global_model", "global_ensemble") else float(args.eps)
        errs = validate_prediction(pred, h, w, eps=val_eps)
        if errs:
            raise RuntimeError(
                f"validation failed seed {seed_idx} (shape {h}x{w}x6, finite, sum≈1):\n"
                + "\n".join(errs[:30])
            )

        seed_summaries.append(summary)

        if args.baseline in ("global_model", "global_ensemble"):
            print(
                f"[seed {seed_idx}] baseline={args.baseline} type={sklearn_model_kind} path={model_loaded_path} "
                f"explore_blend={summary.get('explore_calibration_used')} "
                f"w_g={summary.get('blend_global_weight')} w_e={summary.get('blend_explore_weight')} "
                f"global[min,max]=({summary.get('global_proba_min'):.6g},{summary.get('global_proba_max'):.6g}) "
                f"pre_floor[min,max]=({summary.get('before_floor_min'):.6g},{summary.get('before_floor_max'):.6g}) "
                f"post_floor[min,max]=({summary.get('after_floor_min'):.6g},{summary.get('after_floor_max'):.6g})"
            )
        if not submit:
            if args.save_predictions_dir:
                args.save_predictions_dir.mkdir(parents=True, exist_ok=True)
                out_p = args.save_predictions_dir / f"prediction_seed{seed_idx}.json"
                out_p.write_text(json.dumps(pred), encoding="utf-8")
                print(f"  wrote {out_p}")
            print(
                f"seed {seed_idx}: ok shape={len(pred)}x{len(pred[0])}x{NUM_CLASSES} "
                f"sum[0,0]={sum(pred[0][0]):.6f}"
            )
            continue

        assert client_submit is not None
        r = _submit_prediction_with_retry(
            client_submit,
            round_id=round_id,
            seed_index=seed_idx,
            prediction=pred,
        )
        print(f"seed {seed_idx}: {r.status_code} {r.text[:200]}")

    if args.predict_summary_json:
        args.predict_summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "round_id": round_id,
            "baseline": args.baseline,
            "production_calibrated_preset": bool(args.production_calibrated),
            "debug_explore_hierarchy": bool(args.debug_explore_hierarchy),
            "seed_indices_filter": sorted(selected_seeds) if selected_seeds is not None else None,
            "model_path": model_loaded_path,
            "sklearn_model_kind": sklearn_model_kind,
            "model_load_source": model_load_source,
            "ensemble_members": (
                [{"name": n, "weight": float(wt)} for n, _, wt in ensemble_models]
                if ensemble_models
                else None
            ),
            "ensemble_description": ensemble_description,
            "explore_dir": str(args.explore_dir) if args.explore_dir else None,
            "explore_run_files_count": n_explore_runs,
            "analysis_summary_path": summary_path,
            "calibration_skipped_reason": cal_skip_reason,
            "blend_global_weight_cli": args.global_weight,
            "blend_explore_weight_cli": args.explore_weight,
            "blend_global_weight_effective": wg if args.baseline in ("global_model", "global_ensemble") else None,
            "blend_explore_weight_effective": we if args.baseline in ("global_model", "global_ensemble") else None,
            "prob_floor": prob_floor if args.baseline in ("global_model", "global_ensemble") else float(args.eps),
            "seeds": seed_summaries,
        }
        args.predict_summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote predict summary {args.predict_summary_json}")

    if args.dry_run and args.round_dir:
        meta_path = Path(args.round_dir) / "last_predict_dryrun.json"
        meta_path.write_text(
            json.dumps(
                {
                    "baseline": args.baseline,
                    "eps": args.eps,
                    "prob_floor": prob_floor if args.baseline in ("global_model", "global_ensemble") else args.eps,
                    "map_height": h,
                    "map_width": w,
                    "seed_indices": sorted(selected_seeds) if selected_seeds is not None else None,
                    "model_path": model_loaded_path,
                    "sklearn_model_kind": sklearn_model_kind,
                    "model_load_source": model_load_source,
                    "ensemble_description": ensemble_description,
                    "blend_global_weight_cli": args.global_weight,
                    "blend_explore_weight_cli": args.explore_weight,
                    "blend_global_weight_effective": wg if args.baseline in ("global_model", "global_ensemble") else None,
                    "blend_explore_weight_effective": we if args.baseline in ("global_model", "global_ensemble") else None,
                    "explore_dir": str(args.explore_dir) if args.explore_dir else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

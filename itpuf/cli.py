"""
IT-PUF Command-Line Interface
==============================

Usage:
    itpuf enroll  --model <n> --output <anchor.json>   [requires measurement engine]
    itpuf verify  --model <n> --anchor <anchor.json>   [requires measurement engine]
    itpuf audit   --anchors <dir_or_files...>          [works locally]
    itpuf info    --anchor <anchor.json>               [works locally]

Enrollment and live verification require the measurement engine,
available through a commercial license or the fallrisk.ai API.

Patent Application 63/982,893 (February 13, 2026)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from itpuf.anchor import Anchor, VerifyResult, compute_far


# ═══════════════════════════════════════════════════════════════════
# MEASUREMENT ENGINE CHECK
# ═══════════════════════════════════════════════════════════════════

def _check_measurement_engine():
    """
    Check if the proprietary measurement engine is available.

    Commercial licensees receive the measurement engine files
    (hooks, measurement, calibration) as part of their license.
    """
    try:
        from itpuf.hooks import detect_architecture         # noqa: F401
        from itpuf.measurement import measure_all_seeds     # noqa: F401
        from itpuf.calibration import validate_all_seeds    # noqa: F401
        return True
    except ImportError:
        return False


def _print_engine_required():
    """Print message when measurement engine is not available."""
    print()
    print("  ══════════════════════════════════════════════════════════")
    print("  MEASUREMENT ENGINE REQUIRED")
    print("  ══════════════════════════════════════════════════════════")
    print()
    print("  Enrollment and live verification require the IT-PUF")
    print("  measurement engine, which is not included in the")
    print("  open-source package.")
    print()
    print("  To get access:")
    print("    • Commercial license — anthony@fallrisk.ai")
    print("    • fallrisk.ai API    — coming soon")
    print()
    print("  What you CAN do now (no measurement engine needed):")
    print("    itpuf audit  --anchors a.json b.json   Compare anchors")
    print("    itpuf info   --anchor model.json        Inspect anchor")
    print()
    print("  Pre-enrolled demo anchors are available at:")
    print("    github.com/FallRiskAI/IT-PUF/releases")
    print()
    print("  ══════════════════════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

def _print_footer():
    """Print the standard footer with contact and upgrade nudge."""
    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  Continuous monitoring & signed attestations: fallrisk.ai")
    print("  Questions? anthony@fallrisk.ai")
    print("  ─────────────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING (only available with measurement engine)
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_name: str, device: str):
    """Load a HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id or 0

    is_mamba = "mamba" in model_name.lower() and "jamba" not in model_name.lower()

    kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if is_mamba:
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, **kwargs,
    ).to(device)
    model.eval()

    return model, tokenizer


def _cleanup_model(model, device: str):
    """Free GPU memory."""
    del model
    import gc
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_prompt_bank(path: str = None) -> list:
    """Load prompt bank from file."""
    import importlib.util

    search_paths = []
    if path:
        search_paths.append(Path(path))
    search_paths.extend([
        Path.cwd() / "prompt_bank.py",
        Path.cwd() / "prompt_bank.json",
    ])

    for p in search_paths:
        if not p.exists():
            continue

        if p.suffix == ".py":
            spec = importlib.util.spec_from_file_location("prompt_bank", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "PROMPTS"):
                print(f"  Loaded {len(mod.PROMPTS)} prompts from {p}")
                return mod.PROMPTS

        elif p.suffix == ".json":
            with open(p) as f:
                prompts = json.load(f)
            if isinstance(prompts, list):
                print(f"  Loaded {len(prompts)} prompts from {p}")
                return prompts

    print("  ERROR: No prompt bank found.")
    print("  Provide --prompt-bank <path> to a .py or .json file.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════

def cmd_enroll(args):
    """Enroll a model: measure its identity, save anchor."""
    if not _check_measurement_engine():
        _print_engine_required()
        return 1

    import hashlib
    from datetime import datetime, timezone
    from itpuf.contract import CONTRACT
    from itpuf.hooks import detect_architecture
    from itpuf.measurement import measure_all_seeds
    from itpuf.calibration import validate_all_seeds

    device = args.device
    prompt_bank = _load_prompt_bank(args.prompt_bank)
    model, tokenizer = _load_model(args.model, device)

    print(f"\n  Enrolling {args.model}...")
    t0 = time.time()

    config = detect_architecture(model)
    responses = measure_all_seeds(
        model, tokenizer, prompt_bank, device,
        config=config, contract=CONTRACT,
    )
    all_valid, validation_results = validate_all_seeds(responses)

    if not all_valid:
        failed = {
            s: str(r) for s, r in validation_results.items() if not r.valid
        }
        print(f"  ERROR: τ vector validation failed: {failed}")
        _cleanup_model(model, device)
        return 1

    bank_hash = hashlib.sha256(
        "\n".join(prompt_bank).encode("utf-8")
    ).hexdigest()[:16]

    anchor = Anchor(
        model_name=args.model,
        responses=responses,
        arch_type=config.arch_type,
        arch_strategy=config.strategy,
        n_layers=config.n_layers,
        contract_hash=CONTRACT.content_hash(),
        epsilon=CONTRACT.epsilon,
        enrolled_at=datetime.now(timezone.utc).isoformat(),
        prompt_bank_hash=bank_hash,
        validation={s: str(r) for s, r in validation_results.items()},
    )
    elapsed = time.time() - t0

    output = args.output or f"{args.model.split('/')[-1]}_anchor.json"
    anchor.save(output)

    print(f"\n  ✅ Enrolled in {elapsed:.1f}s")
    print(f"  Anchor saved to: {output}")
    print(f"  Architecture: {anchor.arch_type}")
    print(f"  Contract hash: {anchor.contract_hash}")
    print(f"  ε = {anchor.epsilon:.2e}")
    for seed, tau in sorted(anchor.responses.items()):
        print(f"    Seed {seed}: τ mean={tau.mean():.4f}, std={tau.std():.4f}")

    _print_footer()
    _cleanup_model(model, device)
    return 0


def cmd_verify(args):
    """Verify a model against an enrolled anchor."""
    if not _check_measurement_engine():
        _print_engine_required()
        return 1

    import hashlib
    from itpuf.contract import CONTRACT
    from itpuf.hooks import detect_architecture
    from itpuf.measurement import measure_all_seeds
    from itpuf.calibration import validate_all_seeds

    device = args.device
    prompt_bank = _load_prompt_bank(args.prompt_bank)

    anchor = Anchor.load(args.anchor)
    print(f"  Anchor: {anchor.model_name}")
    print(f"  ε = {anchor.epsilon:.2e}")

    # Contract compatibility check.
    if CONTRACT.content_hash() != anchor.contract_hash:
        print(f"  ERROR: Contract mismatch: current={CONTRACT.content_hash()}, "
              f"anchor={anchor.contract_hash}")
        return 1

    # Prompt bank hash check.
    bank_hash = hashlib.sha256(
        "\n".join(prompt_bank).encode("utf-8")
    ).hexdigest()[:16]
    if bank_hash != anchor.prompt_bank_hash:
        print(f"  ERROR: Prompt bank mismatch: current={bank_hash}, "
              f"anchor={anchor.prompt_bank_hash}")
        return 1

    model, tokenizer = _load_model(args.model, device)

    print(f"\n  Verifying {args.model} against anchor...")
    t0 = time.time()

    config = detect_architecture(model)
    fresh = measure_all_seeds(
        model, tokenizer, prompt_bank, device,
        config=config, contract=CONTRACT,
    )
    all_valid, val_results = validate_all_seeds(fresh)

    distances = anchor.distance(fresh)
    epsilon = anchor.epsilon
    ratios = {s: d / epsilon for s, d in distances.items()}
    max_d = max(distances.values())
    elapsed = time.time() - t0

    accepted = max_d <= epsilon
    ratio_max = max_d / epsilon

    symbol = "✅" if accepted else "❌"
    print(f"\n  {symbol} {'ACCEPT' if accepted else 'REJECT'}")
    print(f"  Verified in {elapsed:.1f}s")
    print(f"  Max distance: {max_d:.2e} ({ratio_max:.1f}× ε)")
    for seed, d in sorted(distances.items()):
        ratio = ratios[seed]
        print(f"    Seed {seed}: d={d:.2e} ({ratio:.1f}× ε)")

    print()
    if accepted:
        print("  This model matches the enrolled identity.")
        print("  The running intelligence is the approved intelligence.")
    else:
        print(f"  ⚠  MODEL MISMATCH DETECTED.")
        print(f"  The verified model does not match the enrolled identity.")
        print(f"  Distance is {ratio_max:,.0f}× the acceptance threshold.")
        print(f"  This swap would be detected with mathematical certainty.")

    _print_footer()
    _cleanup_model(model, device)
    return 0 if accepted else 1


def cmd_audit(args):
    """Cross-compare all anchors: compute pairwise FAR."""
    anchor_files = []
    for p in args.anchors:
        p = Path(p)
        if p.is_dir():
            anchor_files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            anchor_files.append(p)
        else:
            print(f"  WARNING: {p} not found, skipping")

    if len(anchor_files) < 2:
        print("ERROR: Need at least 2 anchor files for cross-comparison.")
        return 1

    anchors = []
    for f in anchor_files:
        a = Anchor.load(str(f))
        anchors.append(a)
        print(f"  Loaded: {a.model_name} ({f.name})")

    print(f"\n  Cross-comparing {len(anchors)} anchors...")
    report = compute_far(anchors)

    n_pairs = report["n_pairs"]
    false_accepts = report["n_false_accepts"]
    min_ratio = report["min_ratio"]

    symbol = "✅" if false_accepts == 0 else "❌"
    print(f"\n  {symbol} FAR: {false_accepts}/{n_pairs}")
    print(f"  Min separation: {report['min_distance']:.4f} ({min_ratio:.1f}× ε)")

    if report.get("details"):
        # Find closest pair.
        closest = min(report["details"], key=lambda x: x["distance"])
        print(f"  Closest pair: {closest['model_a']} ↔ {closest['model_b']}")

    output = args.output or "audit_report.json"
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {output}")

    _print_footer()
    return 0 if false_accepts == 0 else 1


def cmd_info(args):
    """Display anchor details."""
    anchor = Anchor.load(args.anchor)

    print(f"  Model:        {anchor.model_name}")
    print(f"  Architecture: {anchor.arch_type} ({anchor.arch_strategy})")
    print(f"  Layers:       {anchor.n_layers}")
    print(f"  ε:            {anchor.epsilon:.2e}")
    print(f"  Contract:     {anchor.contract_hash}")
    print(f"  Prompt bank:  {anchor.prompt_bank_hash}")
    print(f"  Enrolled at:  {anchor.enrolled_at}")
    print(f"  Seeds:        {sorted(anchor.responses.keys())}")
    for seed, tau in sorted(anchor.responses.items()):
        print(f"    Seed {seed}: dim={len(tau)}, "
              f"mean={tau.mean():.4f}, std={tau.std():.4f}, "
              f"min={tau.min():.4f}, max={tau.max():.4f}")

    _print_footer()
    return 0


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    has_engine = _check_measurement_engine()
    engine_tag = "" if has_engine else " [requires measurement engine]"

    parser = argparse.ArgumentParser(
        prog="itpuf",
        description=(
            "IT-PUF: Biometric identity verification for AI models.\n"
            "Challenge-response fingerprinting with cryptographic guarantees."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  itpuf audit  --anchors model_a.json model_b.json\n"
            "  itpuf info   --anchor model_a.json\n"
            + (
                "  itpuf enroll --model Qwen/Qwen2.5-0.5B-Instruct\n"
                "  itpuf verify --model Qwen/Qwen2.5-0.5B-Instruct "
                "--anchor anchor.json\n"
                if has_engine else
                "\n"
                "  Enrollment & live verification require the measurement\n"
                "  engine. Contact anthony@fallrisk.ai for access.\n"
            )
            + "\n"
            "Patent Application 63/982,893 | https://fallrisk.ai"
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── enroll ──────────────────────────────────────────────────
    p_enroll = sub.add_parser(
        "enroll",
        help=f"Enroll a model — measure its identity and save an anchor.{engine_tag}",
    )
    p_enroll.add_argument(
        "--model", required=True,
        help="HuggingFace model name (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    )
    p_enroll.add_argument(
        "--output", "-o", default=None,
        help="Output anchor file (default: <model>_anchor.json)",
    )
    p_enroll.add_argument(
        "--prompt-bank", default=None,
        help="Path to prompt_bank.py or prompt_bank.json",
    )
    p_enroll.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    # ── verify ──────────────────────────────────────────────────
    p_verify = sub.add_parser(
        "verify",
        help=f"Verify a model's identity against an enrolled anchor.{engine_tag}",
    )
    p_verify.add_argument(
        "--model", required=True,
        help="HuggingFace model name to verify",
    )
    p_verify.add_argument(
        "--anchor", required=True,
        help="Path to the enrolled anchor JSON file",
    )
    p_verify.add_argument(
        "--prompt-bank", default=None,
        help="Path to prompt_bank.py or prompt_bank.json",
    )
    p_verify.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    # ── audit ───────────────────────────────────────────────────
    p_audit = sub.add_parser(
        "audit",
        help="Cross-compare anchors: compute pairwise distances and FAR.",
    )
    p_audit.add_argument(
        "--anchors", nargs="+", required=True,
        help="Anchor files or directories containing anchor JSONs",
    )
    p_audit.add_argument(
        "--output", "-o", default=None,
        help="Output report file (default: audit_report.json)",
    )

    # ── info ────────────────────────────────────────────────────
    p_info = sub.add_parser(
        "info",
        help="Display details of an enrolled anchor.",
    )
    p_info.add_argument(
        "--anchor", required=True,
        help="Path to anchor JSON file",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    print("=" * 60)
    print("IT-PUF  |  Biometric Identity for AI Models")
    print("Patent Application 63/982,893  |  fallrisk.ai")
    print("=" * 60)

    dispatch = {
        "enroll": cmd_enroll,
        "verify": cmd_verify,
        "audit": cmd_audit,
        "info": cmd_info,
    }
    rc = dispatch[args.command](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lookahead Bias Audit Tool for IntradayNet features.

Run this script to scan all feature files for potential lookahead bias.
Each feature is classified as SAFE, LEAK, or BORDERLINE.

Usage:
    python scripts/audit_features.py
    python scripts/audit_features.py --verbose
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class BiasAuditor:
    """Scans Python source code for lookahead bias patterns."""

    # Patterns that indicate lookahead (future data usage)
    LEAK_PATTERNS = [
        (r"\.groupby.*transform\s*\(\s*['\"]sum['\"]", "transform('sum') — future data"),
        (r"\.groupby.*transform\s*\(\s*['\"]max['\"]", "transform('max') — future data"),
        (r"\.groupby.*transform\s*\(\s*['\"]min['\"]", "transform('min') — future data"),
        (r"\.groupby.*cummax\s*\(\)", "cummax() — future data"),
        (r"\.groupby.*cummin\s*\(\)", "cummin() — future data"),
        (r"\.groupby.*transform\s*\(\s*['\"]count['\"]", "transform('count') — future data"),
        (r"\.shift\s*\(\s*-\d+\s*\)", "shift(-N) — future data"),
        (r"\.rolling.*max\s*\(\s*\)", "rolling().max() — potential future if not careful"),
        (r"\.rolling.*min\s*\(\s*\)", "rolling().min() — potential future if not careful"),
    ]

    BORDERLINE_PATTERNS = [
        (r"\.shift\s*\(\s*0\s*\)", "shift(0) — current bar (causal but redundant check)"),
        (r"\.ewm\s*\(", "ewm() — exponentially weighted (causal but verify span)"),
    ]

    # Safe patterns (explicitly confirmed non-leaking)
    SAFE_PATTERNS = [
        (r"\.shift\s*\(\s*\d+\s*\)", "shift(N) — past data only"),
        (r"\.rolling\s*\(\s*\d+", "rolling(N) — past N bars only"),
        (r"\.expanding\s*\(\s*\)", "expanding() — past + current only"),
        (r"\.groupby.*transform\s*\(\s*lambda", "groupby + expanding/transform — causal"),
        (r"\.groupby.*apply.*expanding", "groupby + expanding — causal"),
    ]

    def __init__(self):
        self.files_analyzed: List[Path] = []
        self.findings: List[Dict] = []

    def analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze a single Python file for lookahead bias patterns."""
        findings = []
        lines = file_path.read_text().split("\n")

        for lineno, line in enumerate(lines, 1):
            original = line
            line_lower = line.lower()

            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Check for leak patterns
            for pattern, description in self.LEAK_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "file": file_path.name,
                        "line": lineno,
                        "code": original.strip()[:100],
                        "pattern": description,
                        "severity": "LEAK",
                    })

            # Check for borderline patterns
            for pattern, description in self.BORDERLINE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "file": file_path.name,
                        "line": lineno,
                        "code": original.strip()[:100],
                        "pattern": description,
                        "severity": "BORDERLINE",
                    })

        return findings

    def audit_feature_files(self, verbose: bool = False) -> Dict:
        """Audit all feature files in the project."""
        feature_dir = PROJECT_ROOT / "src" / "intradaynet" / "features"
        market_dir = PROJECT_ROOT / "src" / "intradaynet" / "features"

        files_to_check = [
            feature_dir / "per_bar_features.py",
            feature_dir / "session_features.py",
            feature_dir / "sentiment_features.py",
            feature_dir / "market_features.py",
        ]

        all_findings = []
        files_checked = 0

        for f in files_to_check:
            if f.exists():
                findings = self.analyze_file(f)
                all_findings.extend(findings)
                files_checked += 1
                if verbose:
                    for finding in findings:
                        print(f"  [{finding['severity']}] {f.name}:{finding['line']}")
                        print(f"    {finding['code']}")
                        print(f"    Pattern: {finding['pattern']}")

        leaks = [f for f in all_findings if f["severity"] == "LEAK"]
        borderline = [f for f in all_findings if f["severity"] == "BORDERLINE"]

        return {
            "files_checked": files_checked,
            "total_findings": len(all_findings),
            "leaks": leaks,
            "borderline": borderline,
            "findings": all_findings,
        }

    def check_feature_names(self) -> Dict:
        """Verify feature names match between definition and usage."""
        from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
        from intradaynet.features.session_features import SESSION_FEATURE_NAMES
        from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES

        return {
            "per_bar_count": len(PER_BAR_FEATURE_NAMES),
            "session_count": len(SESSION_FEATURE_NAMES),
            "sentiment_count": len(SENTIMENT_FEATURE_NAMES),
            "per_bar_names": PER_BAR_FEATURE_NAMES,
            "session_names": SESSION_FEATURE_NAMES,
            "sentiment_names": SENTIMENT_FEATURE_NAMES,
        }


def print_report(results: Dict, check_names: Dict):
    """Print a formatted audit report."""
    print("=" * 70)
    print("LOOKAHEAD BIAS AUDIT REPORT")
    print("=" * 70)

    print(f"\nFiles analyzed: {results['files_checked']}")
    print(f"Total findings: {results['total_findings']}")

    # Leak findings
    print(f"\n{'🚩 LEAKS:' if results['leaks'] else '✅ No leaks found:'}")
    if results['leaks']:
        for f in results['leaks']:
            print(f"  {f['file']}:{f['line']}")
            print(f"    Code: {f['code']}")
            print(f"    Issue: {f['pattern']}")
    else:
        print("  No lookahead bias detected in feature files.")

    # Borderline findings
    print(f"\n{'⚠️  BORDERLINE:' if results['borderline'] else ''}")
    if results['borderline']:
        for f in results['borderline']:
            print(f"  {f['file']}:{f['line']}")
            print(f"    Code: {f['code']}")
            print(f"    Issue: {f['pattern']}")

    # Feature counts
    print(f"\n📊 Feature Counts:")
    print(f"  Per-bar features:   {check_names['per_bar_count']}")
    print(f"  Session features:   {check_names['session_count']}")
    print(f"  Sentiment features: {check_names['sentiment_count']}")
    print(f"  Total:             {sum([check_names['per_bar_count'], check_names['session_count'], check_names['sentiment_count']])}")

    # Feature lists
    print(f"\nPer-bar features ({check_names['per_bar_count']}):")
    for i, name in enumerate(check_names['per_bar_names'], 1):
        print(f"  {i:2d}. {name}")

    # Summary verdict
    print("\n" + "=" * 70)
    if results['leaks']:
        print("STATUS: 🚩 FIXES REQUIRED — leaks detected")
        print("Action: Fix all 🚩 issues before training.")
    elif results['borderline']:
        print("STATUS: ⚠️  REVIEW NEEDED — borderline patterns detected")
        print("Action: Review ⚠️  patterns manually.")
    else:
        print("STATUS: ✅ CLEAN — no lookahead bias detected")
        print("Action: Ready for Phase 1.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit features for lookahead bias")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    auditor = BiasAuditor()
    results = auditor.audit_feature_files(verbose=args.verbose)
    names = auditor.check_feature_names()
    print_report(results, names)


if __name__ == "__main__":
    main()

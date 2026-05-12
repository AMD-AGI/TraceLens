###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Parse JUnit XML test results and push per-test duration metrics to Grafana
Cloud via OTLP/HTTP.

Usage:
    python3 push_metrics.py <junit-xml-directory>

Environment variables (required):
    GRAFANA_CLOUD_OTLP_ENDPOINT  Full OTLP HTTP endpoint URL
                                 e.g. https://otlp-gateway-prod-us-east-0.grafana.net/otlp
    GRAFANA_CLOUD_OTLP_TOKEN     Basic-auth credential as "<instance_id>:<sa_token>"
    GITHUB_SHA                   Git commit SHA (set by GitHub Actions)
    GITHUB_REF_NAME              Branch name (set by GitHub Actions)
"""

import base64
import glob
import os
import sys
import traceback
import xml.etree.ElementTree as ET


def parse_junit_xml_dir(junit_dir):
    """Return a list of test-result dicts from all XML files in junit_dir."""
    results = []
    xml_pattern = os.path.join(junit_dir, "*.xml")
    for xml_path in glob.glob(xml_pattern):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            print(
                f"[push_metrics] WARNING: could not parse {xml_path}: {exc}",
                file=sys.stderr,
            )
            continue

        if root.tag == "testsuites":
            suites = root.findall("testsuite")
        elif root.tag == "testsuite":
            suites = [root]
        else:
            suites = root.findall(".//testsuite")

        for suite in suites:
            suite_name = os.path.splitext(os.path.basename(xml_path))[0]
            for tc in suite.findall("testcase"):
                name = tc.get("name", "unknown")
                try:
                    duration = float(tc.get("time", "0") or "0")
                except ValueError:
                    duration = 0.0

                if tc.find("failure") is not None or tc.find("error") is not None:
                    status = "failed"
                elif tc.find("skipped") is not None:
                    status = "skipped"
                else:
                    status = "passed"

                results.append(
                    {
                        "test_name": name,
                        "test_suite": suite_name,
                        "duration_s": duration,
                        "status": status,
                    }
                )
    return results


def push_metrics(results, endpoint, token_raw, branch, commit_sha):
    """Push test duration metrics to Grafana Cloud via OTLP/HTTP."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.metrics import Observation
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    auth_value = "Basic " + base64.b64encode(token_raw.encode()).decode()

    reader = InMemoryMetricReader()

    resource = Resource(
        attributes={
            SERVICE_NAME: "tracelens-ci",
            "branch": branch,
            "commit_sha": commit_sha,
        }
    )

    provider = MeterProvider(metric_readers=[reader], resource=resource)
    meter = provider.get_meter("tracelens.ci.tests")

    snapshot = list(results)

    suite_totals = {}
    for r in snapshot:
        suite_totals[r["test_suite"]] = suite_totals.get(r["test_suite"], 0.0) + r["duration_s"]

    total_duration = sum(r["duration_s"] for r in snapshot)

    def duration_callback(options):
        for r in snapshot:
            yield Observation(
                value=r["duration_s"],
                attributes={
                    "test_name": r["test_name"],
                    "test_suite": r["test_suite"],
                    "status": r["status"],
                    "branch": branch,
                    "commit_sha": commit_sha,
                },
            )

    def suite_duration_callback(options):
        for suite_name, total in suite_totals.items():
            yield Observation(
                value=total,
                attributes={
                    "test_suite": suite_name,
                    "branch": branch,
                    "commit_sha": commit_sha,
                },
            )

    def total_duration_callback(options):
        yield Observation(
            value=total_duration,
            attributes={
                "branch": branch,
                "commit_sha": commit_sha,
            },
        )

    meter.create_observable_gauge(
        name="tracelens_test_duration_seconds",
        callbacks=[duration_callback],
        unit="s",
        description="Per-test duration from pytest JUnit XML results",
    )

    meter.create_observable_gauge(
        name="tracelens_suite_duration_seconds",
        callbacks=[suite_duration_callback],
        unit="s",
        description="Total duration per test suite from pytest JUnit XML results",
    )

    meter.create_observable_gauge(
        name="tracelens_total_duration_seconds",
        callbacks=[total_duration_callback],
        unit="s",
        description="Total duration of all tests from pytest JUnit XML results",
    )

    metrics_data = reader.get_metrics_data()

    exporter = OTLPMetricExporter(
        endpoint=endpoint.rstrip("/") + "/v1/metrics",
        headers={"Authorization": auth_value},
        timeout=30,
    )

    print(
        f"[push_metrics] Exporting {len(snapshot)} metric observations to {endpoint}",
        file=sys.stderr,
    )
    exporter.export(metrics_data)
    exporter.shutdown()
    provider.shutdown()
    print("[push_metrics] Done.", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage: push_metrics.py <junit-xml-directory>", file=sys.stderr)
        sys.exit(0)

    junit_dir = sys.argv[1]
    endpoint = os.environ.get("GRAFANA_CLOUD_OTLP_ENDPOINT", "")
    token_raw = os.environ.get("GRAFANA_CLOUD_OTLP_TOKEN", "")
    branch = os.environ.get("GITHUB_REF_NAME", "unknown")
    commit_sha = os.environ.get("GITHUB_SHA", "unknown")

    try:
        if not endpoint:
            print(
                "[push_metrics] WARNING: GRAFANA_CLOUD_OTLP_ENDPOINT not set, skipping.",
                file=sys.stderr,
            )
            sys.exit(0)
        if not token_raw:
            print(
                "[push_metrics] WARNING: GRAFANA_CLOUD_OTLP_TOKEN not set, skipping.",
                file=sys.stderr,
            )
            sys.exit(0)

        results = parse_junit_xml_dir(junit_dir)
        if not results:
            print(
                f"[push_metrics] No test results found in {junit_dir}, skipping.",
                file=sys.stderr,
            )
            sys.exit(0)

        print(
            f"[push_metrics] Parsed {len(results)} test cases from {junit_dir}",
            file=sys.stderr,
        )
        push_metrics(results, endpoint, token_raw, branch, commit_sha)

    except Exception as exc:  # noqa: BLE001
        print(f"[push_metrics] ERROR (non-fatal): {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()

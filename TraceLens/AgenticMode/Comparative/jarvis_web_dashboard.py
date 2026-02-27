#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Jarvis Web Dashboard - Browser-based interface for GPU trace analysis
Run with: python3 jarvis_web_dashboard.py
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import markdown

# Add Analysis directory to path
sys.path.insert(0, str(Path(__file__).parent / "Analysis"))
from jarvis_analysis import JarvisAnalyzer

app = Flask(__name__)
app.config["SECRET_KEY"] = "jarvis-analysis-secret-key"
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# Hardcoded API key (replace with your actual key)
DEFAULT_API_KEY = ""  # Replace this with your actual API key

# Global state for analysis jobs
analysis_jobs: Dict[str, Dict] = {}

# Job queue configuration
MAX_CONCURRENT_JOBS = 2  # Maximum number of jobs running simultaneously
job_queue = Queue()
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
queue_lock = threading.Lock()
active_jobs = set()  # Track currently running jobs


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    allowed = {".json", ".gz", ".trace"}
    return any(filename.endswith(ext) for ext in allowed)


def get_queue_position(job_id: str) -> int:
    """Get the position of a job in the queue"""
    with queue_lock:
        queue_list = list(job_queue.queue)
        try:
            return queue_list.index(job_id) + 1
        except ValueError:
            return 0


def update_queue_positions():
    """Update queue position for all queued jobs"""
    with queue_lock:
        queue_list = list(job_queue.queue)
        for i, job_id in enumerate(queue_list):
            if job_id in analysis_jobs:
                analysis_jobs[job_id]["queue_position"] = i + 1
                analysis_jobs[job_id]["progress"] = f"Queued (position #{i + 1})"


def run_analysis_background(job_id: str, config: Dict):
    """Run Jarvis analysis in background thread"""
    try:
        # Mark as running
        with queue_lock:
            active_jobs.add(job_id)
            if job_id in analysis_jobs:
                analysis_jobs[job_id]["queue_position"] = 0

        analysis_jobs[job_id]["status"] = "running"
        analysis_jobs[job_id][
            "progress"
        ] = f"Running ({len(active_jobs)}/{MAX_CONCURRENT_JOBS} slots in use)..."
        update_queue_positions()

        # Debug: Log the input configuration
        print(f"\n{'='*70}")
        print(f"WEB DASHBOARD - Starting Analysis Job: {job_id}")
        print(f"Active jobs: {len(active_jobs)}/{MAX_CONCURRENT_JOBS}")
        print(f"{'='*70}")
        print(f"GPU1 Kineto: {config['gpu1_kineto']}")
        print(f"GPU1 ET: {config.get('gpu1_et', 'None')}")
        print(f"GPU2 Kineto: {config['gpu2_kineto']}")
        print(f"GPU2 ET: {config.get('gpu2_et', 'None')}")
        print(f"GPU1 Name: {config.get('gpu1_name', 'GPU1')}")
        print(f"GPU2 Name: {config.get('gpu2_name', 'GPU2')}")
        print(f"{'='*70}\n")

        # Create analyzer
        analyzer = JarvisAnalyzer(
            gpu1_kineto=config["gpu1_kineto"],
            gpu1_et=config.get("gpu1_et"),
            gpu2_kineto=config["gpu2_kineto"],
            gpu2_et=config.get("gpu2_et"),
            gpu1_name=config.get("gpu1_name"),
            gpu2_name=config.get("gpu2_name"),
            output_dir=config.get("output_dir", "web_reports"),
            api_key=config.get("api_key") or DEFAULT_API_KEY,
            generate_plots=config.get("generate_plots", True),
            use_critical_path=config.get("use_critical_path", False),
        )

        analysis_jobs[job_id]["progress"] = "Running Jarvis analysis..."

        # Run analysis
        success = analyzer.run()

        if success:
            analysis_jobs[job_id]["status"] = "completed"
            analysis_jobs[job_id]["progress"] = "Analysis completed!"
            analysis_jobs[job_id]["results"] = {
                "success": True,
                "output_dir": str(analyzer.output_dir),
                "report_path": str(analyzer.output_dir / "analysis_summary.md"),
                "gpu1_name": analyzer.gpu1_name,
                "gpu2_name": analyzer.gpu2_name,
            }
            analysis_jobs[job_id]["output_dir"] = str(analyzer.output_dir)
            analysis_jobs[job_id]["report_path"] = str(
                analyzer.output_dir / "analysis_summary.md"
            )
        else:
            raise Exception("Analysis failed - check logs for details")

    except Exception as e:
        analysis_jobs[job_id]["status"] = "failed"
        analysis_jobs[job_id]["error"] = str(e)
        analysis_jobs[job_id]["progress"] = f"Error: {str(e)}"
    finally:
        # Remove from active jobs
        with queue_lock:
            active_jobs.discard(job_id)
        update_queue_positions()
        print(f"\n{'='*70}")
        print(
            f"Job {job_id} finished. Active jobs: {len(active_jobs)}/{MAX_CONCURRENT_JOBS}"
        )
        print(f"{'='*70}\n")


def process_job_queue():
    """Process jobs from the queue - runs in background thread"""
    while True:
        try:
            # Get next job from queue (blocks until available)
            job_id = job_queue.get()

            if job_id in analysis_jobs:
                config = analysis_jobs[job_id]["config"]
                # Submit to executor (will wait if max workers reached)
                executor.submit(run_analysis_background, job_id, config)

            job_queue.task_done()
        except Exception as e:
            print(f"Error processing job queue: {e}")
            time.sleep(1)


@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("dashboard.html")


@app.route("/api/reports")
def list_reports():
    """List all available reports"""
    reports = []

    # Scan common report directories
    report_dirs = [
        Path("."),
        Path("web_reports"),
        Path("trace_reports"),
    ]

    for base_dir in report_dirs:
        if not base_dir.exists():
            continue

        # Look for report directories (pattern: *_Report or contain analysis_summary.md)
        for item in base_dir.iterdir():
            if item.is_dir() and (
                item.name.endswith("_Report") or item.name == "web_reports"
            ):
                # Check for subdirectories with timestamps
                for subdir in item.iterdir():
                    if subdir.is_dir():
                        summary = subdir / "analysis_summary.md"
                        if summary.exists():
                            # Parse directory name for metadata
                            parts = subdir.name.split("_")
                            reports.append(
                                {
                                    "name": subdir.name,
                                    "path": str(subdir),
                                    "date": summary.stat().st_mtime,
                                    "date_str": datetime.fromtimestamp(
                                        summary.stat().st_mtime
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                                }
                            )

    # Sort by date, newest first
    reports.sort(key=lambda x: x["date"], reverse=True)
    return jsonify(reports)


@app.route("/api/report/<path:report_path>")
def view_report(report_path: str):
    """Get report details"""
    report_dir = Path(report_path)

    if not report_dir.exists():
        return jsonify({"error": "Report not found"}), 404

    # Read analysis summary
    summary_file = report_dir / "analysis_summary.md"
    if not summary_file.exists():
        return jsonify({"error": "Report summary not found"}), 404

    with open(summary_file, "r") as f:
        markdown_content = f.read()

    # Fix image paths in markdown to use report-relative paths
    import re
    from urllib.parse import quote

    def replace_image_path(match):
        img_filename = match.group(1)
        # Only process relative paths (images in the same directory as markdown)
        if not img_filename.startswith("/"):
            img_path = report_dir / img_filename
            if img_path.exists():
                # Create URL with report_path and filename
                encoded_report = quote(str(report_path))
                encoded_filename = quote(img_filename)
                new_url = f"/api/plot/{encoded_report}/{encoded_filename}"
                print(f"DEBUG: Replacing {img_filename} with {new_url}")
                return f"![image]({new_url})"
            else:
                print(f"WARNING: Image not found: {img_path}")
        return match.group(0)

    # Replace markdown image syntax: ![alt](filename.png)
    markdown_content = re.sub(
        r"!\[[^\]]*\]\(([^)]+\.png)\)", replace_image_path, markdown_content
    )

    # Convert markdown to HTML
    html_content = markdown.markdown(
        markdown_content, extensions=["tables", "fenced_code"]
    )

    # Find all plots
    plots = []
    for plot_file in report_dir.glob("*.png"):
        plots.append(
            {
                "name": plot_file.name,
                "path": str(plot_file.absolute()),  # Use absolute path for serving
            }
        )

    return jsonify(
        {
            "markdown": markdown_content,
            "html": html_content,
            "plots": plots,
            "report_dir": str(report_dir),
        }
    )


@app.route("/api/plot/<path:report_path>/<filename>")
def get_plot(report_path: str, filename: str):
    """Serve a plot image from a specific report directory"""
    from urllib.parse import unquote

    # Decode URL-encoded components
    report_path = unquote(report_path)
    filename = unquote(filename)

    # Construct the full path
    report_dir = Path(report_path)
    plot_file = report_dir / filename

    print(f"DEBUG: Serving plot - Report: {report_path}, File: {filename}")
    print(f"DEBUG: Full path: {plot_file}")
    print(f"DEBUG: File exists: {plot_file.exists()}")

    if not plot_file.exists():
        return f"Plot not found: {plot_file}", 404

    return send_file(plot_file, mimetype="image/png")


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    """Start new analysis job"""
    data = request.json

    # Validate required fields
    if not data.get("gpu1_kineto") or not data.get("gpu2_kineto"):
        return jsonify({"error": "Missing required trace files"}), 400

    # Generate job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get current queue size
    with queue_lock:
        queue_size = job_queue.qsize()
        queue_position = queue_size + 1

    # Initialize job
    analysis_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": f"Queued (position #{queue_position})",
        "queue_position": queue_position,
        "created": datetime.now().isoformat(),
        "config": data,
    }

    # Add to queue
    with queue_lock:
        job_queue.put(job_id)

    update_queue_positions()

    print(f"\n{'='*70}")
    print(f"New job queued: {job_id}")
    print(f"Queue position: #{queue_position}")
    print(f"Active jobs: {len(active_jobs)}/{MAX_CONCURRENT_JOBS}")
    print(f"Queue size: {job_queue.qsize()}")
    print(f"{'='*70}\n")

    return jsonify(
        {
            "job_id": job_id,
            "queue_position": queue_position,
            "message": f"Job queued at position #{queue_position}",
        }
    )


@app.route("/api/job/<job_id>")
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in analysis_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = analysis_jobs[job_id]
    return jsonify(
        {
            "id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "queue_position": job.get("queue_position", 0),
            "active_jobs": len(active_jobs),
            "max_concurrent": MAX_CONCURRENT_JOBS,
            "output_dir": job.get("output_dir"),
            "report_path": job.get("report_path"),
            "error": job.get("error"),
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = app.config["UPLOAD_FOLDER"] / filename
    file.save(filepath)

    return jsonify(
        {"filename": filename, "path": str(filepath), "size": filepath.stat().st_size}
    )


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "jarvis-dashboard"})


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Jarvis Web Dashboard Starting...")
    print("=" * 60)
    print(f"📊 Dashboard URL: http://localhost:8080")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER'].absolute()}")
    print(f"⚙️  Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Start queue processor thread
    queue_thread = threading.Thread(target=process_job_queue, daemon=True)
    queue_thread.start()

    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)

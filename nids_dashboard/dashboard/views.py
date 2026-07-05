import csv
import io
from datetime import timedelta

import pandas as pd
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import CSVUploadForm
from .ingest import process_upload
from .ml_engine import get_engine
from .models import Alert, BlockedIP, Detection, EvaluationMetric, UploadBatch


def is_admin(user):
    return user.is_authenticated and user.is_staff


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:index")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.POST.get("next") or request.GET.get("next")
            return redirect(next_url or "dashboard:index")
        messages.error(request, "Invalid username or password.")

    return render(request, "dashboard/login.html", {"next": request.GET.get("next", "")})


def logout_view(request):
    logout(request)
    return redirect("dashboard:login")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

@login_required
def index(request):
    engine = get_engine()
    status = engine.status()

    total_connections = Detection.objects.count()
    active_threats = Alert.objects.filter(status=Alert.STATUS_ACTIVE).count()

    latest_hybrid = (
        EvaluationMetric.objects.filter(model_name__icontains="hybrid").order_by("-recorded_at").first()
    )
    latest_rf = (
        EvaluationMetric.objects.filter(model_name__icontains="random forest")
        .order_by("-recorded_at")
        .first()
    )
    accuracy_metric = latest_hybrid or latest_rf

    recent_detections = Detection.objects.select_related("batch")[:100]

    context = {
        "total_connections": total_connections,
        "active_threats": active_threats,
        "accuracy_pct": f"{accuracy_metric.accuracy * 100:.1f}%" if accuracy_metric else "N/A",
        "accuracy_model_name": accuracy_metric.model_name if accuracy_metric else None,
        "system_health": status["health"],
        "models_loaded": status["rf_ready"],
        "autoencoder_loaded": status["autoencoder_ready"],
        "recent_detections": recent_detections,
    }
    return render(request, "dashboard/index.html", context)


# ---------------------------------------------------------------------------
# JSON APIs consumed by Chart.js on the dashboard page
# ---------------------------------------------------------------------------

@login_required
def api_traffic_overview(request):
    """Line chart: benign vs malicious counts, hourly buckets, last 24h."""
    now = timezone.now()
    start = now - timedelta(hours=24)

    buckets = []
    for h in range(24):
        bucket_start = start + timedelta(hours=h)
        bucket_end = bucket_start + timedelta(hours=1)
        qs = Detection.objects.filter(timestamp__gte=bucket_start, timestamp__lt=bucket_end)
        buckets.append(
            {
                "label": bucket_start.strftime("%H:%M"),
                "benign": qs.filter(verdict=Detection.VERDICT_BENIGN).count(),
                "malicious": qs.exclude(verdict=Detection.VERDICT_BENIGN).count(),
            }
        )

    return JsonResponse(
        {
            "labels": [b["label"] for b in buckets],
            "benign": [b["benign"] for b in buckets],
            "malicious": [b["malicious"] for b in buckets],
        }
    )


@login_required
def api_attack_distribution(request):
    """Doughnut chart: verdict breakdown across all analyzed traffic."""
    qs = Detection.objects.values("verdict").annotate(count=Count("id"))
    counts = {row["verdict"]: row["count"] for row in qs}

    labels_map = dict(Detection.VERDICT_CHOICES)
    data = [
        {"label": labels_map[code], "count": counts.get(code, 0)}
        for code in [
            Detection.VERDICT_BENIGN,
            Detection.VERDICT_KNOWN_ATTACK,
            Detection.VERDICT_ZERO_DAY,
            Detection.VERDICT_CONFIRMED_ATTACK,
        ]
    ]
    return JsonResponse(
        {
            "labels": [d["label"] for d in data],
            "counts": [d["count"] for d in data],
        }
    )


@login_required
def api_algorithm_comparison(request):
    """
    Bar chart: real evaluation metrics only (from evaluation_simple.py /
    EvaluationMetric). We deliberately do NOT fabricate numbers for
    algorithms that were never trained (e.g. SVM, Decision Tree) -- if you
    want those bars, train those baselines first and re-seed this table.
    """
    metrics = EvaluationMetric.objects.order_by("model_name")
    return JsonResponse(
        {
            "labels": [m.model_name for m in metrics],
            "accuracy": [round(m.accuracy * 100, 2) for m in metrics],
            "precision": [round(m.precision * 100, 2) for m in metrics],
            "recall": [round(m.recall * 100, 2) for m in metrics],
        }
    )


@login_required
def api_recent_activity(request):
    detections = Detection.objects.select_related("batch")[:100]
    rows = [
        {
            "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": d.source_ip,
            "destination_ip": d.destination_ip,
            "source_port": d.source_port,
            "destination_port": d.destination_port,
            "verdict": d.get_verdict_display(),
            "severity": d.severity,
            "confidence": round(d.rf_confidence * 100, 1),
            "recon_error": round(d.reconstruction_error, 4),
        }
        for d in detections
    ]
    return JsonResponse({"rows": rows})


# ---------------------------------------------------------------------------
# Upload CSV for analysis
# ---------------------------------------------------------------------------

@login_required
def upload_csv(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data["csv_file"]
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(".parquet"):
                    df = pd.read_parquet(uploaded_file)
                elif fname.endswith(".gz"):
                    df = pd.read_csv(uploaded_file, compression="gzip")
                else:
                    df = pd.read_csv(uploaded_file)
            except Exception as exc:
                messages.error(request, f"Could not parse file: {exc}")
                return render(request, "dashboard/upload.html", {"form": form})

            batch = UploadBatch.objects.create(
                filename=uploaded_file.name,
                uploaded_by=request.user,
            )
            process_upload(batch, df)

            if batch.status == UploadBatch.STATUS_FAILED:
                messages.error(request, f"Analysis failed: {batch.error_message}")
            else:
                messages.success(
                    request,
                    f"Analyzed {batch.row_count} rows: "
                    f"{batch.benign_count} benign, {batch.malicious_count} malicious "
                    f"({batch.processing_ms:.1f} ms).",
                )
            return redirect("dashboard:upload_result", batch_id=batch.id)
    else:
        form = CSVUploadForm()

    recent_batches = UploadBatch.objects.all()[:10]
    return render(request, "dashboard/upload.html", {"form": form, "recent_batches": recent_batches})


@login_required
def upload_result(request, batch_id):
    batch = get_object_or_404(UploadBatch, id=batch_id)
    detections = batch.detections.all()[:500]
    return render(request, "dashboard/upload_result.html", {"batch": batch, "detections": detections})


# ---------------------------------------------------------------------------
# Reports / export
# ---------------------------------------------------------------------------

@login_required
def reports(request):
    batches = UploadBatch.objects.all()
    metrics = EvaluationMetric.objects.order_by("model_name")
    return render(request, "dashboard/reports.html", {"batches": batches, "metrics": metrics})


@login_required
def export_csv(request):
    """Export all Detection rows (optionally filtered by ?batch=<uuid>) as CSV."""
    batch_id = request.GET.get("batch")
    qs = Detection.objects.select_related("batch").order_by("-timestamp")
    if batch_id:
        qs = qs.filter(batch_id=batch_id)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="nids_detections.csv"'
    writer = csv.writer(response)
    writer.writerow(
        ["timestamp", "source_ip", "destination_ip", "source_port", "destination_port",
         "verdict", "severity", "rf_confidence", "reconstruction_error", "true_label"]
    )
    for d in qs.iterator():
        writer.writerow(
            [d.timestamp.isoformat(), d.source_ip, d.destination_ip, d.source_port,
             d.destination_port, d.get_verdict_display(), d.severity,
             f"{d.rf_confidence:.4f}", f"{d.reconstruction_error:.4f}", d.true_label]
        )
    return response


@login_required
def export_pdf(request):
    """One-page PDF summary report using reportlab (pure-python, no system deps)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Hybrid NIDS &mdash; Detection Summary Report", styles["Title"]),
        Paragraph(f"Generated {timezone.now():%Y-%m-%d %H:%M}", styles["Normal"]),
        Spacer(1, 10 * mm),
    ]

    total = Detection.objects.count()
    benign = Detection.objects.filter(verdict=Detection.VERDICT_BENIGN).count()
    malicious = total - benign
    elements.append(Paragraph(
        f"Total analyzed: {total} &nbsp;|&nbsp; Benign: {benign} &nbsp;|&nbsp; "
        f"Malicious/Anomalous: {malicious}", styles["Normal"]
    ))
    elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph("Model Evaluation Metrics", styles["Heading2"]))
    metric_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]]
    for m in EvaluationMetric.objects.order_by("model_name"):
        metric_rows.append([
            m.model_name, f"{m.accuracy:.2%}", f"{m.precision:.2%}", f"{m.recall:.2%}",
            f"{m.f1_score:.2f}", f"{m.false_positive_rate:.2%}", f"{m.false_negative_rate:.2%}",
        ])
    table = Table(metric_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131B2E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph("Most Recent Alerts", styles["Heading2"]))
    alert_rows = [["Time", "Source IP", "Verdict", "Severity"]]
    for a in Alert.objects.select_related("detection").order_by("-created_at")[:20]:
        d = a.detection
        alert_rows.append([d.timestamp.strftime("%Y-%m-%d %H:%M"), d.source_ip,
                            d.get_verdict_display(), d.severity])
    alert_table = Table(alert_rows, hAlign="LEFT")
    alert_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131B2E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    elements.append(alert_table)

    doc.build(elements)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="nids_summary_report.pdf"'
    })


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@login_required
def alerts_view(request):
    status_filter = request.GET.get("status", Alert.STATUS_ACTIVE)
    qs = Alert.objects.select_related("detection")
    if status_filter != "ALL":
        qs = qs.filter(status=status_filter)
    return render(request, "dashboard/alerts.html", {
        "alerts": qs[:200],
        "status_filter": status_filter,
    })


@login_required
@require_POST
def acknowledge_alert(request, alert_id):
    alert = get_object_or_404(Alert, id=alert_id)
    alert.status = Alert.STATUS_ACKNOWLEDGED
    alert.handled_by = request.user
    alert.handled_at = timezone.now()
    alert.save()
    messages.success(request, f"Alert #{alert.id} acknowledged.")
    return redirect("dashboard:alerts")


@login_required
@require_POST
def resolve_alert(request, alert_id):
    alert = get_object_or_404(Alert, id=alert_id)
    alert.status = Alert.STATUS_RESOLVED
    alert.handled_by = request.user
    alert.handled_at = timezone.now()
    alert.save()
    messages.success(request, f"Alert #{alert.id} resolved.")
    return redirect("dashboard:alerts")


# ---------------------------------------------------------------------------
# Access control / IP blocking (admin-only mutations)
# ---------------------------------------------------------------------------

@login_required
def access_control(request):
    blocked = BlockedIP.objects.filter(active=True)
    lifted = BlockedIP.objects.filter(active=False)[:20]
    return render(request, "dashboard/access_control.html", {
        "blocked": blocked, "lifted": lifted, "is_admin": is_admin(request.user),
    })


@login_required
@user_passes_test(is_admin)
@require_POST
def block_ip(request):
    ip_address = request.POST.get("ip_address", "").strip()
    reason = request.POST.get("reason", "").strip()
    if not ip_address:
        messages.error(request, "IP address is required.")
        return redirect("dashboard:access_control")

    BlockedIP.objects.update_or_create(
        ip_address=ip_address,
        defaults={"reason": reason, "blocked_by": request.user, "active": True,
                  "blocked_at": timezone.now()},
    )
    messages.success(request, f"{ip_address} blocked.")
    return redirect("dashboard:access_control")


@login_required
@user_passes_test(is_admin)
@require_POST
def unblock_ip(request, block_id):
    block = get_object_or_404(BlockedIP, id=block_id)
    block.active = False
    block.save(update_fields=["active"])
    messages.success(request, f"{block.ip_address} unblocked.")
    return redirect("dashboard:access_control")

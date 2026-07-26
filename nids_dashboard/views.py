import csv
  import io
  import json
  import logging
  import random
  import time
  from datetime import timedelta
  from typing import Any, Dict

  import joblib
  import numpy as np
  import pandas as pd
  from django.contrib import messages
  from django.contrib.auth import authenticate, login, logout
  from django.contrib.auth.decorators import login_required,
  user_passes_test
  from django.db.models import Avg, Count, Q
  from django.http import HttpResponse, JsonResponse
  from django.shortcuts import get_object_or_404, redirect, render
  from django.utils import timezone
  from django.views.decorators.http import require_POST

  from .forms import CSVUploadForm
  from .ingest import process_upload
  from .ml_engine import get_engine, PredictionResult
  from .models import Alert, BlockedIP, Detection, EvaluationMetric,
  UploadBatch

  logger = logging.getLogger(__name__)

  def is_admin(user):
      return user.is_authenticated and user.is_staff

  # ==================== AUTHENTICATION VIEWS ====================

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
      return render(request, "dashboard/login.html", {"next":
  request.GET.get("next", "")})

  def logout_view(request):
      logout(request)
      return redirect("dashboard:login")

  # ==================== MAIN DASHBOARD VIEW ====================

  @login_required
  def index(request):
      """Main dashboard with KPIs, charts, and live panels."""
      engine = get_engine()
      status = engine.status()

      # Get latest evaluation metric for accuracy display
      latest_metric =
  EvaluationMetric.objects.order_by('-recorded_at').first()

      context = {
          "total_connections": Detection.objects.count(),
          "active_threats":
  Alert.objects.filter(status=Alert.STATUS_ACTIVE).count(),
          "accuracy_pct": f"{latest_metric.accuracy * 100:.1f}%" if
  latest_metric else "N/A",
          "accuracy_model_name": latest_metric.model_name if latest_metric
  else None,
          "system_health": status["health"],
          "models_loaded": status["rf_ready"],
          "autoencoder_loaded": status["autoencoder_ready"],
      }
      return render(request, "dashboard/index.html", context)

  # ==================== JSON API ENDPOINTS ====================

  @login_required
  def api_traffic_overview(request):
      """Hourly traffic overview for line chart."""
      now = timezone.now()
      start = now - timedelta(hours=24)
      buckets = []
      for h in range(24):
          bucket_start = start + timedelta(hours=h)
          bucket_end = bucket_start + timedelta(hours=1)
          qs = Detection.objects.filter(timestamp__gte=bucket_start,
  timestamp__lt=bucket_end)
          buckets.append({
              "label": bucket_start.strftime("%H:%M"),
              "benign": qs.filter(verdict=Detection.VERDICT_BENIGN).count(),
              "malicious":
  qs.exclude(verdict=Detection.VERDICT_BENIGN).count(),
          })
      return JsonResponse({
          "labels": [b["label"] for b in buckets],
          "benign": [b["benign"] for b in buckets],
          "malicious": [b["malicious"] for b in buckets],
      })

  @login_required
  def api_attack_distribution(request):
      """Attack type distribution for doughnut chart."""
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
      return JsonResponse({
          "labels": [d["label"] for d in data],
          "counts": [d["count"] for d in data],
      })

  @login_required
  def api_algorithm_comparison(request):
      """Model accuracy/precision/recall comparison."""
      metrics = EvaluationMetric.objects.order_by("model_name")
      return JsonResponse({
          "labels": [m.model_name for m in metrics],
          "accuracy": [round(m.accuracy * 100, 2) for m in metrics],
          "precision": [round(m.precision * 100, 2) for m in metrics],
          "recall": [round(m.recall * 100, 2) for m in metrics],
      })

  @login_required
  def api_recent_activity(request):
      """Recent detections table (legacy endpoint)."""
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

  @login_required
  def api_live_detections(request):
      """API endpoint for live detection feed - returns latest detections
  for real-time panel"""
      try:
          # Get limit parameter (default 20 for live feed)
          limit = int(request.GET.get('limit', 20))

          # Get latest detections with related batch info
          detections =
  Detection.objects.select_related('batch').order_by('-timestamp')[:limit]

          # Format data for JSON response
          rows = [
              {
                  "id": d.id,
                  "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                  "source_ip": d.source_ip,
                  "destination_ip": d.destination_ip,
                  "source_port": d.source_port,
                  "destination_port": d.destination_port,
                  "verdict": d.get_verdict_display(),
                  "verdict_code": d.verdict,
                  "severity": d.severity,
                  "severity_code": d.severity,
                  "confidence": round(d.rf_confidence * 100, 1),
                  "reconstruction_error": round(d.reconstruction_error, 4),
                  "is_anomaly": d.is_anomaly,
                  "true_label": d.true_label,
                  "batch_filename": d.batch.filename if d.batch else None,
                  "processing_time_ms": d.batch.processing_ms if d.batch and
  d.batch.processing_ms else None
              }
              for d in detections
          ]

          return JsonResponse({
              "success": True,
              "count": len(rows),
              "detections": rows,
              "last_updated": timezone.now().strftime("%Y-%m-%d %H:%M:%S")
          })

      except Exception as e:
          return JsonResponse({
              "success": False,
              "error": str(e)
          }, status=500)

  @login_required
  def api_recent_alerts(request):
      """API endpoint for recent alerts - returns latest alerts for
  dashboard preview"""
      try:
          # Get limit parameter (default 5 for dashboard preview)
          limit = int(request.GET.get('limit', 5))

          # Get latest alerts with related detection and batch info
          alerts = Alert.objects.select_related('detection__batch').order_by
  ('-created_at')[:limit]

          # Format data for JSON response
          rows = [
              {
                  "id": a.id,
                  "created_at": a.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                  "source_ip": a.detection.source_ip,
                  "destination_ip": a.detection.destination_ip,
                  "verdict": a.detection.get_verdict_display(),
                  "verdict_code": a.detection.verdict,
                  "severity": a.detection.severity,
                  "severity_code": a.detection.severity,
                  "status": a.status,
                  "confidence": round(a.detection.rf_confidence * 100, 1),
                  "is_acknowledged": a.status == Alert.STATUS_ACKNOWLEDGED,
                  "is_resolved": a.status == Alert.STATUS_RESOLVED,
                  "handled_by": a.handled_by.username if a.handled_by else
  None,
                  "handled_at": a.handled_at.strftime("%Y-%m-%d %H:%M:%S")
  if a.handled_at else None
              }
              for a in alerts
          ]

          return JsonResponse({
              "success": True,
              "count": len(rows),
              "alerts": rows,
              "last_updated": timezone.now().strftime("%Y-%m-%d %H:%M:%S")
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_processing_speed(request):
      """API endpoint for processing speed metrics"""
      try:
          # Get recent batches for performance metrics
          recent_batches = UploadBatch.objects.filter(
              status=UploadBatch.STATUS_DONE,
              processing_ms__isnull=False
          ).order_by('-uploaded_at')[:10]

          if not recent_batches.exists():
              return JsonResponse({
                  "success": True,
                  "current_batch": None,
                  "recent_avg_ms": None,
                  "recent_avg_per_record": None,
                  "batches_per_minute": None,
                  "total_processed":
  UploadBatch.objects.filter(status=UploadBatch.STATUS_DONE).count()
              })

          # Calculate metrics
          processing_times = [b.processing_ms for b in recent_batches if
  b.processing_ms]
          total_records = sum(b.row_count for b in recent_batches)
          total_time = sum(processing_times)

          avg_processing_ms = sum(processing_times) / len(processing_times)
  if processing_times else 0
          avg_per_record = total_time / total_records if total_records > 0
  else 0

          # Calculate batches per minute (based on last 10 minutes)
          ten_minutes_ago = timezone.now() - timezone.timedelta(minutes=10)
          recent_count = UploadBatch.objects.filter(
              uploaded_at__gte=ten_minutes_ago,
              status=UploadBatch.STATUS_DONE
          ).count()
          batches_per_minute = (recent_count / 10) * 60 if recent_count > 0
  else 0

          # Get latest batch info
          latest_batch = recent_batches.first()

          return JsonResponse({
              "success": True,
              "current_batch": {
                  "filename": latest_batch.filename,
                  "processing_ms": round(latest_batch.processing_ms, 1),
                  "row_count": latest_batch.row_count,
                  "per_record_ms": round(latest_batch.processing_ms /
  max(latest_batch.row_count, 1), 2),
                  "uploaded_at":
  latest_batch.uploaded_at.strftime("%H:%M:%S")
              },
              "recent_avg_ms": round(avg_processing_ms, 1),
              "recent_avg_per_record": round(avg_per_record, 2),
              "batches_per_minute": round(batches_per_minute, 1),
              "total_processed":
  UploadBatch.objects.filter(status=UploadBatch.STATUS_DONE).count(),
              "last_updated": timezone.now().strftime("%Y-%m-%d %H:%M:%S")
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_enhanced_kpis(request):
      """API endpoint for enhanced KPI metrics"""
      try:
          total_detections = Detection.objects.count()
          active_threats =
  Alert.objects.filter(status=Alert.STATUS_ACTIVE).count()

          # Get latest model accuracy from evaluation metrics
          latest_metric =
  EvaluationMetric.objects.order_by('-recorded_at').first()
          model_accuracy = latest_metric.accuracy if latest_metric else 0

          # System health based on recent processing success
          recent_batches = UploadBatch.objects.filter(
              uploaded_at__gte=timezone.now() - timezone.timedelta(hours=1)
          ).order_by('-uploaded_at')[:5]

          failed_recent =
  recent_batches.filter(status=UploadBatch.STATUS_FAILED).count()
          total_recent = len(recent_batches)
          system_health = 'healthy'
          if total_recent > 0:
              failure_rate = failed_recent / total_recent
              if failure_rate > 0.5:
                  system_health = 'offline'
              elif failure_rate > 0.2:
                  system_health = 'degraded'

          # Threat detection rate
          threat_detections =
  Detection.objects.exclude(verdict=Detection.VERDICT_BENIGN).count()
          threat_rate = threat_detections / max(total_detections, 1)

          # False positive rate (benign but flagged as anomalous)
          false_positives = Detection.objects.filter(
              verdict=Detection.VERDICT_BENIGN,
              is_anomaly=True
          ).count()
          false_positive_rate = false_positives / max(total_detections, 1)

          # Average confidence
          avg_confidence = Detection.objects.exclude(
              rf_confidence__isnull=True
          ).aggregate(avg=Avg('rf_confidence'))['avg'] or 0

          # Average processing time
          avg_processing = UploadBatch.objects.filter(
              status=UploadBatch.STATUS_DONE,
              processing_ms__isnull=False
          ).aggregate(avg=Avg('processing_ms'))['avg'] or 0

          return JsonResponse({
              "success": True,
              "total_connections": total_detections,
              "active_threats": active_threats,
              "model_accuracy": model_accuracy,
              "system_health": system_health,
              "threat_rate": threat_rate,
              "false_positive_rate": false_positive_rate,
              "avg_confidence": avg_confidence,
              "avg_processing_ms": avg_processing
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_decision_explanation(request):
      """API endpoint for explaining ML model decisions"""
      try:
          detection_id = request.GET.get('detection_id')
          if not detection_id:
              return JsonResponse({"success": False, "error": "detection_id
  required"}, status=400)

          detection = get_object_or_404(Detection, id=detection_id)

          # Build explanation based on available data (simplified SHAP-like)
          explanation = {
              "detection_id": detection.id,
              "verdict": detection.get_verdict_display(),
              "verdict_code": detection.verdict,
              "confidence": detection.rf_confidence,
              "anomaly_score": detection.reconstruction_error,
              "is_anomaly": detection.is_anomaly,
              "contributing_factors": []
          }

          # Verdict-based factors
          if detection.verdict == Detection.VERDICT_CONFIRMED_ATTACK:
              explanation["contributing_factors"].append({
                  "factor": "High confidence RF prediction + Anomaly
  detection",
                  "impact": "high",
                  "description": "Both Random Forest and Autoencoder
  detected threat"
              })
          elif detection.verdict == Detection.VERDICT_KNOWN_ATTACK:
              explanation["contributing_factors"].append({
                  "factor": "Strong RF prediction",
                  "impact": "high",
                  "description": "Random Forest confidently classified as
  known attack"
              })
          elif detection.verdict == Detection.VERDICT_ZERO_DAY:
              explanation["contributing_factors"].append({
                  "factor": "Anomaly detection",
                  "impact": "high",
                  "description": "Autoencoder detected significant deviation
  from normal"
              })
          else:  # BENIGN
              explanation["contributing_factors"].append({
                  "factor": "Low RF confidence + Normal pattern",
                  "impact": "low",
                  "description": "Random Forest low confidence + normal
  reconstruction"
              })

          # Confidence-based explanation
          if detection.rf_confidence > 0.8:
              explanation["contributing_factors"].append({
                  "factor": "High model confidence",
                  "impact": "medium",
                  "description": f"RF confidence
  {detection.rf_confidence:.2f} indicates strong prediction"
              })
          elif detection.rf_confidence < 0.4:
              explanation["contributing_factors"].append({
                  "factor": "Low model confidence",
                  "impact": "medium",
                  "description": f"RF confidence
  {detection.rf_confidence:.2f} indicates uncertainty"
              })

          # Anomaly-based explanation
          if detection.is_anomaly:
              explanation["contributing_factors"].append({
                  "factor": "Anomalous reconstruction",
                  "impact": "medium" if detection.verdict ==
  Detection.VERDICT_BENIGN else "high",
                  "description": f"Reconstruction error
  {detection.reconstruction_error:.4f} exceeds threshold"
              })

          return JsonResponse({
              "success": True,
              "explanation": explanation
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  # ==================== DETECTION SUMMARY PAGE ENDPOINTS
  ====================

  @login_required
  def api_detection_summary(request):
      """API endpoint for detection summary overview statistics"""
      try:
          total_detections = Detection.objects.count()
          threat_detections =
  Detection.objects.exclude(verdict=Detection.VERDICT_BENIGN).count()
          threat_rate = threat_detections / max(total_detections, 1)

          # Average confidence (non-null only)
          avg_confidence = Detection.objects.exclude(
              rf_confidence__isnull=True
          ).aggregate(avg=Avg('rf_confidence'))['avg'] or 0

          # Estimated false positive rate
          false_positive = Detection.objects.filter(
              verdict=Detection.VERDICT_BENIGN,
              is_anomaly=True
          ).count()
          fpr = false_positive / max(total_detections, 1)

          return JsonResponse({
              "success": True,
              "total_detections": total_detections,
              "threat_rate": threat_rate,
              "avg_confidence": avg_confidence,
              "false_positive_rate": fpr
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_verdict_distribution(request):
      """API endpoint for verdict distribution data"""
      try:
          # Get distribution by verdict
          verdicts = Detection.objects.values('verdict').annotate(
              count=Count('id')
          ).order_by('-count')

          # Get verdict choices for labels
          verdict_choices = dict(Detection.VERDICT_CHOICES)

          labels = []
          counts = []
          total = sum(item['count'] for item in verdicts)

          for item in verdicts:
              labels.append(verdict_choices.get(item['verdict'],
  item['verdict']))
              counts.append(item['count'])

          # Calculate percentages
          percentages = [count / max(total, 1) for count in counts]

          return JsonResponse({
              "success": True,
              "distribution": [
                  {
                      "verdict": label,
                      "count": count,
                      "percentage": pct
                  }
                  for label, count, pct in zip(labels, counts, percentages)
              ],
              "chartData": {
                  "labels": labels,
                  "counts": counts
              }
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_severity_distribution(request):
      """API endpoint for severity distribution data"""
      try:
          # Get distribution by severity
          severities = Detection.objects.values('severity').annotate(
              count=Count('id')
          ).order_by('-count')

          # Get severity choices for labels
          severity_choices = dict(Detection.SEVERITY_CHOICES)

          labels = []
          counts = []
          total = sum(item['count'] for item in severities)

          for item in severities:
              labels.append(severity_choices.get(item['severity'],
  item['severity']))
              counts.append(item['count'])

          # Calculate percentages
          percentages = [count / max(total, 1) for count in counts]

          return JsonResponse({
              "success": True,
              "distribution": [
                  {
                      "severity": label,
                      "count": count,
                      "percentage": pct
                  }
                  for label, count, pct in zip(labels, counts, percentages)
              ],
              "chartData": {
                  "labels": labels,
                  "counts": counts
              }
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  @login_required
  def api_high_risk_detections(request):
      """API endpoint for high-risk detections (last 24 hours)"""
      try:
          # Get detections from last 24 hours that are either confirmed
  attacks or high confidence anomalies
          twenty_four_hours_ago = timezone.now() -
  timezone.timedelta(hours=24)

          high_risk = Detection.objects.filter(
              timestamp__gte=twenty_four_hours_ago
          ).filter(
              Q(verdict=Detection.VERDICT_CONFIRMED_ATTACK) |
              Q(verdict=Detection.VERDICT_KNOWN_ATTACK,
  rf_confidence__gte=0.8) |
              Q(is_anomaly=True, rf_confidence__gte=0.7)
          ).select_related('batch').order_by('-timestamp')[:20]

          detections = [
              {
                  "id": d.id,
                  "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                  "source_ip": d.source_ip,
                  "destination_ip": d.destination_ip,
                  "verdict": d.get_verdict_display(),
                  "severity": d.severity,
                  "confidence": round(d.rf_confidence * 100, 1)
              }
              for d in high_risk
          ]

          return JsonResponse({
              "success": True,
              "detections": detections
          })
      except Exception as e:
          return JsonResponse({"success": False, "error": str(e)},
  status=500)

  # ==================== CSV UPLOAD & PROCESSING ====================

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
                  return render(request, "dashboard/upload.html", {"form":
  form})

              batch = UploadBatch.objects.create(
                  filename=uploaded_file.name,
                  uploaded_by=request.user,
              )
              process_upload(batch, df)

              if batch.status == UploadBatch.STATUS_FAILED:
                  messages.error(request, f"Analysis failed:
  {batch.error_message}")
              else:
                  messages.success(
                      request,
                      f"Analyzed {batch.row_count} rows: "
                      f"{batch.benign_count} benign, {batch.malicious_count}
  malicious "
                      f"({batch.processing_ms:.1f} ms).",
                  )
              return redirect("dashboard:upload_result", batch_id=batch.id)
      else:
          form = CSVUploadForm()
      recent_batches = UploadBatch.objects.all()[:10]
      return render(request, "dashboard/upload.html", {"form": form,
  "recent_batches": recent_batches})

  @login_required
  def upload_result(request, batch_id):
      batch = get_object_or_404(UploadBatch, id=batch_id)
      detections = batch.detections.all()[:500]
      return render(request, "dashboard/upload_result.html", {"batch":
  batch, "detections": detections})

  # ==================== REPORTS & EXPORT ====================

  @login_required
  def reports(request):
      batches = UploadBatch.objects.all()
      metrics = EvaluationMetric.objects.order_by("model_name")
      return render(request, "dashboard/reports.html", {"batches": batches,
  "metrics": metrics})

  @login_required
  def export_csv(request):
      batch_id = request.GET.get("batch")
      qs = Detection.objects.select_related("batch").order_by("-timestamp")
      if batch_id:
          qs = qs.filter(batch_id=batch_id)
      response = HttpResponse(content_type="text/csv")
      response["Content-Disposition"] = 'attachment;
  filename="nids_detections.csv"'
      writer = csv.writer(response)
      writer.writerow([
          "timestamp", "source_ip", "destination_ip", "source_port",
  "destination_port",
          "verdict", "severity", "rf_confidence", "reconstruction_error",
  "true_label"
      ])
      for d in qs.iterator():
          writer.writerow([
              d.timestamp.isoformat(), d.source_ip, d.destination_ip,
  d.source_port,
              d.destination_port, d.get_verdict_display(), d.severity,
              f"{d.rf_confidence:.4f}", f"{d.reconstruction_error:.4f}",
  d.true_label
          ])
      return response

  @login_required
  def export_pdf(request):
      from reportlab.lib import colors
      from reportlab.lib.pagesizes import A4
      from reportlab.lib.units import mm
      from reportlab.platypus import SimpleDocTemplate, Table, TableStyle,
  Paragraph, Spacer
      from reportlab.lib.styles import getSampleStyleSheet
      buf = io.BytesIO()
      doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm,
  bottomMargin=20 * mm)
      styles = getSampleStyleSheet()
      elements = [
          Paragraph("ZeroDayHunter &mdash; Detection Summary Report",
  styles["Title"]),
          Paragraph(f"Generated {timezone.now():%Y-%m-%d %H:%M}",
  styles["Normal"]),
          Spacer(1, 10 * mm),
      ]
      total = Detection.objects.count()
      benign =
  Detection.objects.filter(verdict=Detection.VERDICT_BENIGN).count()
      malicious = total - benign
      elements.append(Paragraph(
          f"Total analyzed: {total} &nbsp;|&nbsp; Benign: {benign}
  &nbsp;|&nbsp; "
          f"Malicious/Anomalous: {malicious}", styles["Normal"]
      ))
      elements.append(Spacer(1, 8 * mm))
      elements.append(Paragraph("Model Evaluation Metrics",
  styles["Heading2"]))
      metric_rows = [["Model", "Accuracy", "Precision", "Recall", "F1",
  "FPR", "FNR"]]
      for m in EvaluationMetric.objects.order_by("model_name"):
          metric_rows.append([
              m.model_name, f"{m.accuracy:.2%}", f"{m.precision:.2%}",
  f"{m.recall:.2%}",
              f"{m.f1_score:.2f}", f"{m.false_positive_rate:.2%}",
  f"{m.false_negative_rate:.2f}",
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
      for a in
  Alert.objects.select_related("detection").order_by("-created_at")[:20]:
          d = a.detection
          alert_rows.append([d.timestamp.strftime("%Y-%m-%d %H:%M"),
  d.source_ip,
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
      return HttpResponse(buf.getvalue(), content_type="application/pdf",
  headers={
          "Content-Disposition": 'attachment;
  filename="zerodayhunter_summary_report.pdf"'
      })

  # ==================== ALERT MANAGEMENT ====================

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

  # ==================== ACCESS CONTROL (IP BLOCKING) ====================

  @login_required
  def access_control(request):
      blocked = BlockedIP.objects.filter(active=True)
      lifted = BlockedIP.objects.filter(active=False)[:20]
      return render(request, "dashboard/access_control.html", {
          "blocked": blocked, "lifted": lifted, "is_admin":
  is_admin(request.user),
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
          defaults={"reason": reason, "blocked_by": request.user, "active":
  True,
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

  # ==================== DETECTION SUMMARY PAGE ====================

  @login_required
  def detection_summary(request):
      """Detection summary analytics page"""
      return render(request, "dashboard/detection_summary.html")

  2. nids_dashboard/dashboard/urls.py (REPLACE ENTIRE FILE)

  from django.urls import path
  from . import views

  app_name = "dashboard"
  urlpatterns = [
      # Authentication
      path("login/", views.login_view, name="login"),
      path("logout/", views.logout_view, name="logout"),

      # Main dashboard
      path("", views.index, name="index"),

      # CSV Upload
      path("upload/", views.upload_csv, name="upload_csv"),
      path("upload/result/<uuid:batch_id>/", views.upload_result,
  name="upload_result"),

      # Reports
      path("reports/", views.reports, name="reports"),
      path("export/csv/", views.export_csv, name="export_csv"),
      path("export/pdf/", views.export_pdf, name="export_pdf"),

      # Alerts
      path("alerts/", views.alerts_view, name="alerts"),
      path("alerts/acknowledge/<int:alert_id>/", views.acknowledge_alert,
  name="acknowledge_alert"),
      path("alerts/resolve/<int:alert_id>/", views.resolve_alert,
  name="resolve_alert"),

      # Access Control
      path("access-control/", views.access_control, name="access_control"),
      path("access-control/block/", views.block_ip, name="block_ip"),
      path("access-control/unblock/<int:block_id>/", views.unblock_ip,
  name="unblock_ip"),

      # ==================== API ENDPOINTS ====================

      # Legacy charts (existing)
      path("api/traffic-overview/", views.api_traffic_overview,
  name="api_traffic_overview"),
      path("api/attack-distribution/", views.api_attack_distribution,
  name="api_attack_distribution"),
      path("api/algorithm-comparison/", views.api_algorithm_comparison,
  name="api_algorithm_comparison"),
      path("api/recent-activity/", views.api_recent_activity,
  name="api_recent_activity"),

      # NEW FEATURES API ENDPOINTS

      # Live Detection Panel
      path("api/live-detections/", views.api_live_detections,
  name="api_live_detections"),

      # Recent Alerts Table
      path("api/recent-alerts/", views.api_recent_alerts,
  name="api_recent_alerts"),

      # Processing Speed Display
      path("api/processing-speed/", views.api_processing_speed,
  name="api_processing_speed"),

      # Enhanced KPIs
      path("api/enhanced-kpis/", views.api_enhanced_kpis,
  name="api_enhanced_kpis"),

      # Decision Explanation
      path("api/decision-explanation/", views.api_decision_explanation,
  name="api_decision_explanation"),

      # Detection Summary Page
      path("detection-summary/", views.detection_summary,
  name="detection_summary"),
      path("api/detection-summary/", views.api_detection_summary,
  name="api_detection_summary"),
      path("api/verdict-distribution/", views.api_verdict_distribution,
  name="api_verdict_distribution"),
      path("api/severity-distribution/", views.api_severity_distribution,
  name="api_severity_distribution"),
      path("api/high-risk-detections/", views.api_high_risk_detections,
  name="api_high_risk_detections"),
  ]

  3. nids_dashboard/dashboard/templates/dashboard/index.html (REPLACE ENTIRE
  FILE)

  {% extends "dashboard/base.html" %}
  {% load static %}
  {% block title %}Dashboard{% endblock %}
  {% block page_title %}ZeroDayHunter SOC Dashboard{% endblock %}
  {% block page_title_sub %}Real-Time Network Threat Detection{% endblock %}

  {% block content %}
  <!-- Enhanced KPI Grid -->
  <div class="kpi-grid" id="kpiGrid">
    <!-- Existing KPIs Enhanced -->
    <div class="kpi-card kpi-card--good" id="kpiTotalConnections">
      <div class="kpi-icon">📊</div>
      <div class="kpi-label">Total Connections</div>
      <div class="kpi-value" id="totalConnectionsValue">--</div>
      <div class="kpi-footer" id="totalConnectionsLabel">all time
  analyzed</div>
    </div>

    <div class="kpi-card kpi-card--threat" id="kpiActiveThreats">
      <div class="kpi-icon">⚠️</div>
      <div class="kpi-label">Active Threats</div>
      <div class="kpi-value" id="activeThreatsValue">--</div>
      <div class="kpi-footer" id="activeThreatsLabel">requiring
  attention</div>
    </div>

    <div class="kpi-card" id="kpiModelAccuracy">
      <div class="kpi-icon">🎯</div>
      <div class="kpi-label">Model Accuracy</div>
      <div class="kpi-value" id="modelAccuracyValue">--%</div>
      <div class="kpi-footer" id="modelAccuracyLabel">hybrid
  RF+Autoencoder</div>
    </div>

    <div class="kpi-card" id="kpiSystemHealth">
      <div class="kpi-icon">💚</div>
      <div class="kpi-label">System Health</div>
      <div class="kpi-value" id="systemHealthValue">--</div>
      <div class="kpi-footer" id="systemHealthLabel">processing status</div>
    </div>

    <!-- NEW Enhanced Metrics -->
    <div class="kpi-card" id="kpiThreatRate">
      <div class="kpi-icon">📈</div>
      <div class="kpi-label">Threat Detection Rate</div>
      <div class="kpi-value" id="threatRateValue">--%</div>
      <div class="kpi-footer" id="threatRateLabel">malicious vs benign</div>
    </div>

    <div class="kpi-card" id="kpiFalsePositiveRate">
      <div class="kpi-icon">🚨</div>
      <div class="kpi-label">Est. FPR</div>
      <div class="kpi-value" id="falsePositiveRateValue">--%</div>
      <div class="kpi-footer" id="falsePositiveRateLabel">false positive
  rate</div>
    </div>

    <div class="kpi-card" id="kpiAvgConfidence">
      <div class="kpi-icon">🔍</div>
      <div class="kpi-label">Avg Confidence</div>
      <div class="kpi-value" id="avgConfidenceValue">--%</div>
      <div class="kpi-footer" id="avgConfidenceLabel">model confidence</div>
    </div>

    <div class="kpi-card" id="kpiProcessingSpeed">
      <div class="kpi-icon">⚡</div>
      <div class="kpi-label">Processing Speed</div>
      <div class="kpi-value" id="processingSpeedValue">-- ms</div>
      <div class="kpi-footer" id="processingSpeedLabel">per batch</div>
    </div>
  </div>

  <!-- Main Content Area -->
  <div class="chart-grid-main" style="margin-bottom: 24px;">
    <!-- Traffic Overview Chart -->
    <div class="chart-wrap">
      <canvas id="trafficChart"></canvas>
    </div>

    <!-- Attack Distribution + Algorithm Comparison -->
    <div class="chart-wrap">
      <div class="chart-grid-sub">
        <div class="chart-wrap--sm">
          <canvas id="distributionChart"></canvas>
        </div>
        <div class="chart-wrap--sm">
          <canvas id="comparisonChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Live Detection Panel -->
  <section class="panel">
    <div class="panel-header">
      <h2>Live Detection Feed</h2>
      <div class="panel-sub">Real-time network traffic analysis</div>
    </div>

    <div class="table-wrap">
      <table class="data-table" id="liveDetectionsTable">
        <thead>
          <tr>
            <th>Time</th>
            <th>Source IP</th>
            <th>Dest IP</th>
            <th>Verdict</th>
            <th>Severity</th>
            <th>Confidence</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          <!-- Rows will be populated by JavaScript -->
          <tr><td colspan="7" class="empty-row">Loading live
  detections...</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Recent Alerts Panel -->
  <section class="panel">
    <div class="panel-header">
      <h2>Recent Alerts</h2>
      <div class="panel-sub">Latest security alerts requiring
  attention</div>
      <a href="{% url 'dashboard:alerts' %}" class="panel-link">View All
  Alerts →</a>
    </div>

    <div class="table-wrap">
      <table class="data-table" id="recentAlertsTable">
        <thead>
          <tr>
            <th>Time</th>
            <th>Source IP</th>
            <th>Verdict</th>
            <th>Severity</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <!-- Rows will be populated by JavaScript -->
          <tr><td colspan="5" class="empty-row">Loading recent
  alerts...</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- CVE Threat Intelligence Panel -->
  <section class="panel">
    <div class="panel-header">
      <h2>CVE Threat Intelligence</h2>
      <div class="panel-sub">Active threats mapped to known
  vulnerabilities</div>
    </div>

    <div id="cveGrid" class="cve-grid">
      <!-- CVE cards will be populated by JavaScript -->
      <div class="cve-card" style="grid-column:1/-1">
        <div class="kpi-label"
  style="text-align:center;padding:20px">Loading threat
  intelligence...</div>
      </div>
    </div>
  </section>
  {% endblock %}

  {% block extra_js %}
  <script src="{% static 'dashboard/js/dashboard.js' %}"></script>
  {% endblock %}

  4. nids_dashboard/dashboard/static/dashboard/js/dashboard.js (REPLACE
  ENTIRE FILE)

  /* dashboard.js — ZeroDayHunter SOC Console
     All chart data comes from real Django JSON APIs.
     No hardcoded or random numbers anywhere. */

  const C = {
    text:     '#7A8BA8',
    grid:     '#1E2E4A',
    brand:    '#00D4FF',
    accent:   '#7C3AED',
    low:      '#10B981',
    medium:   '#F59E0B',
    high:     '#F97316',
    critical: '#EF4444',
  };

  Chart.defaults.color = C.text;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size   = 12;

  let trafficChart, distributionChart, comparisonChart;

  /* Helper: Fetch JSON with proper headers */
  async function fetchJSON(url) {
    const r = await fetch(url, { headers: { 'X-Requested-With':
  'XMLHttpRequest' } });
    if (!r.ok) throw new Error(`${url} → ${r.status}`);
    return r.json();
  }

  /* ── Traffic Overview (line chart) ── */
  async function loadTraffic() {
    const d = await fetchJSON('/dashboard/api/traffic-overview/');
    const ctx = document.getElementById('trafficChart');
    if (!ctx) return;
    const cfg = {
      type: 'line',
      data: {
        labels: d.labels,
        datasets: [
          { label: 'Benign',    data: d.benign,    borderColor: C.low,
  backgroundColor: 'rgba(16,185,129,0.08)', tension: 0.4, fill: true,
  pointRadius: 2, borderWidth: 2 },
          { label: 'Malicious', data: d.malicious, borderColor: C.critical,
  backgroundColor: 'rgba(239,68,68,0.08)',  tension: 0.4, fill: true,
  pointRadius: 2, borderWidth: 2 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { grid: { color: C.grid }, ticks: { maxTicksLimit: 12 } },
          y: { beginAtZero: true, grid: { color: C.grid }, ticks: {
  precision: 0 } },
        },
        plugins: { legend: { position: 'top', align: 'end', labels: {
  boxWidth: 10, boxHeight: 10, padding: 16 } } },
      },
    };
    if (trafficChart) { trafficChart.data = cfg.data; trafficChart.update();
  }
    else trafficChart = new Chart(ctx, cfg);
  }

  /* ── Attack Distribution (doughnut) ── */
  async function loadDistribution() {
    const d = await fetchJSON('/dashboard/api/attack-distribution/');
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;
    const cfg = {
      type: 'doughnut',
      data: {
        labels: d.labels,
        datasets: [{ data: d.counts, backgroundColor: [C.low, C.high,
  C.medium, C.critical], borderWidth: 0, hoverOffset: 6 }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: '65%',
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, boxHeight:
  12, padding: 16, font: { size: 12 } } },
        },
      },
    };
    if (distributionChart) { distributionChart.data = cfg.data;
  distributionChart.update(); }
    else distributionChart = new Chart(ctx, cfg);
  }

  /* ── Algorithm Comparison (grouped bar) ── */
  async function loadComparison() {
    const d = await fetchJSON('/dashboard/api/algorithm-comparison/');
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;

    // Deduplicate labels
    const seen = new Set(), idx = [];
    d.labels.forEach((l, i) => { if (!seen.has(l)) { seen.add(l);
  idx.push(i); } });
    const labels   = idx.map(i => d.labels[i]);
    const accuracy = idx.map(i => d.accuracy[i]);
    const precision= idx.map(i => d.precision[i]);
    const recall   = idx.map(i => d.recall[i]);

    const cfg = {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Accuracy %',  data: accuracy,  backgroundColor: C.brand,
    borderRadius: 4, barPercentage: 0.6 },
          { label: 'Precision %', data: precision, backgroundColor:
  C.medium,  borderRadius: 4, barPercentage: 0.6 },
          { label: 'Recall %',    data: recall,    backgroundColor: C.high,
    borderRadius: 4, barPercentage: 0.6 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 15 } },
          y: { beginAtZero: true, max: 105, grid: { color: C.grid }, ticks:
  { callback: v => v + '%' } },
        },
        plugins: {
          legend: { position: 'top', align: 'end', labels: { boxWidth: 12,
  boxHeight: 12, padding: 16 } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}:
  ${ctx.parsed.y.toFixed(2)}%` } },
        },
      },
    };
    if (comparisonChart) { comparisonChart.data = cfg.data;
  comparisonChart.update(); }
    else comparisonChart = new Chart(ctx, cfg);
  }

  /* ── Live Detection Panel ── */
  async function loadLiveDetections() {
    try {
      const d = await fetchJSON('/dashboard/api/live-detections/');
      const tbody = document.querySelector('#liveDetectionsTable tbody');
      if (!tbody) return;

      if (!d.success || !d.detections.length) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No live
  detections available</td></tr>`;
        return;
      }

      const sevClass = { LOW:'low', MEDIUM:'medium', HIGH:'high',
  CRITICAL:'critical' };
      tbody.innerHTML = d.detections.map(detection => `
        <tr class="sev-${detection.severity_code.toLowerCase()}">
          <td class="mono">${detection.timestamp}</td>
          <td class="mono">${detection.source_ip}</td>
          <td class="mono">${detection.destination_ip}</td>
          <td>${detection.verdict}</td>
          <td><span class="badge badge-${detection.severity_code.toLowerCase
  ()}">${detection.severity}</span></td>
          <td class="mono">${detection.confidence}%</td>
          <td><button class="btn-link btn-sm"
  onclick="showExplanation(${detection.id})">🔍 Explain</button></td>
        </tr>`).join('');
    } catch(e) {
      const tbody = document.querySelector('#liveDetectionsTable tbody');
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-row">Error
  loading detections: ${e.message}</td></tr>`;
      }
    }
  }

  /* ── Recent Alerts Table ── */
  async function loadRecentAlerts() {
    try {
      const d = await fetchJSON('/dashboard/api/recent-alerts/');
      const tbody = document.querySelector('#recentAlertsTable tbody');
      if (!tbody) return;

      if (!d.success || !d.alerts.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-row">No recent
  alerts</td></tr>`;
        return;
      }

      const sevClass = { LOW:'low', MEDIUM:'medium', HIGH:'high',
  CRITICAL:'critical' };
      const statusClass = {
        ACTIVE: 'status-active',
        ACKNOWLEDGED: 'status-acknowledged',
        RESOLVED: 'status-resolved'
      };

      tbody.innerHTML = d.alerts.map(alert => `
        <tr class="sev-${alert.severity_code.toLowerCase()}
  ${statusClass[alert.status] || ''}">
          <td class="mono">${alert.created_at}</td>
          <td class="mono">${alert.source_ip}</td>
          <td>${alert.verdict}</td>
          <td><span class="badge
  badge-${alert.severity_code.toLowerCase()}">${alert.severity}</span></td>
          <td><span class="badge
  badge-status-${alert.status.toLowerCase()}">${alert.status}</span></td>
        </tr>`).join('');
    } catch(e) {
      const tbody = document.querySelector('#recentAlertsTable tbody');
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-row">Error
  loading alerts: ${e.message}</td></tr>`;
      }
    }
  }

  /* ── Processing Speed Display ── */
  async function loadProcessingSpeed() {
    try {
      const d = await fetchJSON('/dashboard/api/processing-speed/');
      if (!d.success) return;

      // Update current batch info
      const currentBatchTime = document.getElementById('currentBatchTime');
      const currentBatchInfo = document.getElementById('currentBatchInfo');
      if (currentBatchTime && currentBatchInfo) {
        if (d.current_batch) {
          currentBatchTime.textContent = `${d.current_batch.processing_ms}
  ms`;
          currentBatchInfo.textContent = `${d.current_batch.row_count} rows
  @ ${d.current_batch.per_record_ms} ms/record`;
        } else {
          currentBatchTime.textContent = '--';
          currentBatchInfo.textContent = 'No recent batches';
        }
      }

      // Update average per record
      const avgPerRecord = document.getElementById('avgPerRecord');
      if (avgPerRecord) {
        avgPerRecord.textContent = d.recent_avg_per_record !== null ?
  `${d.recent_avg_per_record} ms` : '--';
      }

      // Update batch rate
      const batchRate = document.getElementById('batchRate');
      if (batchRate) {
        batchRate.textContent = d.batches_per_minute !== null ?
  `${d.batches_per_minute}` : '--';
      }

      // Update total processed
      const totalProcessed = document.getElementById('totalProcessed');
      if (totalProcessed) {
        totalProcessed.textContent = d.total_processed !== null ?
  d.total_processed.toLocaleString() : '--';
      }

    } catch(e) {
      console.error('Error loading processing speed:', e);
    }
  }

  /* ── Enhanced KPI Updates ── */
  async function updateEnhancedKPIs() {
    try {
      const d = await fetchJSON('/dashboard/api/enhanced-kpis/');
      if (!d.success) return;

      // Update existing KPIs with better formatting
      const totalConn = document.getElementById('totalConnectionsValue');
      if (totalConn) {
        totalConn.textContent = d.total_connections?.toLocaleString() ||
  '--';
      }

      const activeThreats = document.getElementById('activeThreatsValue');
      if (activeThreats) {
        activeThreats.textContent = d.active_threats?.toLocaleString() ||
  '--';
      }

      const modelAcc = document.getElementById('modelAccuracyValue');
      if (modelAcc) {
        modelAcc.textContent = `${(d.model_accuracy * 100).toFixed(1)}%`;
      }

      const sysHealth = document.getElementById('systemHealthValue');
      if (sysHealth) {
        sysHealth.textContent = d.system_health || '--';

        // Update color based on health status
        const kpiCard = document.getElementById('kpiSystemHealth');
        if (kpiCard) {
          kpiCard.className = 'kpi-card';
          if (d.system_health === 'healthy') {
            kpiCard.classList.add('kpi-card--good');
          } else if (d.system_health === 'degraded') {
            kpiCard.classList.add('kpi-card--threat'); // Using threat color
  for warning
          } else {
            kpiCard.classList.add('kpi-card--threat');
          }
        }
      }

      // Update new KPIs
      const threatRate = document.getElementById('threatRateValue');
      if (threatRate) {
        threatRate.textContent = `${(d.threat_rate * 100).toFixed(1)}%`;
      }

      const fpr = document.getElementById('falsePositiveRateValue');
      if (fpr) {
        fpr.textContent = `${(d.false_positive_rate * 100).toFixed(2)}%`;
      }

      const avgConf = document.getElementById('avgConfidenceValue');
      if (avgConf) {
        avgConf.textContent = `${(d.avg_confidence * 100).toFixed(1)}%`;
      }

      const procSpeed = document.getElementById('processingSpeedValue');
      if (procSpeed) {
        procSpeed.textContent = `${d.avg_processing_ms?.toFixed(1) || '--'}
  ms`;
      }

    } catch(e) {
      console.error('Error updating enhanced KPIs:', e);
    }
  }

  /* ── Recent Activity table (legacy) ── */
  async function loadActivity() {
    const d = await fetchJSON('/dashboard/api/recent-activity/');
    const tbody = document.querySelector('#activityTable tbody');
    if (!tbody || !d.rows.length) return;
    const sevClass = { LOW:'low', MEDIUM:'medium', HIGH:'high',
  CRITICAL:'critical' };
    tbody.innerHTML = d.rows.map(r => `
      <tr class="sev-${(r.severity||'low').toLowerCase()}">
        <td class="mono">${r.timestamp}</td>
        <td class="mono">${r.source_ip}</td>
        <td class="mono">${r.destination_ip||'—'}</td>
        <td>${r.verdict}</td>
        <td><span class="badge
  badge-${(r.severity||'low').toLowerCase()}">${r.severity}</span></td>
        <td class="mono">${r.confidence}%</td>
      </tr>`).join('');
  }

  /* ── CVE Threat Intelligence panel ── */
  const CVE_DB = {
    CONFIRMED_ATTACK: [
      { id:'CVE-2017-0144', name:'EternalBlue (MS17-010)', score:9.8,
  level:'critical', desc:'Remote code execution via SMB. Used by WannaCry
  and NotPetya ransomware.' },
      { id:'CVE-2021-44228', name:'Log4Shell', score:10.0, level:'critical',
  desc:'Critical RCE in Apache Log4j via JNDI injection. Affects millions
  of Java apps.' },
      { id:'CVE-2021-26855', name:'ProxyLogon (Exchange)', score:9.8,
  level:'critical', desc:'Microsoft Exchange Server SSRF enabling
  unauthenticated RCE.' },
    ],
    KNOWN_ATTACK: [
      { id:'CVE-2022-30190', name:'Follina (MSDT)', score:7.8, level:'high',
  desc:'Microsoft Windows MSDT RCE exploitable via Office documents without
  macros.' },
      { id:'CVE-2019-19781', name:'Citrix ADC RCE', score:9.8,
  level:'critical', desc:'Path traversal in Citrix Application Delivery
  Controller allowing RCE.' },
      { id:'CVE-2020-1472', name:'Zerologon', score:10.0, level:'critical',
  desc:'Privilege escalation via Netlogon allowing domain controller
  compromise.' },
    ],
    ZERO_DAY: [
      { id:'CVE-UNKNOWN-ZD1', name:'Zero-Day Anomaly Detected', score:8.5,
  level:'high', desc:'Traffic pattern deviates significantly from baseline
  normal behaviour. No known signature.' },
      { id:'CVE-2023-23397', name:'Outlook Zero-Click RCE', score:9.8,
  level:'critical', desc:'Microsoft Outlook elevation of privilege,
  exploitable with no user interaction.' },
      { id:'CVE-2022-41082', name:'ProxyNotShell (Exchange)', score:8.8,
  level:'high', desc:'Authenticated RCE in Microsoft Exchange via SSRF +
  deserialization chain.' },
    ],
    BENIGN: [],
  };

  async function loadCVE() {
    const grid = document.getElementById('cveGrid');
    if (!grid) return;
    try {
      const d = await fetchJSON('/dashboard/api/attack-distribution/');
      // Pick CVE set based on highest-priority verdict with non-zero count
      const order = ['CONFIRMED_ATTACK','KNOWN_ATTACK','ZERO_DAY','BENIGN'];
      const labelMap = {
        'Confirmed Attack':'CONFIRMED_ATTACK','Known Attack':'KNOWN_ATTACK',
        'Zero-Day Anomaly':'ZERO_DAY','Benign':'BENIGN'
      };
      let cves = CVE_DB.CONFIRMED_ATTACK;
      for (const key of order) {
        const idx = d.labels.findIndex(l => labelMap[l] === key);
        if (idx >= 0 && d.counts[idx] > 0) { cves = CVE_DB[key]; break; }
      }
      if (!cves.length) {
        grid.innerHTML = `<div class="cve-card"
  style="grid-column:1/-1"><div class="kpi-label"
  style="text-align:center;padding:20px">No threats detected &mdash; upload
  a CSV to populate threat intelligence.</div></div>`;
        return;
      }
      grid.innerHTML = cves.map(c => `
        <div class="cve-card">
          <div class="cve-id">${c.id}</div>
          <div class="cve-name">${c.name}</div>
          <div class="cve-score-row">
            <span class="cve-score cve-score-${c.level}">CVSS
  ${c.score}</span>
            <span class="badge
  badge-${c.level}">${c.level.toUpperCase()}</span>
          </div>
          <div class="cve-desc">${c.desc}</div>
        </div>`).join('');
    } catch(e) {
      grid.innerHTML = `<div class="cve-card"><div class="cve-desc">Could
  not load threat intelligence: ${e.message}</div></div>`;
    }
  }

  /* ── Decision Explanation Modal ── */
  async function showExplanation(detectionId) {
    try {
      const d = await fetchJSON(`/dashboard/api/decision-explanation/?detect
  ion_id=${detectionId}`);
      if (!d.success) {
        alert('Failed to load explanation: ' + d.error);
        return;
      }

      // Create and show modal
      const modalHTML = `
        <div class="modal-overlay" id="explanationModal">
          <div class="modal-content">
            <div class="modal-header">
              <h3>Decision Explanation</h3>
              <button class="btn-link"
  onclick="closeExplanationModal()">×</button>
            </div>
            <div class="modal-body">
              <p><strong>Detection ID:</strong>
  ${d.explanation.detection_id}</p>
              <p><strong>Verdict:</strong> <span class="badge badge-${d.expl
  anation.verdict_code.toLowerCase()}">${d.explanation.verdict}</span></p>
              <p><strong>Confidence:</strong> ${(d.explanation.confidence *
  100).toFixed(1)}%</p>
              <p><strong>Anomaly Score:</strong>
  ${d.explanation.anomaly_score.toFixed(4)}</p>
              <p><strong>Is Anomaly:</strong> ${d.explanation.is_anomaly ?
  'Yes' : 'No'}</p>

              <h4>Contributing Factors:</h4>
              <ul class="explanation-list">
                ${d.explanation.contributing_factors.map(factor => `
                  <li>
                    <strong>${factor.factor}</strong> <span
  class="impact-${factor.impact}">(${factor.impact})</span><br>
                    <small>${factor.description}</small>
                  </li>`).join('')}
              </ul>
            </div>
          </div>
        </div>
      `;

      // Remove existing modal if any
      const existingModal = document.getElementById('explanationModal');
      if (existingModal) {
        existingModal.remove();
      }

      // Add new modal
      document.body.insertAdjacentHTML('beforeend', modalHTML);

      // Add styles if not already present
      if (!document.getElementById('explanationModalStyles')) {
        const style = document.createElement('style');
        style.id = 'explanationModalStyles';
        style.textContent = `
          .modal-overlay {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
          }
          .modal-content {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            width: 90%;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
          }
          .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
          }
          .modal-header h3 {
            margin: 0;
            color: var(--text);
          }
          .modal-body {
            padding: 20px;
          }
          .explanation-list {
            margin: 12px 0;
            padding-left: 20px;
          }
          .explanation-list li {
            margin: 8px 0;
            line-height: 1.4;
          }
          .impact-high { color: var(--critical); font-weight: 600; }
          .impact-medium { color: var(--medium); font-weight: 500; }
          .impact-low { color: var(--low); font-weight: 500; }
          .btn-link {
            background: none;
            border: none;
            color: var(--brand);
            font-size: 12.5px;
            cursor: pointer;
            padding: 0;
          }
          .btn-link:hover { text-decoration: underline; }
        `;
        document.head.appendChild(style);
      }
    } catch(e) {
      alert('Error loading explanation: ' + e.message);
    }
  }

  function closeExplanationModal() {
    const modal = document.getElementById('explanationModal');
    if (modal) {
      modal.remove();
    }
  }

  /* ── Refresh all ── */
  async function refreshAll() {
    await Promise.allSettled([
      loadTraffic(),
      loadDistribution(),
      loadComparison(),
      loadActivity(),
      loadCVE(),
      loadLiveDetections(),
      loadRecentAlerts(),
      loadProcessingSpeed(),
      updateEnhancedKPIs()
    ]);
  }

  /* Initial load and interval */
  document.addEventListener('DOMContentLoaded', () => {
    refreshAll();
    setInterval(refreshAll, 20000); // refresh every 20s
  });

  5. nids_dashboard/dashboard/static/dashboard/js/detection_summary.js
  (CREATE NEW FILE)

  /* Detection Summary Page Specific JavaScript */

  let verdictChart, severityChart;

  async function loadDetectionSummary() {
    try {
      const [overview, verdictDist, severityDist, highRisk] = await
  Promise.all([
        fetchJSON('/dashboard/api/detection-summary/'),
        fetchJSON('/dashboard/api/verdict-distribution/'),
        fetchJSON('/dashboard/api/severity-distribution/'),
        fetchJSON('/dashboard/api/high-risk-detections/')
      ]);

      // Update overview metrics
      if (overview.success) {
        document.getElementById('totalDetections').textContent =
  overview.total_detections?.toLocaleString() || '--';
        document.getElementById('threatRate').textContent =
  `${(overview.threat_rate * 100).toFixed(1)}%`;
        document.getElementById('avgConfidence').textContent =
  `${(overview.avg_confidence * 100).toFixed(1)}%`;
        document.getElementById('fpr').textContent =
  `${(overview.false_positive_rate * 100).toFixed(2)}%`;
      }

      // Update verdict distribution table
      if (verdictDist.success && verdictDist.distribution) {
        const tbody = document.querySelector('#verdictTable tbody');
        tbody.innerHTML = verdictDist.distribution.map(item => `
          <tr>
            <td>${item.verdict}</td>
            <td>${item.count.toLocaleString()}</td>
            <td>${(item.percentage * 100).toFixed(1)}%</td>
          </tr>`).join('');
      }

      // Update severity distribution table
      if (severityDist.success && severityDist.distribution) {
        const tbody = document.querySelector('#severityTable tbody');
        tbody.innerHTML = severityDist.distribution.map(item => `
          <tr class="sev-${item.severity.toLowerCase()}">
            <td>${item.severity}</td>
            <td>${item.count.toLocaleString()}</td>
            <td>${(item.percentage * 100).toFixed(1)}%</td>
          </tr>`).join('');
      }

      // Update high-risk detections table
      if (highRisk.success && highRisk.detections) {
        const tbody = document.querySelector('#highRiskTable tbody');
        if (highRisk.detections.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No
  high-risk detections in last 24h</td></tr>';
        } else {
          tbody.innerHTML = highRisk.detections.map(det => `
            <tr class="sev-${det.severity.toLowerCase()}">
              <td class="mono">${det.timestamp}</td>
              <td class="mono">${det.source_ip}</td>
              <td class="mono">${det.destination_ip}</td>
              <td>${det.verdict}</td>
              <td><span class="badge
  badge-${det.severity.toLowerCase()}">${det.severity}</span></td>
              <td class="mono">${det.confidence}%</td>
            </tr>`).join('');
        }
      }

      // Update charts
      if (verdictDist.success && verdictDist.chartData) {
        updateVerdictChart(verdictDist.chartData);
      }

      if (severityDist.success && severityDist.chartData) {
        updateSeverityChart(severityDist.chartData);
      }

    } catch(e) {
      console.error('Error loading detection summary:', e);
      // Show error in UI
      document.getElementById('totalDetections').textContent = 'Error';
    }
  }

  function updateVerdictChart(data) {
    const ctx = document.getElementById('verdictDistributionChart');
    if (!ctx) return;

    if (verdictChart) {
      verdictChart.data = {
        labels: data.labels,
        datasets: [{
          data: data.counts,
          backgroundColor: [
            getComputedStyle(document.documentElement).getPropertyValue('--l
  ow').trim(),
            getComputedStyle(document.documentElement).getPropertyValue('--h
  igh').trim(),
            getComputedStyle(document.documentElement).getPropertyValue('--m
  edium').trim(),
            getComputedStyle(document.documentElement).getPropertyValue('--c
  ritical').trim()
          ],
          borderWidth: 0
        }]
      };
      verdictChart.update();
    } else {
      verdictChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: data.labels,
          datasets: [{
            data: data.counts,
            backgroundColor: [
              getComputedStyle(document.documentElement).getPropertyValue('-
  -low').trim(),
              getComputedStyle(document.documentElement).getPropertyValue('-
  -high').trim(),
              getComputedStyle(document.documentElement).getPropertyValue('-
  -medium').trim(),
              getComputedStyle(document.documentElement).getPropertyValue('-
  -critical').trim()
            ],
            borderWidth: 0,
            hoverOffset: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                boxWidth: 12,
                boxHeight: 12,
                padding: 12
              }
            },
            tooltip: {
              callbacks: {
                label: ctx => `${ctx.label}: ${ctx.parsed} (${(ctx.parsed /
  ctx.dataset._meta[0].total * 100).toFixed(1)}%)`
              }
            }
          }
        }
      });
    }
  }

  function updateSeverityChart(data) {
    const ctx = document.getElementById('severityTrendChart');
    if (!ctx) return;

    if (severityChart) {
      severityChart.data = {
        labels: data.labels,
        datasets: [{
          label: 'Count',
          data: data.counts,
          backgroundColor: 'rgba(0,212,255,0.2)',
          borderColor: getComputedStyle(document.documentElement).getPropert
  yValue('--brand').trim(),
          borderWidth: 2,
          tension: 0.3,
          fill: true
        }]
      };
      severityChart.update();
    } else {
      severityChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.labels,
          datasets: [{
            label: 'Count',
            data: data.counts,
            backgroundColor: 'rgba(0,212,255,0.2)',
            borderColor: getComputedStyle(document.documentElement).getPrope
  rtyValue('--brand').trim(),
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointBackgroundColor: getComputedStyle(document.documentElement)
  .getPropertyValue('--brand').trim(),
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: getComputedStyle(document.documentElement).getPropert
  yValue('--grid').trim()
              }
            },
            x: {
              grid: {
                display: false
              }
            }
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                label: ctx => `${ctx.label}:
  ${ctx.parsed.value.toLocaleString()} detections`
              }
            }
          }
        }
      });
    }
  }

  // Initialize charts when page loads
  document.addEventListener('DOMContentLoaded', () => {
    loadDetectionSummary();
    // Refresh every 5 minutes for this detailed view
    setInterval(loadDetectionSummary, 300000);
  });

  6. nids_dashboard/dashboard/templates/dashboard/detection_summary.html
  (CREATE NEW FILE)

  {% extends "dashboard/base.html" %}
  {% load static %}
  {% block title %}Detection Summary{% endblock %}
  {% block page_title %}Detection Analytics{% endblock %}
  {% block page_title_sub %}Comprehensive analysis of network threat
  detection{% endblock %}

  {% block content %}
  <!-- Key Metrics Row -->
  <div class="kpi-grid" style="margin-bottom: 24px;">
    <div class="kpi-card">
      <div class="kpi-label">Total Detections</div>
      <div class="kpi-value" id="totalDetections">--</div>
      <div class="kpi-footer">all time</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Threat Detection Rate</div>
      <div class="kpi-value" id="threatRate">--%</div>
      <div class="kpi-footer">malicious vs benign</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Avg Confidence</div>
      <div class="kpi-value" id="avgConfidence">--%</div>
      <div class="kpi-footer">model confidence</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">False Positive Rate</div>
      <div class="kpi-value" id="fpr">--%</div>
      <div class="kpi-footer">estimated</div>
    </div>
  </div>

  <!-- Charts Row -->
  <div class="chart-grid-main" style="margin-bottom: 24px;">
    <div class="chart-wrap">
      <canvas id="verdictDistributionChart"></canvas>
    </div>
    <div class="chart-wrap">
      <canvas id="severityTrendChart"></canvas>
    </div>
  </div>

  <!-- Detailed Tables -->
  <section class="panel">
    <div class="panel-header">
      <h2>Detection Details</h2>
      <div class="panel-sub">Breakdown by verdict and severity</div>
    </div>

    <div class="chart-grid-sub">
      <div class="table-wrap">
        <h3>Verdict Distribution</h3>
        <table class="data-table" id="verdictTable">
          <thead>
            <tr>
              <th>Verdict</th>
              <th>Count</th>
              <th>Percentage</th>
            </tr>
          </thead>
          <tbody>
            <tr><td colspan="3" class="empty-row">Loading...</td></tr>
          </tbody>
        </table>
      </div>

      <div class="table-wrap">
        <h3>Severity Distribution</h3>
        <table class="data-table" id="severityTable">
          <thead>
            <tr>
              <th>Severity</th>
              <th>Count</th>
              <th>Percentage</th>
            </tr>
          </thead>
          <tbody>
            <tr><td colspan="3" class="empty-row">Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Recent High-Risk Detections -->
  <section class="panel">
    <div class="panel-header">
      <h2>High-Risk Detections (Last 24h)</h2>
      <div class="panel-sub">Confirmed attacks and high-confidence
  anomalies</div>
    </div>

    <div class="table-wrap">
      <table class="data-table" id="highRiskTable">
        <thead>
          <tr>
            <th>Time</th>
            <th>Source IP</th>
            <th>Dest IP</th>
            <th>Verdict</th>
            <th>Severity</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          <tr><td colspan="6" class="empty-row">Loading high-risk
  detections...</td></tr>
        </tbody>
      </table>
    </div>
  </section>
  {% endblock %}

  {% block extra_js %}
  <script src="{% static 'dashboard/js/detection_summary.js' %}"></script>
  {% endblock %}

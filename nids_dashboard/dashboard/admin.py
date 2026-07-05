from django.contrib import admin

from .models import Alert, BlockedIP, Detection, EvaluationMetric, UploadBatch


@admin.register(UploadBatch)
class UploadBatchAdmin(admin.ModelAdmin):
    list_display = ("filename", "uploaded_by", "uploaded_at", "status", "row_count",
                     "benign_count", "malicious_count", "processing_ms")
    list_filter = ("status",)


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "source_ip", "destination_ip", "verdict", "severity",
                     "rf_confidence", "is_anomaly")
    list_filter = ("verdict", "severity")
    search_fields = ("source_ip", "destination_ip")


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("id", "detection", "status", "created_at", "handled_by")
    list_filter = ("status",)


@admin.register(BlockedIP)
class BlockedIPAdmin(admin.ModelAdmin):
    list_display = ("ip_address", "active", "reason", "blocked_by", "blocked_at")
    list_filter = ("active",)


@admin.register(EvaluationMetric)
class EvaluationMetricAdmin(admin.ModelAdmin):
    list_display = ("model_name", "accuracy", "precision", "recall", "f1_score",
                     "false_positive_rate", "false_negative_rate", "recorded_at")

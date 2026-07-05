import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class UploadBatch(models.Model):
    """One CSV upload/analysis run."""

    STATUS_PENDING = "PENDING"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_DONE = "DONE"
    STATUS_FAILED = "FAILED"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    uploaded_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(blank=True, default="")

    row_count = models.PositiveIntegerField(default=0)
    benign_count = models.PositiveIntegerField(default=0)
    known_attack_count = models.PositiveIntegerField(default=0)
    zero_day_count = models.PositiveIntegerField(default=0)
    confirmed_attack_count = models.PositiveIntegerField(default=0)

    processing_ms = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.filename} ({self.uploaded_at:%Y-%m-%d %H:%M})"

    @property
    def malicious_count(self):
        return self.known_attack_count + self.zero_day_count + self.confirmed_attack_count


class Detection(models.Model):
    """A single analyzed network record (one CSV row)."""

    VERDICT_BENIGN = "BENIGN"
    VERDICT_KNOWN_ATTACK = "KNOWN_ATTACK"
    VERDICT_ZERO_DAY = "ZERO_DAY"
    VERDICT_CONFIRMED_ATTACK = "CONFIRMED_ATTACK"
    VERDICT_CHOICES = [
        (VERDICT_BENIGN, "Benign"),
        (VERDICT_KNOWN_ATTACK, "Known Attack"),
        (VERDICT_ZERO_DAY, "Zero-Day Anomaly"),
        (VERDICT_CONFIRMED_ATTACK, "Confirmed Attack"),
    ]

    SEVERITY_LOW = "LOW"
    SEVERITY_MEDIUM = "MEDIUM"
    SEVERITY_HIGH = "HIGH"
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_CHOICES = [
        (SEVERITY_LOW, "Low"),
        (SEVERITY_MEDIUM, "Medium"),
        (SEVERITY_HIGH, "High"),
        (SEVERITY_CRITICAL, "Critical"),
    ]

    batch = models.ForeignKey(UploadBatch, on_delete=models.CASCADE, related_name="detections")
    row_index = models.PositiveIntegerField()
    timestamp = models.DateTimeField(default=timezone.now)

    source_ip = models.GenericIPAddressField()
    destination_ip = models.GenericIPAddressField()
    source_port = models.PositiveIntegerField()
    destination_port = models.PositiveIntegerField()

    rf_prediction = models.PositiveSmallIntegerField()  # 0 / 1
    rf_confidence = models.FloatField()
    reconstruction_error = models.FloatField()
    anomaly_threshold = models.FloatField()
    is_anomaly = models.BooleanField(default=False)

    true_label = models.PositiveSmallIntegerField(null=True, blank=True)  # if CSV had a label col

    verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["-timestamp"]),
            models.Index(fields=["verdict"]),
            models.Index(fields=["severity"]),
        ]

    def __str__(self):
        return f"{self.source_ip} -> {self.destination_ip} [{self.verdict}]"

    @property
    def is_malicious(self):
        return self.verdict != self.VERDICT_BENIGN


class Alert(models.Model):
    """Raised automatically for any non-benign Detection."""

    STATUS_ACTIVE = "ACTIVE"
    STATUS_ACKNOWLEDGED = "ACKNOWLEDGED"
    STATUS_RESOLVED = "RESOLVED"
    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_ACKNOWLEDGED, "Acknowledged"),
        (STATUS_RESOLVED, "Resolved"),
    ]

    detection = models.OneToOneField(Detection, on_delete=models.CASCADE, related_name="alert")
    created_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    handled_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    handled_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Alert #{self.pk} ({self.status})"


class BlockedIP(models.Model):
    """Dashboard-level IP block list (access-control simulation)."""

    ip_address = models.GenericIPAddressField(unique=True)
    reason = models.CharField(max_length=255, blank=True, default="")
    blocked_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    blocked_at = models.DateTimeField(default=timezone.now)
    active = models.BooleanField(default=True)
    source_alert = models.ForeignKey(
        Alert, on_delete=models.SET_NULL, null=True, blank=True, related_name="resulting_blocks"
    )

    class Meta:
        ordering = ["-blocked_at"]

    def __str__(self):
        return f"{self.ip_address} ({'active' if self.active else 'lifted'})"


class EvaluationMetric(models.Model):
    """Offline evaluation results for a trained model (from evaluation_simple.py)."""

    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    false_positive_rate = models.FloatField()
    false_negative_rate = models.FloatField()
    recorded_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["model_name"]
        get_latest_by = "recorded_at"

    def __str__(self):
        return f"{self.model_name} ({self.accuracy:.2%} acc)"

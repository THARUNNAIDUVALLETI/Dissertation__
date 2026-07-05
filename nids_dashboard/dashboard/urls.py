from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("", views.index, name="index"),

    # JSON APIs for Chart.js
    path("api/traffic-overview/", views.api_traffic_overview, name="api_traffic_overview"),
    path("api/attack-distribution/", views.api_attack_distribution, name="api_attack_distribution"),
    path("api/algorithm-comparison/", views.api_algorithm_comparison, name="api_algorithm_comparison"),
    path("api/recent-activity/", views.api_recent_activity, name="api_recent_activity"),

    # CSV upload & analysis
    path("upload/", views.upload_csv, name="upload"),
    path("upload/<uuid:batch_id>/", views.upload_result, name="upload_result"),

    # Reports / export
    path("reports/", views.reports, name="reports"),
    path("reports/export/csv/", views.export_csv, name="export_csv"),
    path("reports/export/pdf/", views.export_pdf, name="export_pdf"),

    # Alerts
    path("alerts/", views.alerts_view, name="alerts"),
    path("alerts/<int:alert_id>/acknowledge/", views.acknowledge_alert, name="acknowledge_alert"),
    path("alerts/<int:alert_id>/resolve/", views.resolve_alert, name="resolve_alert"),

    # Access control / IP blocking
    path("access-control/", views.access_control, name="access_control"),
    path("access-control/block/", views.block_ip, name="block_ip"),
    path("access-control/unblock/<int:block_id>/", views.unblock_ip, name="unblock_ip"),
]

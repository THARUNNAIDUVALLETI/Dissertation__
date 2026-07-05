"""
manage.py seed_metrics
Loads results/metrics/comparison.csv into EvaluationMetric.
Uses update_or_create so running it multiple times never creates duplicates.
"""
from pathlib import Path
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from dashboard.models import EvaluationMetric


def _find_csv(base_dir):
    candidates = [
        base_dir.parent / "results" / "metrics" / "comparison.csv",
        base_dir / "results" / "metrics" / "comparison.csv",
        Path.home() / "Desktop" / "Dissertation-" / "results" / "metrics" / "comparison.csv",
        Path.home() / "project" / "results" / "metrics" / "comparison.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


class Command(BaseCommand):
    help = "Load results/metrics/comparison.csv (no duplicates — safe to re-run)."

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default=None)

    def handle(self, *args, **options):
        from django.conf import settings
        csv_path = Path(options['path']) if options['path'] else _find_csv(Path(settings.BASE_DIR))
        if not csv_path or not csv_path.exists():
            raise CommandError("Cannot find comparison.csv. Use --path explicitly.")

        df = pd.read_csv(csv_path)
        required = {"model","accuracy","precision","recall","f1_score","false_positive_rate","false_negative_rate"}
        if missing := required - set(df.columns):
            raise CommandError(f"Missing columns: {missing}")

        created = updated = 0
        for _, row in df.iterrows():
            obj, was_created = EvaluationMetric.objects.update_or_create(
                model_name=row["model"],
                defaults={
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "false_positive_rate": row["false_positive_rate"],
                    "false_negative_rate": row["false_negative_rate"],
                },
            )
            if was_created: created += 1
            else: updated += 1

        self.stdout.write(self.style.SUCCESS(
            f"Done: {created} created, {updated} updated from {csv_path}"
        ))

from django import forms


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Network traffic file",
        help_text=(
            "Accepts .csv, .parquet, or .csv.gz files. "
            "Raw UNSW-NB15 training/testing files are cleaned automatically. "
            "Max file size: 500 MB."
        ),
    )

    def clean_csv_file(self):
        f = self.cleaned_data["csv_file"]
        allowed = (".csv", ".parquet", ".gz")
        if not any(f.name.lower().endswith(ext) for ext in allowed):
            raise forms.ValidationError(
                "Please upload a .csv, .parquet, or .csv.gz file."
            )
        max_size_mb = 500
        if f.size > max_size_mb * 1024 * 1024:
            raise forms.ValidationError(f"File too large (max {max_size_mb} MB).")
        return f

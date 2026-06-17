# Data Directory

Place the cleaned EPB input CSV here, or pass its path with `--input-csv`.

Default expected filename:

```text
merged_author_2809_cleaned_v617.csv
```

The CSV should include at least:

- `title`
- `year`
- `authors`
- `authors_full_final` or `authors_stand`
- `doi`

The author column can also be selected explicitly with `--author-col`.

Optional quality-tracking columns are used when present:

- `authors_source_final`
- `paper_needs_review`
- `paper_review_reason`

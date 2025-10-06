#!/bin/bash

# Get repo root
repo_root=$(git rev-parse --show-toplevel)
echo "Repo root: $repo_root"

# Move to repo root
cd "$repo_root" || exit 1
echo "Running from: $(pwd)"

# If DIR not set, default to current directory (entire repo)
DIR="${DIR:-.}"
echo "Scanning directory: $DIR"

max_size=$((100 * 1024 * 1024)) # 100 MB
skipped_files=()

# Collect all file paths into an array (excluding .git)
mapfile -d '' all_files < <(find "$DIR" -type f -not -path "*/.git/*" -print0)
total_files=${#all_files[@]}
processed=0
bar_width=50

# Function to update progress bar
update_progress_bar() {
    percent=$((processed * 100 / total_files))
    filled=$((processed * bar_width / total_files))
    empty=$((bar_width - filled))

    bar=$(printf "%${filled}s" | tr ' ' '#')
    bar+=$(printf "%${empty}s" | tr ' ' '-')

    printf "\r[%s] %d%% (%d/%d)" "$bar" "$percent" "$processed" "$total_files"
}

# Process files
for f in "${all_files[@]}"; do
    actual_size=$(stat -c%s "$f")

    if git check-ignore -q "$f"; then
        skipped_files+=("$f | ignored | $actual_size bytes")
    elif (( actual_size > max_size )); then
        skipped_files+=("$f | too large | $actual_size bytes")
    else
        git add "$f"
    fi

    ((processed++))
    update_progress_bar
done

echo "" # New line after progress bar

# Save skipped files
if [ ${#skipped_files[@]} -gt 0 ]; then
    printf "%s\n" "${skipped_files[@]}" > skipped_files.txt
    git add skipped_files.txt
    echo "Skipped files saved and added in skipped_files.txt"
fi

echo "Completed adding files."

# Check if skipped_files.txt exists
if [ ! -f skipped_files.txt ]; then
    echo "No skipped_files.txt found, exiting."
    exit 1
fi

# Empty or create the output file
> skipped_files_abs_paths.txt

# Resolve absolute paths
while IFS='|' read -r filepath _; do
    filepath=$(echo "$filepath" | xargs)
    abs_path=$(readlink -f "$filepath" 2>/dev/null)
    if [ -n "$abs_path" ]; then
        echo "$abs_path" >> skipped_files_abs_paths.txt
    else
        echo "Warning: Could not resolve absolute path for '$filepath'"
    fi
done < skipped_files.txt

echo "Absolute paths of skipped files saved in skipped_files_abs_paths.txt"

# add all deleted files ready for commit
git add -u :/
echo "Added all potentially deleted files ready for commit"

echo "Commit ready"
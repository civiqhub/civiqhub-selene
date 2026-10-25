#!/usr/bin/env bash
# Fails if any non-merge commit in BASE..HEAD lacks a Signed-off-by trailer
# matching its author email (Developer Certificate of Origin).
# An empty or all-zero BASE (first push of a branch) checks every commit of HEAD.
set -euo pipefail

base="${1:-}"
head="$2"
status=0

if [ -z "$base" ] || [ -z "${base//0/}" ]; then
  range="$head"
else
  range="${base}..${head}"
fi

for sha in $(git rev-list --no-merges "$range"); do
  email="$(git log -1 --format='%ae' "$sha")"
  # Case-insensitive match in pure bash: `grep -iF` crashes in Git for Windows.
  signoffs="$(git log -1 --format='%(trailers:key=Signed-off-by,valueonly)' "$sha")"
  if [[ "${signoffs,,}" != *"<${email,,}>"* ]]; then
    echo "::error::Commit ${sha} has no Signed-off-by line for ${email}. Fix with: git rebase --signoff ${base:-<base>}"
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "All commits are signed off."
fi
exit "$status"

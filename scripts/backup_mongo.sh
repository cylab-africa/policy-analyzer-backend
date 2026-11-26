#!/usr/bin/env bash
# Backs up the bundled MongoDB (the pdp-mongo container) to a compressed file and removes old copies.
# The login is read from inside the container, so no password is stored here.
#
# Run it by hand:   scripts/backup_mongo.sh
# Run it every night at 02:30 (crontab -e):
#   30 2 * * * /path/to/policy-analyzer-backend/scripts/backup_mongo.sh >> $HOME/pdp-backups/backup.log 2>&1
#
# Settings (all optional): BACKUP_DIR (default ~/pdp-backups), KEEP_DAYS (default 14),
# MONGO_CONTAINER (default pdp-mongo).
set -euo pipefail

CONTAINER="${MONGO_CONTAINER:-pdp-mongo}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/pdp-backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"

umask 077
mkdir -p "$BACKUP_DIR"
stamp="$(date +%Y-%m-%d_%H%M%S)"
tmp="$BACKUP_DIR/.mongo_$stamp.tmp"
out="$BACKUP_DIR/mongo_$stamp.archive.gz"
trap 'rm -f "$tmp"' EXIT

docker exec "$CONTAINER" sh -c '
  if [ -n "$MONGO_INITDB_ROOT_USERNAME" ]; then
    mongodump --gzip --archive -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin
  else
    mongodump --gzip --archive
  fi' > "$tmp"

# A failed dump can leave an empty file, which would look like a good backup.
if [ ! -s "$tmp" ]; then
  echo "$(date -Is) backup failed: the dump is empty" >&2
  exit 1
fi

mv "$tmp" "$out"
find "$BACKUP_DIR" -maxdepth 1 -name 'mongo_*.archive.gz' -mtime +"$KEEP_DAYS" -delete
echo "$(date -Is) backup written: $out ($(du -h "$out" | cut -f1))"

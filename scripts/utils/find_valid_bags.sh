#!/bin/bash

# The target directory (defaults to the current directory if none is provided)
SEARCH_DIR="${1:-.}"

echo "Searching for MCAP files with >= 1 minute of data in: $SEARCH_DIR"
echo "--------------------------------------------------------"

find "$SEARCH_DIR" -type f -name "*.mcap" | while read -r file; do
    # Run mcap info and pipe to awk to safely parse start and end times
    mcap info "$file" 2>/dev/null | awk -v filepath="$file" '
        /^start:/ {
            # If the format includes a datetime and unix timestamp in parentheses
            if (index($0, "(") > 0) {
                split($0, a, "(")
                split(a[2], b, ")")
                start_time = b[1]
            } else {
                # If the format is just a raw float (e.g., start: 0.000)
                start_time = $2
            }
        }
        /^end:/ {
            if (index($0, "(") > 0) {
                split($0, a, "(")
                split(a[2], b, ")")
                end_time = b[1]
            } else {
                end_time = $2
            }
        }
        END {
            # If both timestamps were successfully parsed
            if (start_time != "" && end_time != "") {
                duration = end_time - start_time
                if (duration <= 60.0) {
                    printf "%s (Duration: %.2f seconds)\n", filepath, duration
                }
            }
        }
    '
done

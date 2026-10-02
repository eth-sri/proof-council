#!/bin/sh
# Checksums from the versioned Anthropic release manifest, not a mutable installer.
set -eu
version=${1:?usage: install_claude_native.sh VERSION DESTINATION}
destination=${2:?usage: install_claude_native.sh VERSION DESTINATION}
case "$version:$(uname -s):$(uname -m)" in
    2.1.251:Linux:x86_64)
        platform=linux-x64
        checksum=fd5f10ff0eb58daec04900466b143ea98aab50abf208a422bc008eaec13f61f7 ;;
    2.1.251:Linux:aarch64|2.1.251:Linux:arm64)
        platform=linux-arm64
        checksum=65445bd4dd042079cc3fa43791b561370a05c8599e8ec47580e25a81050abbdd ;;
    *) echo "Unsupported Claude version/platform; review and pin its checksum first" >&2; exit 1 ;;
esac
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 30 --max-time 600 --retry 3 \
    "https://downloads.claude.ai/claude-code-releases/$version/$platform/claude" \
    --output "$temporary/claude"
printf '%s  %s\n' "$checksum" "$temporary/claude" | sha256sum --check --status
install -m 0755 "$temporary/claude" "$destination"

#!/usr/bin/env bash
# ECC GLM Proxy Compatibility Patch
#
# Applies custom patches to the everything-claude-code plugin for GLM proxy
# compatibility. Must be re-applied after ECC plugin updates.
#
# What this patches:
#   1. config.json          — enable observer
#   2. session-start.js     — fix stdin reading for pipe compatibility
#   3. observer-loop.sh     — structured text output prompt + parse-instincts
#   4. parse-instincts.js   — NEW file, post-processor for instinct extraction
#   5. settings.json        — add ECC_OBSERVER_* env vars
#
# Usage:
#   bash ecc-glm-proxy-patch.sh           # Apply patches
#   bash ecc-glm-proxy-patch.sh --check   # Check if patches are already applied
#   bash ecc-glm-proxy-patch.sh --undo    # Restore backups

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[PATCH]${NC} $*"; }
warn()  { echo -e "${YELLOW}[PATCH]${NC} $*"; }
error() { echo -e "${RED}[PATCH]${NC} $*"; }

# ── Find ECC plugin directory ──
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

find_ecc_root() {
  for slug in everything-claude-code ecc; do
    cache_base="$CLAUDE_DIR/plugins/cache/$slug"
    if [ -d "$cache_base" ]; then
      for org_dir in "$cache_base"/*/; do
        [ -d "$org_dir" ] || continue
        for ver_dir in "$org_dir"*/; do
          [ -d "$ver_dir" ] || continue
          if [ -f "$ver_dir/scripts/hooks/run-with-flags.js" ]; then
            echo "$ver_dir"
            return 0
          fi
        done
      done
    fi
  done

  if [ -f "$CLAUDE_DIR/scripts/hooks/run-with-flags.js" ]; then
    echo "$CLAUDE_DIR"
    return 0
  fi

  return 1
}

ECC_ROOT="$(find_ecc_root)" || {
  error "Could not find ECC plugin directory"
  exit 1
}
info "Found ECC at: $ECC_ROOT"

BACKUP_DIR="$CLAUDE_DIR/patches/backup"

backup_file() {
  local src="$1"
  if [ ! -f "$src" ]; then return; fi
  mkdir -p "$BACKUP_DIR"
  local base="$(basename "$src")"
  if [ ! -f "$BACKUP_DIR/$base" ]; then
    cp "$src" "$BACKUP_DIR/$base"
    info "Backed up $base"
  fi
}

# ── Check ──
check_patched() {
  local ok=true

  if grep -q '"enabled": true' "$ECC_ROOT/skills/continuous-learning-v2/config.json" 2>/dev/null; then
    info "config.json: observer enabled"
  else
    warn "config.json: observer NOT enabled"
    ok=false
  fi

  if grep -q 'fs.readSync(0, buf, 0, buf.length, null)' "$ECC_ROOT/scripts/hooks/session-start.js" 2>/dev/null; then
    info "session-start.js: pipe-compatible stdin"
  else
    warn "session-start.js: NOT patched"
    ok=false
  fi

  if [ -f "$ECC_ROOT/skills/continuous-learning-v2/agents/parse-instincts.js" ]; then
    info "parse-instincts.js: exists"
  else
    warn "parse-instincts.js: NOT found"
    ok=false
  fi

  if grep -q '<<<INSTINCT:' "$ECC_ROOT/skills/continuous-learning-v2/agents/observer-loop.sh" 2>/dev/null; then
    info "observer-loop.sh: new prompt"
  else
    warn "observer-loop.sh: NOT patched"
    ok=false
  fi

  if [ "$ok" = true ]; then
    info "All patches applied"
    return 0
  else
    return 1
  fi
}

# ── Undo ──
undo_patches() {
  for f in config.json session-start.js observer-loop.sh; do
    if [ -f "$BACKUP_DIR/$f" ]; then
      case "$f" in
        config.json) dest="$ECC_ROOT/skills/continuous-learning-v2/config.json" ;;
        session-start.js) dest="$ECC_ROOT/scripts/hooks/session-start.js" ;;
        observer-loop.sh) dest="$ECC_ROOT/skills/continuous-learning-v2/agents/observer-loop.sh" ;;
      esac
      cp "$BACKUP_DIR/$f" "$dest"
      info "Restored $f"
    fi
  done
  rm -f "$ECC_ROOT/skills/continuous-learning-v2/agents/parse-instincts.js"
  info "Removed parse-instincts.js"
  info "Undo complete"
}

# ── Patch 1: config.json ──
patch_config() {
  local file="$ECC_ROOT/skills/continuous-learning-v2/config.json"
  [ ! -f "$file" ] && { warn "config.json not found"; return; }
  backup_file "$file"

  if grep -q '"enabled": false' "$file"; then
    sed -i.bak 's/"enabled": false/"enabled": true/' "$file" && rm -f "$file.bak"
    info "Patched config.json: observer enabled"
  elif grep -q '"enabled": true' "$file"; then
    info "config.json already enabled"
  else
    warn "Could not find 'enabled' field in config.json"
  fi
}

# ── Patch 2: session-start.js ──
patch_session_start() {
  local file="$ECC_ROOT/scripts/hooks/session-start.js"
  [ ! -f "$file" ] && { warn "session-start.js not found"; return; }
  backup_file "$file"

  grep -q 'fs.readSync(0, buf, 0, buf.length, null)' "$file" && {
    info "session-start.js already patched"
    return
  }

  python3 << 'PYEOF'
import re, sys
filepath = sys.argv[1] if len(sys.argv) > 1 else ""
if not filepath:
    # Read from env
    import os
    filepath = os.environ.get('_PATCH_FILE', '')

with open(filepath, 'r') as f:
    content = f.read()

old = r'''  // Read stdin for session_id \(Claude Code passes it via hook JSON, not always as env var\)
  let stdinSessionId = '';
  try \{
    // run-with-flags\.js spawns this script with input: raw \(stdin JSON from Claude Code\)
    // Try to read piped stdin synchronously to extract session_id
    const fd = 0; // stdin
    const stat = fs\.fstatSync\(fd\);
    if \(stat\.isFile\(\) \|\| stat\.isFIFO\(\)\) \{
      const buf = Buffer\.alloc\(Math\.min\(stat\.size \|\| 65536, 1048576\)\);
      const bytesRead = fs\.readSync\(fd, buf, 0, buf\.length, 0\);
      const raw = buf\.toString\('utf8', 0, bytesRead\)\.trim\(\);
      if \(raw\) \{
        const parsed = JSON\.parse\(raw\);
        stdinSessionId = parsed\.session_id \|\| '';
        log\(\`\[SessionStart\] Extracted session_id from stdin: \$\{stdinSessionId \|\| '\(empty\)'\}`\);
      \}
    \}
  \} catch \(e\) \{
    // stdin not available or not JSON — fall through to env var
    log\(\`\[SessionStart\] Could not read session_id from stdin: \$\{e\.message\}`\);
  \}'''

new = '''  // Read stdin for session_id (Claude Code passes it via hook JSON, not always as env var)
  let stdinSessionId = '';
  try {
    // run-with-flags.js spawns this script with input: raw (stdin JSON from Claude Code).
    // Use null position for pipe compatibility — pipes don't support position-based reads.
    const chunks = [];
    const buf = Buffer.alloc(65536);
    let bytesRead;
    while ((bytesRead = fs.readSync(0, buf, 0, buf.length, null)) > 0) {
      chunks.push(buf.toString('utf8', 0, bytesRead));
      if (chunks.length > 16) break; // safety limit: ~1MB
    }
    const raw = chunks.join('').trim();
    if (raw) {
      const parsed = JSON.parse(raw);
      stdinSessionId = parsed.session_id || '';
      log(`[SessionStart] Extracted session_id from stdin: ${stdinSessionId || '(empty)'}`);
    }
  } catch (e) {
    // stdin not available or not JSON — fall through to env var
    log(`[SessionStart] Could not read session_id from stdin: ${e.message}`);
  }'''

if not re.search(old, content):
    print('WARN: old pattern not found in session-start.js')
    sys.exit(1)

content = re.sub(old, new, content, count=1)
with open(filepath, 'w') as f:
    f.write(content)
print('OK: session-start.js patched')
PYEOF

  export _PATCH_FILE="$file"
  python3 "$file" 2>/dev/null || python3 -c "
import re, sys
filepath = '$file'
with open(filepath, 'r') as f:
    content = f.read()

# Fallback: just check if patched
if 'fs.readSync(0, buf, 0, buf.length, null)' in content:
    print('Already patched')
    sys.exit(0)
print('WARN: Python heredoc patch failed, trying sed fallback')
sys.exit(1)
" && info "session-start.js patched" || warn "session-start.js patch failed"
}

# ── Patch 3: observer-loop.sh ──
patch_observer_loop() {
  local file="$ECC_ROOT/skills/continuous-learning-v2/agents/observer-loop.sh"
  [ ! -f "$file" ] && { warn "observer-loop.sh not found"; return; }
  backup_file "$file"

  grep -q '<<<INSTINCT:' "$file" && {
    info "observer-loop.sh already patched"
    return
  }

  # Use a standalone Python patch file to avoid heredoc escaping issues
  local patch_py="$CLAUDE_DIR/patches/_patch_observer.py"
  cat > "$patch_py" << 'PYSCRIPT'
import re, sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# Check old prompt exists
if 'IMPORTANT: You are running in non-interactive --print mode. You MUST use the Write tool' not in content:
    print('WARN: old prompt pattern not found')
    sys.exit(1)

# Replace prompt section (between cat > "$prompt_file" <<PROMPT and PROMPT)
old_prompt_match = re.search(
    r'(  cat > "\$prompt_file" <<PROMPT\n)(.*?)(\nPROMPT)',
    content, re.DOTALL
)
if not old_prompt_match:
    print('WARN: could not locate PROMPT heredoc')
    sys.exit(1)

new_prompt = """You are analyzing developer observations for project ${PROJECT_NAME}.
Read ${analysis_relpath} and identify patterns (user corrections, error resolutions, repeated workflows, tool preferences).

For each pattern with 3+ occurrences, output an instinct block using EXACTLY this format:

<<<INSTINCT:kebab-case-id.md>>>
---
id: kebab-case-id
trigger: when <specific condition>
confidence: <0.3-0.85 based on frequency: 3-5 times=0.5, 6-10=0.7, 11+=0.85>
domain: <one of: code-style, testing, git, debugging, workflow, file-patterns>
source: session-observation
scope: <project or global>
project_id: ${PROJECT_ID}
project_name: ${PROJECT_NAME}
---

# Title

## Action
<what to do, one clear sentence>

## Evidence
- Observed N times in session <id>
- Pattern: <description>
- Last observed: <date>
<<<END_INSTINCT>>>

Rules:
- Output ONLY instinct blocks — no summaries, no explanations, no tool calls
- Be conservative, only clear patterns with 3+ observations
- Use narrow, specific triggers
- Never include actual code snippets, only describe patterns
- If no qualifying patterns exist, output nothing
- If a pattern seems universal (not project-specific), set scope to global instead of project"""

content = content[:old_prompt_match.start()] + \
    old_prompt_match.group(1) + new_prompt + old_prompt_match.group(3) + \
    content[old_prompt_match.end():]

# Replace claude call section
old_call_marker = 'ECC_SKIP_OBSERVE=1 ECC_HOOK_PROFILE=minimal claude --model haiku --max-turns "$max_turns" --print \\\n    --allowedTools "Read,Write" \\\n    -p "$prompt_content" >> "$LOG_FILE" 2>&1 &'
new_call_marker = '# Save analysis output to a temp file for post-processing (GLM proxy compatibility).\n  output_file="$(mktemp "${observer_tmp_dir}/ecc-observer-output.XXXXXX")"\n\n  # Prevent observe.sh from recording this automated Haiku session as observations.\n  ECC_SKIP_OBSERVE=1 ECC_HOOK_PROFILE=minimal claude --model haiku --max-turns "$max_turns" --print \\\n    --allowedTools "Read,Write" \\\n    -p "$prompt_content" > "$output_file" 2>&1 &'

if old_call_marker in content:
    content = content.replace(old_call_marker, new_call_marker, 1)
else:
    print('WARN: old claude call pattern not found')
    sys.exit(1)

# Replace log-only output with log + parse-instincts
old_post = '''  rm -f "$analysis_file"

  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date)] Claude analysis failed (exit $exit_code)" >> "$LOG_FILE"
  fi'''

new_post = '''  rm -f "$analysis_file"

  # Append analysis output to log for debugging
  if [ -f "$output_file" ]; then
    cat "$output_file" >> "$LOG_FILE"
  fi

  # Post-process: extract instinct blocks from text output and write files.
  SCRIPT_DIR_OBS="$(cd "$(dirname "$0")" && pwd)"
  if [ -f "$output_file" ] && [ -s "$output_file" ]; then
    parse_result="$(node "${SCRIPT_DIR_OBS}/parse-instincts.js" "$output_file" "$INSTINCTS_DIR" 2>&1)" || true
    if [ -n "$parse_result" ]; then
      echo "[$(date)] $parse_result" >> "$LOG_FILE"
    fi
  fi
  rm -f "$output_file"

  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date)] Claude analysis failed (exit $exit_code)" >> "$LOG_FILE"
  fi'''

if old_post in content:
    content = content.replace(old_post, new_post, 1)
else:
    print('WARN: old post-processing pattern not found')
    sys.exit(1)

with open(filepath, 'w') as f:
    f.write(content)
print('OK: observer-loop.sh patched')
PYSCRIPT

  python3 "$patch_py" "$file" && {
    info "observer-loop.sh patched"
    rm -f "$patch_py"
  } || {
    warn "observer-loop.sh patch failed (pattern may differ in this version)"
    rm -f "$patch_py"
  }
}

# ── Patch 4: Create parse-instincts.js ──
patch_create_parse_instincts() {
  local file="$ECC_ROOT/skills/continuous-learning-v2/agents/parse-instincts.js"
  if [ -f "$file" ] && grep -q '<<<INSTINCT:' "$file"; then
    info "parse-instincts.js already exists"
    return
  fi

  cat > "$file" << 'PARSEOF'
#!/usr/bin/env node
/**
 * Post-processor for Observer analysis output.
 *
 * Extracts instinct file content from text output (using structured
 * delimiters) and writes them to the instincts directory. This bypasses
 * the need for the LLM to call Claude's Write tool, which some API
 * proxies (e.g. GLM) do not support.
 *
 * Usage:
 *   node parse-instincts.js <analysis-output-file> <instincts-dir>
 *
 * Input format expected in analysis output:
 *   <<<INSTINCT:filename.md>>>
 *   ---
 *   id: kebab-case-name
 *   ... YAML frontmatter ...
 *   ---
 *   # Title
 *   ... content ...
 *   <<<END_INSTINCT>>>
 */

'use strict';

const fs = require('fs');
const path = require('path');

function main() {
  const inputFile = process.argv[2];
  const instinctsDir = process.argv[3];

  if (!inputFile || !instinctsDir) {
    process.stderr.write('Usage: node parse-instincts.js <analysis-output-file> <instincts-dir>\n');
    process.exit(1);
  }

  if (!fs.existsSync(inputFile)) {
    process.stderr.write(`Input file not found: ${inputFile}\n`);
    process.exit(1);
  }

  const content = fs.readFileSync(inputFile, 'utf8');
  const regex = /<<<INSTINCT:(.+?)>>>\n([\s\S]*?)<<<END_INSTINCT>>>/g;
  let match;
  let count = 0;

  while ((match = regex.exec(content)) !== null) {
    const filename = match[1].trim();
    const instinctContent = match[2].trim();

    // Validate filename safety (no path traversal)
    if (filename.includes('/') || filename.includes('\\') || filename.includes('..')) {
      process.stderr.write(`Skipping unsafe filename: ${filename}\n`);
      continue;
    }

    // Must end with .md or .yaml
    if (!/\.ya?ml$|\.md$/i.test(filename)) {
      process.stderr.write(`Skipping non-markdown/yaml filename: ${filename}\n`);
      continue;
    }

    // Validate content has YAML frontmatter
    if (!instinctContent.startsWith('---')) {
      process.stderr.write(`Skipping instinct without frontmatter: ${filename}\n`);
      continue;
    }

    fs.mkdirSync(instinctsDir, { recursive: true });
    const filePath = path.join(instinctsDir, filename);
    fs.writeFileSync(filePath, instinctContent + '\n');
    process.stdout.write(`[parse-instincts] Written: ${filePath}\n`);
    count++;
  }

  if (count === 0) {
    process.stdout.write('[parse-instincts] No instinct blocks found in analysis output\n');
  } else {
    process.stdout.write(`[parse-instincts] Total instincts written: ${count}\n`);
  }
}

main();
PARSEOF

  info "Created parse-instincts.js"
}

# ── Patch 5: settings.json env vars ──
patch_settings_env() {
  local file="$CLAUDE_DIR/settings.json"
  [ ! -f "$file" ] && { warn "settings.json not found"; return; }

  grep -q 'ECC_OBSERVER_MAX_ANALYSIS_LINES' "$file" && \
  grep -q 'ECC_OBSERVER_TIMEOUT_SECONDS' "$file" && {
    info "settings.json: ECC_OBSERVER vars already set"
    return
  }

  python3 -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
if 'env' not in data:
    data['env'] = {}
data['env']['ECC_OBSERVER_MAX_ANALYSIS_LINES'] = '500'
data['env']['ECC_OBSERVER_TIMEOUT_SECONDS'] = '300'
with open('$file', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print('OK: added ECC_OBSERVER env vars')
"

  info "settings.json: added ECC_OBSERVER_* env vars"
}

# ── Main ──
case "${1:-}" in
  --check)
    check_patched
    exit $?
    ;;
  --undo)
    undo_patches
    exit 0
    ;;
  "")
    info "Applying GLM proxy compatibility patches..."
    echo ""
    patch_config
    patch_session_start
    patch_observer_loop
    patch_create_parse_instincts
    patch_settings_env
    echo ""
    info "Done. Run with --check to verify."
    info "Note: Re-run this script after ECC plugin updates."
    ;;
  *)
    echo "Usage: $0 [--check|--undo]"
    exit 1
    ;;
esac

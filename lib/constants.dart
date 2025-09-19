const List<String> severityOrder = ['low', 'medium', 'high', 'block'];

const String reviewPromptTemplate =
    '''Review the following STAGED git diff and produce:

1) A concise, GitHub-like review in Markdown:
   - High-level summary
   - Per-file sections with inline comments and line references when possible
   - Suggestions using fenced code blocks with language and minimal context
   - Label each comment with Severity: low|medium|high|block

2) A machine-readable JSON summary at the end, enclosed in a fenced block:
```json
{
  "counts": {"low": n, "medium": n, "high": n, "block": n},
  "max_severity": "low|medium|high|block",
  "block": true|false
}
```

Guidelines:
- Be precise and actionable; avoid boilerplate.
- Only refer to lines from the diff. If unsure, say so.
- Keep the Markdown review readable in a terminal.

Here is the git diff:
```diff
{diff}
```
''';

const String preCommitHookScript = r'''#!/usr/bin/env sh
echo "Running code review bot (pre-commit)..."
review-bot pre-commit
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "Pre-commit blocked by code review bot."
  exit $STATUS
fi
exit 0
''';

final RegExp jsonBlockRegExp =
    RegExp(r"```json\s*([\s\S]*?)\s*```", multiLine: true);

const String noStagedChangesMessage =
    'No staged changes found. Stage files before committing.';

const String defaultSeverity = 'low';

const String reviewHeader = '=== Code Review (model: {model}) ===';
const String reviewSubHeader =
    'Staged diff analyzed. Threshold: {failOnSeverity}{strict}';
const String maxSeverityDetected = 'Max severity detected: {maxSeverity}';
const String lineSeparator = '-----------------------------------';

import 'dart:convert';
import 'dart:io';

import 'package:code_review_bot/git.dart';
import 'package:code_review_bot/ollama_client.dart';

class ReviewResult {
  final String renderedOutput;
  final String maxSeverity;
  final bool blockCommit;

  ReviewResult(
      {required this.renderedOutput,
      required this.maxSeverity,
      required this.blockCommit});
}

class Reviewer {
  final String model;
  final String host;
  final int port;
  final bool strict;
  final String failOnSeverity; // low|medium|high|block
  final int maxChars;
  final String systemPrompt;

  Reviewer({
    required this.model,
    required this.host,
    required this.port,
    required this.strict,
    required this.failOnSeverity,
    required this.maxChars,
    required this.systemPrompt,
  });

  static const _severityOrder = ['low', 'medium', 'high', 'block'];

  static int _severityRank(String s) {
    final idx = _severityOrder.indexOf(s.toLowerCase());
    return idx < 0 ? 0 : idx;
  }

  Future<ReviewResult> reviewStagedChanges() async {
    final diff = await Git.getStagedDiff();
    if (diff.trim().isEmpty) {
      return ReviewResult(
        renderedOutput:
            'No staged changes found. Stage files before committing.',
        maxSeverity: 'low',
        blockCommit: false,
      );
    }

    final clipped = diff.length > maxChars ? diff.substring(0, maxChars) : diff;
    final prompt = _buildPrompt(clipped);

    final client = OllamaClient(host: host, port: port);
    String response;
    try {
      response = await client.chat(
          model: model, systemPrompt: systemPrompt, userPrompt: prompt);
    } catch (e) {
      // Fallback: prepend system prompt to user prompt using generate API
      try {
        final combined = 'System Instruction:\n$systemPrompt\n\nTask:\n$prompt';
        response = await client.generate(model: model, prompt: combined);
      } catch (e2) {
        final msg = e is SocketException || e2 is SocketException
            ? 'Failed to connect to Ollama at $host:$port. Is Ollama running?'
            : 'Error calling Ollama: $e2';
        return ReviewResult(
            renderedOutput: msg, maxSeverity: 'block', blockCommit: false);
      }
    }

    final parse = _parseResponse(response);
    final maxSeverity = parse.maxSeverity ?? 'low';
    final shouldBlock =
        _severityRank(maxSeverity) >= _severityRank(failOnSeverity);

    final header = StringBuffer()
      ..writeln('=== Code Review (model: $model) ===')
      ..writeln(
          'Staged diff analyzed. Threshold: $failOnSeverity${strict ? ' (strict)' : ''}')
      ..writeln('Max severity detected: $maxSeverity')
      ..writeln('-----------------------------------');

    final out = StringBuffer()
      ..write(header.toString())
      ..writeln(parse.reviewMarkdown?.trim() ?? response.trim());

    return ReviewResult(
        renderedOutput: out.toString(),
        maxSeverity: maxSeverity,
        blockCommit: shouldBlock);
  }

  String _buildPrompt(String diff) {
    return '''Review the following STAGED git diff and produce:

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
$diff
```
''';
  }

  _ParsedResponse _parseResponse(String response) {
    final jsonBlock = RegExp(r"```json\s*([\s\S]*?)```", multiLine: true)
        .firstMatch(response);
    String? reviewMarkdown;
    String? maxSeverity;
    if (jsonBlock != null) {
      final jsonText = jsonBlock.group(1);
      try {
        final data = json.decode(jsonText!) as Map<String, dynamic>;
        maxSeverity = (data['max_severity'] ?? data['severity'] ?? 'low')
            .toString()
            .toLowerCase();
      } catch (_) {
        // ignore parse errors
      }
      reviewMarkdown = response.replaceFirst(jsonBlock.group(0)!, '').trim();
    } else {
      reviewMarkdown = response.trim();
    }
    return _ParsedResponse(
        reviewMarkdown: reviewMarkdown, maxSeverity: maxSeverity);
  }

  static Future<void> installHook() async {
    final repoRoot = Directory.current;
    final gitDir = Directory('${repoRoot.path}${Platform.pathSeparator}.git');
    if (!await gitDir.exists()) {
      stderr.writeln(
          'No .git directory found at ${repoRoot.path}. Run this inside a Git repository.');
      exitCode = 2;
      return;
    }
    final hooksDir = Directory('${gitDir.path}${Platform.pathSeparator}hooks');
    if (!await hooksDir.exists()) {
      await hooksDir.create(recursive: true);
    }
    final hookFile =
        File('${hooksDir.path}${Platform.pathSeparator}pre-commit');
    final script = r'''#!/usr/bin/env sh
echo "Running code review bot (pre-commit)..."
code_review_bot pre-commit
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "Pre-commit blocked by code review bot."
  exit $STATUS
fi
exit 0
''';

    await hookFile.writeAsString(script);
    // Try to make it executable on Unix.
    if (!Platform.isWindows) {
      try {
        await Process.run('chmod', ['+x', hookFile.path]);
      } catch (_) {}
    }
    stdout.writeln('Installed pre-commit hook at ${hookFile.path}.');
    stdout.writeln(
        'Ensure the executable "code_review_bot" is in PATH (dart pub global activate).');
  }
}

class _ParsedResponse {
  final String? reviewMarkdown;
  final String? maxSeverity;
  _ParsedResponse({this.reviewMarkdown, this.maxSeverity});
}

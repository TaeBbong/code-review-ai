import 'dart:convert';
import 'dart:io';

import 'package:code_review_bot/git.dart';
import 'package:code_review_bot/ollama_client.dart';
import 'package:code_review_bot/constants.dart';

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

  static int _severityRank(String s) {
    final idx = severityOrder.indexOf(s.toLowerCase());
    return idx < 0 ? 0 : idx;
  }

  Future<ReviewResult> reviewStagedChanges() async {
    final diff = await Git.getStagedDiff();
    if (diff.trim().isEmpty) {
      return ReviewResult(
        renderedOutput: noStagedChangesMessage,
        maxSeverity: defaultSeverity,
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
    final maxSeverity = parse.maxSeverity ?? defaultSeverity;
    final shouldBlock =
        _severityRank(maxSeverity) >= _severityRank(failOnSeverity);

    final header = StringBuffer()
      ..writeln(reviewHeader.replaceFirst('{model}', model))
      ..writeln(reviewSubHeader
          .replaceFirst('{failOnSeverity}', failOnSeverity)
          .replaceFirst('{strict}', strict ? ' (strict)' : ''))
      ..writeln(maxSeverityDetected.replaceFirst('{maxSeverity}', maxSeverity))
      ..writeln(lineSeparator);

    final out = StringBuffer()
      ..write(header.toString())
      ..writeln(parse.reviewMarkdown?.trim() ?? response.trim());

    return ReviewResult(
        renderedOutput: out.toString(),
        maxSeverity: maxSeverity,
        blockCommit: shouldBlock);
  }

  String _buildPrompt(String diff) {
    return reviewPromptTemplate.replaceFirst('{diff}', diff);
  }

  _ParsedResponse _parseResponse(String response) {
    final jsonBlock = jsonBlockRegExp.firstMatch(response);
    String? reviewMarkdown;
    String? maxSeverity;
    if (jsonBlock != null) {
      final jsonText = jsonBlock.group(1);
      try {
        final data = json.decode(jsonText!) as Map<String, dynamic>;
        maxSeverity = (data['max_severity'] ?? data['severity'] ?? defaultSeverity)
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

  static Future<void> installHook({String? failOn}) async {
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

    var command = 'code_review_bot pre-commit';
    if (failOn != null) {
      command += ' --fail-on $failOn';
    }

    final script = preCommitHookScript.replaceFirst(
      'code_review_bot pre-commit',
      command,
    );

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

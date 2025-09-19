import 'dart:io';

import 'package:args/args.dart';
import 'package:code_review_bot/reviewer.dart';

void main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('model',
        abbr: 'm',
        help: 'Ollama model name',
        defaultsTo: Platform.environment['OLLAMA_MODEL'] ?? 'qwen2.5-coder:7b')
    ..addOption('host',
        help: 'Ollama host',
        defaultsTo: Platform.environment['OLLAMA_HOST'] ?? '127.0.0.1')
    ..addOption('port',
        help: 'Ollama port',
        defaultsTo: Platform.environment['OLLAMA_PORT'] ?? '11434')
    ..addOption('system-prompt',
        help: 'System prompt to set reviewer persona',
        defaultsTo: Platform.environment['OLLAMA_SYSTEM_PROMPT'] ??
            'You are an expert senior software engineer who performs thorough, constructive code reviews for junior developers. You focus on correctness, readability, security, performance, and maintainability. Be concise and actionable. Use clear severity labels.')
    ..addOption('fail-on',
        help: 'Fail commit on severity >= level (low|medium|high|block)',
        defaultsTo: 'block')
    ..addFlag('strict',
        help: 'Enable strict mode (treat high as blocking)',
        negatable: true,
        defaultsTo: false)
    ..addOption('max-chars',
        help: 'Max diff chars per review request', defaultsTo: '60000')
    ..addFlag('help', abbr: 'h', help: 'Show help', negatable: false);

  final commands = <String>{'review', 'pre-commit', 'install-hook'};

  // Executes --help option when no arguments are provided.
  if (arguments.isEmpty) {
    _printUsage(parser, commands);
    return;
  }

  ArgResults args;
  try {
    args = parser.parse(arguments);
  } catch (e) {
    stderr.writeln('Argument error: $e');
    _printUsage(parser, commands);
    exitCode = 2;
    return;
  }

  if (args['help'] == true) {
    _printUsage(parser, commands);
    return;
  }

  final rest = args.rest;
  final command = rest.isEmpty ? 'review' : rest.first;
  if (!commands.contains(command)) {
    stderr.writeln('Unknown command: $command');
    _printUsage(parser, commands);
    exitCode = 2;
    return;
  }

  final model = args['model'] as String;
  final host = args['host'] as String;
  final port = int.tryParse(args['port'] as String) ?? 11434;
  final systemPrompt = args['system-prompt'] as String;
  final strict = (args['strict'] as bool) || command == 'pre-commit';
  final maxChars = int.tryParse(args['max-chars'] as String) ?? 60000;
  var failOn = (args['fail-on'] as String).toLowerCase().trim();
  if (strict && failOn == 'block') failOn = 'high';

  switch (command) {
    case 'install-hook':
      await Reviewer.installHook(failOn: failOn);
      return;
    case 'review':
    case 'pre-commit':
      final reviewer = Reviewer(
        model: model,
        host: host,
        port: port,
        strict: strict,
        failOnSeverity: failOn,
        maxChars: maxChars,
        systemPrompt: systemPrompt,
      );
      final result = await reviewer.reviewStagedChanges();
      stdout.writeln(result.renderedOutput);
      if (result.blockCommit) {
        stderr.writeln(
            '\nCommit blocked: severity threshold met (${result.maxSeverity}).');
        exitCode = 1;
      }
      return;
  }
}

void _printUsage(ArgParser parser, Set<String> commands) {
  stdout.writeln('Code Review Bot (Dart + Ollama)');
  stdout.writeln('Usage: review-bot [options] [command]');
  stdout.writeln('Commands: ${commands.join(', ')}');
  stdout.writeln(parser.usage);
}

import 'dart:io';

class Git {
  static Future<String> getStagedDiff() async {
    // Prefer --cached; some installations prefer --staged alias
    final args = ['diff', '--cached', '--unified=3', '--no-color'];
    final result = await Process.run('git', args);
    if (result.exitCode != 0) {
      final alt = await Process.run('git', ['diff', '--staged', '--unified=3', '--no-color']);
      if (alt.exitCode != 0) {
        throw ProcessException('git', args, (result.stderr ?? alt.stderr).toString(), result.exitCode);
      }
      return (alt.stdout as String?) ?? '';
    }
    return (result.stdout as String?) ?? '';
  }
}


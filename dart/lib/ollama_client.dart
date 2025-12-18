import 'dart:convert';
import 'package:http/http.dart' as http;

class OllamaClient {
  final String host;
  final int port;

  OllamaClient({required this.host, required this.port});

  Uri get _generateUri => Uri.parse('http://$host:$port/api/generate');
  Uri get _chatUri => Uri.parse('http://$host:$port/api/chat');

  Future<String> generate(
      {required String model, required String prompt}) async {
    final body =
        jsonEncode({'model': model, 'prompt': prompt, 'stream': false});
    final resp = await http.post(_generateUri,
        headers: {'Content-Type': 'application/json'}, body: body);
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw Exception('Ollama error ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final text = data['response'] as String?;
    if (text == null) {
      throw Exception('Malformed Ollama response: missing "response"');
    }
    return text;
  }

  Future<String> chat({
    required String model,
    required String systemPrompt,
    required String userPrompt,
  }) async {
    final body = jsonEncode({
      'model': model,
      'stream': false,
      'messages': [
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': userPrompt},
      ]
    });
    final resp = await http.post(_chatUri,
        headers: {'Content-Type': 'application/json'}, body: body);
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw Exception('Ollama error ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    // Prefer chat response shape
    final message = data['message'];
    if (message is Map && message['content'] is String) {
      return message['content'] as String;
    }
    // Fallback to non-chat shape if some models/plugins return 'response'
    final text = data['response'] as String?;
    if (text != null) return text;
    throw Exception(
        'Malformed Ollama chat response: missing message.content/response');
  }
}

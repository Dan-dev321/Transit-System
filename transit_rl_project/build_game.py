"""python build_game.py [viewer_data.json] -> game_standalone.html (data embedded, double-click to play)."""
import sys
d = open(sys.argv[1] if len(sys.argv) > 1 else 'viewer_data.json', encoding='utf-8').read().replace('</', '<\\/')
h = open('game.html', encoding='utf-8').read().replace('<script id="data" type="application/json"></script>', f'<script id="data" type="application/json">{d}</script>')
open('game_standalone.html', 'w', encoding='utf-8').write(h); print('wrote game_standalone.html', len(h) // 1024, 'KB')

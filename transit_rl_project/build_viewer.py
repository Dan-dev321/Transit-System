"""python build_viewer.py [viewer_data.json] -> viewer_standalone.html (data embedded, double-click to open)."""
import sys
d = open(sys.argv[1] if len(sys.argv) > 1 else 'viewer_data.json', encoding='utf-8').read().replace('</', '<\\/')
h = open('viewer.html', encoding='utf-8').read().replace('<script id="data" type="application/json"></script>', f'<script id="data" type="application/json">{d}</script>')
open('viewer_standalone.html', 'w', encoding='utf-8').write(h); print('wrote viewer_standalone.html', len(h) // 1024, 'KB')
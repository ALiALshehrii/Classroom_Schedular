import os
import re

color_map = {
    '#0d1b2a': '#ffffff',
    '#112240': '#f8f9fa',
    '#1e3a5c': '#dee2e6',
    '#ccd6f6': '#212529',
    '#e6f1ff': '#0056b3',
    '#8892b0': '#6c757d',
    'rgba(255, 255, 255,': 'rgba(0, 0, 0,',
    'rgba(255,255,255,': 'rgba(0,0,0,'
}

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for old, new in color_map.items():
        content = content.replace(old, new)
        # Also handle uppercase hex
        if old.startswith('#'):
            content = content.replace(old.upper(), new)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")

# Update templates
for d, _, files in os.walk('templates'):
    for f in files:
        if f.endswith('.html'):
            process_file(os.path.join(d, f))

# Also update static/style.css
process_file('static/style.css')

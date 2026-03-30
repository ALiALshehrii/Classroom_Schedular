import re

with open('static/style.css', 'r') as f:
    css = f.read()

# Replace root variables
new_root = """:root {
  --primary: #0056b3;
  --primary-light: #e6f0fa;
  --accent: #007bff;
  --accent-2: #0056b3;
  --success: #28a745;
  --danger: #dc3545;
  --warning: #ffc107;
  --info: #17a2b8;
  --dark-bg: #f4f7f6;
  --card-bg: #ffffff;
  --card-border: #dee2e6;
  --text-light: #212529;
  --text-muted: #6c757d;
  --white: #0056b3;
}"""

css = re.sub(r':root\s*\{[^}]+\}', new_root, css)

# Update background of navbar
css = re.sub(r'background:\s*linear-gradient\([^)]+\);', 'background: #ffffff;', css)
# Update text colors in components where white is expected to be actual white
css = css.replace('color: var(--white) !important;', 'color: var(--primary) !important;')

css = css.replace('background: #0d1b2a;', 'background: #ffffff;')

with open('static/style.css', 'w') as f:
    f.write(css)

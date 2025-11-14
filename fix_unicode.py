#!/usr/bin/env python3
"""Helper script to replace Unicode symbols with ASCII-safe alternatives"""

# Read the file
with open('test_pipe.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all Unicode symbols with ASCII-safe alternatives
replacements = [
    ('\u2713', '[OK]'),      # ✓
    ('\u2717', '[ERROR]'),   # ✗
    ('\u26a0', '[WARN]'),    # ⚠
    ('\u2298', '[SKIP]'),    # ⊘
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open('test_pipe.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully replaced all Unicode symbols with ASCII-safe alternatives')

#!/usr/bin/env python3
# Release management script for ARKHE Framework
import argparse, re, sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent

def get_current_version():
    init_file = project_root / 'src' / 'math_research' / '__init__.py'
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
    raise ValueError('Could not find __version__')

def bump_version(current, bump_type):
    parts = current.split('.')
    major, minor, patch = map(int, parts)
    if bump_type == 'major':
        return f'{major+1}.0.0'
    elif bump_type == 'minor':
        return f'{major}.{minor+1}.0'
    elif bump_type == 'patch':
        return f'{major}.{minor}.{patch+1}'
    raise ValueError(f'Invalid bump type: {bump_type}')

def update_version_in_file(file_path, old_version, new_version):
    if not file_path.exists():
        return False
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_version not in content:
        return False
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content.replace(old_version, new_version))
    return True

def main():
    parser = argparse.ArgumentParser(description='ARKHE Framework Release Management')
    parser.add_argument('--version', type=str, help='Set specific version')
    parser.add_argument('--bump', choices=['major', 'minor', 'patch'], help='Bump version type')
    parser.add_argument('--tag-only', action='store_true', help='Only create git tag')
    parser.add_argument('--skip-validation', action='store_true', help='Skip CHANGELOG validation')
    parser.add_argument('--skip-tag', action='store_true', help='Skip git tag creation')
    parser.add_argument('--message', type=str, help='Custom tag message')
    args = parser.parse_args()
    
    current_version = get_current_version()
    print(f'📦 Current version: {current_version}')
    
    if args.version:
        new_version = args.version
    elif args.bump:
        new_version = bump_version(current_version, args.bump)
    elif args.tag_only:
        new_version = current_version
    else:
        print('❌ Error: Must specify --version, --bump, or --tag-only')
        parser.print_help()
        sys.exit(1)
    
    print(f'🚀 New version: {new_version}')
    
    if not args.tag_only:
        files_to_update = [
            project_root / 'src' / 'math_research' / '__init__.py',
            project_root / 'src' / 'apps' / 'cli' / '__init__.py',
        ]
        updated = []
        for file_path in files_to_update:
            if update_version_in_file(file_path, current_version, new_version):
                updated.append(file_path)
        if updated:
            print(f'✅ Updated {len(updated)} file(s)')
    
    if not args.skip_tag:
        tag_name = f'v{new_version}'
        tag_message = args.message or f'Release version {new_version}'
        result = subprocess.run(['git', 'tag', '-a', tag_name, '-m', tag_message], cwd=project_root)
        if result.returncode == 0:
            print(f'✅ Created git tag: {tag_name}')
            print(f'\n📋 Next steps:')
            if not args.tag_only:
                print(f'   1. Commit: git add . && git commit -m \"chore: bump version to {new_version}\"')
            print(f'   2. Push tag: git push origin {tag_name}')

if __name__ == '__main__':
    main()

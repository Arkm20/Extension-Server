import os
import re

# 1. Find all files, excluding .git and __pycache__
def get_all_files(base_dir):
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        # Exclude .git and __pycache__
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__']]
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, base_dir)
            # Convert to web root-relative path
            web_path = '/' + rel_path.replace('\\', '/').replace(' ', '%20')
            file_list.append(web_path)
    return sorted(file_list)

# 2. Update PRECACHE_URLS in sw.js
def update_precache_urls(sw_path, new_urls):
    with open(sw_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Regex to find the PRECACHE_URLS array
    pattern = re.compile(r'(const PRECACHE_URLS = \[)(.*?)(\];)', re.DOTALL)
    # Format new URLs
    url_lines = [f"  '{url}'," for url in new_urls]
    new_array = '\n' + '\n'.join(url_lines) + '\n'
    new_content = pattern.sub(r"\\1" + new_array + r"\\3", content)
    with open(sw_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sw_path = os.path.join(base_dir, 'sw.js')
    files = get_all_files(base_dir)
    # Optionally filter out sw.js itself if you don't want to cache it
    files = [f for f in files if not f.endswith('/sw.js')]
    update_precache_urls(sw_path, files)
    print(f'Updated PRECACHE_URLS in {sw_path} with {len(files)} files.')

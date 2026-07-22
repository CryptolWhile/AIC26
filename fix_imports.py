import os

backend_dir = "/Users/lallannkhann/Documents/AIC26/backend"

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "backend.src" in content:
            new_content = content.replace("backend.src", "src")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated imports in {filepath}")
    except Exception as e:
        print(f"Failed processing {filepath}: {e}")

for root, dirs, files in os.walk(backend_dir):
    for name in files:
        if name.endswith('.py') or name.endswith('.ipynb'):
            process_file(os.path.join(root, name))

print("Import correction completed.")
